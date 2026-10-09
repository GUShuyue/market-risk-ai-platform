"""
LangGraph Orchestrator — 多Agent工作流编排

工作流图:
  collector → transcriber → analyze_episodes → reporter

每集独立分析:
  - Transcriber 转录完成后，按集拆分转录文本
  - analyze_episodes 对每集独立运行 Sentiment → Analyst → CrossMarket
  - 长文本通过 Map-Reduce 分块处理，避免截断

Map-Reduce 分块策略:
  - 将全文按句边界切分为重叠块 (chunk_size=4000, overlap=500)
  - 每块独立运行 Sentiment + Analyst
  - 汇总: 情感加权平均、实体/主题去重合并
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional

from langgraph.graph import StateGraph, END

from ..agents.collector import CollectorAgent
from ..agents.transcriber import TranscriberAgent
from ..agents.retriever import RetrieverAgent
from ..agents.sentiment import SentimentAgent
from ..agents.analyst import AnalystAgent
from ..agents.crossmarket import CrossMarketAgent
from ..agents.reporter import ReporterAgent
from ..config import get_config
from ..models.messages import (
    AgentStatus, EpisodeAnalysis, RetrieverOutput, SentimentOutput,
    AnalystOutput, CrossMarketOutput, ReporterInput, Sentiment,
)
from .state import OrchestratorState, create_initial_state

logger = logging.getLogger(__name__)

# ================================================================
# Map-Reduce 分块参数
# ================================================================
CHUNK_SIZE = 4000       # 每块目标字符数（英文约 2000 tokens）
CHUNK_OVERLAP = 500     # 块间重叠字符数，保持上下文连续性
MIN_CHUNK_SIZE = 500    # 小于此值不切分，直接全文处理

# ================================================================
# 并发基础设施：共享 Agent 实例 + 全局 LLM 并发信号量
#
# - 共享实例: Agent 无状态，可跨线程复用；FinBERT 模型只加载一次
#   （旧实现每块新建 SentimentAgent，导致 FinBERT 每块重复加载）
# - 信号量: 块级(集内)与集级(集间)并发共用一个上限，
#   无论线程池如何嵌套，同时在途的 LLM 调用总数 ≤ llm_concurrency
# ================================================================
_shared_agents: dict = {}
_shared_lock = threading.Lock()
_llm_semaphore: threading.BoundedSemaphore | None = None


def _get_shared_agent(key: str, factory):
    agent = _shared_agents.get(key)
    if agent is None:
        with _shared_lock:
            agent = _shared_agents.get(key)
            if agent is None:
                agent = factory()
                _shared_agents[key] = agent
    return agent


def _get_llm_semaphore() -> threading.BoundedSemaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        with _shared_lock:
            if _llm_semaphore is None:
                _llm_semaphore = threading.BoundedSemaphore(
                    max(1, get_config().llm_concurrency)
                )
    return _llm_semaphore


# ================================================================
# 分块工具 & 聚合函数
# ================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    将长文本按句边界切分为重叠块

    策略:
      - 按句号/问号/感叹号分段，保持句子完整性
      - 相邻块之间有 overlap 字符的重叠，维持上下文连续性
      - 短文本 (< MIN_CHUNK_SIZE) 不切分
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    # 按句边界拆分
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(current_chunk) + len(sent) + 1 <= chunk_size:
            current_chunk = (current_chunk + " " + sent).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)

            # 新块从上一块的末尾 overlap 字符开始
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                # 找到 overlap 文本中的第一个完整句子起点
                boundary = overlap_text.find(" ")
                if boundary > 0:
                    current_chunk = overlap_text[boundary:].strip() + " " + sent
                else:
                    current_chunk = sent
            else:
                current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk)

    # 聚合过短的块
    merged = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_CHUNK_SIZE:
            if len(merged[-1]) + len(chunk) < chunk_size * 1.5:
                merged[-1] = merged[-1] + " " + chunk
                continue
        merged.append(chunk)

    logger.info(
        f"  分块: {len(text)} 字符 → {len(merged)} 块 "
        f"(avg {sum(len(c) for c in merged)//max(len(merged),1)} 字符/块)"
    )
    return merged


def _aggregate_sentiment_results(
    results: list[SentimentOutput], chunk_texts: list[str]
) -> SentimentOutput:
    """
    汇聚多块情感分析结果

    策略:
      - FinBERT 概率: 按块长度加权平均
      - 情感标签: 多数投票
      - 综合分数: 加权平均
      - 正负面因素: 合并去重
    """
    if len(results) == 1:
        r = results[0]
        r.finbert_chunk_coverage = 1.0 if r.finbert_scores else 0.0
        r.llm_chunk_coverage = 1.0 if (r.engines_used and "llm" in r.engines_used) else 0.0
        return r

    # 块权重（按字符数）
    weights = [len(t) for t in chunk_texts]
    total_weight = sum(weights)

    # 加权平均 FinBERT 概率
    finbert_pos = 0.0
    finbert_neg = 0.0
    finbert_neu = 0.0
    sentiment_score = 0.0
    confidence = 0.0
    has_finbert = False

    sentiment_labels = []
    all_positive_factors = []
    all_negative_factors = []
    all_hedging = []
    divergence_rates = []
    finbert_cov = sum(1 for r in results if r.finbert_scores) / len(results)
    llm_cov = sum(1 for r in results if r.engines_used and "llm" in r.engines_used) / len(results)

    for i, result in enumerate(results):
        w = weights[i] / total_weight

        if result.finbert_scores:
            finbert_pos += result.finbert_scores.get("positive", 0) * w
            finbert_neg += result.finbert_scores.get("negative", 0) * w
            finbert_neu += result.finbert_scores.get("neutral", 0) * w
            has_finbert = True

        sentiment_score += result.sentiment_score * w
        confidence += result.confidence * w
        sentiment_labels.append(result.overall_sentiment.value)
        if result.engine_divergence_rate is not None:
            divergence_rates.append(result.engine_divergence_rate)

        if result.key_positive_factors:
            all_positive_factors.extend(result.key_positive_factors)
        if result.key_negative_factors:
            all_negative_factors.extend(result.key_negative_factors)
        if result.hedging_language_detected:
            all_hedging.append(result.hedging_language_detected)

    # 多数投票决定情感标签
    label_counts = Counter(sentiment_labels)
    majority_label = label_counts.most_common(1)[0][0]
    label_map = {"bullish": Sentiment.BULLISH, "bearish": Sentiment.BEARISH, "neutral": Sentiment.NEUTRAL}
    voted_sentiment = label_map.get(majority_label, Sentiment.NEUTRAL)

    # 计算共识度（所有块的情感一致性）
    consensus = label_counts[majority_label] / len(sentiment_labels)

    # 冲突检测: 投票标签与加权分数方向不一致时, 以分数为准重判, 共识度减半
    if sentiment_score > 0.15:
        scored_sentiment = Sentiment.BULLISH
    elif sentiment_score < -0.15:
        scored_sentiment = Sentiment.BEARISH
    else:
        scored_sentiment = Sentiment.NEUTRAL

    if voted_sentiment != scored_sentiment:
        logger.warning(
            "聚合冲突: 投票标签=%s 与加权分数=%.4f (%s) 不一致, 以分数为准",
            majority_label, sentiment_score, scored_sentiment.value,
        )
        voted_sentiment = scored_sentiment
        consensus = consensus / 2

    # 合并并去重因素
    seen_pos = set()
    unique_positive = []
    for f in all_positive_factors:
        if f not in seen_pos:
            seen_pos.add(f)
            unique_positive.append(f)

    seen_neg = set()
    unique_negative = []
    for f in all_negative_factors:
        if f not in seen_neg:
            seen_neg.add(f)
            unique_negative.append(f)

    aggregated = SentimentOutput(
        agent_status=AgentStatus.SUCCESS,
        episode_id=results[0].episode_id,
        overall_sentiment=voted_sentiment,
        sentiment_score=round(sentiment_score, 4),
        confidence=round(confidence, 4),
        key_positive_factors=unique_positive[:10],
        key_negative_factors=unique_negative[:10],
        hedging_language_detected=any(all_hedging),
        finbert_scores=(
            {"positive": round(finbert_pos, 4), "negative": round(finbert_neg, 4), "neutral": round(finbert_neu, 4)}
            if has_finbert else {}
        ),
        chunk_consensus=round(consensus, 2),
        chunk_count=len(results),
        engine_divergence_rate=(
            round(sum(divergence_rates) / len(divergence_rates), 2)
            if divergence_rates else None
        ),
        engines_used=(
            "llm+finbert" if llm_cov == 1.0 and finbert_cov == 1.0
            else "mixed" if (0 < llm_cov < 1.0 or 0 < finbert_cov < 1.0)
            else "llm" if llm_cov == 1.0
            else "finbert" if finbert_cov == 1.0
            else "none"
        ),
        finbert_chunk_coverage=round(finbert_cov, 2),
        llm_chunk_coverage=round(llm_cov, 2),
    )
    return aggregated


def _aggregate_analyst_results(
    results: list[AnalystOutput]
) -> AnalystOutput:
    """
    汇聚多块分析师结果

    策略:
      - 实体: 按名称去重合并
      - 主题: 去重合并
      - 中国关联度: 取最大值
      - 逻辑链: 合并
    """
    if len(results) == 1:
        return results[0]

    # 合并实体 (Entity 对象按 name 去重)
    seen_entities = {}
    for r in results:
        for ent in (r.entities or []):
            key = ent.name.lower().strip()
            if key and key not in seen_entities:
                seen_entities[key] = ent

    # 合并宏观主题
    seen_topics = set()
    all_topics = []
    for r in results:
        for t in (r.macro_topics or []):
            key = t.lower().strip()
            if key and key not in seen_topics:
                seen_topics.add(key)
                all_topics.append(t)

    # 中国关联度: 取最大值
    china_relevance = max(
        (r.china_relevance or 0.0 for r in results),
        default=0.0
    )

    # 合并逻辑链
    logic_chains = []
    for r in results:
        if r.logic_chain and r.logic_chain not in logic_chains:
            logic_chains.append(r.logic_chain)

    combined_logic = " | ".join(logic_chains) if logic_chains else ""

    # 摘要: 合并所有非空摘要
    summary_en = " ".join(
        r.summary_en for r in results if r.summary_en
    )
    summary_cn = " ".join(
        r.summary_cn for r in results if r.summary_cn
    )

    return AnalystOutput(
        agent_status=AgentStatus.SUCCESS,
        entities=list(seen_entities.values()),
        macro_topics=all_topics,
        china_relevance=round(china_relevance, 4),
        logic_chain=combined_logic,
        summary_cn=summary_cn,
        summary_en=summary_en,
        chunk_count=len(results),
    )


# ================================================================
# Agent 节点函数
# ================================================================

def collector_node(state: OrchestratorState) -> OrchestratorState:
    """Collector Agent 节点"""
    logger.info("=" * 50)
    logger.info("🚀 Stage: Collector Agent")

    config = get_config()
    agent = CollectorAgent()
    result = agent.execute({
        "search_query": state.get("search_query", ""),
        "max_results": state.get("max_results", 10),
        "podcast_names": state.get("podcast_names") or config.target_podcasts,
        "episodes_per_podcast": state.get("episodes_per_podcast") or config.episodes_per_podcast,
    })

    state["collector_output"] = result
    state["current_stage"] = "collector"

    if result.agent_status == AgentStatus.FAILED:
        state["errors"].append(f"Collector: {result.error_message}")

    logger.info(f"  → 找到 {result.total_found} 集, 过滤后 {result.filtered_count} 集")
    return state


def transcriber_node(state: OrchestratorState) -> OrchestratorState:
    """Transcriber Agent 节点 — 转录所有集并记录每集独立文本"""
    logger.info("-" * 40)
    logger.info("🎙️ Stage: Transcriber Agent")

    if state.get("skip_transcription"):
        logger.info("  → 跳过转录 (skip_transcription=True)")
        return state

    collector_output = state.get("collector_output")
    if not collector_output or not collector_output.episodes:
        state["errors"].append("Transcriber: 无播客数据")
        return state

    agent = TranscriberAgent()
    result = agent.execute(collector_output)
    state["transcriber_output"] = result
    state["current_stage"] = "transcriber"

    if result.agent_status == AgentStatus.FAILED:
        state["errors"].append(f"Transcriber: {result.error_message}")

    # 使用 Transcriber 返回的每集独立转录结果
    episode_transcripts = []
    if result.episode_transcriptions:
        for et in result.episode_transcriptions:
            episode_transcripts.append({
                "episode_id": et.episode_id,
                "podcast_title": et.podcast_title,
                "episode_title": et.episode_title,
                "text": et.full_text,
                "description": et.description,
            })
    else:
        # 回退：用 description
        for ep in collector_output.episodes:
            episode_transcripts.append({
                "episode_id": ep.episode_id,
                "podcast_title": ep.podcast_title,
                "episode_title": ep.episode_title,
                "text": ep.description or "",
                "description": ep.description or "",
            })

    state["episode_transcripts"] = episode_transcripts
    logger.info(f"  → 记录 {len(episode_transcripts)} 集独立转录文本")

    return state


def analyze_episodes_node(state: OrchestratorState) -> OrchestratorState:
    """
    单集分析节点 — 对每集独立运行 Sentiment → Analyst → CrossMarket

    这是核心改造：每集单独分析，保留各自独特观点

    并发: 集间并行 (episode_concurrency)，集内分块并行 (llm_concurrency)，
    两者共享全局 LLM 信号量，总并发受 llm_concurrency 封顶
    """
    logger.info("-" * 40)
    logger.info("🔬 Stage: Per-Episode Analysis")

    episode_transcripts = state.get("episode_transcripts", [])
    collector_output = state.get("collector_output")

    if not episode_transcripts and collector_output and collector_output.episodes:
        # 跳过转录时，用 description 构造
        episode_transcripts = [
            {
                "episode_id": ep.episode_id,
                "podcast_title": ep.podcast_title,
                "episode_title": ep.episode_title,
                "text": ep.description or "",
                "description": ep.description or "",
            }
            for ep in collector_output.episodes
        ]

    if not episode_transcripts:
        state["errors"].append("Analyze: 无转录数据")
        return state

    total = len(episode_transcripts)
    episode_concurrency = max(1, get_config().episode_concurrency)

    if total > 1 and episode_concurrency > 1:
        logger.info(
            f"  集级并发: {min(episode_concurrency, total)}/{total} 集同时分析 "
            f"(LLM 全局并发上限 {get_config().llm_concurrency})"
        )
        with ThreadPoolExecutor(max_workers=min(episode_concurrency, total)) as pool:
            futures = {
                pool.submit(_analyze_one_episode, i, total, et): i
                for i, et in enumerate(episode_transcripts)
            }
            indexed_results = []
            for fut in as_completed(futures):
                idx = futures[fut]
                result = fut.result()
                if result is not None:
                    indexed_results.append((idx, result))
        # 保持与输入顺序一致，报告展示稳定
        indexed_results.sort(key=lambda x: x[0])
        episode_analyses = [r for _, r in indexed_results]
    else:
        episode_analyses = []
        for i, et in enumerate(episode_transcripts):
            result = _analyze_one_episode(i, total, et)
            if result is not None:
                episode_analyses.append(result)

    state["episode_analyses"] = episode_analyses
    state["current_stage"] = "analyze"

    # 同时更新旧版字段（兼容性）
    if episode_analyses:
        first = episode_analyses[0]
        state["sentiment_output"] = first.sentiment_output
        state["analyst_output"] = first.analyst_output
        state["crossmarket_output"] = first.crossmarket_output

    logger.info(f"  → 完成 {len(episode_analyses)} 集分析")
    return state


def _analyze_one_episode(i: int, total: int, et: dict):
    """分析单集: 分块 → Sentiment(块并发) → Analyst(块并发) → CrossMarket"""
    episode_id = et.get("episode_id", "")
    episode_title = et.get("episode_title", "")[:60]
    podcast_title = et.get("podcast_title", "")
    text = et.get("text", "") or et.get("description", "")

    logger.info(f"  [{i+1}/{total}] {episode_title}")

    if not text.strip():
        logger.warning(f"    → 无文本，跳过")
        return None

    start = time.time()

    # Map-Reduce 分块: 长文本切分为重叠块，避免信息丢失
    chunks = chunk_text(text)
    logger.info(f"    原文 {len(text)} 字符 → {len(chunks)} 块")

    # --- Sentiment (分块处理) ---
    sentiment_result = _run_sentiment_chunked(chunks, episode_id)

    # --- Analyst (分块处理) ---
    analyst_result = _run_analyst_chunked(chunks, sentiment_result)

    # --- CrossMarket (单次，基于聚合结果) ---
    crossmarket_result = _run_crossmarket(analyst_result)

    logger.info(
        f"    [{i+1}/{total}] {episode_title} 完成 ({time.time()-start:.1f}s) → "
        f"情感: {sentiment_result.overall_sentiment.value} "
        f"({sentiment_result.sentiment_score:+.2f}), "
        f"板块: {len(crossmarket_result.a_share_impacts)} 个"
    )

    return EpisodeAnalysis(
        episode_id=episode_id,
        podcast_title=podcast_title,
        episode_title=et.get("episode_title", ""),
        full_text=text,  # 保存完整文本
        description=et.get("description", ""),
        sentiment_output=sentiment_result,
        analyst_output=analyst_result,
        crossmarket_output=crossmarket_result,
    )


def _run_sentiment(text: str, episode_id: str):
    """运行 Sentiment Agent（单块，线程池 worker 内调用，共享 Agent 实例）"""
    retriever_input = RetrieverOutput(
        agent_status=AgentStatus.SKIPPED,
        current_text=text,
        episode_id=episode_id,
    )
    agent = _get_shared_agent("sentiment", SentimentAgent)
    try:
        with _get_llm_semaphore():
            return agent.execute(retriever_input)
    except Exception as e:
        logger.warning(f"    Sentiment 失败: {e}")
        return SentimentOutput(agent_status=AgentStatus.FAILED, error_message=str(e))


def _run_sentiment_chunked(chunks: list[str], episode_id: str) -> SentimentOutput:
    """
    Map-Reduce: 对每块独立运行 Sentiment，然后聚合

    - 短文本（1块）: 直接运行，不走聚合
    - 长文本（多块）: 每块并发分析（线程池）→ 加权平均聚合
    """
    if len(chunks) == 1:
        return _run_sentiment(chunks[0], episode_id)

    config = get_config()
    max_workers = max(1, min(config.llm_concurrency, len(chunks)))
    logger.info(f"    [Sentiment] 分块并发处理: {len(chunks)} 块 x {max_workers} 并发")
    start = time.time()
    results: list = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_sentiment, chunk, episode_id): i
            for i, chunk in enumerate(chunks)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            logger.info(
                f"      [Sentiment] 块 [{i+1}/{len(chunks)}] 完成 "
                f"({done}/{len(chunks)}, 累计 {time.time()-start:.1f}s)"
            )

    aggregated = _aggregate_sentiment_results(results, chunks)
    logger.info(
        f"    [Sentiment] 聚合: {aggregated.overall_sentiment.value} "
        f"({aggregated.sentiment_score:+.2f}), 共识度 {aggregated.chunk_consensus:.0%} "
        f"(总耗时 {time.time()-start:.1f}s)"
    )
    return aggregated


def _run_analyst(sentiment_result):
    """运行 Analyst Agent（单块，线程池 worker 内调用，共享 Agent 实例）"""
    agent = _get_shared_agent("analyst", AnalystAgent)
    try:
        with _get_llm_semaphore():
            return agent.execute(sentiment_result)
    except Exception as e:
        logger.warning(f"    Analyst 失败: {e}")
        return AnalystOutput(agent_status=AgentStatus.FAILED, error_message=str(e))


def _run_analyst_chunked(
    chunks: list[str], sentiment_aggregated: SentimentOutput
) -> AnalystOutput:
    """
    Map-Reduce: 对每块独立运行 Analyst，然后聚合

    每块使用各自的文本内容 + 聚合后的情感作为输入上下文
    - 短文本（1块）: 直接运行
    - 长文本（多块）: 每块并发分析（线程池）→ 实体/主题去重合并
    """
    if len(chunks) == 1:
        return _run_analyst(sentiment_aggregated)

    config = get_config()
    max_workers = max(1, min(config.llm_concurrency, len(chunks)))
    logger.info(f"    [Analyst] 分块并发处理: {len(chunks)} 块 x {max_workers} 并发")
    start = time.time()

    # 为每块构造独立的情感输出（使用聚合情感标签，但保留块文本）
    chunk_sentiments = [
        SentimentOutput(
            agent_status=AgentStatus.SUCCESS,
            overall_sentiment=sentiment_aggregated.overall_sentiment,
            sentiment_score=sentiment_aggregated.sentiment_score,
            confidence=sentiment_aggregated.confidence,
            # 把块原文作为显式原文传入 Analyst（语义正确，不再借道 llm_sentiment_analysis）
            source_text=chunk,
        )
        for chunk in chunks
    ]

    results: list = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_analyst, cs): i
            for i, cs in enumerate(chunk_sentiments)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            logger.info(
                f"      [Analyst] 块 [{i+1}/{len(chunks)}] 完成 "
                f"({done}/{len(chunks)}, 累计 {time.time()-start:.1f}s)"
            )

    aggregated = _aggregate_analyst_results(results)
    logger.info(
        f"    [Analyst] 聚合: {len(aggregated.entities)} 实体, "
        f"{len(aggregated.macro_topics)} 主题, 中国关联度 {aggregated.china_relevance:.2f} "
        f"(总耗时 {time.time()-start:.1f}s)"
    )
    return aggregated


def _run_crossmarket(analyst_result):
    """运行 CrossMarket Agent（共享 Agent 实例）"""
    agent = _get_shared_agent("crossmarket", CrossMarketAgent)
    try:
        with _get_llm_semaphore():
            return agent.execute(analyst_result)
    except Exception as e:
        logger.warning(f"    CrossMarket 失败: {e}")
        return CrossMarketOutput(agent_status=AgentStatus.FAILED, error_message=str(e))


def retriever_node(state: OrchestratorState) -> OrchestratorState:
    """Retriever Agent 节点（保留兼容）"""
    logger.info("-" * 40)
    logger.info("🔍 Stage: Retriever Agent")
    if state.get("skip_retrieval") or state.get("skip_transcription"):
        logger.info("  → 跳过RAG检索")
        return state
    transcriber_output = state.get("transcriber_output")
    if not transcriber_output:
        return state
    agent = RetrieverAgent()
    result = agent.execute(transcriber_output)
    state["retriever_output"] = result
    state["current_stage"] = "retriever"
    return state


def reporter_node(state: OrchestratorState) -> OrchestratorState:
    """Reporter Agent 节点 — 汇总多集分析结果"""
    logger.info("-" * 40)
    logger.info("📊 Stage: Reporter Agent")

    from ..models.messages import ReporterInput

    episode_analyses = state.get("episode_analyses", [])
    collector_output = state.get("collector_output")

    # 构建 ReporterInput
    if episode_analyses:
        reporter_input = ReporterInput(
            episode_analyses=episode_analyses,
        )
    else:
        # 兼容旧版：单集模式
        transcriber_output = state.get("transcriber_output")
        sentiment_output = state.get("sentiment_output")
        analyst_output = state.get("analyst_output")
        crossmarket_output = state.get("crossmarket_output")

        podcast_title = ""
        episode_title = ""
        episode_id = ""
        full_text = ""

        if transcriber_output:
            podcast_title = transcriber_output.podcast_title
            episode_title = transcriber_output.episode_title
            episode_id = transcriber_output.episode_id
            full_text = transcriber_output.full_text
        elif collector_output and collector_output.episodes:
            ep = collector_output.episodes[0]
            podcast_title = ep.podcast_title
            episode_title = ep.episode_title
            episode_id = ep.episode_id

        reporter_input = ReporterInput(
            episode_id=episode_id,
            podcast_title=podcast_title,
            episode_title=episode_title,
            full_text=full_text,
            sentiment_output=sentiment_output,
            analyst_output=analyst_output,
            crossmarket_output=crossmarket_output,
        )

    agent = ReporterAgent()
    result = agent.execute(reporter_input)
    state["reporter_output"] = result
    state["current_stage"] = "reporter"
    state["pipeline_complete"] = True
    state["final_status"] = "success"

    if result.report_html_path:
        logger.info(f"  📄 报告已保存: {result.report_html_path}")

    logger.info("=" * 50)
    logger.info("✅ 管道执行完成")

    return state


# ================================================================
# 条件路由
# ================================================================

def should_continue_after_collector(state: OrchestratorState) -> Literal["transcriber", "analyze", END]:
    """Collector 之后的路由决策"""
    collector_output = state.get("collector_output")
    if not collector_output or collector_output.agent_status == AgentStatus.FAILED:
        logger.warning("Collector 失败或无数据，结束管道")
        return END

    if not collector_output.episodes:
        logger.info("无匹配播客节目，结束管道")
        return END

    if state.get("skip_transcription"):
        logger.info("跳过转录，直接进入单集分析")
        return "analyze"

    return "transcriber"


def should_continue_after_transcriber(state: OrchestratorState) -> Literal["retriever", "analyze"]:
    """Transcriber 之后的路由"""
    if state.get("skip_retrieval") or state.get("skip_transcription"):
        return "analyze"
    return "retriever"


# ================================================================
# 构建工作流图
# ================================================================

def build_orchestrator_graph() -> StateGraph:
    """
    构建 LangGraph 工作流图

    新工作流:
      collector → transcriber → analyze_episodes → reporter → END

    每集独立分析:
      analyze_episodes 内部对每集运行 Sentiment → Analyst → CrossMarket
    """
    workflow = StateGraph(OrchestratorState)

    # 添加节点
    workflow.add_node("collector", collector_node)
    workflow.add_node("transcriber", transcriber_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyze", analyze_episodes_node)
    workflow.add_node("reporter", reporter_node)

    # 设置入口
    workflow.set_entry_point("collector")

    # Collector → 条件路由
    workflow.add_conditional_edges(
        "collector",
        should_continue_after_collector,
        {
            "transcriber": "transcriber",
            "analyze": "analyze",
            END: END,
        }
    )

    # Transcriber → retriever/analyze
    workflow.add_conditional_edges(
        "transcriber",
        should_continue_after_transcriber,
        {
            "retriever": "retriever",
            "analyze": "analyze",
        }
    )

    # Retriever → analyze → reporter → END
    workflow.add_edge("retriever", "analyze")
    workflow.add_edge("analyze", "reporter")
    workflow.add_edge("reporter", END)

    return workflow


# ================================================================
# 运行管道
# ================================================================

def run_pipeline(
    search_query: str = "",
    max_results: int = 10,
    skip_transcription: bool = False,
    podcast_names: Optional[list] = None,
    episodes_per_podcast: Optional[int] = None,
) -> OrchestratorState:
    """
    运行完整的多Agent管道

    Args:
        search_query: 指定播客名称（空则使用 podcast_names / config.target_podcasts）
        max_results: 采集总集数上限
        skip_transcription: 是否跳过转录
        podcast_names: 指定采集的播客名列表（可选，空则用 config.target_podcasts）
        episodes_per_podcast: 每播客采集集数（可选，空则用 config）

    Returns:
        最终管道状态
    """
    config = get_config()

    logger.info("=" * 60)
    logger.info("🎯 Podcast Agent Orchestrator 启动")
    logger.info(f"   GPU: {'可用' if config.gpu_available else '不可用 (CPU模式)'}")
    logger.info(f"   Listen Notes API: {'已配置' if config.listennotes_available else '未配置 (Demo模式)'}")
    logger.info(f"   LLM API: {'已配置' if config.llm_available else '未配置'} ({config.llm_base_url})")
    logger.info(f"   LLM 模型: {config.llm_model}")
    logger.info(f"   Embedding: {config.embedding_provider} ({config.embedding_model})")
    logger.info(f"   跳过转录: {skip_transcription}")
    logger.info(f"   RAG 检索: {'启用' if config.enable_rag else '关闭（跳过 retriever）'}")
    logger.info(f"   模式: 多集独立分析 + 汇总报告")
    logger.info("=" * 60)

    # 构建图
    workflow = build_orchestrator_graph()
    app = workflow.compile()

    # 创建初始状态
    initial_state = create_initial_state(
        search_query=search_query,
        max_results=max_results,
        podcast_names=podcast_names,
        episodes_per_podcast=episodes_per_podcast,
    )
    initial_state["skip_transcription"] = skip_transcription
    # RAG 默认关闭，仅当 config.enable_rag=true 时启用
    initial_state["skip_retrieval"] = not config.enable_rag

    # 运行
    final_state = app.invoke(initial_state)

    return final_state


# ================================================================
# 本地文件管道（绕过 Collector / Listen Notes API）
# ================================================================

def run_local_file_pipeline(file_path: str) -> OrchestratorState:
    """
    从本地 MP3 文件直接运行分析管道（绕过 Collector）

    工作流:
      Transcriber (本地文件) → Analyze (单集) → Reporter

    Args:
        file_path: 本地 MP3 文件的绝对路径

    Returns:
        最终管道状态
    """
    import os
    import uuid

    config = get_config()

    logger.info("=" * 60)
    logger.info("🎯 Podcast Agent — 本地文件模式")
    logger.info(f"  文件: {file_path}")
    logger.info(f"  LLM API: {'已配置' if config.llm_available else '未配置'} ({config.llm_base_url})")
    logger.info(f"  LLM 模型: {config.llm_model}")
    logger.info("  模式: 单文件 → 转录 → 分析 → 报告")
    logger.info("=" * 60)

    file_name = os.path.basename(file_path)
    session_id = str(uuid.uuid4())[:8]

    state: OrchestratorState = {
        "session_id": session_id,
        "search_query": file_name,
        "max_results": 1,
        "collector_output": None,
        "transcriber_output": None,
        "retriever_output": None,
        "sentiment_output": None,
        "analyst_output": None,
        "crossmarket_output": None,
        "reporter_output": None,
        "episode_analyses": [],
        "episode_transcripts": [],
        "current_stage": "transcriber",
        "errors": [],
        "skip_transcription": False,
        "skip_retrieval": True,
        "pipeline_complete": False,
        "final_status": "pending",
    }

    # ================================================================
    # Stage 1: Transcriber — 本地文件转录
    # ================================================================
    logger.info("-" * 40)
    logger.info("🎙️ Stage 1: Transcriber (本地文件)")

    transcriber = TranscriberAgent()
    transcriber_output = transcriber.transcribe_local_file(file_path)
    state["transcriber_output"] = transcriber_output
    state["current_stage"] = "transcriber"

    if transcriber_output.agent_status == AgentStatus.FAILED:
        state["errors"].append(f"Transcriber: {transcriber_output.error_message}")
        logger.error(f"  转录失败，终止管道")
        state["pipeline_complete"] = True
        state["final_status"] = "failed"
        return state

    # 提取单集转录文本
    episode_transcripts = []
    if transcriber_output.episode_transcriptions:
        for et in transcriber_output.episode_transcriptions:
            episode_transcripts.append({
                "episode_id": et.episode_id,
                "podcast_title": et.podcast_title,
                "episode_title": et.episode_title,
                "text": et.full_text,
                "description": et.description,
            })
    elif transcriber_output.full_text:
        episode_transcripts.append({
            "episode_id": file_name,
            "podcast_title": os.path.splitext(file_name)[0],
            "episode_title": file_name,
            "text": transcriber_output.full_text,
            "description": "",
        })

    state["episode_transcripts"] = episode_transcripts
    logger.info(f"  → 转录完成: {len(transcriber_output.full_text)} 字符")

    # ================================================================
    # Stage 2: Analyze — 单集分析 (Sentiment → Analyst → CrossMarket)
    # ================================================================
    logger.info("-" * 40)
    logger.info("🔬 Stage 2: Per-Episode Analysis (单集)")

    if not episode_transcripts:
        state["errors"].append("Analyze: 无转录数据")
        state["pipeline_complete"] = True
        state["final_status"] = "failed"
        return state

    et = episode_transcripts[0]
    text = et.get("text", "") or et.get("description", "")
    episode_id = et.get("episode_id", file_name)
    podcast_title = et.get("podcast_title", "")
    episode_title = et.get("episode_title", file_name)

    logger.info(f"  [{1}/{1}] {episode_title[:60]}")

    if not text.strip():
        logger.warning("  → 无文本，跳过分析")
        state["errors"].append("Analyze: 转录文本为空")
        state["pipeline_complete"] = True
        state["final_status"] = "failed"
        return state

    # Map-Reduce 分块: 长文本切分为重叠块，避免信息丢失
    chunks = chunk_text(text)
    logger.info(f"  原文 {len(text)} 字符 → {len(chunks)} 块")

    # Sentiment (分块)
    sentiment_result = _run_sentiment_chunked(chunks, episode_id)

    # Analyst (分块)
    analyst_result = _run_analyst_chunked(chunks, sentiment_result)

    # CrossMarket (单次)
    crossmarket_result = _run_crossmarket(analyst_result)

    episode_analysis = EpisodeAnalysis(
        episode_id=episode_id,
        podcast_title=podcast_title,
        episode_title=et.get("episode_title", ""),
        full_text=text,  # 保存完整文本
        description=et.get("description", ""),
        sentiment_output=sentiment_result,
        analyst_output=analyst_result,
        crossmarket_output=crossmarket_result,
    )

    state["episode_analyses"] = [episode_analysis]
    state["sentiment_output"] = sentiment_result
    state["analyst_output"] = analyst_result
    state["crossmarket_output"] = crossmarket_result
    state["current_stage"] = "analyze"

    logger.info(
        f"  → 情感: {sentiment_result.overall_sentiment.value} "
        f"({sentiment_result.sentiment_score:+.2f}), "
        f"板块: {len(crossmarket_result.a_share_impacts)} 个"
    )

    # ================================================================
    # Stage 3: Reporter — 单集报告生成
    # ================================================================
    logger.info("-" * 40)
    logger.info("📊 Stage 3: Reporter (单集报告)")

    reporter_input = ReporterInput(
        episode_id=episode_id,
        podcast_title=podcast_title,
        episode_title=episode_title,
        full_text=text,  # 使用完整文本
        sentiment_output=sentiment_result,
        analyst_output=analyst_result,
        crossmarket_output=crossmarket_result,
    )

    reporter = ReporterAgent()
    reporter_output = reporter.execute(reporter_input)
    state["reporter_output"] = reporter_output
    state["current_stage"] = "reporter"
    state["pipeline_complete"] = True
    state["final_status"] = "success"

    if reporter_output.report_html_path:
        logger.info(f"  📄 报告: {reporter_output.report_html_path}")

    logger.info("=" * 50)
    logger.info("✅ 本地文件管道完成")

    return state


def run_local_files_pipeline(file_paths: list) -> OrchestratorState:
    """
    本地多文件管道 — 每个文件独立转写分析，汇总为一份综合报告

    复用多集管道: episode_transcripts 列表 → analyze_episodes_node → reporter_node
    """
    config = get_config()

    session_id = str(uuid.uuid4())[:8]

    state: OrchestratorState = {
        "session_id": session_id,
        "search_query": f"本地文件 x{len(file_paths)}",
        "max_results": len(file_paths),
        "collector_output": None,
        "transcriber_output": None,
        "retriever_output": None,
        "sentiment_output": None,
        "analyst_output": None,
        "crossmarket_output": None,
        "reporter_output": None,
        "episode_analyses": [],
        "episode_transcripts": [],
        "current_stage": "transcriber",
        "errors": [],
        "skip_transcription": False,
        "skip_retrieval": True,
        "pipeline_complete": False,
        "final_status": "pending",
    }

    # ================================================================
    # Stage 1: Transcriber — 多文件并行转录
    # (并发上传+提交 ASR 任务，统一轮询，总耗时 ≈ 最慢文件而非之和)
    # ================================================================
    logger.info("=" * 60)
    logger.info(f"🎯 Podcast Agent — 本地多文件模式 ({len(file_paths)} 个文件)")
    logger.info(f"   LLM API: {'已配置' if config.llm_available else '未配置'} ({config.llm_base_url})")
    logger.info(f"   LLM 模型: {config.llm_model}")
    logger.info("-" * 40)
    logger.info("🎙️ Stage 1: Transcriber (并行转录)")

    transcriber = TranscriberAgent()
    transcribe_results = transcriber.transcribe_local_files(file_paths)

    episode_transcripts = []
    failed_files = []
    last_success_output = None

    for fp in file_paths:  # 按输入顺序组装，报告展示稳定
        file_name = os.path.basename(fp)
        transcriber_output = transcribe_results.get(fp)

        if transcriber_output is None or transcriber_output.agent_status == AgentStatus.FAILED:
            logger.warning(f"  → 转录失败，跳过: {file_name}")
            failed_files.append(file_name)
            state["errors"].append(f"Transcriber: {file_name} 转录失败")
            continue

        if transcriber_output.episode_transcriptions:
            for et in transcriber_output.episode_transcriptions:
                episode_transcripts.append({
                    "episode_id": et.episode_id,
                    "podcast_title": et.podcast_title,
                    "episode_title": et.episode_title,
                    "text": et.full_text,
                    "description": et.description,
                })
        elif transcriber_output.full_text:
            episode_transcripts.append({
                "episode_id": file_name,
                "podcast_title": os.path.splitext(file_name)[0],
                "episode_title": file_name,
                "text": transcriber_output.full_text,
                "description": "",
            })

        last_success_output = transcriber_output

    state["transcriber_output"] = last_success_output  # 最后一个成功文件（兼容展示）
    state["episode_transcripts"] = episode_transcripts
    logger.info(f"  → 转录完成: {len(episode_transcripts)}/{len(file_paths)} 个文件")

    if not episode_transcripts:
        state["pipeline_complete"] = True
        state["final_status"] = "failed"
        return state

    # ================================================================
    # Stage 2: 每集分析 (Sentiment → Analyst → CrossMarket)
    # ================================================================
    logger.info("-" * 40)
    logger.info("🔬 Stage 2: Per-Episode Analysis (多文件)")

    state = analyze_episodes_node(state)

    # ================================================================
    # Stage 3: 多集综合报告
    # ================================================================
    logger.info("-" * 40)
    logger.info("📊 Stage 3: Reporter (多文件综合报告)")

    state = reporter_node(state)

    if failed_files:
        logger.warning(f"  未纳入分析的文件: {', '.join(failed_files)}")

    return state