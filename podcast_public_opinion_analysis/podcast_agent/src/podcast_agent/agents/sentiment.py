"""
Sentiment Agent — 金融情感分析 (Day 5-6 实现)

负责:
  1. 使用 LLM (DashScope Qwen) 深度分析情感因素（主引擎）
  2. 提取正面/负面关键因素
  3. 检测对冲语言 (hedging language)
  4. 输出结构化情感分析结果

技术栈:
  - LLM: DashScope Qwen (深度分析，主引擎)
  - FinBERT: 本地金融情感分类 (CPU推理, 兜底)
  - 回退: LLM不可用时用 FinBERT 兜底
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from ..config import get_config
from ..models.messages import (
    AgentStatus, RetrieverOutput, SentimentOutput, Sentiment,
)
from .base import BaseAgent

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent[RetrieverOutput, SentimentOutput]):
    """金融情感分析Agent"""

    agent_name = "sentiment"

    def __init__(self):
        super().__init__()
        self._finbert = None
        self._finbert_tokenizer = None
        self._finbert_lock = None

    def execute(self, input_data: RetrieverOutput) -> SentimentOutput:
        self.log_start(
            text_length=len(input_data.current_text),
            context_length=len(input_data.augmented_context),
        )

        output = SentimentOutput(
            agent_status=AgentStatus.RUNNING,
            episode_id=input_data.episode_id,
        )

        try:
            # 使用增强上下文进行分析
            text = input_data.augmented_context or input_data.current_text
            # 保存原文，供下游 Analyst 直接读取（修复原文断链）
            output.source_text = text
            if not text.strip():
                output.agent_status = AgentStatus.SUCCESS
                output.overall_sentiment = Sentiment.NEUTRAL
                return output

            # 1. LLM 深度分析（主引擎，优先）
            llm_analysis = self._llm_sentiment_analysis(text)

            if llm_analysis:
                # LLM 可用：分数与标签完全由 LLM 决定，FinBERT 不参与
                output = self._apply_llm_result(output, llm_analysis)
            else:
                # LLM 不可用：FinBERT 兜底
                finbert_scores = self._finbert_sentiment(text)
                output = self._apply_finbert_fallback(output, finbert_scores)

            output.agent_status = AgentStatus.SUCCESS
            self.log_done(
                sentiment=output.overall_sentiment.value,
                score=f"{output.sentiment_score:.3f}",
                confidence=f"{output.confidence:.3f}",
            )

        except Exception as e:
            self.log_error(str(e))
            output.agent_status = AgentStatus.FAILED
            output.error_message = str(e)

        return output

    # ================================================================
    # FinBERT 本地情感分类
    # ================================================================

    def _load_finbert(self):
        """延迟加载 FinBERT 模型（本地缓存优先，无网络依赖；缺失时走国内镜像下载）"""
        if self._finbert is not None:
            return self._finbert, self._finbert_tokenizer

        # 并发分块时多个线程同时首次加载 → 加锁保证只加载一次
        import threading
        if self._finbert_lock is None:
            self._finbert_lock = threading.Lock()
        with self._finbert_lock:
            if self._finbert is not None:
                return self._finbert, self._finbert_tokenizer

            # 必须在 import transformers 之前设置，huggingface_hub 读取该环境变量构建下载地址
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch

                name = self.config.finbert_model
                self.logger.info(f"加载 FinBERT 模型: {name} (本地缓存优先) ...")
                try:
                    # 纯离线：只读本地 HF 缓存，不发起任何网络请求
                    self._finbert_tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
                    self._finbert = AutoModelForSequenceClassification.from_pretrained(name, local_files_only=True)
                except Exception:
                    # 缓存未命中：经 hf-mirror.com 镜像下载（无需 VPN）
                    self.logger.info("本地缓存未命中，从镜像站下载 FinBERT ...")
                    self._finbert_tokenizer = AutoTokenizer.from_pretrained(name)
                    self._finbert = AutoModelForSequenceClassification.from_pretrained(name)
                self._finbert.eval()
                # 多线程并发推理时限制单次前向的内部线程数，避免 CPU 超额订阅
                torch.set_num_threads(min(4, os.cpu_count() or 4))
                self.logger.info("FinBERT 模型加载完成")
                return self._finbert, self._finbert_tokenizer

            except ImportError as e:
                self.logger.warning(f"transformers 不可用: {e}")
                return None, None
            except Exception as e:
                self.logger.warning(f"FinBERT 加载失败: {e}")
                return None, None

    def _finbert_sentiment(self, text: str) -> dict:
        """
        使用 FinBERT 分析金融文本情感（滑动窗口，覆盖头尾）

        分块文本可能超过 FinBERT 的 512 token 限制，
        因此取头尾两个窗口各跑一次取平均，确保覆盖更多内容。

        Returns:
            {"positive": float, "negative": float, "neutral": float, "score": float}
            score 范围: -1.0 (极度负面) ~ 1.0 (极度正面)
        """
        model, tokenizer = self._load_finbert()
        if model is None:
            return {}

        try:
            import torch

            def _run_window(window_text: str) -> dict:
                """对单个窗口运行 FinBERT"""
                inputs = tokenizer(
                    window_text[:2000],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                return {
                    "positive": float(probs[0]),
                    "negative": float(probs[1]),
                    "neutral": float(probs[2]),
                }

            text_len = len(text)

            if text_len <= 2000:
                # 短文本: 单窗口
                windows = [text]
            else:
                # 长文本: 头尾双窗口
                windows = [text[:2000], text[-2000:]]

            # 收集所有窗口结果
            all_results = []
            for i, win in enumerate(windows):
                if len(win.strip()) < 50:
                    continue  # 跳过太短的窗口
                try:
                    result = _run_window(win)
                    all_results.append(result)
                except Exception:
                    pass

            if not all_results:
                return {}

            # 平均
            n = len(all_results)
            pos = sum(r["positive"] for r in all_results) / n
            neg = sum(r["negative"] for r in all_results) / n
            neu = sum(r["neutral"] for r in all_results) / n
            score = pos - neg

            self.logger.info(
                f"  FinBERT ({n}窗口): pos={pos:.3f} neg={neg:.3f} neu={neu:.3f} score={score:.3f}"
            )

            return {"positive": pos, "negative": neg, "neutral": neu, "score": score}

        except Exception as e:
            self.logger.warning(f"FinBERT 推理失败: {e}")
            return {}

    # ================================================================
    # LLM 深度情感分析
    # ================================================================

    def _llm_sentiment_analysis(self, text: str) -> dict:
        """
        使用 LLM 进行深度情感分析

        提取: 正面/负面因素, 对冲语言检测, 整体情感判断
        """
        client = self.get_llm_client()
        if client is None:
            self.logger.warning("LLM 不可用，仅使用 FinBERT 结果")
            return {}

        model = self.get_llm_model()

        prompt = f"""Analyze the following financial podcast transcript for sentiment. 
Provide a structured analysis in JSON format only (no other text).

Text:
{text}

Return ONLY a JSON object with these fields:
- overall_sentiment: "bullish", "bearish", or "neutral"
- sentiment_score: float from -1.0 (extremely bearish) to 1.0 (extremely bullish)
- confidence: float from 0.0 to 1.0
- key_positive_factors: list of strings (max 5)
- key_negative_factors: list of strings (max 5)
- hedging_language_detected: boolean
- brief_analysis: one sentence summary in Chinese

JSON:"""

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )

            content = resp.choices[0].message.content.strip()
            # 清理可能的前缀
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]

            import json
            result = json.loads(content)
            self.logger.info(f"  LLM 分析: {result.get('overall_sentiment', 'unknown')}")
            return result

        except Exception as e:
            self.logger.warning(f"LLM 情感分析失败: {e}")
            return {}

    # ================================================================
    # 结果合并
    # ================================================================

    def _apply_llm_result(
        self,
        output: SentimentOutput,
        llm: dict,
    ) -> SentimentOutput:
        """LLM 可用：情感分数与标签完全由 LLM 决定（FinBERT 不参与）"""
        raw_sentiment = llm.get("overall_sentiment", "neutral")
        try:
            output.overall_sentiment = Sentiment(raw_sentiment)
        except ValueError:
            output.overall_sentiment = Sentiment.NEUTRAL

        output.sentiment_score = llm.get("sentiment_score", 0.0)
        output.confidence = llm.get("confidence", 0.5)
        output.key_positive_factors = llm.get("key_positive_factors", [])
        output.key_negative_factors = llm.get("key_negative_factors", [])
        output.hedging_language_detected = llm.get("hedging_language_detected", False)
        output.llm_sentiment_analysis = llm.get("brief_analysis", "")
        output.engines_used = "llm"
        return output

    def _apply_finbert_fallback(
        self,
        output: SentimentOutput,
        finbert: dict,
    ) -> SentimentOutput:
        """LLM 不可用：FinBERT 兜底（分数、标签均由 FinBERT 决定）"""
        if finbert:
            output.finbert_scores = {
                "positive": finbert.get("positive", 0),
                "negative": finbert.get("negative", 0),
                "neutral": finbert.get("neutral", 0),
            }
            score = finbert.get("score", 0)
            output.sentiment_score = score
            output.confidence = max(
                finbert.get("positive", 0),
                finbert.get("negative", 0),
                finbert.get("neutral", 0),
            )
            if score > 0.15:
                output.overall_sentiment = Sentiment.BULLISH
            elif score < -0.15:
                output.overall_sentiment = Sentiment.BEARISH
            else:
                output.overall_sentiment = Sentiment.NEUTRAL
            output.engines_used = "finbert"
        else:
            # FinBERT 也不可用
            output.overall_sentiment = Sentiment.NEUTRAL
            output.confidence = 0.0
            output.engines_used = "none"
        return output