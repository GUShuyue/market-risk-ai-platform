"""
并发改造冒烟测试 — 不调用真实 API / 不加载 FinBERT

覆盖:
  1. 块级并发: _run_sentiment_chunked / _run_analyst_chunked 并行执行且聚合结果与串行一致
  2. 全局 LLM 信号量: 并发在途调用数 ≤ llm_concurrency
  3. 集级并发: analyze_episodes_node 多集并行且结果顺序与输入一致
  4. 共享 Agent 实例: 同一 key 返回同一实例
  5. ASR 批量转录: transcribe_local_files 并发上传提交 + 统一轮询逻辑

运行: python test_concurrency_smoke.py  (在 podcast_agent/ 目录下)
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from podcast_agent.config import get_config
from podcast_agent.models.messages import (
    AgentStatus, Sentiment, SentimentOutput, AnalystOutput, Entity,
    CrossMarketOutput, AShareImpact, TranscriptSegment,
)
from podcast_agent.agents.sentiment import SentimentAgent
from podcast_agent.agents.analyst import AnalystAgent
from podcast_agent.agents.crossmarket import CrossMarketAgent
from podcast_agent.agents.transcriber import TranscriberAgent
from podcast_agent.orchestration import graph

# ================================================================
# 打桩: 记录在途并发数，模拟耗时
# ================================================================

_lock = threading.Lock()
_inflight = 0
max_inflight = 0
SENTIMENT_CALLS = 0

STUB_DELAY = 0.15  # 每次模拟调用的耗时


def _track_inflight():
    global _inflight, max_inflight
    with _lock:
        _inflight += 1
        max_inflight = max(max_inflight, _inflight)


def _release_inflight():
    global _inflight
    with _lock:
        _inflight -= 1


def stub_sentiment_execute(self, input_data):
    global SENTIMENT_CALLS
    SENTIMENT_CALLS += 1
    # 从文本中提取块编号 marker: "CHUNK3"
    idx = 0
    for part in input_data.current_text.split():
        if part.startswith("CHUNK"):
            idx = int(part[5:])
    _track_inflight()
    time.sleep(STUB_DELAY)
    _release_inflight()
    score = (idx % 5 - 2) * 0.2  # -0.4 ~ 0.4 与块相关，用于验证加权聚合
    label = Sentiment.BULLISH if score > 0.15 else (
        Sentiment.BEARISH if score < -0.15 else Sentiment.NEUTRAL
    )
    return SentimentOutput(
        agent_status=AgentStatus.SUCCESS,
        episode_id=input_data.episode_id,
        overall_sentiment=label,
        sentiment_score=score,
        confidence=0.8,
        key_positive_factors=[f"pos_{idx}"],
        engines_used="llm",
        finbert_scores={"positive": 0.4, "negative": 0.3, "neutral": 0.3},
    )


def stub_analyst_execute(self, input_data):
    _track_inflight()
    time.sleep(STUB_DELAY)
    _release_inflight()
    text = input_data.llm_sentiment_analysis or ""
    idx = 0
    for part in text.split():
        if part.startswith("CHUNK"):
            idx = int(part[5:])
    return AnalystOutput(
        agent_status=AgentStatus.SUCCESS,
        entities=[Entity(name=f"Company_{idx}")],
        macro_topics=[f"topic_{idx}"],
        china_relevance=0.5,
    )


def stub_crossmarket_execute(self, input_data):
    _track_inflight()
    time.sleep(STUB_DELAY * 0.5)
    _release_inflight()
    return CrossMarketOutput(
        agent_status=AgentStatus.SUCCESS,
        a_share_impacts=[AShareImpact(sector_name="半导体", impact_strength=0.6)],
    )


def install_stubs():
    SentimentAgent.execute = stub_sentiment_execute
    AnalystAgent.execute = stub_analyst_execute
    CrossMarketAgent.execute = stub_crossmarket_execute


# ================================================================
# 测试用例
# ================================================================

def test_shared_agent_reuse():
    a1 = graph._get_shared_agent("sentiment", SentimentAgent)
    a2 = graph._get_shared_agent("sentiment", SentimentAgent)
    assert a1 is a2, "共享 Agent 实例应复用"
    print("  [PASS] 共享 Agent 实例复用")


def test_semaphore_caps_concurrency():
    config = get_config()
    config.llm_concurrency = 3
    graph._llm_semaphore = None  # 重置，让信号量按新值创建

    global max_inflight, SENTIMENT_CALLS
    max_inflight = 0
    SENTIMENT_CALLS = 0

    chunks = [f"CHUNK{i} filler text. " * 50 for i in range(9)]
    start = time.time()
    result = graph._run_sentiment_chunked(chunks, "ep_test")
    elapsed = time.time() - start

    serial_time = 9 * STUB_DELAY
    assert max_inflight <= 3, f"在途并发 {max_inflight} 超过上限 3"
    assert SENTIMENT_CALLS == 9, f"应调用 9 次, 实际 {SENTIMENT_CALLS}"
    assert elapsed < serial_time * 0.8, (
        f"耗时 {elapsed:.2f}s 未显著低于串行 {serial_time:.2f}s"
    )
    assert result.chunk_count == 9
    print(f"  [PASS] 信号量限流: 9 块最大在途 {max_inflight}, "
          f"耗时 {elapsed:.2f}s (串行约 {serial_time:.2f}s)")

    # 聚合结果应与块分数的加权平均一致
    scores = [(i % 5 - 2) * 0.2 for i in range(9)]
    weights = [len(c) for c in chunks]
    expected = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    assert abs(result.sentiment_score - expected) < 1e-3, (
        f"聚合分数 {result.sentiment_score} != 加权平均 {expected:.4f}"
    )
    print(f"  [PASS] 聚合正确: 加权分数 {result.sentiment_score:+.4f}")


def test_episode_level_parallelism():
    config = get_config()
    config.llm_concurrency = 4
    config.episode_concurrency = 3
    graph._llm_semaphore = None

    global max_inflight
    max_inflight = 0

    # 3 集 × 4 块（块 marker 藏在长文本里）
    episode_transcripts = [
        {
            "episode_id": f"ep{i}",
            "podcast_title": f"Podcast {i}",
            "episode_title": f"EPISODE_{i}",
            "text": " ".join(
                f"CHUNK{i}{j} " + "market moves higher today. " * 40
                for j in range(4)
            ),
        }
        for i in range(3)
    ]

    state = {
        "episode_transcripts": episode_transcripts,
        "errors": [],
        "current_stage": "",
    }

    start = time.time()
    state = graph.analyze_episodes_node(state)
    elapsed = time.time() - start

    analyses = state["episode_analyses"]
    assert len(analyses) == 3, f"应完成 3 集分析, 实际 {len(analyses)}"
    # 顺序必须与输入一致
    for i, ea in enumerate(analyses):
        assert ea.episode_id == f"ep{i}", (
            f"顺序错乱: 位置 {i} 是 {ea.episode_id}"
        )
    assert max_inflight <= 4, f"全局在途并发 {max_inflight} 超过上限 4"

    # 串行基线: 3集 × (4块×0.15 + 4块×0.15 + 0.075) ≈ 3.8s
    serial_time = 3 * (4 * STUB_DELAY + 4 * STUB_DELAY + STUB_DELAY * 0.5)
    assert elapsed < serial_time * 0.6, (
        f"耗时 {elapsed:.2f}s 未显著低于串行 {serial_time:.2f}s"
    )
    print(f"  [PASS] 集级并发: 3 集 × 4 块, 最大在途 {max_inflight}, "
          f"耗时 {elapsed:.2f}s (串行约 {serial_time:.2f}s)")


def test_transcribe_local_files_orchestration():
    """打桩 ASR 网络层，验证 并发上传提交 + 统一轮询 的编排逻辑"""
    config = get_config()
    config.asr_poll_interval = 0.05
    config.asr_max_wait = 10
    config.asr_concurrency = 3

    agent = TranscriberAgent()

    upload_order = []
    upload_lock = threading.Lock()
    submit_count = {"n": 0}
    poll_rounds = {"n": 0}

    def stub_upload(fp):
        with upload_lock:
            upload_order.append(os.path.basename(fp))
        time.sleep(STUB_DELAY)  # 模拟上传耗时
        return f"oss://bucket/{os.path.basename(fp)}"

    def stub_submit(api_key, audio_url):
        submit_count["n"] += 1
        return f"task_{submit_count['n']}"

    def stub_query(query_url, headers):
        poll_rounds["n"] += 1
        # 每个任务轮询 2 次后成功
        task_no = int(query_url.split("_")[-1])
        if poll_rounds["n"] <= 2 * task_no - 1:  # 前几轮 running
            return "running", None, None
        return "succeeded", f"https://fake/{task_no}.json", None

    def stub_parse(url):
        seg = TranscriptSegment(speaker="A", start_time=0.0, end_time=1.0,
                                text="hello world", segment_index=0)
        return [seg], "hello world", 1.0

    agent._upload_to_dashscope = stub_upload
    agent._submit_asr_task = stub_submit
    agent._query_asr_task = stub_query
    agent._parse_transcription = stub_parse
    agent.config.asr_api_key = "fake-key"

    files = [f"C:/fake/file_{i}.mp3" for i in range(3)]
    start = time.time()
    results = agent.transcribe_local_files(files)
    elapsed = time.time() - start

    assert len(results) == 3, f"应成功 3 个, 实际 {len(results)}"
    for fp in files:
        out = results[fp]
        assert out.agent_status == AgentStatus.SUCCESS
        assert out.full_text == "hello world"
        assert len(out.episode_transcriptions) == 1
    assert submit_count["n"] == 3

    # 串行上传 = 3 × 0.15 = 0.45s；并发应 ≈ 0.15s + 轮询
    assert elapsed < 0.45, f"上传未并发? 耗时 {elapsed:.2f}s"
    print(f"  [PASS] ASR 批量编排: 3 文件并发上传+统一轮询, "
          f"耗时 {elapsed:.2f}s (串行上传约 0.45s)")


def main():
    print("=" * 60)
    print("并发改造冒烟测试 (stub, 无网络调用)")
    print("=" * 60)
    install_stubs()

    test_shared_agent_reuse()
    test_semaphore_caps_concurrency()
    test_episode_level_parallelism()
    test_transcribe_local_files_orchestration()

    print("=" * 60)
    print("全部通过 ✅")


if __name__ == "__main__":
    main()
