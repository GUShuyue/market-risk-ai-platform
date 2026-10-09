"""
LangGraph 状态定义 — 管道全程流转的上下文

使用 TypedDict 定义 LangGraph 状态，确保类型安全
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

from ..models.messages import (
    AgentStatus,
    CollectorOutput,
    TranscriberOutput,
    RetrieverOutput,
    SentimentOutput,
    AnalystOutput,
    CrossMarketOutput,
    ReporterOutput,
    EpisodeAnalysis,
)


class OrchestratorState(TypedDict, total=False):
    """
    LangGraph Orchestrator 状态

    在整个管道中流转，每个Agent节点读取/写入对应字段
    """

    # === 会话信息 ===
    session_id: str
    search_query: str
    max_results: int
    podcast_names: List[str]      # 指定采集的播客名（可选，空则用 config.target_podcasts）
    episodes_per_podcast: int     # 每播客采集集数（可选，空则用 config）

    # === 各Agent输出 ===
    collector_output: Optional[CollectorOutput]
    transcriber_output: Optional[TranscriberOutput]
    retriever_output: Optional[RetrieverOutput]
    sentiment_output: Optional[SentimentOutput]
    analyst_output: Optional[AnalystOutput]
    crossmarket_output: Optional[CrossMarketOutput]
    reporter_output: Optional[ReporterOutput]

    # === 多集分析结果 ===
    episode_analyses: List[EpisodeAnalysis]  # 每集独立分析结果
    episode_transcripts: List[dict]           # 每集转录文本 {episode_id, text, ...}

    # === 控制字段 ===
    current_stage: str          # 当前阶段: collector/transcriber/retriever/sentiment/analyst/crossmarket/reporter
    errors: List[str]           # 错误列表
    skip_transcription: bool    # 是否跳过转录
    skip_retrieval: bool        # 是否跳过RAG检索
    pipeline_complete: bool     # 管道是否完成
    final_status: str           # 最终状态: success/partial/failed


def create_initial_state(
    search_query: str = "",
    max_results: int = 10,
    session_id: str = "",
    podcast_names: Optional[List[str]] = None,
    episodes_per_podcast: Optional[int] = None,
) -> OrchestratorState:
    """创建初始状态"""
    import uuid

    return OrchestratorState(
        session_id=session_id or str(uuid.uuid4())[:8],
        search_query=search_query,
        max_results=max_results,
        podcast_names=podcast_names or [],
        episodes_per_podcast=episodes_per_podcast or 0,
        collector_output=None,
        transcriber_output=None,
        retriever_output=None,
        sentiment_output=None,
        analyst_output=None,
        crossmarket_output=None,
        reporter_output=None,
        episode_analyses=[],
        episode_transcripts=[],
        current_stage="collector",
        errors=[],
        skip_transcription=False,  # Day 3-4 启用转录
        skip_retrieval=True,       # 默认跳过 RAG 检索（enable_rag 开启时才启用）
        pipeline_complete=False,
        final_status="pending",
    )