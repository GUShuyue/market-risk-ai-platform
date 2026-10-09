"""
Agent间通信协议 — 所有Agent通过Pydantic严格类型化消息通信

消息流向:
  Collector → Transcriber → Retriever → Sentiment → Analyst → CrossMarket → Reporter
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# 基础枚举
# ============================================================

class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ChinaRelevance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================
# Collector Agent 消息
# ============================================================

class PodcastMetadata(BaseModel):
    """单集播客元数据"""
    podcast_id: str = Field(description="Listen Notes podcast ID")
    episode_id: str = Field(description="Listen Notes episode ID")
    podcast_title: str
    episode_title: str
    description: str = ""
    audio_url: str = ""
    audio_length_sec: int = 0
    pub_date_ms: int = 0
    publisher: str = ""
    language: str = "en"
    listennotes_url: str = ""
    image_url: str = ""


class CollectorOutput(BaseModel):
    """Collector Agent 输出 → 传给 Transcriber"""
    agent_status: AgentStatus = AgentStatus.PENDING
    collected_at: datetime = Field(default_factory=datetime.now)
    total_found: int = 0
    filtered_count: int = 0
    episodes: List[PodcastMetadata] = Field(default_factory=list)
    search_query: str = ""
    error_message: Optional[str] = None


# ============================================================
# Transcriber Agent 消息
# ============================================================

class TranscriptSegment(BaseModel):
    """转录文本段落（带时间戳）"""
    speaker: str = "UNKNOWN"
    start_time: float = 0.0  # 秒
    end_time: float = 0.0
    text: str = ""
    segment_index: int = 0


class EpisodeTranscription(BaseModel):
    """单集转录结果"""
    episode_id: str = ""
    podcast_title: str = ""
    episode_title: str = ""
    full_text: str = ""
    segments: List[TranscriptSegment] = Field(default_factory=list)
    word_count: int = 0
    duration_seconds: float = 0.0
    description: str = ""  # 回退用


class TranscriberOutput(BaseModel):
    """Transcriber Agent 输出 → 传给 Retriever"""
    agent_status: AgentStatus = AgentStatus.PENDING
    episode_id: str = ""
    podcast_title: str = ""
    episode_title: str = ""
    full_text: str = ""
    segments: List[TranscriptSegment] = Field(default_factory=list)
    word_count: int = 0
    duration_seconds: float = 0.0
    language: str = "en"
    transcribed_at: Optional[datetime] = None
    # 多集转录：每集独立转录结果
    episode_transcriptions: List[EpisodeTranscription] = Field(default_factory=list)
    error_message: Optional[str] = None


# ============================================================
# Retriever Agent 消息
# ============================================================

class RetrievedChunk(BaseModel):
    """检索到的相关段落"""
    chunk_id: str
    text: str = ""
    episode_id: str = ""
    episode_title: str = ""
    podcast_title: str = ""
    similarity_score: float = 0.0
    pub_date: Optional[str] = None


class RetrieverOutput(BaseModel):
    """Retriever Agent 输出 → 传给 Sentiment"""
    agent_status: AgentStatus = AgentStatus.PENDING
    episode_id: str = ""
    current_text: str = ""  # 当前分析的文本
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    augmented_context: str = ""  # 拼接后的增强上下文
    error_message: Optional[str] = None


# ============================================================
# Sentiment Agent 消息
# ============================================================

class SentimentOutput(BaseModel):
    """Sentiment Agent 输出 → 传给 Analyst"""
    agent_status: AgentStatus = AgentStatus.PENDING
    episode_id: str = ""
    overall_sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_score: float = 0.0       # -1.0 (极度负面) ~ 1.0 (极度正面)
    confidence: float = 0.0             # FinBERT 置信度
    key_positive_factors: List[str] = Field(default_factory=list)
    key_negative_factors: List[str] = Field(default_factory=list)
    hedging_language_detected: bool = False
    finbert_scores: Dict[str, float] = Field(default_factory=dict)  # {pos, neg, neu}
    llm_sentiment_analysis: Optional[str] = None  # LLM深度分析文本
    source_text: str = ""  # 原始转录文本（供下游 Analyst 直接读取原文，避免断链）
    error_message: Optional[str] = None
    # FinBERT 与 LLM 引擎分歧率 (0=一致, 1=分歧; None=仅单引擎可用)
    engine_divergence_rate: Optional[float] = None
    # 该块实际使用的引擎 ("llm+finbert" / "llm" / "finbert" / "none")
    engines_used: Optional[str] = None
    # 聚合层: FinBERT 可用块占比 (0-1; None=未统计)
    finbert_chunk_coverage: Optional[float] = None
    # 聚合层: LLM 可用块占比 (0-1; None=未统计)
    llm_chunk_coverage: Optional[float] = None
    # Map-Reduce 分块聚合字段
    chunk_consensus: Optional[float] = None  # 各块情感一致性 (0-1)
    chunk_count: int = 1                     # 分块数量


# ============================================================
# Analyst Agent 消息
# ============================================================

class Entity(BaseModel):
    """提取的实体"""
    name: str
    ticker: Optional[str] = None
    entity_type: str = "company"  # company | sector | indicator | person
    sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_score: float = 0.0
    key_points: List[str] = Field(default_factory=list)
    evidence: str = ""  # 原文证据句（引用转录原文中提及该实体的原句）


class AnalystOutput(BaseModel):
    """Analyst Agent 输出 → 传给 CrossMarket"""
    agent_status: AgentStatus = AgentStatus.PENDING
    episode_id: str = ""
    overall_sentiment: Sentiment = Sentiment.NEUTRAL
    confidence: float = 0.0
    entities: List[Entity] = Field(default_factory=list)
    macro_topics: List[str] = Field(default_factory=list)
    china_relevance: float = 0.0       # 0.0 ~ 1.0 与中国市场的关联度
    summary_cn: str = ""               # 中文摘要
    summary_en: str = ""               # 英文摘要
    logic_chain: str = ""              # 逻辑推演链
    cot_reasoning: Optional[str] = None  # Chain-of-Thought 原始推理
    error_message: Optional[str] = None
    # Map-Reduce 分块聚合字段
    chunk_count: int = 1               # 分块数量


# ============================================================
# CrossMarket Agent 消息
# ============================================================

class AShareImpact(BaseModel):
    """A股影响评估"""
    sector_name: str = ""               # A股板块名称
    sector_code: Optional[str] = None   # 行业代码
    affected_stocks: List[str] = Field(default_factory=list)
    top_stocks: List[Dict[str, Any]] = Field(default_factory=list)  # 真实成分股 [{code,name,price,change_pct,market_cap}]
    top_stocks_note: str = ""  # 成分股为空时的原因说明
    representative_stocks: List[Dict[str, str]] = Field(default_factory=list)  # LLM 生成的代表性个股 [{code,name,reason}]
    impact_direction: Sentiment = Sentiment.NEUTRAL
    impact_strength: float = 0.0        # 0.0 ~ 1.0
    reasoning: str = ""
    northbound_flow_alignment: Optional[str] = None  # 北向资金验证结果


class CrossMarketOutput(BaseModel):
    """CrossMarket Agent 输出 → 传给 Reporter"""
    agent_status: AgentStatus = AgentStatus.PENDING
    episode_id: str = ""
    a_share_impacts: List[AShareImpact] = Field(default_factory=list)
    northbound_data: Optional[Dict[str, Any]] = None
    overall_assessment: str = ""        # 综合评估
    risk_warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


# ============================================================
# 单集分析结果（多集汇总用）
# ============================================================

class EpisodeAnalysis(BaseModel):
    """单集完整分析结果 — 聚合 Sentiment + Analyst + CrossMarket"""
    episode_id: str = ""
    podcast_title: str = ""
    episode_title: str = ""
    full_text: str = ""       # 转录文本
    description: str = ""     # 播客描述（回退用）
    sentiment_output: Optional[SentimentOutput] = None
    analyst_output: Optional[AnalystOutput] = None
    crossmarket_output: Optional[CrossMarketOutput] = None


# ============================================================
# Reporter Agent 消息
# ============================================================

class ReporterInput(BaseModel):
    """Reporter Agent 输入 — 聚合多集分析 + 兼容单集"""
    episode_analyses: List[EpisodeAnalysis] = Field(default_factory=list)
    # 兼容旧版单集字段
    episode_id: str = ""
    podcast_title: str = ""
    episode_title: str = ""
    full_text: str = ""
    sentiment_output: Optional[SentimentOutput] = None
    analyst_output: Optional[AnalystOutput] = None
    crossmarket_output: Optional[CrossMarketOutput] = None


class ReporterOutput(BaseModel):
    """Reporter Agent 最终输出"""
    agent_status: AgentStatus = AgentStatus.PENDING
    report_id: str = ""
    report_html_path: str = ""       # HTML 报告文件路径
    report_markdown: str = ""
    episode_id: str = ""
    podcast_title: str = ""
    episode_title: str = ""
    generated_at: Optional[datetime] = None
    error_message: Optional[str] = None


# ============================================================
# 管道状态 — 全程流转的上下文
# ============================================================

class PipelineState(BaseModel):
    """LangGraph 管道全程状态"""
    # 会话信息
    session_id: str = ""
    started_at: datetime = Field(default_factory=datetime.now)

    # 各阶段输出
    collector_output: Optional[CollectorOutput] = None
    transcriber_output: Optional[TranscriberOutput] = None
    retriever_output: Optional[RetrieverOutput] = None
    sentiment_output: Optional[SentimentOutput] = None
    analyst_output: Optional[AnalystOutput] = None
    crossmarket_output: Optional[CrossMarketOutput] = None
    reporter_output: Optional[ReporterOutput] = None

    # 控制
    current_agent: str = ""
    errors: List[str] = Field(default_factory=list)
    skip_transcription: bool = False  # 如果已有转录文本则跳过
    skip_retrieval: bool = False      # 如果是首集分析则跳过RAG