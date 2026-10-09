"""
SQLite 数据库模型 — 使用 SQLAlchemy ORM
"""

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean,
    ForeignKey, JSON, create_engine, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship, Session


class Base(DeclarativeBase):
    pass


# ============================================================
# 播客
# ============================================================

class Podcast(Base):
    __tablename__ = "podcasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    podcast_id = Column(String(100), unique=True, index=True, comment="Listen Notes podcast ID")
    title = Column(String(500), nullable=False)
    publisher = Column(String(300), default="")
    description = Column(Text, default="")
    image_url = Column(String(1000), default="")
    language = Column(String(10), default="en")
    total_episodes = Column(Integer, default=0)
    listennotes_url = Column(String(1000), default="")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    episodes = relationship("Episode", back_populates="podcast")


# ============================================================
# 单集节目
# ============================================================

class Episode(Base):
    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String(100), unique=True, index=True, comment="Listen Notes episode ID")
    podcast_id = Column(Integer, ForeignKey("podcasts.id"), index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, default="")
    audio_url = Column(String(1000), default="")
    audio_length_sec = Column(Integer, default=0)
    pub_date_ms = Column(Integer, default=0)
    listennotes_url = Column(String(1000), default="")
    image_url = Column(String(1000), default="")

    # 处理状态
    is_transcribed = Column(Boolean, default=False)
    is_analyzed = Column(Boolean, default=False)
    china_relevance = Column(Float, default=0.0)  # 0~1

    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    podcast = relationship("Podcast", back_populates="episodes")
    transcript = relationship("Transcript", back_populates="episode", uselist=False)
    analysis_results = relationship("AnalysisResult", back_populates="episode")


# ============================================================
# 转录文本
# ============================================================

class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(Integer, ForeignKey("episodes.id"), unique=True, index=True)
    full_text = Column(Text, default="")
    segments_json = Column(Text, default="[]")  # JSON: List[TranscriptSegment]
    word_count = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    language = Column(String(10), default="en")
    whisper_model = Column(String(50), default="")
    created_at = Column(DateTime, default=datetime.now)

    episode = relationship("Episode", back_populates="transcript")


# ============================================================
# 分析结果
# ============================================================

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(Integer, ForeignKey("episodes.id"), index=True)

    # 情感
    overall_sentiment = Column(String(20), default="neutral")
    sentiment_score = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)

    # 实体
    entities_json = Column(Text, default="[]")       # JSON: List[Entity]
    macro_topics_json = Column(Text, default="[]")   # JSON: List[str]

    # A股传导
    a_share_impacts_json = Column(Text, default="[]") # JSON: List[AShareImpact]
    northbound_data_json = Column(Text, default="{}")

    # 摘要
    summary_cn = Column(Text, default="")
    summary_en = Column(Text, default="")
    logic_chain = Column(Text, default="")
    risk_warnings_json = Column(Text, default="[]")

    # 报告
    report_html = Column(Text, default="")
    report_markdown = Column(Text, default="")

    created_at = Column(DateTime, default=datetime.now)

    episode = relationship("Episode", back_populates="analysis_results")

    __table_args__ = (
        Index("idx_episode_sentiment", "episode_id", "overall_sentiment"),
    )


# ============================================================
# 数据库初始化
# ============================================================

def init_database(db_path: str) -> Session:
    """初始化数据库，返回 SQLAlchemy Session"""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)