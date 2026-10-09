"""
配置系统 — 集中管理所有环境变量和参数
针对无GPU环境优化
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# 自动加载 .env 文件
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


def _env(*names: str, default: str = "") -> str:
    """按优先级返回第一个非空环境变量"""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


# 默认采集的播客名称（未配置 TARGET_PODCASTS 环境变量时使用）
DEFAULT_TARGET_PODCASTS = [
    "FT News Briefing",
    "WSJ What's News",
    "Thoughts on the Market",
]


def _parse_podcast_list(raw: str) -> list:
    """解析逗号分隔的播客名称列表；为空时返回默认列表"""
    if raw and raw.strip():
        return [p.strip() for p in raw.split(",") if p.strip()]
    return list(DEFAULT_TARGET_PODCASTS)


@dataclass
class Config:
    """全局配置，从环境变量和默认值加载"""

    # === 项目路径 ===
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data"
    )

    # === 硬件环境 ===
    gpu_available: bool = False  # 无GPU
    device: str = "cpu"

    # === Listen Notes API ===
    listennotes_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("LISTENNOTES_API_KEY")
    )
    listennotes_base_url: str = "https://listen-api.listennotes.com/api/v2"

    # === LLM (OpenAI Compatible 协议) ===
    llm_api_key: Optional[str] = field(
        default_factory=lambda: _env("LLM_API_KEY", "DASHSCOPE_API_KEY") or None
    )
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    llm_model: str = field(
        default_factory=lambda: _env("LLM_MODEL", default="qwen3.7-plus")
    )

    # === Embedding (OpenAI Compatible 协议) ===
    embedding_api_key: Optional[str] = field(
        default_factory=lambda: _env("EMBEDDING_API_KEY", "DASHSCOPE_API_KEY") or None
    )
    embedding_base_url: str = field(
        default_factory=lambda: _env("EMBEDDING_BASE_URL", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", default="text-embedding-v4")
    )
    embedding_provider: str = field(
        default_factory=lambda: _env("EMBEDDING_PROVIDER", default="api")  # "api" | "local"
    )

    # === Rerank (HTTP API; OpenAI 协议无 rerank 标准, 走供应商 REST) ===
    rerank_api_key: Optional[str] = field(
        default_factory=lambda: _env("RERANK_API_KEY", "DASHSCOPE_API_KEY") or None
    )
    rerank_base_url: str = field(
        default_factory=lambda: _env("RERANK_BASE_URL", default="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank")
    )
    rerank_model: str = field(
        default_factory=lambda: _env("RERANK_MODEL", default="qwen3-rerank")
    )

    # === 语音转录 (HTTP filetrans 协议: 提交任务 + 轮询) ===
    asr_api_key: Optional[str] = field(
        default_factory=lambda: _env("ASR_API_KEY", "DASHSCOPE_API_KEY") or None
    )
    asr_upload_url: str = field(
        default_factory=lambda: _env("ASR_UPLOAD_URL", default="https://dashscope.aliyuncs.com/api/v1/uploads")
    )
    asr_submit_url: str = field(
        default_factory=lambda: _env("ASR_SUBMIT_URL", default="https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription")
    )
    asr_task_url: str = field(
        default_factory=lambda: _env("ASR_TASK_URL", default="https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}")
    )
    asr_model: str = field(
        default_factory=lambda: _env("ASR_MODEL", default="qwen-audio-3.0-asr-flash-filetrans")
    )
    asr_language: str = "en"
    asr_poll_interval: int = 3  # 轮询间隔（秒）
    asr_max_wait: int = 300  # 最大等待时间（秒）

    # === 本地 Embedding 模型 (embedding_provider=local 时使用) ===
    embedding_local_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 1024  # text-embedding-v4 输出维度

    # === ChromaDB ===
    chroma_persist_dir: str = field(
        default_factory=lambda: str(
            Path(__file__).parent.parent.parent / "data" / "chroma_db"
        )
    )
    chroma_collection_name: str = "podcast_transcripts"

    # === SQLite ===
    sqlite_path: str = field(
        default_factory=lambda: str(
            Path(__file__).parent.parent.parent / "data" / "podcast_agent.db"
        )
    )

    # === Podcast 采集配置 ===
    # 指定采集的播客名称（逗号分隔），从环境变量 TARGET_PODCASTS 读取；
    # 未配置时使用 DEFAULT_TARGET_PODCASTS 默认列表
    target_podcasts: list = field(
        default_factory=lambda: _parse_podcast_list(
            _env("TARGET_PODCASTS", default="")
        )
    )
    # 每个播客采集的最新集数
    episodes_per_podcast: int = field(
        default_factory=lambda: int(_env("EPISODES_PER_PODCAST", default="1"))
    )

    # === Agent 配置 ===
    max_retries: int = 3
    request_timeout: int = 120  # qwen3.7-plus CoT推理需要较长时间
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # === 并发配置 ===
    # 全局并发的 LLM/Agent 调用上限（块级 x 集级共用一个信号量，防止打爆 API 限流）
    llm_concurrency: int = field(
        default_factory=lambda: int(_env("LLM_CONCURRENCY", default="6"))
    )
    # 同时分析的播客集数上限（每集内部仍按块并发）
    episode_concurrency: int = field(
        default_factory=lambda: int(_env("EPISODE_CONCURRENCY", default="2"))
    )
    # 多文件 ASR 并发上传/提交数
    asr_concurrency: int = field(
        default_factory=lambda: int(_env("ASR_CONCURRENCY", default="3"))
    )

    # === 情感分析 ===
    sentiment_threshold: float = 0.6  # FinBERT置信度阈值
    finbert_model: str = "ProsusAI/finbert"

    # === RAG 配置 ===
    # 是否启用 RAG 检索增强（默认关闭，跳过 retriever 节点以省 embedding/rerank 配额；
    # 需要长期运行、积累跨期语料时再设为 true）
    enable_rag: bool = field(
        default_factory=lambda: _env("ENABLE_RAG", default="false").lower()
        in ("1", "true", "yes", "on")
    )
    rag_top_k: int = 5
    chunk_size: int = 2000  # tokens
    chunk_overlap: int = 200
    hybrid_dense_weight: float = 0.6  # 混合检索中语义检索的权重

    def __post_init__(self):
        """初始化后处理"""
        # 确保数据目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def listennotes_available(self) -> bool:
        return bool(self.listennotes_api_key)

    @property
    def llm_available(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def embedding_available(self) -> bool:
        return bool(self.embedding_api_key)

    @property
    def rerank_available(self) -> bool:
        return bool(self.rerank_api_key)

    @property
    def asr_available(self) -> bool:
        return bool(self.asr_api_key)


# 全局单例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = Config()
    return _config