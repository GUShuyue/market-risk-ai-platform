"""
Agent 基类 — 所有Agent的抽象基类
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar

from ..config import Config, get_config

# 泛型：输入类型和输出类型
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")

logger = logging.getLogger(__name__)


class BaseAgent(ABC, Generic[InputType, OutputType]):
    """
    Agent 抽象基类

    每个Agent子类需要实现:
      - agent_name: 唯一标识
      - execute(input_data) -> output_data: 核心执行逻辑
    """

    agent_name: str = "base"

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"podcast_agent.{self.agent_name}")
        self._llm_client = None

    @abstractmethod
    def execute(self, input_data: InputType) -> OutputType:
        """
        执行Agent核心逻辑

        Args:
            input_data: 上游Agent的输出（类型化消息）

        Returns:
            类型化的输出消息，传递给下游Agent
        """
        ...

    def log_start(self, **kwargs: Any) -> None:
        """记录Agent开始执行"""
        self.logger.info(
            f"[{self.agent_name}] 开始执行 | {self._format_kwargs(**kwargs)}"
        )

    def log_done(self, status: str = "SUCCESS", **kwargs: Any) -> None:
        """记录Agent执行完成"""
        self.logger.info(
            f"[{self.agent_name}] 完成 [{status}] | {self._format_kwargs(**kwargs)}"
        )

    def log_error(self, error: str, **kwargs: Any) -> None:
        """记录Agent执行错误"""
        self.logger.error(
            f"[{self.agent_name}] 错误: {error} | {self._format_kwargs(**kwargs)}"
        )

    @staticmethod
    def _format_kwargs(**kwargs: Any) -> str:
        """格式化日志参数"""
        if not kwargs:
            return ""
        return " | ".join(f"{k}={v}" for k, v in kwargs.items())

    def get_llm_client(self):
        """获取LLM客户端 (OpenAI Compatible 协议，实例级缓存，线程安全)"""
        from openai import OpenAI

        if self._llm_client is not None:
            return self._llm_client

        if self.config.llm_available:
            # OpenAI SDK 内置 429/连接错误自动重试 (max_retries)
            client = OpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                timeout=self.config.request_timeout,
                max_retries=self.config.max_retries,
            )
            self._llm_client = client
            return client

        return None

    def get_llm_model(self) -> str:
        """获取当前使用的LLM模型名"""
        if self.config.llm_available:
            return self.config.llm_model
        return ""