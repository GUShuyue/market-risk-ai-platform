"""
LLM Client - 统一大模型API客户端

Qwen(通义千问)和OpenAI都使用OpenAI兼容的Chat Completions接口，
请求格式完全一致，因此客户端统一为单一实现，只需配置不同的
api_key / base_url / model 即可切换provider。

密钥管理:
    API Key和Base URL通过 .env 文件管理，不硬编码在配置文件中。
    .env 文件示例见 .env.example
"""

import json
import os
import logging
from typing import Dict, List, Optional
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class LLMClient:
    """
    统一大模型客户端 (OpenAI兼容协议)

    通过 .env 配置以下环境变量:
        LLM_API_KEY   - API密钥
        LLM_BASE_URL  - API地址 (如 https://dashscope.aliyuncs.com/compatible-mode/v1)
        LLM_MODEL     - 模型名称 (如 qwen-max)
        LLM_TEMPERATURE - 温度参数 (可选，默认0.3)
        LLM_MAX_TOKENS  - 最大token数 (可选，默认4096)
        LLM_TIMEOUT     - 超时秒数 (可选，默认60)
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化客户端

        参数:
            config: 可选配置覆盖，支持以下key:
                api_key, base_url, model, temperature, max_tokens, timeout
                未提供的字段从环境变量读取
        """
        config = config or {}

        self.api_key = config.get("api_key") or os.getenv("LLM_API_KEY", "")
        self.base_url = config.get("base_url") or os.getenv(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = config.get("model") or os.getenv("LLM_MODEL", "qwen-max")
        self.temperature = float(
            config.get("temperature") or os.getenv("LLM_TEMPERATURE", "0.3")
        )
        self.max_tokens = int(
            config.get("max_tokens") or os.getenv("LLM_MAX_TOKENS", "4096")
        )
        self.timeout = int(
            config.get("timeout") or os.getenv("LLM_TIMEOUT", "60")
        )

        if not self.api_key:
            logger.warning(
                "未配置LLM_API_KEY，请在 .env 文件中设置 "
                "(参考 .env.example)"
            )

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        调用大模型对话

        参数:
            prompt: 用户提示词
            system_prompt: 系统提示词 (可选)
            temperature: 温度参数 (可选，覆盖默认值)
            max_tokens: 最大输出token数 (可选，覆盖默认值)

        返回:
            模型返回的文本

        异常:
            RuntimeError: API调用失败时抛出
        """
        url = f"{self.base_url}/chat/completions"

        messages = self._build_messages(prompt, system_prompt)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        return self._send_request(url, headers, payload)

    def chat_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        调用大模型并解析JSON响应

        适用于需要结构化输出的场景 (意图识别、定性分析等)。
        prompt中应明确要求模型返回JSON格式。

        参数:
            prompt: 用户提示词 (应要求模型返回JSON)
            system_prompt: 系统提示词
            temperature: 温度参数

        返回:
            解析后的JSON字典

        异常:
            RuntimeError: API调用失败
            json.JSONDecodeError: 响应无法解析为JSON
        """
        response_text = self.chat(prompt, system_prompt, temperature)
        json_str = self._extract_json(response_text)
        return json.loads(json_str)

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str]
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _send_request(self, url: str, headers: dict, payload: dict) -> str:
        """
        发送HTTP请求并解析响应

        统一处理:
            - HTTP错误 (非200状态码)
            - 网络异常
            - 响应解析
            - API级错误 (限流、余额不足等)
        """
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )

            if response.status_code != 200:
                error_msg = self._parse_error_response(response)
                logger.error(f"LLM API调用失败 [{response.status_code}]: {error_msg}")
                raise RuntimeError(
                    f"LLM API调用失败 (HTTP {response.status_code}): {error_msg}"
                )

            data = response.json()

            # 检查API级错误 (有时返回200但body里有error字段)
            if "error" in data:
                error_info = data["error"]
                error_msg = error_info.get("message", str(error_info))
                error_code = error_info.get("code", "unknown")
                logger.error(f"LLM API返回错误 [{error_code}]: {error_msg}")
                raise RuntimeError(f"LLM API错误 ({error_code}): {error_msg}")

            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("LLM API返回为空: 无choices")

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise RuntimeError("LLM API返回为空: 无content")

            usage = data.get("usage", {})
            if usage:
                logger.debug(
                    f"LLM调用完成 - "
                    f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
                    f"completion_tokens={usage.get('completion_tokens', '?')}, "
                    f"total_tokens={usage.get('total_tokens', '?')}"
                )

            return content

        except requests.exceptions.Timeout:
            logger.error(f"LLM API请求超时 ({self.timeout}s)")
            raise RuntimeError(f"LLM API请求超时 ({self.timeout}秒)")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"LLM API连接失败: {e}")
            raise RuntimeError(f"LLM API连接失败: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API请求异常: {e}")
            raise RuntimeError(f"LLM API请求异常: {e}")

    def _parse_error_response(self, response: requests.Response) -> str:
        """解析错误响应，提取可读的错误信息"""
        try:
            data = response.json()
            if "error" in data:
                return data["error"].get("message", str(data["error"]))
            if "message" in data:
                return data.get("message", str(data))
            return str(data)
        except (json.JSONDecodeError, ValueError):
            return response.text[:500] if response.text else "无响应内容"

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        从模型回复中提取JSON字符串

        模型有时会在JSON前后添加markdown标记或说明文字:
            "```json\n{...}\n```"
            "好的，结果如下：\n{...}"
        """
        text = text.strip()

        # 尝试直接解析
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 块
        if "```json" in text:
            start = text.find("```json") + len("```json")
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # 尝试提取 ``` ... ``` 块 (无json标记)
        if "```" in text:
            start = text.find("```") + len("```")
            if text[start:start+4].strip().lower() in ("json", ""):
                start = text.find("\n", start) + 1
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # 尝试提取第一个 { 到最后一个 }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1]

        return text
