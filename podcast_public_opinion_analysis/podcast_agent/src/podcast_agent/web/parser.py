"""
自然语言意图解析 — 把用户指令解析为结构化操作

优先用 LLM（DashScope Qwen）解析，失败时回退到简单正则。

输出结构:
    {
        "action": "collect" | "upload" | "unknown",
        "podcast_names": ["播客名1", ...],
        "episodes_per_podcast": 1,
    }
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..config import get_config

logger = logging.getLogger(__name__)

# 上传/本地文件类意图的关键词
_UPLOAD_HINTS = ("上传", "本地文件", "本地", "这个文件", "这些文件", "mp3", "MP3")
# 采集类意图的关键词
_COLLECT_HINTS = ("采集", "抓取", "分析", "最新", "集", "播客")


def parse_intent(user_input: str) -> Dict[str, Any]:
    """
    解析用户自然语言输入，返回结构化意图。

    优先 LLM 解析，失败回退正则，最终兜底 unknown。
    """
    user_input = (user_input or "").strip()
    if not user_input:
        return _empty("unknown")

    # 1. LLM 解析
    try:
        result = _llm_parse(user_input)
        if result and result.get("action"):
            return _normalize(result)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"LLM 意图解析失败: {e}")

    # 2. 正则兜底
    result = _regex_parse(user_input)
    if result and result.get("action") != "unknown":
        return _normalize(result)

    return _empty("unknown")


def _llm_parse(user_input: str) -> Optional[Dict[str, Any]]:
    """用 LLM 解析意图"""
    config = get_config()
    if not config.llm_available:
        return None

    from openai import OpenAI

    client = OpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        timeout=config.request_timeout,
        max_retries=config.max_retries,
    )

    prompt = f"""你是播客分析系统的意图解析器。把用户指令解析成 JSON。

指令类型:
- collect: 采集指定播客的最新节目并分析
- upload: 上传本地播客文件分析
- unknown: 无法识别

规则:
1. 用户提到具体播客名(如 Odd Lots、FT News Briefing、WSJ What's News)时，action=collect，提取播客名列表和集数(未指定集数默认 1)。
2. 用户提到"上传/本地文件/这个文件/mp3"等时，action=upload。
3. 其他情况 action=unknown。

只输出 JSON，不要输出其他文字:
{{"action": "collect", "podcast_names": ["播客名"], "episodes_per_podcast": 1}}

用户指令: {user_input}
"""

    resp = client.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    content = resp.choices[0].message.content.strip()

    # 提取 JSON
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        return None
    return json.loads(m.group(0))


def _regex_parse(user_input: str) -> Dict[str, Any]:
    """正则兜底：识别 upload/collect 意图 + 集数"""
    # 上传意图
    if any(h in user_input for h in _UPLOAD_HINTS):
        return {"action": "upload", "podcast_names": [], "episodes_per_podcast": 1}

    # 集数
    m_ep = re.search(r"(\d+)\s*集", user_input)
    episodes = int(m_ep.group(1)) if m_ep else 1

    # 播客名：优先引号包裹的内容
    quoted = re.findall(r"[\"'「」]([^\"'「」]+)[\"'「」]", user_input)
    if quoted:
        return {"action": "collect", "podcast_names": quoted, "episodes_per_podcast": episodes}

    # 播客名：匹配已知播客名（config.target_podcasts 里出现过的）
    config = get_config()
    known = [p for p in config.target_podcasts if p.lower() in user_input.lower()]
    if known:
        return {"action": "collect", "podcast_names": known, "episodes_per_podcast": episodes}

    # 有采集类关键词但无明确播客名 → collect（播客名留给表单）
    if any(h in user_input for h in _COLLECT_HINTS):
        return {"action": "collect", "podcast_names": [], "episodes_per_podcast": episodes}

    return _empty("unknown")


def _normalize(result: Dict[str, Any]) -> Dict[str, Any]:
    """规整字段，保证结构完整"""
    action = str(result.get("action", "unknown")).strip().lower()
    if action not in ("collect", "upload", "unknown"):
        action = "unknown"

    names = result.get("podcast_names") or []
    if isinstance(names, str):
        names = [names]
    names = [str(n).strip() for n in names if str(n).strip()]

    episodes = result.get("episodes_per_podcast") or 1
    try:
        episodes = int(episodes)
    except (TypeError, ValueError):
        episodes = 1
    episodes = max(1, min(episodes, 20))

    return {
        "action": action,
        "podcast_names": names,
        "episodes_per_podcast": episodes,
    }


def _empty(action: str) -> Dict[str, Any]:
    return {"action": action, "podcast_names": [], "episodes_per_podcast": 1}
