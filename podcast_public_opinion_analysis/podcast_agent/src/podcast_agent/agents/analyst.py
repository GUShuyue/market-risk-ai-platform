"""
Analyst Agent — LLM深度推理 (Day 5-6 实现)

负责:
  1. 实体提取 (公司/行业/指标/人物)
  2. 宏观主题识别
  3. 中国市场关联度评分 (0-1)
  4. Chain-of-Thought 逻辑推演
  5. 中英文双语摘要

技术栈:
  - LLM: DashScope Qwen
  - CoT: 逐步推理，展示从信息到结论的完整逻辑链
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from ..config import get_config
from ..models.messages import (
    AgentStatus, SentimentOutput, AnalystOutput, Entity, Sentiment,
)
from .base import BaseAgent

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent[SentimentOutput, AnalystOutput]):
    """LLM深度推理Agent"""

    agent_name = "analyst"

    def execute(self, input_data: SentimentOutput) -> AnalystOutput:
        self.log_start(
            sentiment=input_data.overall_sentiment.value,
            score=input_data.sentiment_score,
        )

        output = AnalystOutput(
            agent_status=AgentStatus.RUNNING,
            episode_id=input_data.episode_id,
            overall_sentiment=input_data.overall_sentiment,
            confidence=input_data.confidence,
        )

        client = self.get_llm_client()
        if client is None:
            output.agent_status = AgentStatus.FAILED
            output.error_message = "LLM 不可用"
            return output

        try:
            # 构建分析上下文
            context = self._build_context(input_data)

            # 1. CoT 深度推理
            cot_result = self._cot_reasoning(client, context)

            # 2. 结构化提取
            output = self._parse_cot_result(output, cot_result, input_data)

            output.agent_status = AgentStatus.SUCCESS
            self.log_done(
                entities=len(output.entities),
                topics=len(output.macro_topics),
                china_relevance=f"{output.china_relevance:.2f}",
            )

        except Exception as e:
            self.log_error(str(e))
            output.agent_status = AgentStatus.FAILED
            output.error_message = str(e)

        return output

    def _build_context(self, input_data: SentimentOutput) -> str:
        """构建 LLM 分析上下文（显式携带原文，供实体提取引用证据）"""
        parts = [
            "=== Financial Podcast Analysis Context ===",
            f"Overall Sentiment: {input_data.overall_sentiment.value}",
            f"Sentiment Score: {input_data.sentiment_score:.3f} (-1 bearish to +1 bullish)",
            f"Confidence: {input_data.confidence:.3f}",
        ]

        # 原文（显式字段）——Analyst 提取实体/话题的依据
        if input_data.source_text:
            parts.append("")
            parts.append("=== PODCAST TRANSCRIPT (SOURCE TEXT) ===")
            parts.append(input_data.source_text)
            parts.append("=== END TRANSCRIPT ===")

        if input_data.key_positive_factors:
            parts.append(f"Positive Factors: {', '.join(input_data.key_positive_factors)}")
        if input_data.key_negative_factors:
            parts.append(f"Negative Factors: {', '.join(input_data.key_negative_factors)}")
        if input_data.hedging_language_detected:
            parts.append("Note: Hedging language detected — uncertainty in analysis")
        if input_data.llm_sentiment_analysis:
            parts.append(f"Brief Analysis: {input_data.llm_sentiment_analysis}")
        if input_data.finbert_scores:
            parts.append(
                f"FinBERT Scores: pos={input_data.finbert_scores.get('positive', 0):.3f} "
                f"neg={input_data.finbert_scores.get('negative', 0):.3f} "
                f"neu={input_data.finbert_scores.get('neutral', 0):.3f}"
            )

        return "\n".join(parts)

    def _cot_reasoning(self, client, context: str) -> str:
        """
        Chain-of-Thought 逐步推理

        让 LLM 展示从信息到结论的完整逻辑链
        """
        model = self.get_llm_model()

        prompt = f"""{context}

You are a senior financial analyst specializing in cross-market analysis between US markets and China's A-share market.

Please perform Chain-of-Thought reasoning step by step:

Step 1 — Entity Extraction: Identify all mentioned companies, sectors, economic indicators, and key persons. For each, note: name, ticker (if known), type (company/sector/indicator/person), sentiment, and evidence (quote the EXACT sentence(s) from the TRANSCRIPT above that mention this entity — verbatim, do not paraphrase).

sentiment_score scoring rubric (-1.0 extremely bearish ~ +1.0 extremely bullish, reflecting the podcast's attitude toward that entity):
- +0.7 ~ +1.0: strongly bullish / clearly positive tone
- +0.3 ~ +0.7: leaning positive
- -0.3 ~ +0.3: neutral or mixed
- -0.7 ~ -0.3: leaning negative
- -1.0 ~ -0.7: strongly bearish / clearly negative tone

Step 2 — Macro Topic Identification: List the macro-economic themes discussed (e.g., "Fed monetary policy", "semiconductor supply chain", "China stimulus").

Step 3 — China Relevance Assessment: Rate how relevant this content is to China's A-share market on a scale of 0.0 to 1.0. Explain your reasoning.

Step 4 — Logic Chain: Build a step-by-step logic chain from the podcast content to potential A-share market impact. Format: "IF [condition] THEN [implication] BECAUSE [reasoning]".

Step 5 — Summary: Provide a concise summary in Chinese (中文) and English.

Finally, output the results in this JSON format:
```json
{{
  "entities": [
    {{"name": "...", "ticker": "..." or null, "entity_type": "company"|"sector"|"indicator"|"person", "sentiment": "bullish"|"bearish"|"neutral", "sentiment_score": -1.0 to 1.0, "key_points": ["..."], "evidence": "verbatim sentence(s) from the transcript that mention this entity"}}
  ],
  "macro_topics": ["topic1", "topic2", ...],
  "china_relevance": 0.0 to 1.0,
  "china_relevance_reasoning": "...",
  "logic_chain": "IF ... THEN ... BECAUSE ...",
  "summary_cn": "中文摘要",
  "summary_en": "English summary"
}}
```

ONLY output the JSON, no other text."""

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )

        content = resp.choices[0].message.content.strip()
        self.logger.info(f"  CoT 推理完成 ({len(content)} 字符)")
        return content

    def _parse_cot_result(
        self,
        output: AnalystOutput,
        raw: str,
        input_data: SentimentOutput,
    ) -> AnalystOutput:
        """解析 CoT 推理结果"""
        # 保存原始推理
        output.cot_reasoning = raw

        # 提取 JSON
        json_str = self._extract_json(raw)

        try:
            data = json.loads(json_str)

            # 实体
            for ent in data.get("entities", []):
                try:
                    sent = Sentiment(ent.get("sentiment", "neutral"))
                except ValueError:
                    sent = Sentiment.NEUTRAL

                output.entities.append(Entity(
                    name=ent.get("name", ""),
                    ticker=ent.get("ticker"),
                    entity_type=ent.get("entity_type", "company"),
                    sentiment=sent,
                    sentiment_score=ent.get("sentiment_score", 0.0),
                    key_points=ent.get("key_points", []),
                    evidence=ent.get("evidence", ""),
                ))

            # 宏观主题
            output.macro_topics = data.get("macro_topics", [])

            # 中国关联度
            output.china_relevance = data.get("china_relevance", 0.5)

            # 逻辑链
            output.logic_chain = data.get("logic_chain", "")

            # 摘要
            output.summary_cn = data.get("summary_cn", "")
            output.summary_en = data.get("summary_en", "")

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"JSON 解析失败: {e}，使用原始文本")
            output.summary_en = raw[:500]
            output.summary_cn = raw[:500]

        return output

    @staticmethod
    def _extract_json(text: str) -> str:
        """从文本中提取 JSON 块"""
        # 尝试匹配 ```json ... ``` 块
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            return match.group(1).strip()

        # 尝试匹配 { ... } 块
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return match.group(0).strip()

        return text.strip()