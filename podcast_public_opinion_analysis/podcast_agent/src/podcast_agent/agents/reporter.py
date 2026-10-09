"""
Reporter Agent — 报告生成与可视化

负责:
  1. LLM 综合分析多集单集分析结果，生成聚合报告
  2. 计算多集共识/分歧，展示情感一致性
  3. 生成包含 ECharts 可视化的自包含 HTML 报告
  4. 保存报告到 data/reports/ 目录

技术栈:
  - LLM: DashScope Qwen (报告文本生成)
  - ECharts: 情感仪表盘、A股板块影响柱状图、共识散点图
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import get_config
from ..models.messages import (
    AgentStatus, EpisodeAnalysis, ReporterInput, ReporterOutput, Sentiment,
    AShareImpact,
)
from .base import BaseAgent

logger = logging.getLogger(__name__)

_ECHARTS_LIB_PATH = Path(__file__).resolve().parent.parent / "assets" / "echarts.min.js"
_ECHARTS_CDN_TAG = (
    '<script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.5.0/echarts.min.js"\n'
    '    onerror="var s=document.createElement(\'script\');'
    "s.src='https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js';"
    'document.head.appendChild(s);">\n</script>'
)


class ReporterAgent(BaseAgent[ReporterInput, ReporterOutput]):
    """报告生成与可视化 Agent"""

    agent_name = "reporter"

    def execute(self, input_data: ReporterInput) -> ReporterOutput:
        self.log_start(
            episode_count=len(input_data.episode_analyses),
        )

        output = ReporterOutput(
            agent_status=AgentStatus.RUNNING,
            report_id=str(uuid.uuid4())[:8],
            generated_at=datetime.now(),
        )

        # 多集模式
        if input_data.episode_analyses:
            output.episode_title = f"{len(input_data.episode_analyses)} 集播客聚合分析"
            output.podcast_title = self._collect_podcast_names(input_data.episode_analyses)
        else:
            output.episode_id = input_data.episode_id
            output.podcast_title = input_data.podcast_title
            output.episode_title = input_data.episode_title

        try:
            # 1. LLM 生成聚合分析报告
            report_text = self._generate_report_text(input_data)

            # 2. 生成 HTML 报告
            html_path = self._generate_html_report(input_data, report_text, output.report_id)

            output.report_html_path = html_path
            output.report_markdown = report_text
            output.agent_status = AgentStatus.SUCCESS

            self.log_done(
                report_id=output.report_id,
                html_path=html_path,
            )

        except Exception as e:
            self.log_error(str(e))
            output.agent_status = AgentStatus.FAILED
            output.error_message = str(e)

        return output

    def _collect_podcast_names(self, analyses: list) -> str:
        """收集所有播客名称"""
        names = list(set(a.podcast_title for a in analyses if a.podcast_title))
        if len(names) <= 3:
            return ", ".join(names)
        return ", ".join(names[:3]) + f" 等 {len(names)} 个播客"

    # ================================================================
    # LLM 报告生成
    # ================================================================

    def _generate_report_text(self, input_data: ReporterInput) -> str:
        """使用 LLM 生成聚合分析报告"""
        client = self.get_llm_client()
        if client is None:
            return self._fallback_report_text(input_data)

        model = self.get_llm_model()

        if input_data.episode_analyses:
            context = self._build_multi_episode_context(input_data.episode_analyses)
            prompt = self._build_aggregation_prompt(context)
        else:
            context = self._build_report_context(input_data)
            prompt = self._build_single_prompt(context)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"LLM 报告生成失败: {e}")
            return self._fallback_report_text(input_data)

    def _build_aggregation_prompt(self, context: str) -> str:
        """多集聚合报告 prompt（固定 6 个小节，含各播客内容概览 + EP 对应关系）"""
        return f"""{context}

You are a senior financial analyst writing a cross-market impact report for analysts/investment advisors at a Chinese securities firm.

You have analyzed multiple podcast episodes individually. Now synthesize all findings into ONE report in Chinese.

IMPORTANT — fixed structure: The report MUST follow EXACTLY the 6 sections below, with these EXACT headings (in Markdown "## 标题" form), in this exact order. Do NOT add, remove, rename, or reorder any section.

## 执行摘要
- 分析了几集、来自哪些播客
- 各播客的情感倾向概览（各自针对自己的主题；主题相近时指出倾向是否一致，主题不同则分别说明，不要跨主题求平均）
- 跨集共同主题
- 对 A 股最重要的启示

## 各播客内容概览
- 用列表逐一说明每集聊了什么核心内容，格式统一为 "EP1（播客名）：1~3 句话概括主题与关键观点"
- 明确建立 EP 编号与播客名的对应关系；此后全文引用某集时都用 "EP1（播客名）" 这种格式，禁止只写光秃秃的 "EP1"

## 各播客情感
- 逐一说明每个播客各自的情感倾向（看多/看空/中性）及简要理由
- 情感是针对各自主题的，不要跨主题求平均或比较"共识"；只有当多个播客讨论同一主题时，才比较它们的情感倾向

## A股相关板块提示
- 对每个关联板块：提及集数（标注是哪几集 EP 提及）、方向、影响程度（强度）
- 标注高共识板块（多集一致提及）
- 每个板块必须写清「传导逻辑」：播客中的具体内容 → 通过什么路径 → 影响该板块，要求具体、可追溯，禁止笼统一句"相关"

## 关键实体与主题
- 多集共同提及的实体/主题
- 单集独有的实体/主题
- 跨集宏观主题

## 风险提示
- 汇总风险，按影响程度排序，突出高影响风险

Format in Markdown. Use Chinese throughout. 面向券商研究员，内容要具体、有出处、可追溯，杜绝空泛表述。Base every claim strictly on the podcast content above — do NOT fabricate or over-extrapolate.

Return ONLY the markdown report text, starting with the heading "## 执行摘要"."""

    def _build_single_prompt(self, context: str) -> str:
        """单集报告 prompt（固定 5 个小节结构，兼容旧版）"""
        return f"""{context}

You are a senior financial analyst writing a cross-market impact report for analysts at a Chinese securities firm.

Based on the analysis data above, write a report in Chinese with EXACTLY these 5 sections, in this exact order (Markdown "## 标题" form):
1. ## 执行摘要
2. ## 各播客情感
3. ## A股相关板块提示
4. ## 关键实体与主题
5. ## 风险提示

Do NOT add, remove, or rename any section. 每个板块要写清「传导逻辑」（播客内容 → 路径 → 板块），内容具体、可追溯。Base every claim on the content above.

Format in Markdown. Return ONLY the markdown report text, starting with "## 执行摘要"."""

    def _build_multi_episode_context(self, analyses: List[EpisodeAnalysis]) -> str:
        """构建多集分析上下文"""
        parts = [
            "=== Multi-Episode Podcast Analysis Context ===",
            f"Total Episodes: {len(analyses)}",
            "",
        ]

        # 每集摘要
        for i, ea in enumerate(analyses):
            parts.append(f"--- Episode {i+1}: {ea.podcast_title} - {ea.episode_title[:60]} ---")

            sent = ea.sentiment_output
            if sent and sent.agent_status == AgentStatus.SUCCESS:
                parts.append(f"Sentiment: {sent.overall_sentiment.value} (score={sent.sentiment_score:+.2f}, conf={sent.confidence:.2f})")
                if sent.finbert_scores:
                    parts.append(f"FinBERT: pos={sent.finbert_scores.get('positive', 0):.2f}, "
                                f"neg={sent.finbert_scores.get('negative', 0):.2f}, "
                                f"neu={sent.finbert_scores.get('neutral', 0):.2f}")
                if sent.key_positive_factors:
                    parts.append(f"Positive: {', '.join(sent.key_positive_factors[:3])}")
                if sent.key_negative_factors:
                    parts.append(f"Negative: {', '.join(sent.key_negative_factors[:3])}")

            analyst = ea.analyst_output
            if analyst and analyst.agent_status == AgentStatus.SUCCESS:
                parts.append(f"China Relevance: {analyst.china_relevance:.2f}")
                if analyst.entities:
                    entity_strs = [f"{e.name}({e.sentiment.value})" for e in analyst.entities[:5]]
                    parts.append(f"Entities: {', '.join(entity_strs)}")
                if analyst.macro_topics:
                    parts.append(f"Topics: {', '.join(analyst.macro_topics[:5])}")
                if analyst.summary_cn:
                    parts.append(f"Summary: {analyst.summary_cn[:200]}")

            cm = ea.crossmarket_output
            if cm and cm.agent_status == AgentStatus.SUCCESS:
                if cm.a_share_impacts:
                    impact_strs = [
                        f"{imp.sector_name}({imp.impact_direction.value}, strength={imp.impact_strength:.2f})"
                        for imp in cm.a_share_impacts[:5]
                    ]
                    parts.append(f"Impacts: {', '.join(impact_strs)}")
                if cm.risk_warnings:
                    parts.append(f"Risks: {'; '.join(cm.risk_warnings[:3])}")
                if cm.overall_assessment:
                    parts.append(f"Assessment: {cm.overall_assessment[:200]}")

            parts.append("")

        # 汇总统计：只保留高频板块/实体与同板块方向分歧，不做跨主题情感平均/共识
        consensus = self._compute_consensus(analyses)
        parts.append("--- Cross-Episode Summary ---")
        if consensus['top_sectors']:
            parts.append(f"Top Sectors: {', '.join(consensus['top_sectors'])}")
        if consensus['top_entities']:
            parts.append(f"Top Entities: {', '.join(consensus['top_entities'])}")
        if consensus['divergences']:
            parts.append(f"Sector Direction Divergences: {'; '.join(consensus['divergences'][:3])}")
        parts.append("")

        return "\n".join(parts)

    def _compute_consensus(self, analyses: List[EpisodeAnalysis]) -> Dict[str, Any]:
        """计算跨集汇总统计（高频板块/实体、同板块方向分歧）"""
        sentiments = []
        scores = []
        all_sectors = []
        all_entities = []
        divergences = []

        for ea in analyses:
            sent = ea.sentiment_output
            if sent and sent.agent_status == AgentStatus.SUCCESS:
                sentiments.append(sent.overall_sentiment.value)
                scores.append(sent.sentiment_score)

            cm = ea.crossmarket_output
            if cm and cm.agent_status == AgentStatus.SUCCESS:
                for imp in (cm.a_share_impacts or []):
                    all_sectors.append((imp.sector_name, imp.impact_direction.value))

            analyst = ea.analyst_output
            if analyst and analyst.agent_status == AgentStatus.SUCCESS:
                for e in (analyst.entities or []):
                    all_entities.append(e.name)

        # 情感分布
        sentiment_counts = Counter(sentiments)
        dist = ", ".join(f"{k}={v}" for k, v in sentiment_counts.most_common())

        # 共识程度
        if len(sentiments) >= 2:
            dominant = sentiment_counts.most_common(1)[0]
            consensus_ratio = dominant[1] / len(sentiments)
            if consensus_ratio >= 0.8:
                consensus_level = "高"
            elif consensus_ratio >= 0.5:
                consensus_level = "中"
            else:
                consensus_level = "低"
        else:
            consensus_level = "N/A (单集)"

        avg_score = sum(scores) / len(scores) if scores else 0.0

        # 高频板块
        sector_counts = Counter(name for name, _ in all_sectors)
        top_sectors = [f"{name}({count})" for name, count in sector_counts.most_common(5)]

        # 高频实体
        entity_counts = Counter(all_entities)
        top_entities = [f"{name}({count})" for name, count in entity_counts.most_common(5)]

        # 分歧检测
        sector_directions = {}
        for name, direction in all_sectors:
            if name not in sector_directions:
                sector_directions[name] = set()
            sector_directions[name].add(direction)
        for name, dirs in sector_directions.items():
            if len(dirs) > 1:
                divergences.append(f"{name}: {', '.join(dirs)}")

        return {
            "sentiment_distribution": dist,
            "consensus_level": consensus_level,
            "avg_score": avg_score,
            "top_sectors": top_sectors,
            "top_entities": top_entities,
            "divergences": divergences,
            "sentiment_counts": dict(sentiment_counts),
            "episode_count": len(analyses),
        }

    def _build_report_context(self, input_data: ReporterInput) -> str:
        """构建单集报告上下文（兼容旧版）"""
        parts = [
            "=== Podcast Analysis Report Context ===",
            f"Podcast: {input_data.podcast_title}",
            f"Episode: {input_data.episode_title}",
            f"Episode ID: {input_data.episode_id}",
            "",
        ]

        sent = input_data.sentiment_output
        if sent and sent.agent_status == AgentStatus.SUCCESS:
            parts.append("--- Sentiment Analysis ---")
            parts.append(f"Overall Sentiment: {sent.overall_sentiment.value}")
            parts.append(f"Sentiment Score: {sent.sentiment_score:.2f}")
            parts.append(f"Confidence: {sent.confidence:.2f}")
            if sent.finbert_scores:
                parts.append(f"FinBERT Scores: pos={sent.finbert_scores.get('positive', 0):.2f}, "
                            f"neg={sent.finbert_scores.get('negative', 0):.2f}, "
                            f"neu={sent.finbert_scores.get('neutral', 0):.2f}")
            if sent.key_positive_factors:
                parts.append(f"Positive Factors: {', '.join(sent.key_positive_factors)}")
            if sent.key_negative_factors:
                parts.append(f"Negative Factors: {', '.join(sent.key_negative_factors)}")
            if sent.llm_sentiment_analysis:
                parts.append(f"LLM Analysis: {sent.llm_sentiment_analysis[:500]}")
            parts.append("")

        analyst = input_data.analyst_output
        if analyst and analyst.agent_status == AgentStatus.SUCCESS:
            parts.append("--- Analyst Entities & Topics ---")
            parts.append(f"China Relevance: {analyst.china_relevance:.2f}")
            if analyst.entities:
                for e in analyst.entities:
                    parts.append(f"  - {e.name} ({e.entity_type}): {e.sentiment.value} score={e.sentiment_score:.2f}")
            if analyst.macro_topics:
                parts.append(f"Macro Topics: {', '.join(analyst.macro_topics)}")
            if analyst.summary_cn:
                parts.append(f"Summary (CN): {analyst.summary_cn}")
            if analyst.logic_chain:
                parts.append(f"Logic Chain: {analyst.logic_chain}")
            parts.append("")

        cm = input_data.crossmarket_output
        if cm and cm.agent_status == AgentStatus.SUCCESS:
            parts.append("--- Cross-Market A-Share Impact ---")
            parts.append(f"Overall Assessment: {cm.overall_assessment}")
            if cm.a_share_impacts:
                for imp in cm.a_share_impacts:
                    parts.append(f"  - {imp.sector_name}: {imp.impact_direction.value} "
                                f"(strength={imp.impact_strength:.2f})")
                    parts.append(f"    Reasoning: {imp.reasoning}")
            if cm.risk_warnings:
                for w in cm.risk_warnings:
                    parts.append(f"  Risk: {w}")
            if cm.northbound_data:
                parts.append(f"Northbound Data: CSI 300={cm.northbound_data.get('csi300', 'N/A')}")
            parts.append("")

        return "\n".join(parts)

    def _fallback_report_text(self, input_data: ReporterInput) -> str:
        """LLM 不可用时的回退报告"""
        if input_data.episode_analyses:
            return self._fallback_multi_report(input_data.episode_analyses)
        return self._fallback_single_report(input_data)

    def _fallback_multi_report(self, analyses: List[EpisodeAnalysis]) -> str:
        """多集回退报告"""
        parts = [
            "# 播客舆情聚合分析报告",
            "",
            f"**分析集数**: {len(analyses)}",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # 每集摘要
        parts.append("## 各集摘要")
        for i, ea in enumerate(analyses):
            parts.append(f"### {i+1}. {ea.podcast_title} — {ea.episode_title[:60]}")
            sent = ea.sentiment_output
            if sent and sent.agent_status == AgentStatus.SUCCESS:
                parts.append(f"- 情感: {sent.overall_sentiment.value} (score={sent.sentiment_score:+.2f})")
            cm = ea.crossmarket_output
            if cm and cm.a_share_impacts:
                for imp in cm.a_share_impacts[:3]:
                    parts.append(f"- {imp.sector_name}: {imp.impact_direction.value} (强度={imp.impact_strength:.0%})")
            parts.append("")

        # 分歧
        if consensus["divergences"]:
            parts.append("## 分歧观点")
            for d in consensus["divergences"]:
                parts.append(f"- {d}")
            parts.append("")

        parts.extend([
            "---",
            "*本报告由 Podcast Agent 系统自动生成，仅供参考，不构成投资建议。*",
        ])
        return "\n".join(parts)

    def _fallback_single_report(self, input_data: ReporterInput) -> str:
        """单集回退报告"""
        parts = [
            "# 播客舆情分析报告",
            f"**播客**: {input_data.podcast_title}",
            f"**节目**: {input_data.episode_title}",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        sent = input_data.sentiment_output
        if sent and sent.agent_status == AgentStatus.SUCCESS:
            parts.extend([
                "## 情感分析",
                f"- 整体情感: **{sent.overall_sentiment.value}**",
                f"- 情感分数: {sent.sentiment_score:.2f}",
                f"- 置信度: {sent.confidence:.0%}",
            ])
        cm = input_data.crossmarket_output
        if cm and cm.a_share_impacts:
            parts.append("## A股传导分析")
            for imp in cm.a_share_impacts:
                parts.append(f"- **{imp.sector_name}**: {imp.impact_direction.value} (强度={imp.impact_strength:.0%})")
        parts.extend([
            "---",
            "*本报告由 Podcast Agent 系统自动生成，仅供参考，不构成投资建议。*",
        ])
        return "\n".join(parts)

    # ================================================================
    # HTML 报告生成
    # ================================================================

    def _generate_html_report(
        self, input_data: ReporterInput, report_text: str, report_id: str
    ) -> str:
        config = get_config()
        report_dir = config.data_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        chart_data = self._prepare_chart_data(input_data)
        html = self._build_html(input_data, report_text, chart_data, report_id)

        file_path = report_dir / f"report_{report_id}.html"
        file_path.write_text(html, encoding="utf-8")

        self.logger.info(f"  HTML 报告已保存: {file_path}")
        return str(file_path)

    def _prepare_chart_data(self, input_data: ReporterInput) -> Dict[str, Any]:
        """准备 ECharts 图表数据 — 多集聚合"""
        data: Dict[str, Any] = {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "finbert_scores": {"positive": 0, "negative": 0, "neutral": 0},
            "sector_impacts": [],
            "csi300_trend": [],
            "market_breadth": {},
            "southbound_net": None,
            "northbound_note": "",
            "key_factors": {"positive": [], "negative": []},
            "entities": [],
            "risk_warnings": [],
            "episode_summaries": [],
            "consensus": {},
        }

        if input_data.episode_analyses:
            data.update(self._prepare_multi_chart_data(input_data.episode_analyses))
        else:
            data.update(self._prepare_single_chart_data(input_data))

        return data

    def _prepare_multi_chart_data(self, analyses: List[EpisodeAnalysis]) -> Dict[str, Any]:
        """多集聚合图表数据"""
        consensus = self._compute_consensus(analyses)

        # 聚合情感分数
        all_scores = []
        all_finbert = {"positive": 0, "negative": 0, "neutral": 0}
        finbert_count = 0
        all_pos_factors = []
        all_neg_factors = []
        entities_by_podcast = {}  # podcast_title -> [entity dict]，跨播客均衡取实体
        all_sector_impacts = []
        all_risks = []
        episode_summaries = []
        northbound_data = None

        for ea in analyses:
            sent = ea.sentiment_output
            if sent and sent.agent_status == AgentStatus.SUCCESS:
                all_scores.append(sent.sentiment_score)
                if sent.finbert_scores:
                    for k in all_finbert:
                        all_finbert[k] += sent.finbert_scores.get(k, 0)
                    finbert_count += 1
                all_pos_factors.extend(sent.key_positive_factors)
                all_neg_factors.extend(sent.key_negative_factors)

            analyst = ea.analyst_output
            if analyst and analyst.agent_status == AgentStatus.SUCCESS:
                lst = entities_by_podcast.setdefault(ea.podcast_title, [])
                seen = {x["name"].lower().strip() for x in lst}
                for e in (analyst.entities or []):
                    # 去掉 person 类型实体（主持人/记者等噪声，对 A 股板块分析无意义）
                    if e.entity_type == "person":
                        continue
                    key = e.name.lower().strip()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    lst.append({
                        "name": e.name,
                        "type": e.entity_type,
                        "sentiment": e.sentiment.value,
                        "score": e.sentiment_score,
                        "podcast": ea.podcast_title,
                        "evidence": e.evidence or "",
                    })

            cm = ea.crossmarket_output
            if cm and cm.agent_status == AgentStatus.SUCCESS:
                for imp in (cm.a_share_impacts or []):
                    all_sector_impacts.append({
                        "name": imp.sector_name,
                        "direction": imp.impact_direction.value,
                        "strength": imp.impact_strength,
                        "reasoning": imp.reasoning,
                        "podcast": ea.podcast_title,
                        "top_stocks": imp.top_stocks or [],
                        "top_stocks_note": imp.top_stocks_note or "",
                        "representative_stocks": imp.representative_stocks or [],
                    })
                all_risks.extend(cm.risk_warnings or [])
                if not northbound_data:
                    northbound_data = cm.northbound_data

            episode_summaries.append({
                "podcast": ea.podcast_title,
                "episode": ea.episode_title[:50],
                "sentiment": sent.overall_sentiment.value if (sent and sent.agent_status == AgentStatus.SUCCESS) else "unknown",
                "score": sent.sentiment_score if (sent and sent.agent_status == AgentStatus.SUCCESS) else 0,
            })

        # 去掉 person 后，其余实体全部保留（不截断）
        all_entities = [e for _lst in entities_by_podcast.values() for e in _lst]

        # 平均 FinBERT
        if finbert_count > 0:
            for k in all_finbert:
                all_finbert[k] = round(all_finbert[k] / finbert_count, 3)

        # 合并板块影响（同名合并，取平均强度）
        merged_sectors = self._merge_sector_impacts(all_sector_impacts)

        # 去重风险
        unique_risks = list(dict.fromkeys(all_risks))

        # 北向资金 → 改为沪深300走势 + 市场宽度
        market_data = northbound_data or {}
        csi300_trend = market_data.get("csi300_trend", [])
        csi300 = market_data.get("csi300", 0)
        csi300_change = market_data.get("csi300_change", 0)
        market_breadth = market_data.get("market_breadth", {})
        southbound_net = market_data.get("southbound_net")
        northbound_note = market_data.get("northbound_note", "")

        return {
            "sentiment_score": consensus["avg_score"],
            "sentiment_label": "bullish" if consensus["avg_score"] > 0.2 else (
                "bearish" if consensus["avg_score"] < -0.2 else "neutral"
            ),
            "finbert_scores": all_finbert,
            "sector_impacts": merged_sectors,
            "csi300_trend": csi300_trend,
            "market_breadth": market_breadth,
            "southbound_net": southbound_net,
            "northbound_note": northbound_note,
            "key_factors": {
                "positive": list(dict.fromkeys(all_pos_factors))[:8],
                "negative": list(dict.fromkeys(all_neg_factors))[:8],
            },
            "entities": all_entities,
            "risk_warnings": unique_risks[:10],
            "episode_summaries": episode_summaries,
            "consensus": consensus,
            "csi300": csi300,
            "csi300_change": csi300_change,
            "episode_count": len(analyses),
            "engine_divergence_rate": self._avg_field(analyses, "engine_divergence_rate"),
            "finbert_chunk_coverage": self._avg_field(analyses, "finbert_chunk_coverage"),
            "llm_chunk_coverage": self._avg_field(analyses, "llm_chunk_coverage"),
        }

    def _avg_field(self, analyses: List[EpisodeAnalysis], field: str) -> Optional[float]:
        """各集 SentimentOutput 某数值字段的平均值；无数据返回 None"""
        values = [
            getattr(ea.sentiment_output, field)
            for ea in analyses
            if ea.sentiment_output and getattr(ea.sentiment_output, field) is not None
        ]
        return round(sum(values) / len(values), 2) if values else None

    def _merge_sector_impacts(self, impacts: list) -> list:
        """合并同名板块影响（取平均强度，统计提及次数）"""
        merged = {}
        for imp in impacts:
            name = imp["name"]
            if name not in merged:
                merged[name] = {
                    "name": name,
                    "directions": [],
                    "strengths": [],
                    "reasonings": [],
                    "podcasts": [],
                    "top_stocks": [],
                    "top_stocks_note": "",
                    "representative_stocks": [],
                }
            merged[name]["directions"].append(imp["direction"])
            merged[name]["strengths"].append(imp["strength"])
            merged[name]["reasonings"].append(imp["reasoning"])
            merged[name]["podcasts"].append(imp.get("podcast", ""))
            if imp.get("top_stocks") and not merged[name]["top_stocks"]:
                merged[name]["top_stocks"] = imp["top_stocks"]
            if imp.get("top_stocks_note") and not merged[name]["top_stocks_note"]:
                merged[name]["top_stocks_note"] = imp["top_stocks_note"]
            if imp.get("representative_stocks") and not merged[name]["representative_stocks"]:
                merged[name]["representative_stocks"] = imp["representative_stocks"]

        result = []
        for name, m in merged.items():
            # 多数方向
            dir_counts = Counter(m["directions"])
            dominant_dir = dir_counts.most_common(1)[0][0]
            avg_strength = sum(m["strengths"]) / len(m["strengths"])

            result.append({
                "name": name,
                "direction": dominant_dir,
                "strength": round(avg_strength, 2),
                "reasoning": m["reasonings"][0][:200],
                "mention_count": len(m["podcasts"]),
                "podcasts": list(set(m["podcasts"])),
                "top_stocks": m["top_stocks"],
                "top_stocks_note": m["top_stocks_note"],
                "representative_stocks": m["representative_stocks"],
            })

        # 按提及次数排序
        result.sort(key=lambda x: x["mention_count"], reverse=True)
        return result

    def _prepare_single_chart_data(self, input_data: ReporterInput) -> Dict[str, Any]:
        """单集图表数据（兼容旧版）"""
        data: Dict[str, Any] = {
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "finbert_scores": {"positive": 0, "negative": 0, "neutral": 0},
            "sector_impacts": [],
            "csi300_trend": [],
            "market_breadth": {},
            "southbound_net": None,
            "northbound_note": "",
            "key_factors": {"positive": [], "negative": []},
            "entities": [],
            "risk_warnings": [],
            "episode_summaries": [],
            "consensus": {"consensus_level": "N/A"},
            "episode_count": 1,
        }

        sent = input_data.sentiment_output
        if sent and sent.agent_status == AgentStatus.SUCCESS:
            data["sentiment_score"] = sent.sentiment_score
            data["sentiment_label"] = sent.overall_sentiment.value
            data["finbert_scores"] = sent.finbert_scores or data["finbert_scores"]
            data["key_factors"]["positive"] = sent.key_positive_factors
            data["key_factors"]["negative"] = sent.key_negative_factors
            data["engine_divergence_rate"] = sent.engine_divergence_rate
            data["finbert_chunk_coverage"] = sent.finbert_chunk_coverage
            data["llm_chunk_coverage"] = sent.llm_chunk_coverage

        analyst = input_data.analyst_output
        if analyst and analyst.agent_status == AgentStatus.SUCCESS:
            data["entities"] = [
                {"name": e.name, "type": e.entity_type, "sentiment": e.sentiment.value,
                 "score": e.sentiment_score, "podcast": input_data.podcast_title,
                 "evidence": e.evidence or ""}
                for e in (analyst.entities or [])
                if e.entity_type != "person"
            ]

        cm = input_data.crossmarket_output
        if cm and cm.agent_status == AgentStatus.SUCCESS:
            data["sector_impacts"] = [
                {"name": imp.sector_name, "direction": imp.impact_direction.value,
                 "strength": imp.impact_strength,
                 "reasoning": imp.reasoning, "mention_count": 1,
                 "top_stocks": imp.top_stocks or [],
                 "top_stocks_note": imp.top_stocks_note or "",
                 "representative_stocks": imp.representative_stocks or [],
                 "podcast": input_data.podcast_title}
                for imp in (cm.a_share_impacts or [])
            ]
            data["risk_warnings"] = cm.risk_warnings or []
            nb = cm.northbound_data or {}
            if nb.get("csi300_trend"):
                data["csi300_trend"] = nb["csi300_trend"]
                data["csi300"] = nb.get("csi300", 0)
                data["csi300_change"] = nb.get("csi300_change", 0)
            if nb.get("market_breadth"):
                data["market_breadth"] = nb["market_breadth"]
            if nb.get("southbound_net") is not None:
                data["southbound_net"] = nb["southbound_net"]
            data["northbound_note"] = nb.get("northbound_note", "")

        return data

    # ================================================================
    # HTML 模板
    # ================================================================

    def _echarts_tag(self) -> str:
        """内嵌本地 ECharts 库（报告自包含、离线可用）；库文件缺失时回退 CDN"""
        try:
            lib = _ECHARTS_LIB_PATH.read_text(encoding="utf-8")
        except OSError:
            logger.warning("本地 ECharts 库缺失 (%s)，回退 CDN 加载", _ECHARTS_LIB_PATH)
            return _ECHARTS_CDN_TAG
        return "<script>\n" + lib + "\n</script>"

    def _build_html(
        self, input_data: ReporterInput, report_text: str,
        chart_data: Dict[str, Any], report_id: str,
    ) -> str:
        report_html = self._markdown_to_html(report_text)
        chart_json = json.dumps(chart_data, ensure_ascii=False)
        echarts_tag = self._echarts_tag()

        is_multi = bool(input_data.episode_analyses)
        title = "播客舆情聚合分析报告" if is_multi else "播客舆情分析报告"
        podcast_info = (f"{chart_data.get('episode_count', 1)} 集播客聚合分析"
                       if is_multi else input_data.podcast_title)

        return f"""<!-- Generated by Trae Work -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — {podcast_info}</title>
    {echarts_tag}
    <style>
        :root {{
            --bg: #eef1f6;
            --bg2: #ffffff;
            --ink: #1e293b;
            --muted: #64748b;
            --rule: #e2e8f0;
            --accent: #2563eb;
            --accent2: #0ea5e9;
            --bullish: #059669;
            --bearish: #dc2626;
            --neutral: #f59e0b;
            --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
            --shadow-md: 0 4px 16px rgba(15, 23, 42, 0.08);
            --shadow-lg: 0 12px 32px rgba(15, 23, 42, 0.14);
            --radius: 14px;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html {{ scroll-behavior: smooth; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif;
            background: var(--bg);
            background-image: radial-gradient(circle at 12% -5%, rgba(37, 99, 235, 0.06) 0%, transparent 42%),
                              radial-gradient(circle at 88% 5%, rgba(14, 165, 233, 0.06) 0%, transparent 42%);
            color: var(--ink);
            line-height: 1.75;
            font-size: 15px;
            -webkit-font-smoothing: antialiased;
        }}
        .container {{ max-width: 1140px; margin: 0 auto; padding: 24px; }}

        .report-header {{
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, #1e293b 0%, #334155 52%, #1d4ed8 135%);
            color: white;
            padding: 44px 48px;
            border-radius: 18px;
            margin-bottom: 28px;
            box-shadow: var(--shadow-lg);
        }}
        .report-header::before {{
            content: '';
            position: absolute;
            top: -70px; right: -50px;
            width: 280px; height: 280px;
            background: radial-gradient(circle, rgba(255,255,255,0.16) 0%, transparent 70%);
            border-radius: 50%;
        }}
        .report-header::after {{
            content: '';
            position: absolute;
            bottom: -90px; left: 28%;
            width: 340px; height: 200px;
            background: radial-gradient(circle, rgba(14, 165, 233, 0.28) 0%, transparent 70%);
            border-radius: 50%;
        }}
        .report-header h1 {{
            position: relative;
            font-size: 30px;
            font-weight: 800;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
            z-index: 1;
        }}
        .report-header .subtitle {{
            position: relative;
            font-size: 14px;
            opacity: 0.82;
            margin-bottom: 16px;
            z-index: 1;
        }}
        .report-header .meta {{
            position: relative;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px 22px;
            font-size: 13px;
            opacity: 0.9;
            z-index: 1;
        }}
        .report-header .meta span {{ display: inline-flex; align-items: center; gap: 6px; }}
        .report-header .meta .dot {{
            width: 4px; height: 4px;
            border-radius: 50%;
            background: rgba(255,255,255,0.45);
        }}

        .card {{
            background: var(--bg2);
            border-radius: var(--radius);
            padding: 26px 28px;
            margin-bottom: 22px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--rule);
            transition: box-shadow 0.25s ease, transform 0.25s ease;
        }}
        .card:hover {{
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }}
        .card h2 {{
            font-size: 19px;
            font-weight: 700;
            margin-bottom: 18px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--rule);
            color: var(--ink);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .card h2::before {{
            content: '';
            width: 4px;
            height: 20px;
            border-radius: 2px;
            background: linear-gradient(180deg, var(--accent), var(--accent2));
            flex-shrink: 0;
        }}
        .card h3 {{ font-size: 16px; font-weight: 600; margin: 16px 0 8px; color: var(--ink); }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            position: relative;
            overflow: hidden;
            background: var(--bg2);
            border-radius: var(--radius);
            padding: 22px 16px;
            text-align: center;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--rule);
            transition: box-shadow 0.25s ease, transform 0.25s ease;
        }}
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent), var(--accent2));
            opacity: 0.9;
        }}
        .metric-card:hover {{
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }}
        .metric-card .value {{ font-size: 30px; font-weight: 800; color: var(--accent); letter-spacing: -0.5px; }}
        .metric-card .label {{ font-size: 13px; color: var(--muted); margin-top: 5px; }}
        .metric-card.bullish .value {{ color: var(--bullish); }}
        .metric-card.bullish::before {{ background: linear-gradient(90deg, var(--bullish), #34d399); }}
        .metric-card.bearish .value {{ color: var(--bearish); }}
        .metric-card.bearish::before {{ background: linear-gradient(90deg, var(--bearish), #f87171); }}
        .metric-card.consensus-high {{ box-shadow: inset 0 0 0 2px rgba(5, 150, 105, 0.18); }}
        .metric-card.consensus-low {{ box-shadow: inset 0 0 0 2px rgba(220, 38, 38, 0.18); }}

        .chart-figure {{ margin: 8px 0 4px; }}
        .chart-figure figcaption {{ font-size: 13px; font-weight: 500; color: var(--muted); margin-bottom: 10px; }}
        .chart-container {{ width: 100%; min-height: 400px; }}

        .risk-list {{ list-style: none; padding: 0; }}
        .risk-list li {{
            padding: 13px 16px;
            margin-bottom: 10px;
            border-radius: 8px;
            background: #fef2f2;
            border-left: 4px solid var(--bearish);
            font-size: 14px;
            line-height: 1.6;
        }}
        .risk-list li.high {{ background: #fffbeb; border-left-color: #f59e0b; }}

        .entity-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .entity-table th {{
            background: #f8fafc;
            padding: 11px 14px;
            text-align: left;
            font-weight: 600;
            color: var(--muted);
            font-size: 12px;
            letter-spacing: 0.3px;
            border-bottom: 2px solid var(--rule);
        }}
        .entity-table td {{ padding: 11px 14px; border-bottom: 1px solid var(--rule); }}
        .entity-table tbody tr {{ transition: background 0.15s ease; }}
        .entity-table tbody tr:hover {{ background: #f8fafc; }}
        .tag {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .tag-bullish {{ background: #d1fae5; color: #065f46; }}
        .tag-bearish {{ background: #fee2e2; color: #991b1b; }}
        .tag-neutral {{ background: #fef3c7; color: #92400e; }}

        .episode-bar {{
            display: flex;
            align-items: center;
            padding: 9px 0;
            border-bottom: 1px solid var(--rule);
        }}
        .episode-bar:last-child {{ border-bottom: none; }}
        .episode-bar .ep-label {{ width: 300px; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .episode-bar .ep-bar-bg {{
            flex: 1;
            height: 18px;
            background: #eef2f7;
            border-radius: 9px;
            margin: 0 14px;
            position: relative;
            overflow: hidden;
        }}
        .episode-bar .ep-bar-fill {{
            height: 100%;
            border-radius: 9px;
            transition: width 0.6s ease;
        }}
        .episode-bar .ep-score {{ font-size: 13px; font-weight: 700; width: 56px; text-align: right; }}

        .report-content {{ line-height: 1.9; }}
        .report-content h2 {{ font-size: 20px; font-weight: 700; margin: 24px 0 12px; color: var(--ink); }}
        .report-content h3 {{ font-size: 16px; font-weight: 600; margin: 16px 0 8px; color: var(--ink); }}
        .report-content p {{ margin-bottom: 12px; }}
        .report-content ul, .report-content ol {{ margin: 8px 0 12px 24px; }}
        .report-content li {{ margin-bottom: 4px; }}
        .report-content strong {{ color: var(--accent); }}
        .report-content hr {{ border: none; border-top: 1px solid var(--rule); margin: 24px 0; }}
        .report-content blockquote {{
            border-left: 4px solid var(--accent);
            padding: 10px 18px;
            margin: 12px 0;
            background: #f8fafc;
            border-radius: 0 8px 8px 0;
        }}
        .report-content table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 14px;
        }}
        .report-content th {{
            background: #f8fafc;
            padding: 10px 12px;
            text-align: left;
            border-bottom: 2px solid var(--rule);
            font-weight: 600;
        }}
        .report-content td {{ padding: 10px 12px; border-bottom: 1px solid var(--rule); }}

        .report-footer {{
            text-align: center;
            padding: 28px;
            color: var(--muted);
            font-size: 13px;
            border-top: 1px solid var(--rule);
            margin-top: 30px;
        }}
        .report-footer strong {{ color: var(--accent); }}

        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
        @media (max-width: 768px) {{
            .two-col {{ grid-template-columns: 1fr; }}
            .report-header {{ padding: 28px 22px; }}
            .report-header h1 {{ font-size: 23px; }}
            .container {{ padding: 14px; }}
            .card {{ padding: 20px; }}
        }}
    </style>
</head>
<body>
<div class="container">

    <div class="report-header">
        <h1>{title}</h1>
        <div class="subtitle">{podcast_info}</div>
        <div class="meta">
            <span>报告ID: {report_id}</span>
            <span class="dot"></span>
            <span>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
    </div>

    <div class="metrics-grid" id="metrics-grid"></div>

    <div class="card" id="report-summary-card">
        <h2>综合分析报告</h2>
        <div class="report-content">
            {report_html}
        </div>
    </div>

    <div class="card" id="episode-bars-card">
        <h2>各播客情感一览</h2>
        <div id="episode-bars"></div>
    </div>

    <div class="card" id="sector-card">
        <h2>A股相关板块提示</h2>
        <figure class="chart-figure">
            <figcaption>播客话题关联的 A 股板块及影响方向 (数字=提及集数, 绿色=利好, 红色=利空, 黄色=中性)</figcaption>
            <div class="chart-container" id="chart-sector-impact" style="min-height:400px"></div>
        </figure>
        <div id="sector-detail-list"></div>
    </div>

    <div class="card" id="northbound-card">
        <h2>沪深300 近5日走势</h2>
        <figure class="chart-figure">
            <figcaption id="nb-note">沪深300指数收盘价</figcaption>
            <div class="chart-container" id="chart-northbound" style="min-height:350px"></div>
        </figure>
    </div>

    <div class="card" id="entity-card">
        <h2>关键实体分析</h2>
        <div id="entity-table-container"></div>
    </div>

    <div class="card" id="risk-card">
        <h2>风险提示</h2>
        <ul class="risk-list" id="risk-list"></ul>
    </div>

    <div class="report-footer">
        <p>本报告由 <strong>Podcast Agent 多智能体系统</strong> 自动生成</p>
        <p>基于 DashScope Qwen LLM + FinBERT + ECharts 技术栈</p>
        <p>仅供参考，不构成投资建议 | Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</div>

<script>
(function() {{
    var chartData = {chart_json};

    // ============================================================
    // Key Metrics
    // ============================================================
    var metrics = [
        {{ value: chartData.sector_impacts.length, label: '关联板块', cls: '' }},
        {{ value: chartData.entities.length, label: '关键实体', cls: '' }},
        {{ value: chartData.risk_warnings.length, label: '风险提示', cls: '' }},
    ];
    var grid = document.getElementById('metrics-grid');
    metrics.forEach(function(m) {{
        grid.innerHTML += '<div class="metric-card ' + m.cls + '"><div class="value">' + m.value + '</div><div class="label">' + m.label + '</div></div>';
    }});

    // ============================================================
    // Episode sentiment bars
    // ============================================================
    var barsContainer = document.getElementById('episode-bars');
    if (barsContainer && chartData.episode_summaries && chartData.episode_summaries.length > 0) {{
        chartData.episode_summaries.forEach(function(ep) {{
            var pct = ((ep.score + 1) / 2 * 100).toFixed(0);
            var color = ep.score > 0.2 ? '#059669' : (ep.score < -0.2 ? '#dc2626' : '#f59e0b');
            barsContainer.innerHTML +=
                '<div class="episode-bar">' +
                '<div class="ep-label" title="' + ep.podcast + '">' + ep.podcast + ' — ' + ep.episode + '</div>' +
                '<div class="ep-bar-bg"><div class="ep-bar-fill" style="width:' + pct + '%;background:' + color + ';"></div></div>' +
                '<div class="ep-score" style="color:' + color + '">' + (ep.score > 0 ? '+' : '') + ep.score.toFixed(2) + '</div>' +
                '</div>';
        }});
    }}

    // ============================================================
    // Sector Impact Bar Chart
    // ============================================================
    var sectorCard = document.getElementById('sector-card');
    if (chartData.sector_impacts.length > 0) {{
        var sectors = chartData.sector_impacts;
        var names = sectors.map(function(s) {{ return s.name + ' (' + (s.mention_count || 1) + '集)'; }});
        var strengths = sectors.map(function(s) {{ return (s.strength * 100).toFixed(0); }});
        var colors = sectors.map(function(s) {{
            return s.direction === 'bullish' ? '#059669' : (s.direction === 'bearish' ? '#dc2626' : '#f59e0b');
        }});
        var sectorDom = document.getElementById('chart-sector-impact');
        if (sectorDom) {{
            var sectorChart = echarts.init(sectorDom);
            sectorChart.setOption({{
                tooltip: {{
                    trigger: 'axis', axisPointer: {{ type: 'shadow' }},
                    formatter: function(params) {{
                        var s = sectors[params[0].dataIndex];
                        return '<b>' + s.name + '</b><br/>'
                            + '强度 ' + (s.strength * 100).toFixed(0) + '%';
                    }}
                }},
                grid: {{ left: '3%', right: '15%', bottom: '3%', containLabel: true }},
                xAxis: {{ type: 'value', max: 100, axisLabel: {{ formatter: '{{value}}%' }} }},
                yAxis: {{ type: 'category', data: names, axisLabel: {{ fontSize: 13 }} }},
                series: [{{
                    type: 'bar',
                    data: strengths.map(function(v, i) {{ return {{ value: v, itemStyle: {{ color: colors[i], borderRadius: [0, 6, 6, 0] }} }}; }}),
                    barWidth: '60%',
                    label: {{ show: true, position: 'right', formatter: '{{c}}%' }},
                }}]
            }});
            window.addEventListener('resize', function() {{ sectorChart.resize(); }});
        }}

        // 板块详情：完整展示分析文字（避免 tooltip 过长被截断）
        var detailDom = document.getElementById('sector-detail-list');
        if (detailDom) {{
            var detailHtml = '<h3 style="margin-top:16px;margin-bottom:6px">板块详情</h3>'
                + '<p style="color:var(--muted);font-size:12px;margin-bottom:12px">本部分基于播客内容识别可能相关的 A 股板块，属「关联提示」而非严格的传导分析或涨跌预测；「强度」为主观判断的关联/影响程度（非预测涨跌幅），0~100%。</p>';
            sectors.forEach(function(s) {{
                var dirColor = s.direction === 'bullish' ? '#059669' : (s.direction === 'bearish' ? '#dc2626' : '#f59e0b');
                var dirLabel = s.direction === 'bullish' ? '利好' : (s.direction === 'bearish' ? '利空' : '中性');
                detailHtml += '<div style="border:1px solid var(--rule);border-radius:8px;padding:12px 16px;margin-bottom:10px;background:#f8fafc">';
                detailHtml += '<div style="font-weight:600;margin-bottom:6px">' + s.name
                    + ' <span style="color:' + dirColor + '">' + dirLabel + '</span>'
                    + ' · 强度 ' + (s.strength * 100).toFixed(0) + '%'
                    + ' · 提及 ' + (s.mention_count || 1) + ' 集'
                    + '</div>';
                detailHtml += '<div style="color:var(--muted);font-size:13px;line-height:1.7">' + s.reasoning + '</div>';
                var repStocks = s.representative_stocks || [];
                if (repStocks.length) {{
                    var stockNames = repStocks.map(function(st) {{
                        return st.code ? (st.name + ' (' + st.code + ')') : st.name;
                    }}).join('、');
                    detailHtml += '<div style="margin-top:6px;font-size:12px;color:var(--muted)">代表性个股（大模型推荐，非实时行情）: ' + stockNames + '</div>';
                }}
                var source = (s.podcasts && s.podcasts.length) ? s.podcasts.join('、') : (s.podcast || '');
                if (source) {{
                    detailHtml += '<div style="margin-top:6px;font-size:12px;color:var(--muted)">出处: ' + source + '</div>';
                }}
                detailHtml += '</div>';
            }});
            detailDom.innerHTML = detailHtml;
        }}
    }} else if (sectorCard) {{
        sectorCard.innerHTML = '<div class="card"><h2>A股相关板块提示</h2><p style="color:var(--muted);text-align:center;padding:40px">暂无板块影响数据</p></div>';
    }}

    // ============================================================
    // CSI 300 Trend Chart (replaces northbound flow)
    // ============================================================
    var nbCard = document.getElementById('northbound-card');
    var nbNote = document.getElementById('nb-note');
    if (chartData.csi300_trend && chartData.csi300_trend.length > 0) {{
        var trend = chartData.csi300_trend;
        var dates = trend.map(function(d) {{ return d.date; }});
        var csiValues = trend.map(function(d) {{ return d.csi300; }});
        var changes = trend.map(function(d) {{ return d.change_pct; }});

        // 更新说明
        var noteText = '沪深300指数收盘价';
        if (chartData.market_breadth && chartData.market_breadth.up) {{
            noteText += ' | 今日市场宽度: ↑' + chartData.market_breadth.up + ' ↓' + chartData.market_breadth.down;
        }}
        if (chartData.southbound_net !== null && chartData.southbound_net !== undefined) {{
            noteText += ' | 南向资金: ' + (chartData.southbound_net > 0 ? '+' : '') + chartData.southbound_net.toFixed(1) + '亿';
        }}
        if (chartData.northbound_note) {{
            noteText += ' | ' + chartData.northbound_note;
        }}
        if (nbNote) nbNote.textContent = noteText;

        var nbDom = document.getElementById('chart-northbound');
        if (nbDom) {{
            // 纵坐标自适应数据区间（ECharts 默认从 0 起，曲线会看起来平直）:
            // 以数据 min/max 中值为中心，窗口宽度 = 数据波幅 x 1.5（不小于指数 0.4%），
            // 无论指数在 3000 还是 5000 点位，折线波动都清晰可见
            var dataMin = Math.min.apply(null, csiValues);
            var dataMax = Math.max.apply(null, csiValues);
            var span = Math.max(dataMax - dataMin, dataMax * 0.004);
            var mid = (dataMax + dataMin) / 2;
            var yMin = Math.floor(mid - span * 0.75);
            var yMax = Math.ceil(mid + span * 0.75);

            var nbChart = echarts.init(nbDom);
            nbChart.setOption({{
                tooltip: {{
                    trigger: 'axis',
                    formatter: function(params) {{
                        var d = trend[params[0].dataIndex];
                        return '<b>' + d.date + '</b><br/>'
                            + '沪深300: ' + d.csi300.toFixed(0) + '<br/>'
                            + '涨跌幅: ' + (d.change_pct > 0 ? '+' : '') + d.change_pct.toFixed(2) + '%';
                    }}
                }},
                grid: {{ left: '3%', right: '8%', bottom: '3%', containLabel: true }},
                xAxis: {{ type: 'category', data: dates, axisLabel: {{ rotate: 30, fontSize: 12 }} }},
                yAxis: {{
                    type: 'value',
                    name: '指数点位',
                    min: yMin,
                    max: yMax,
                    axisLabel: {{ formatter: '{{value}}' }}
                }},
                series: [{{
                    type: 'line',
                    data: csiValues,
                    smooth: true,
                    lineStyle: {{ color: '#2563eb', width: 3 }},
                    itemStyle: {{ color: '#2563eb' }},
                    symbol: 'circle',
                    symbolSize: 10,
                    areaStyle: {{
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            {{ offset: 0, color: 'rgba(37, 99, 235, 0.3)' }},
                            {{ offset: 1, color: 'rgba(37, 99, 235, 0.05)' }}
                        ])
                    }},
                    markLine: {{
                        silent: true,
                        data: [{{ type: 'average', name: '均值', label: {{ formatter: '均值: {{c}}' }} }}]
                    }}
                }}]
            }});
            window.addEventListener('resize', function() {{ nbChart.resize(); }});
        }}
    }} else if (nbCard) {{
        nbCard.innerHTML = '<div class="card"><h2>沪深300 近5日走势</h2><p style="color:var(--muted);text-align:center;padding:40px">暂无沪深300走势数据</p></div>';
    }}

    // ============================================================
    // Entity Table
    // ============================================================
    var entityContainer = document.getElementById('entity-table-container');
    if (entityContainer && chartData.entities.length > 0) {{
        var html = '<table class="entity-table"><thead><tr><th>实体名称</th><th>类型</th><th>情感</th><th>情感分数</th><th>来源播客</th></tr></thead><tbody>';
        chartData.entities.forEach(function(e) {{
            var tagClass = e.sentiment === 'bullish' ? 'tag-bullish' : (e.sentiment === 'bearish' ? 'tag-bearish' : 'tag-neutral');
            var tagLabel = e.sentiment === 'bullish' ? '利好' : (e.sentiment === 'bearish' ? '利空' : '中性');
            var podcastName = e.podcast || '-';
            var evText = e.evidence || '';
            if (evText.length > 150) {{ evText = evText.substring(0, 150) + '…'; }}
            var evidenceHtml = evText ? '<div style="font-size:12px;color:var(--muted);margin-top:3px;line-height:1.5">' + evText + '</div>' : '';
            html += '<tr><td><strong>' + e.name + '</strong>' + evidenceHtml + '</td><td>' + e.type + '</td><td><span class="tag ' + tagClass + '">' + tagLabel + '</span></td><td>' + (e.score * 100).toFixed(0) + '%</td><td style="color:var(--muted)">' + podcastName + '</td></tr>';
        }});
        html += '</tbody></table>';
        html += '<p style="color:var(--muted);font-size:12px;margin-top:8px">情感分数表示播客对该实体的情感倾向（-100% 极度负面 ~ +100% 极度正面，0 为中性），为主观判断，非行情预测。</p>';
        entityContainer.innerHTML = html;
    }} else if (entityContainer) {{
        entityContainer.innerHTML = '<p style="color:var(--muted);text-align:center;padding:20px">暂无实体数据</p>';
    }}

    // ============================================================
    // Risk Warnings
    // ============================================================
    var riskList = document.getElementById('risk-list');
    if (riskList && chartData.risk_warnings.length > 0) {{
        chartData.risk_warnings.forEach(function(w) {{
            var cls = w.indexOf('[高影响]') >= 0 ? 'high' : '';
            riskList.innerHTML += '<li class="' + cls + '">' + w + '</li>';
        }});
    }} else if (riskList) {{
        riskList.innerHTML = '<li style="background:#f8f9fa;border-left-color:#b2bec3">暂无风险提示</li>';
    }}
}})();
</script>
</body>
</html>"""

    def _markdown_to_html(self, md_text: str) -> str:
        """简易 Markdown 转 HTML"""
        lines = md_text.split('\n')
        html_lines = []
        in_list = False
        in_ol = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_list: html_lines.append('</ul>'); in_list = False
                if in_ol: html_lines.append('</ol>'); in_ol = False
                continue

            if stripped.startswith('### '):
                if in_list: html_lines.append('</ul>'); in_list = False
                if in_ol: html_lines.append('</ol>'); in_ol = False
                html_lines.append(f'<h3>{stripped[4:]}</h3>')
            elif stripped.startswith('## '):
                if in_list: html_lines.append('</ul>'); in_list = False
                if in_ol: html_lines.append('</ol>'); in_ol = False
                html_lines.append(f'<h2>{stripped[3:]}</h2>')
            elif stripped.startswith('# '):
                if in_list: html_lines.append('</ul>'); in_list = False
                if in_ol: html_lines.append('</ol>'); in_ol = False
                html_lines.append(f'<h1>{stripped[2:]}</h1>')
            elif stripped in ('---', '***', '___'):
                if in_list: html_lines.append('</ul>'); in_list = False
                if in_ol: html_lines.append('</ol>'); in_ol = False
                html_lines.append('<hr>')
            elif stripped.startswith('> '):
                if in_list: html_lines.append('</ul>'); in_list = False
                if in_ol: html_lines.append('</ol>'); in_ol = False
                html_lines.append(f'<blockquote>{self._inline_md(stripped[2:])}</blockquote>')
            elif stripped.startswith('- ') or stripped.startswith('* '):
                if in_ol: html_lines.append('</ol>'); in_ol = False
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                html_lines.append(f'<li>{self._inline_md(stripped[2:])}</li>')
            elif stripped[0].isdigit() and '. ' in stripped[:4]:
                if in_list: html_lines.append('</ul>'); in_list = False
                if not in_ol:
                    html_lines.append('<ol>')
                    in_ol = True
                content = stripped.split('. ', 1)[1]
                html_lines.append(f'<li>{self._inline_md(content)}</li>')
            else:
                if in_list: html_lines.append('</ul>'); in_list = False
                if in_ol: html_lines.append('</ol>'); in_ol = False
                html_lines.append(f'<p>{self._inline_md(stripped)}</p>')

        if in_list: html_lines.append('</ul>')
        if in_ol: html_lines.append('</ol>')
        return '\n'.join(html_lines)

    def _inline_md(self, text: str) -> str:
        """处理行内 Markdown"""
        import re
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
        return text