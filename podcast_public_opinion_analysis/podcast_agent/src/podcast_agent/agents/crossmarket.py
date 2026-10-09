"""
CrossMarket Agent — A股跨市场传导分析 (Day 5-6 实现)

负责:
  1. LLM 忠实分析播客内容对A股板块的影响
  2. AKShare 获取北向资金、行业指数等实时数据
  3. 评估对具体板块/个股的影响方向和强度
  4. 用东财真实成分股填充受影响个股

技术栈:
  - LLM: DashScope Qwen (传导分析)
  - AKShare: A股实时数据 (北向资金/行业指数/板块行情)
  - 回退: AKShare不可用时仅LLM分析
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..config import get_config
from ..models.messages import (
    AgentStatus, AnalystOutput, CrossMarketOutput, AShareImpact,
    Sentiment,
)
from .base import BaseAgent

logger = logging.getLogger(__name__)


class CrossMarketAgent(BaseAgent[AnalystOutput, CrossMarketOutput]):
    """A股跨市场传导Agent"""

    agent_name = "crossmarket"

    # 市场数据短时缓存: 并行分析多集时共享同一份快照，避免重复拉取 AKShare
    _market_data_cache: Optional[Dict[str, Any]] = None
    _market_data_cache_time: float = 0.0
    _MARKET_DATA_TTL: float = 300.0  # 秒

    def execute(self, input_data: AnalystOutput) -> CrossMarketOutput:
        self.log_start(
            entities=len(input_data.entities),
            china_relevance=input_data.china_relevance,
        )

        output = CrossMarketOutput(
            agent_status=AgentStatus.RUNNING,
            episode_id=input_data.episode_id,
        )

        try:
            # 1. 获取A股市场数据（并行分析时命中缓存）
            market_data = self._get_market_data()

            # 2. LLM 跨市场传导分析
            impacts = self._analyze_crossmarket_impact(input_data, market_data)

            # 注：成分股获取已暂时停用（东财成分股接口限流严重，取不到），
            # 后续数据源稳定可恢复 _enrich_top_stocks(impacts)

            output.a_share_impacts = impacts
            output.northbound_data = market_data if market_data.get("available") else None

            # 3. 综合评估
            output.overall_assessment = self._build_overall_assessment(impacts, input_data)
            output.risk_warnings = self._extract_risk_warnings(impacts)

            output.agent_status = AgentStatus.SUCCESS
            self.log_done(
                impacts=len(impacts),
                warnings=len(output.risk_warnings),
            )

        except Exception as e:
            self.log_error(str(e))
            output.agent_status = AgentStatus.FAILED
            output.error_message = str(e)

        return output

    # ================================================================
    # AKShare 市场数据获取
    # ================================================================

    def _get_market_data(self) -> Dict[str, Any]:
        """带短时缓存的市场数据获取（TTL 内并行调用共享同一份快照）"""
        import time as _time

        now = _time.time()
        if (
            self._market_data_cache is not None
            and now - self._market_data_cache_time < self._MARKET_DATA_TTL
        ):
            self.logger.info("  市场数据: 命中缓存（并行分析共享）")
            return self._market_data_cache

        data = self._fetch_market_data()
        type(self)._market_data_cache = data
        type(self)._market_data_cache_time = now
        return data

    def _fetch_market_data(self) -> Dict[str, Any]:
        """
        获取A股实时市场数据

        数据源说明:
        - stock_zh_index_daily (新浪): 沪深300指数日K（收盘价自算涨跌幅）
          注意: 曾用 stock_hsgt_hist_em(北向资金) 的"沪深300"列，但 2026-08 实测
          该列返回的是上证综指点位（3982.65=上证收盘，真实沪深300=4741.10），
          属东方财富接口字段错位，不可用于指数行情
        - stock_hsgt_fund_flow_summary_em: 市场宽度（沪深两市涨跌家数合计）、
          南向资金净买额（港股通沪+深合计）
        - 北向资金日频净买额自2024年8月19日起交易所永久停止披露，不可用
        """
        import math

        data: Dict[str, Any] = {
            "csi300_trend": [],       # 近5日沪深300走势
            "csi300": None,           # 最新沪深300指数
            "csi300_change": None,    # 沪深300涨跌幅
            "market_breadth": {},     # 市场宽度 {up, down, flat}
            "southbound_net": None,   # 南向资金净买额（港股通，作为跨境资金参考）
            "northbound_note": "北向资金日频净买额自2024年8月19日起交易所停止披露",
            "available": False,
        }

        try:
            import akshare as ak
        except ImportError:
            self.logger.warning("akshare 未安装，跳过市场数据获取")
            return data

        # 1. 获取沪深300历史走势（近5日，新浪指数日线）——独立容错
        try:
            df = ak.stock_zh_index_daily(symbol="sh000300")
            if df is not None and not df.empty:
                # 涨跌幅需要前一日收盘，多取1天
                recent = df.tail(6).reset_index(drop=True)
                for i in range(1, len(recent)):
                    prev = self._safe_float(recent.loc[i - 1, "close"])
                    close = self._safe_float(recent.loc[i, "close"])
                    if prev > 0 and close > 0:
                        data["csi300_trend"].append({
                            "date": str(recent.loc[i, "date"]),
                            "csi300": round(close, 2),
                            "change_pct": round((close / prev - 1) * 100, 2),
                        })

                if data["csi300_trend"]:
                    last = data["csi300_trend"][-1]
                    data["csi300"] = last["csi300"]
                    data["csi300_change"] = last["change_pct"]

                data["available"] = True
                self.logger.info(
                    f"  沪深300: {data['csi300']:.0f} ({data['csi300_change']:+.2f}%), "
                    f"近5日走势: {len(data['csi300_trend'])} 条"
                )
        except Exception as e:
            self.logger.warning(f"  沪深300数据获取失败: {e}")

        # 2. 获取市场宽度和南向资金（东财）——独立容错，不受沪深300失败影响
        try:
            flow_df = ak.stock_hsgt_fund_flow_summary_em()
            if flow_df is not None and not flow_df.empty:
                # 市场宽度 = 沪深两市 A 股涨跌家数之和。
                # 接口返回 4 行: 沪股通(北向/上证指数)、深股通(北向/深证成指)、
                # 港股通(沪)/港股通(深)(南向/恒生指数)。南向行挂靠恒生指数，
                # 其"上涨/下跌/持平数"是港股口径，不计入 A 股市场宽度。
                north_rows = flow_df[flow_df["资金方向"] == "北向"]
                if not north_rows.empty:
                    data["market_breadth"] = {
                        "up": int(north_rows["上涨数"].apply(self._safe_float).sum()),
                        "down": int(north_rows["下跌数"].apply(self._safe_float).sum()),
                        "flat": int(north_rows["持平数"].apply(self._safe_float).sum()),
                    }
                # 南向资金（港股通）净买额 = 港股通(沪) + 港股通(深) 两通道合计
                south_rows = flow_df[flow_df["资金方向"] == "南向"]
                if not south_rows.empty:
                    data["southbound_net"] = round(
                        south_rows["成交净买额"].apply(self._safe_float).sum(), 2
                    )
                data["available"] = True
                self.logger.info(
                    f"  市场宽度: ↑{data['market_breadth'].get('up', 0)} "
                    f"↓{data['market_breadth'].get('down', 0)}, "
                    f"南向净买额: {data['southbound_net']}"
                )
        except Exception as e:
            self.logger.warning(f"  市场宽度/南向数据获取失败: {e}")

        return data

    @staticmethod
    def _safe_float(value: Any) -> float:
        """安全转换为 float，NaN 返回 0"""
        import math
        try:
            v = float(value) if value is not None else 0.0
            return 0.0 if math.isnan(v) else v
        except (ValueError, TypeError):
            return 0.0

    # ================================================================
    # LLM 跨市场传导分析
    # ================================================================

    def _analyze_crossmarket_impact(
        self, input_data: AnalystOutput, market_data: Dict[str, Any]
    ) -> List[AShareImpact]:
        """
        Retrieve-then-generate 两段式传导分析:

        1. 第一段 LLM: 输出受影响「主题词」(topic，中文关键词) + 方向 + 强度 + 理由
        2. 代码检索: 用主题词在东财概念板块名单里检索 top-k 候选
        3. 第二段 LLM: 从候选中选最贴切的标准板块名（名字必中，避免自由发挥匹配失败）
        """
        client = self.get_llm_client()
        if client is None:
            return self._rule_based_impact(input_data, market_data)

        model = self.get_llm_model()

        # 构建上下文
        context = self._build_impact_context(input_data, market_data)

        # 第一段：提取受影响主题词
        topics = self._extract_topics(client, model, context)
        if not topics:
            self.logger.info("  第一段未识别出受影响主题，回退规则分析")
            return self._rule_based_impact(input_data, market_data)

        # 第二段：检索候选 + 从候选选板块
        impacts = self._resolve_sectors(client, model, topics)
        if not impacts:
            self.logger.info("  第二段板块选择失败，回退规则分析")
            return self._rule_based_impact(input_data, market_data)

        # 第三段：LLM 为每个板块生成代表性个股（替代被限流的真实成分股接口）
        impacts = self._gen_representative_stocks(client, model, impacts)

        return impacts

    def _extract_topics(self, client, model: str, context: str) -> List[Dict[str, Any]]:
        """第一段：LLM 输出受影响主题词（中文关键词）而非板块名"""
        prompt = f"""{context}

You are a cross-market analyst specializing in how US/global events impact China's A-share market.

CRITICAL RULES — You MUST follow these strictly:
1. STAY FAITHFUL to the podcast. ONLY identify impacts that are EXPLICITLY discussed in the podcast, or DIRECTLY and specifically implied by a named entity/topic in it. Do NOT over-extrapolate.
2. DO NOT list an impact just because it is broadly "related" to a mentioned topic. If the podcast did not actually talk about it, leave it out.
3. Prefer fewer, higher-confidence impacts over many speculative ones — 3 to 6 impacts is usually enough.
4. Each impact's reasoning MUST cite the specific podcast entity/topic that justifies the connection.
5. If a sector is commonly mentioned in Chinese finance (e.g., 白酒, 新能源) but has NO real connection to this podcast, DO NOT include it.
6. The northbound net flow daily data is NO LONGER available since August 2024; use CSI 300 trend and market breadth instead.

For each impact, output a SHORT Chinese keyword as the "topic" — it should be close to an A-share concept sector name (e.g., 英伟达, 半导体, 低空经济, 人工智能, 新能源车, 算力). It MUST be a bare keyword (no full sentence, no English).

impact_strength scoring rubric (0.0~1.0, the subjective relevance/impact degree of this topic on the sector — NOT a prediction of price change):
- 0.8~1.0: the podcast explicitly and heavily discusses it, directly touching the sector's core business (e.g., names a company's earnings, a policy aimed squarely at the industry)
- 0.6~0.8: explicitly discussed, indirectly affects the sector (e.g., a macro factor clearly acting on it)
- 0.4~0.6: mentioned, has a moderate connection to the sector (needs some reasoning)
- 0.2~0.4: mentioned in passing, weak connection
- 0.0~0.2: only tenuously related, needs heavy reasoning to connect

Return ONLY a JSON array (no other text):
```json
[
  {{
    "topic": "short Chinese keyword, e.g. 英伟达 or 半导体",
    "impact_direction": "bullish"|"bearish"|"neutral",
    "impact_strength": 0.0 to 1.0,
    "reasoning": "MUST cite the specific podcast entity/topic that justifies this impact. Format: '基于播客中讨论的[XX实体/话题]，传导至A股[XX板块]...'",
    "northbound_flow_alignment": "aligned"|"divergent"|"unknown"
  }}
]
```

ONLY output the JSON array."""

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1200,
            )

            content = resp.choices[0].message.content.strip()
            return self._parse_topics(content)

        except Exception as e:
            self.logger.warning(f"LLM 主题提取失败: {e}")
            return []

    def _build_impact_context(
        self, input_data: AnalystOutput, market_data: Dict[str, Any]
    ) -> str:
        """构建传导分析上下文"""
        parts = [
            "=== Cross-Market Impact Analysis Context ===",
            f"China Relevance: {input_data.china_relevance:.2f}",
            f"Overall Sentiment: {input_data.overall_sentiment.value}",
        ]

        if input_data.entities:
            parts.append("Entities Identified (FROM THE PODCAST - use these as basis):")
            for e in input_data.entities:
                parts.append(
                    f"  - {e.name} ({e.entity_type}): {e.sentiment.value} "
                    f"score={e.sentiment_score:.2f}"
                )
                if e.evidence:
                    parts.append(f"    Evidence: {e.evidence}")

        if input_data.macro_topics:
            parts.append(f"Macro Topics (FROM THE PODCAST): {', '.join(input_data.macro_topics)}")

        if input_data.logic_chain:
            parts.append(f"Logic Chain: {input_data.logic_chain}")

        if input_data.summary_cn:
            parts.append(f"Summary (CN): {input_data.summary_cn}")

        # 市场数据
        if market_data.get("available"):
            parts.append("A-Share Market Data (最新市场数据):")
            parts.append(
                f"  CSI 300: {market_data.get('csi300', 'N/A')} "
                f"({market_data.get('csi300_change', 0):+.2f}%)"
            )

            # 近5日沪深300走势
            trend = market_data.get("csi300_trend", [])
            if trend:
                parts.append("  CSI 300 Trend (近5日):")
                for t in trend:
                    parts.append(f"    {t['date']}: {t['csi300']:.0f} ({t['change_pct']:+.2f}%)")

            # 市场宽度
            mb = market_data.get("market_breadth", {})
            if mb:
                parts.append(
                    f"  Market Breadth: ↑{mb.get('up', 0)} ↓{mb.get('down', 0)} "
                    f"→{mb.get('flat', 0)}"
                )

            # 南向资金
            sb = market_data.get("southbound_net")
            if sb is not None:
                parts.append(f"  Southbound Net Flow (港股通): {sb:.1f}亿")

            # 数据说明
            parts.append(f"  Note: {market_data.get('northbound_note', '')}")

        return "\n".join(parts)

    def _parse_topics(self, raw: str) -> List[Dict[str, Any]]:
        """解析第一段 LLM 返回的主题列表"""
        import json
        import re

        match = re.search(r'\[[\s\S]*\]', raw)
        if match:
            raw = match.group(0)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("主题 JSON 解析失败")
            return []

        topics = []
        for item in data:
            topic = (item.get("topic") or "").strip()
            if not topic:
                continue
            try:
                direction = Sentiment(item.get("impact_direction", "neutral"))
            except ValueError:
                direction = Sentiment.NEUTRAL
            topics.append({
                "topic": topic,
                "impact_direction": direction,
                "impact_strength": float(item.get("impact_strength", 0.3)),
                "reasoning": item.get("reasoning", ""),
                "northbound_flow_alignment": item.get("northbound_flow_alignment"),
            })

        self.logger.info(
            f"  第一段识别主题: {len(topics)} 个 -> {[t['topic'] for t in topics]}"
        )
        return topics

    def _resolve_sectors(
        self, client, model: str, topics: List[Dict[str, Any]]
    ) -> List[AShareImpact]:
        """第二段：每个主题检索候选概念板块，LLM 从候选中选标准板块名"""
        from ..utils.sector_data import retrieve_candidate_sectors

        # 1. 检索候选（保持 topic 顺序）
        topic_candidates: List[Dict[str, Any]] = []
        for t in topics:
            cands = retrieve_candidate_sectors(t["topic"], top_k=5)
            topic_candidates.append({"topic": t["topic"], "candidates": cands})
            if cands:
                self.logger.info(
                    f"  [{t['topic']}] 候选: {[c['name'] for c in cands[:3]]}"
                )
            else:
                self.logger.info(f"  [{t['topic']}] 无候选（子串未匹配）")

        with_candidates = [tc for tc in topic_candidates if tc["candidates"]]
        if not with_candidates:
            return []

        # 2. 第二段 LLM 从候选中选
        pick_prompt = self._build_sector_pick_prompt(with_candidates)
        picked_map: Dict[str, Dict[str, str]] = {}
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": pick_prompt}],
                temperature=0.0,
                max_tokens=800,
            )
            picked_map = self._parse_picked(resp.choices[0].message.content.strip())
        except Exception as e:
            self.logger.warning(f"  第二段板块选择失败: {e}")
            picked_map = {}

        # 3. 合并
        impacts: List[AShareImpact] = []
        for t in topics:
            topic = t["topic"]
            picked = picked_map.get(topic)
            if picked and picked.get("sector_name"):
                sector_name = picked["sector_name"]
                sector_code = picked.get("sector_code")
            else:
                # 降级：LLM 未选或失败时，用检索第一候选
                cands = next(
                    (tc["candidates"] for tc in topic_candidates if tc["topic"] == topic),
                    [],
                )
                if not cands:
                    continue
                sector_name = cands[0]["name"]
                sector_code = cands[0]["code"]

            impacts.append(AShareImpact(
                sector_name=sector_name,
                sector_code=sector_code,
                impact_direction=t["impact_direction"],
                impact_strength=t["impact_strength"],
                reasoning=t["reasoning"],
                northbound_flow_alignment=t["northbound_flow_alignment"],
            ))

        self.logger.info(f"  第二段选定板块: {[i.sector_name for i in impacts]}")
        return impacts

    def _build_sector_pick_prompt(
        self, topic_candidates: List[Dict[str, Any]]
    ) -> str:
        """构建第二段「从候选中选板块」的 prompt"""
        lines = [
            "Below are topics extracted from a podcast. For each topic, a list of candidate A-share concept sectors is given.",
            "For EACH topic, pick the SINGLE most relevant sector. If no candidate fits, set sector_name to null.",
            "",
        ]
        for tc in topic_candidates:
            lines.append(f"Topic: {tc['topic']}")
            for c in tc["candidates"]:
                lines.append(f"  - {c['name']} ({c['code']})")
            lines.append("")

        lines.append("Return ONLY a JSON array (no other text):")
        lines.append("```json")
        lines.append("[")
        lines.append('  {"topic": "<topic verbatim>", "sector_name": "<chosen sector name>", "sector_code": "<chosen sector code>"}')
        lines.append("]")
        lines.append("```")
        lines.append("Use the EXACT sector name and code from the candidate list. ONLY output the JSON array.")
        return "\n".join(lines)

    def _parse_picked(self, raw: str) -> Dict[str, Dict[str, str]]:
        """解析第二段「选板块」结果 -> {topic: {sector_name, sector_code}}"""
        import json
        import re

        match = re.search(r'\[[\s\S]*\]', raw)
        if match:
            raw = match.group(0)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("选板块 JSON 解析失败")
            return {}

        result: Dict[str, Dict[str, str]] = {}
        for item in data:
            topic = (item.get("topic") or "").strip()
            name = (item.get("sector_name") or "").strip()
            if not topic or not name or name.lower() == "null":
                continue
            result[topic] = {
                "sector_name": name,
                "sector_code": item.get("sector_code"),
            }
        return result

    def _gen_representative_stocks(
        self, client, model: str, impacts: List[AShareImpact]
    ) -> List[AShareImpact]:
        """第三段：LLM 为每个板块生成代表性 A 股（替代被限流的真实成分股接口）"""
        if not impacts:
            return impacts

        sector_names = [imp.sector_name for imp in impacts]
        prompt = self._build_stocks_prompt(sector_names)

        stocks_map: Dict[str, List[Dict[str, str]]] = {}
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1200,
            )
            stocks_map = self._parse_stocks(resp.choices[0].message.content.strip())
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"  第三段代表性个股生成失败: {e}")
            stocks_map = {}

        for imp in impacts:
            stocks = stocks_map.get(imp.sector_name)
            if stocks is None:
                # 宽松匹配：容忍 LLM 对板块名的微调（key 与板块名互为子串）
                for key, val in stocks_map.items():
                    if key and (key in imp.sector_name or imp.sector_name in key):
                        stocks = val
                        break
            if stocks:
                imp.representative_stocks = stocks
                names = "、".join(s["name"] for s in stocks[:3])
                self.logger.info(f"  [{imp.sector_name}] 代表性个股: {names}")
            else:
                imp.representative_stocks = []
        return impacts

    def _build_stocks_prompt(self, sector_names: List[str]) -> str:
        """构建第三段「生成代表性个股」的 prompt"""
        lines = [
            "Below are A-share concept sectors identified as potentially impacted by a podcast.",
            "For EACH sector, provide 3-5 most representative A-share stocks (industry leaders / heavyweights).",
            "",
            "CRITICAL RULES:",
            "1. ONLY list REAL, currently-listed A-share stocks (Shanghai/Shenzhen/Beijing exchanges). Do NOT fabricate.",
            "2. Prefer sector leaders with the largest market cap or clearest business linkage to the sector.",
            "3. 'code' is the 6-digit stock code. If you are NOT certain of the exact code, set it to empty string.",
            "4. 'reason' is ONE short Chinese sentence on why this stock represents the sector.",
            "",
            "Sectors:",
        ]
        for s in sector_names:
            lines.append(f"  - {s}")
        lines.append("")
        lines.append("Return ONLY a JSON object (no other text), keyed by sector name verbatim:")
        lines.append("```json")
        lines.append("{")
        lines.append('  "<sector name>": [')
        lines.append('    {"code": "600519", "name": "贵州茅台", "reason": "白酒行业绝对龙头"}')
        lines.append("  ]")
        lines.append("}")
        lines.append("```")
        return "\n".join(lines)

    def _parse_stocks(self, raw: str) -> Dict[str, List[Dict[str, str]]]:
        """解析第三段「代表性个股」结果 -> {sector: [{code,name,reason}]}"""
        import json
        import re

        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            raw = match.group(0)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("代表性个股 JSON 解析失败")
            return {}

        result: Dict[str, List[Dict[str, str]]] = {}
        for sector, stocks in data.items():
            if not isinstance(stocks, list):
                continue
            cleaned: List[Dict[str, str]] = []
            for s in stocks:
                if not isinstance(s, dict):
                    continue
                name = (s.get("name") or "").strip()
                if not name:
                    continue
                cleaned.append({
                    "code": (s.get("code") or "").strip(),
                    "name": name,
                    "reason": (s.get("reason") or "").strip(),
                })
            if cleaned:
                result[sector.strip()] = cleaned
        return result

    def _enrich_top_stocks(self, impacts: List[AShareImpact]) -> List[AShareImpact]:
        """
        用东财真实成分股填充 top_stocks / affected_stocks，替代 LLM 编的个股。

        板块名无法解析或接口失败时降级为空，并在 top_stocks_note 记录原因。
        """
        from ..utils.sector_data import get_board_components, resolve_board_code

        for imp in impacts:
            if imp.top_stocks:
                continue

            # 先判断板块名能否解析到东财板块
            if not resolve_board_code(imp.sector_name):
                imp.top_stocks_note = "板块名未匹配到东财板块"
                self.logger.info(f"  [{imp.sector_name}] 板块名未匹配，跳过成分股")
                continue

            try:
                stocks = get_board_components(imp.sector_name, top_n=5)
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"  [{imp.sector_name}] 成分股获取失败: {e}")
                stocks = []

            if stocks:
                imp.top_stocks = stocks
                # 兼容旧字段：也填真实股票名
                imp.affected_stocks = [s["name"] for s in stocks]
                names = "、".join(s["name"] for s in stocks[:3])
                self.logger.info(f"  [{imp.sector_name}] 真实成分股: {names}")
            else:
                imp.top_stocks_note = "成分股获取失败（数据源限流或临时不可用）"
        return impacts

    # ================================================================
    # 规则回退
    # ================================================================

    def _rule_based_impact(
        self, input_data: AnalystOutput, market_data: Dict[str, Any]
    ) -> List[AShareImpact]:
        """
        基于规则的粗粒度传导分析（LLM不可用时的回退）
        """
        impacts = []

        # 按实体类型映射到A股板块
        sector_map = {
            "semiconductor": ("半导体", "BK1036"),
            "chip": ("半导体", "BK1036"),
            "AI": ("AI人工智能", "BK0800"),
            "electric vehicle": ("新能源车", "BK0900"),
            "EV": ("新能源车", "BK0900"),
            "real estate": ("房地产", "BK0451"),
            "property": ("房地产", "BK0451"),
            "commodity": ("有色金属", "BK0478"),
            "oil": ("石油石化", "BK0465"),
            "bank": ("银行", "BK0475"),
            "technology": ("科技", "BK1037"),
            "supply chain": ("制造业", "BK0493"),
        }

        for entity in input_data.entities:
            name_lower = entity.name.lower()
            for keyword, (sector, code) in sector_map.items():
                if keyword in name_lower:
                    impacts.append(AShareImpact(
                        sector_name=sector,
                        sector_code=code,
                        affected_stocks=[],
                        impact_direction=entity.sentiment,
                        impact_strength=abs(entity.sentiment_score) * input_data.china_relevance,
                        reasoning=f"基于实体 {entity.name} 的产业链关联",
                    ))
                    break

        if not impacts and input_data.china_relevance > 0.3:
            impacts.append(AShareImpact(
                sector_name="A股大盘",
                sector_code=None,
                affected_stocks=[],
                impact_direction=input_data.overall_sentiment,
                impact_strength=input_data.china_relevance * 0.5,
                reasoning="基于整体情绪和市场关联度的传导评估",
            ))

        self.logger.info(f"  规则传导分析: {len(impacts)} 个板块受影响")
        return impacts

    # ================================================================
    # 综合评估
    # ================================================================

    def _build_overall_assessment(
        self, impacts: List[AShareImpact], input_data: AnalystOutput
    ) -> str:
        """构建综合评估"""
        if not impacts:
            return "当前播客内容对A股市场无明显传导影响。"

        bullish = [i for i in impacts if i.impact_direction == Sentiment.BULLISH]
        bearish = [i for i in impacts if i.impact_direction == Sentiment.BEARISH]

        parts = [f"综合分析：本播客对A股市场关联度 {input_data.china_relevance:.0%}。"]

        if bullish:
            names = "、".join(i.sector_name for i in bullish[:3])
            parts.append(f"利好板块: {names}。")
        if bearish:
            names = "、".join(i.sector_name for i in bearish[:3])
            parts.append(f"利空板块: {names}。")

        return "".join(parts)

    def _extract_risk_warnings(self, impacts: List[AShareImpact]) -> List[str]:
        """提取风险提示"""
        warnings = []

        high_impact = [i for i in impacts if i.impact_strength > 0.7]
        if high_impact:
            for i in high_impact:
                warnings.append(
                    f"[高影响] {i.sector_name}: {i.reasoning} (强度: {i.impact_strength:.0%})"
                )

        bearish_impacts = [i for i in impacts if i.impact_direction == Sentiment.BEARISH]
        if len(bearish_impacts) >= 2:
            warnings.append(
                f"多个板块面临下行压力: {', '.join(i.sector_name for i in bearish_impacts)}"
            )

        return warnings