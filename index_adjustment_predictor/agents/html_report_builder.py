"""
HTML Report Builder - 预测结果可视化HTML报告生成器

生成自包含的HTML页面，包含：
    - 预测摘要（指数、日期、置信度、模式）
    - 调入/调出股票卡片（按置信度着色）
    - 备选调入候选
    - 市值对比图表
    - 财务数据表格
    - 分析说明
"""

from datetime import datetime
import html
import json
import sys
import os

# 添加项目根目录到路径，用于导入 date_utils
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from utils.date_utils import get_review_period


class HtmlReportBuilder:
    """预测结果HTML可视化报告生成器"""

    def build(
        self, prediction: dict, rule_result: dict, market_data: dict
    ) -> str:
        """
        构建完整的HTML报告

        返回:
            自包含的HTML字符串
        """
        index_name = prediction.get(
            "index_name", rule_result.get("index_name", "未知")
        )
        target_date = prediction.get(
            "target_date", market_data.get("target_date", "未知")
        )
        cutoff_date = market_data.get("cutoff_date", "未知")
        confidence = prediction.get("confidence", 0)
        reasoning = prediction.get("reasoning", "")
        details = prediction.get("details", [])
        predicted_add = prediction.get("predicted_add", [])
        alternative_add = prediction.get("alternative_add", [])
        predicted_remove = prediction.get("predicted_remove", [])
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 计算考察期
        review_period = ""
        try:
            review_start, review_end = get_review_period(cutoff_date)
            review_period = f"{review_start} ~ {review_end}"
        except Exception:
            review_period = "未知"

        # 判断预测模式
        reasoning_text = reasoning or ""
        is_quant_only = "纯量化" in reasoning_text or "未启用LLM" in reasoning_text

        # 获取排名数据用于补充信息
        rankings = rule_result.get("rankings", [])
        rank_map = {}
        for r in rankings:
            code = r.get("code", "")
            if code:
                rank_map[code] = r

        # 获取财务数据
        financial_flags = market_data.get("financial_flags", {})

        # 清洗reasoning
        cleaned_reasoning = self._clean_reasoning(reasoning)

        # 构建各部分HTML
        parts = []
        parts.append(self._build_head(index_name, target_date))
        parts.append(self._build_header(
            index_name, target_date, cutoff_date, review_period, now,
            confidence, is_quant_only, len(predicted_add),
            len(predicted_remove), len(alternative_add)
        ))

        # 置信度分布图
        parts.append(self._build_confidence_chart(
            predicted_add, predicted_remove, alternative_add
        ))

        # 调入卡片
        parts.append(self._build_stock_section(
            "调入", predicted_add, rank_map, financial_flags, "add"
        ))

        # 调出卡片
        parts.append(self._build_stock_section(
            "调出", predicted_remove, rank_map, financial_flags, "remove"
        ))

        # 备选调入
        if alternative_add:
            parts.append(self._build_stock_section(
                "备选调入", alternative_add, rank_map, financial_flags, "alt"
            ))

        # 市值对比图
        parts.append(self._build_market_cap_chart(
            predicted_add, predicted_remove, alternative_add, rank_map
        ))

        # 分析说明
        if cleaned_reasoning:
            parts.append(self._build_reasoning_section(cleaned_reasoning))

        # 风险因素
        if details:
            parts.append(self._build_risks_section(details))

        parts.append(self._build_footer())

        return "\n".join(parts)

    def _clean_reasoning(self, reasoning: str) -> str:
        """清洗reasoning文本，过滤掉原始LLM JSON输出"""
        if not reasoning:
            return ""
        cleaned = reasoning.strip()
        is_raw = (
            cleaned.startswith("{")
            or cleaned.startswith("```")
            or cleaned.startswith("[")
            or '"predicted_add"' in cleaned
            or '"predicted_remove"' in cleaned
            or '"stock_code"' in cleaned
            or "```json" in cleaned
            or (len(cleaned) > 2000 and cleaned.count("{") > 5)
        )
        return "" if is_raw else cleaned

    def _get_confidence_level(self, conf) -> str:
        """根据置信度返回级别: high/medium/low"""
        if isinstance(conf, str):
            conf_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
            conf = conf_map.get(conf.lower(), 0.5)
        if not isinstance(conf, (int, float)):
            conf = 0.5
        if conf >= 0.75:
            return "high"
        elif conf >= 0.5:
            return "medium"
        else:
            return "low"

    def _format_confidence(self, conf) -> str:
        """格式化置信度为可读字符串"""
        if isinstance(conf, str):
            conf_map = {"high": "高", "medium": "中", "low": "低"}
            return conf_map.get(conf.lower(), conf)
        if isinstance(conf, (int, float)):
            return f"{conf:.0%}"
        return str(conf)

    def _format_market_cap(self, mv) -> str:
        """格式化市值为亿元"""
        if not mv or not isinstance(mv, (int, float)):
            return "-"
        if mv >= 1e8:
            return f"{mv / 1e8:.0f}亿"
        elif mv >= 1e4:
            return f"{mv / 1e4:.0f}万"
        return str(mv)

    def _build_head(self, index_name: str, target_date: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(index_name)} 成分股调整预测 {target_date}</title>
<style>
:root {{
  --bg: #f0f2f5;
  --bg2: #ffffff;
  --ink: #1a1a2e;
  --muted: #6b7280;
  --rule: #e5e7eb;
  --accent: #2563eb;
  --accent2: #7c3aed;
  --green: #10b981;
  --green-bg: #d1fae5;
  --green-border: #6ee7b7;
  --orange: #f59e0b;
  --orange-bg: #fef3c7;
  --orange-border: #fcd34d;
  --red: #ef4444;
  --red-bg: #fee2e2;
  --red-border: #fca5a5;
  --gray: #9ca3af;
  --gray-bg: #f3f4f6;
  --gray-border: #d1d5db;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", "PingFang SC", "Noto Sans CJK SC",
    "Microsoft YaHei", sans-serif;
  background: var(--bg);
  color: var(--ink);
  line-height: 1.6;
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}}
.section {{ margin-bottom: 32px; }}
.section-title {{
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--rule);
  display: flex;
  align-items: center;
  gap: 8px;
}}
.section-title .count {{
  font-size: 14px;
  font-weight: 500;
  color: var(--muted);
  background: var(--bg);
  padding: 2px 10px;
  border-radius: 12px;
}}
/* 摘要卡片 */
.summary {{
  background: var(--bg2);
  border-radius: 12px;
  padding: 28px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  margin-bottom: 32px;
}}
.summary-header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 20px;
}}
.summary-title {{
  font-size: 24px;
  font-weight: 700;
}}
.summary-subtitle {{
  font-size: 14px;
  color: var(--muted);
  margin-top: 4px;
}}
.summary-badge {{
  display: inline-block;
  padding: 4px 12px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
}}
.badge-llm {{ background: var(--accent); color: white; }}
.badge-quant {{ background: var(--gray); color: white; }}
.summary-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 16px;
}}
.summary-item {{
  text-align: center;
  padding: 16px;
  background: var(--bg);
  border-radius: 8px;
}}
.summary-item .value {{
  font-size: 28px;
  font-weight: 700;
  color: var(--accent);
}}
.summary-item .value.green {{ color: var(--green); }}
.summary-item .value.red {{ color: var(--red); }}
.summary-item .label {{
  font-size: 13px;
  color: var(--muted);
  margin-top: 4px;
}}
.confidence-bar {{
  margin-top: 16px;
  height: 8px;
  background: var(--rule);
  border-radius: 4px;
  overflow: hidden;
}}
.confidence-bar-fill {{
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}}
/* 股票列表（从上到下） */
.stock-list {{
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.stock-card {{
  background: var(--bg2);
  border-radius: 8px;
  padding: 14px 18px;
  border-left: 4px solid var(--gray-border);
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
  transition: box-shadow 0.2s, transform 0.15s;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}}
.stock-card:hover {{
  box-shadow: 0 3px 10px rgba(0,0,0,0.10);
  transform: translateX(2px);
}}
.stock-card.conf-high {{ border-left-color: var(--green); }}
.stock-card.conf-medium {{ border-left-color: var(--orange); }}
.stock-card.conf-low {{ border-left-color: var(--red); }}
/* 左侧：代码+名称 */
.card-stock-info {{
  display: flex;
  align-items: baseline;
  gap: 8px;
  min-width: 180px;
  flex-shrink: 0;
}}
.card-code {{
  font-size: 12px;
  color: var(--muted);
  font-family: "SF Mono", Monaco, Consolas, monospace;
}}
.card-name {{
  font-size: 15px;
  font-weight: 600;
}}
/* 中间：指标 */
.card-meta {{
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  flex: 1;
  min-width: 0;
}}
.meta-item {{
  font-size: 13px;
  white-space: nowrap;
}}
.meta-item .meta-label {{
  color: var(--muted);
}}
.meta-item .meta-value {{
  font-weight: 600;
}}
/* 财务标签 */
.card-financial {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}}
/* 右侧：置信度 */
.conf-badge {{
  padding: 3px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
  flex-shrink: 0;
  margin-left: auto;
}}
.conf-badge.high {{ background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }}
.conf-badge.medium {{ background: var(--orange-bg); color: var(--orange); border: 1px solid var(--orange-border); }}
.conf-badge.low {{ background: var(--red-bg); color: var(--red); border: 1px solid var(--red-border); }}
/* 分析理由（换行显示在下方） */
.card-reason {{
  font-size: 13px;
  color: var(--muted);
  line-height: 1.5;
  width: 100%;
  margin-top: 2px;
  padding-left: 4px;
}}
.fin-tag {{
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 500;
}}
.fin-tag.loss {{ background: var(--red-bg); color: var(--red); }}
.fin-tag.normal {{ background: var(--green-bg); color: var(--green); }}
.fin-tag.neutral {{ background: var(--gray-bg); color: var(--gray); }}
/* 图表容器 */
.chart-container {{
  background: var(--bg2);
  border-radius: 10px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.bar-chart {{
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 16px;
}}
.bar-row {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.bar-label {{
  width: 140px;
  font-size: 13px;
  font-weight: 500;
  text-align: right;
  flex-shrink: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.bar-track {{
  flex: 1;
  height: 24px;
  background: var(--bg);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}}
.bar-fill {{
  height: 100%;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 8px;
  font-size: 12px;
  font-weight: 600;
  color: white;
  min-width: 40px;
}}
.bar-fill.add {{ background: var(--green); }}
.bar-fill.remove {{ background: var(--red); }}
.bar-fill.alt {{ background: var(--orange); }}
.bar-value {{
  width: 60px;
  font-size: 13px;
  font-weight: 600;
  flex-shrink: 0;
}}
/* 置信度分布 */
.conf-dist {{
  display: flex;
  gap: 20px;
  margin-top: 16px;
  flex-wrap: wrap;
}}
.conf-dist-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}}
.conf-dot {{
  width: 12px;
  height: 12px;
  border-radius: 50%;
}}
.conf-dot.high {{ background: var(--green); }}
.conf-dot.medium {{ background: var(--orange); }}
.conf-dot.low {{ background: var(--red); }}
/* 分析说明 */
.reasoning-box {{
  background: var(--bg2);
  border-radius: 10px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  font-size: 15px;
  line-height: 1.8;
  white-space: pre-wrap;
}}
/* 风险因素 */
.risk-list {{
  background: var(--bg2);
  border-radius: 10px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.risk-list ul {{
  list-style: none;
}}
.risk-list li {{
  padding: 8px 0;
  font-size: 14px;
  border-bottom: 1px solid var(--rule);
  display: flex;
  align-items: flex-start;
  gap: 8px;
}}
.risk-list li:last-child {{ border-bottom: none; }}
.risk-list li::before {{
  content: "⚠";
  color: var(--orange);
  flex-shrink: 0;
}}
/* 空状态 */
.empty-state {{
  text-align: center;
  padding: 40px;
  color: var(--muted);
  font-size: 15px;
}}
/* 页脚 */
.footer {{
  text-align: center;
  padding: 24px;
  color: var(--muted);
  font-size: 13px;
  border-top: 1px solid var(--rule);
  margin-top: 40px;
}}
@media (max-width: 768px) {{
  body {{ padding: 12px; }}
  .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .bar-label {{ width: 100px; }}
  .card-stock-info {{ min-width: 140px; }}
}}
</style>
</head>"""

    def _build_header(
        self, index_name, target_date, cutoff_date, review_period, now,
        confidence, is_quant_only, add_count, remove_count, alt_count
    ) -> str:
        conf_pct = confidence if isinstance(confidence, (int, float)) else 0.5
        mode_class = "badge-quant" if is_quant_only else "badge-llm"
        mode_text = "纯量化" if is_quant_only else "量化+LLM"
        conf_color = (
            "green" if conf_pct >= 0.75
            else "red" if conf_pct < 0.5
            else ""
        )
        return f"""
<body>
<div class="summary">
  <div class="summary-header">
    <div>
      <div class="summary-title">{html.escape(str(index_name))} 成分股调整预测</div>
      <div class="summary-subtitle">
        预测目标: {html.escape(str(target_date))} ｜
        数据截止日: {html.escape(str(cutoff_date))} ｜
        考察期: {html.escape(str(review_period))} ｜
        生成时间: {html.escape(now)}
      </div>
    </div>
    <span class="summary-badge {mode_class}">{mode_text}</span>
  </div>
  <div class="summary-grid">
    <div class="summary-item">
      <div class="value green">{add_count}</div>
      <div class="label">预测调入</div>
    </div>
    <div class="summary-item">
      <div class="value red">{remove_count}</div>
      <div class="label">预测调出</div>
    </div>
    <div class="summary-item">
      <div class="value">{alt_count}</div>
      <div class="label">备选调入</div>
    </div>
    <div class="summary-item">
      <div class="value {conf_color}">{conf_pct:.0%}</div>
      <div class="label">整体置信度</div>
    </div>
  </div>
  <div class="confidence-bar">
    <div class="confidence-bar-fill" style="width:{conf_pct*100:.0f}%; background: {'var(--green)' if conf_pct >= 0.75 else 'var(--orange)' if conf_pct >= 0.5 else 'var(--red)'};"></div>
  </div>
</div>"""

    def _build_confidence_chart(
        self, predicted_add, predicted_remove, alternative_add
    ) -> str:
        """构建置信度分布统计"""
        all_stocks = []
        for s in predicted_add:
            all_stocks.append(("调入", s))
        for s in predicted_remove:
            all_stocks.append(("调出", s))
        for s in alternative_add:
            all_stocks.append(("备选", s))

        high_count = sum(
            1 for _, s in all_stocks
            if self._get_confidence_level(s.get("confidence", 0.5)) == "high"
        )
        med_count = sum(
            1 for _, s in all_stocks
            if self._get_confidence_level(s.get("confidence", 0.5)) == "medium"
        )
        low_count = sum(
            1 for _, s in all_stocks
            if self._get_confidence_level(s.get("confidence", 0.5)) == "low"
        )

        return f"""
<div class="section">
  <div class="section-title">置信度分布 <span class="count">共{len(all_stocks)}只</span></div>
  <div class="chart-container">
    <div class="conf-dist">
      <div class="conf-dist-item">
        <span class="conf-dot high"></span>
        <span>高置信度: <strong>{high_count}</strong> 只</span>
      </div>
      <div class="conf-dist-item">
        <span class="conf-dot medium"></span>
        <span>中置信度: <strong>{med_count}</strong> 只</span>
      </div>
      <div class="conf-dist-item">
        <span class="conf-dot low"></span>
        <span>低置信度: <strong>{low_count}</strong> 只</span>
      </div>
    </div>
  </div>
</div>"""

    def _build_stock_section(
        self, title, stocks, rank_map, financial_flags, section_type
    ) -> str:
        """构建股票卡片区域"""
        if not stocks:
            return f"""
<div class="section">
  <div class="section-title">{title} <span class="count">0只</span></div>
  <div class="chart-container">
    <div class="empty-state">无{title}候选</div>
  </div>
</div>"""

        # 按置信度从高到低排序
        def _conf_val(s):
            c = s.get("confidence", 0.5)
            if isinstance(c, str):
                return {"high": 0.9, "medium": 0.6, "low": 0.3}.get(c.lower(), 0.5)
            return c if isinstance(c, (int, float)) else 0.5

        sorted_stocks = sorted(stocks, key=_conf_val, reverse=True)

        cards = []
        for s in sorted_stocks:
            cards.append(self._build_stock_card(
                s, rank_map, financial_flags
            ))

        return f"""
<div class="section">
  <div class="section-title">{title} <span class="count">{len(stocks)}只</span></div>
  <div class="stock-list">
    {''.join(cards)}
  </div>
</div>"""

    def _build_stock_card(self, stock, rank_map, financial_flags) -> str:
        """构建单只股票卡片"""
        code = stock.get("stock_code", stock.get("code", ""))
        name = stock.get("stock_name", stock.get("name", ""))
        rank = stock.get("rank", "-")
        conf = stock.get("confidence", 0.5)
        reason = stock.get("reasoning", "")
        market_cap = stock.get("market_cap", 0)

        # 从rank_map补充市值信息
        rank_info = rank_map.get(code, {})
        if not market_cap:
            market_cap = rank_info.get("avg_total_mv", rank_info.get("total_mv", 0))
        if rank == "-":
            rank = rank_info.get("rank", "-")

        conf_level = self._get_confidence_level(conf)
        conf_text = self._format_confidence(conf)
        mv_text = self._format_market_cap(market_cap)

        reason_html = ""
        if reason:
            reason_html = (
                f'<div class="card-reason">{html.escape(str(reason))}</div>'
            )

        return f"""    <div class="stock-card conf-{conf_level}">
      <div class="card-stock-info">
        <span class="card-code">{html.escape(str(code))}</span>
        <span class="card-name">{html.escape(str(name))}</span>
      </div>
      <div class="card-meta">
        <div class="meta-item"><span class="meta-label">排名</span> <span class="meta-value">{rank}</span></div>
        <div class="meta-item"><span class="meta-label">市值</span> <span class="meta-value">{mv_text}</span></div>
      </div>
      <span class="conf-badge {conf_level}">{conf_text}</span>
      {reason_html}
    </div>"""

    def _build_market_cap_chart(
        self, predicted_add, predicted_remove, alternative_add, rank_map
    ) -> str:
        """构建市值对比柱状图"""
        all_items = []
        for s in predicted_add:
            code = s.get("stock_code", s.get("code", ""))
            name = s.get("stock_name", s.get("name", ""))
            mv = s.get("market_cap", 0) or rank_map.get(code, {}).get(
                "avg_total_mv", rank_map.get(code, {}).get("total_mv", 0)
            )
            if mv:
                all_items.append((name, mv, "add"))
        for s in predicted_remove:
            code = s.get("stock_code", s.get("code", ""))
            name = s.get("stock_name", s.get("name", ""))
            mv = s.get("market_cap", 0) or rank_map.get(code, {}).get(
                "avg_total_mv", rank_map.get(code, {}).get("total_mv", 0)
            )
            if mv:
                all_items.append((name, mv, "remove"))
        for s in alternative_add:
            code = s.get("stock_code", s.get("code", ""))
            name = s.get("stock_name", s.get("name", ""))
            mv = s.get("market_cap", 0) or rank_map.get(code, {}).get(
                "avg_total_mv", rank_map.get(code, {}).get("total_mv", 0)
            )
            if mv:
                all_items.append((name, mv, "alt"))

        if not all_items:
            return ""

        # 按市值排序，取前20
        all_items.sort(key=lambda x: x[1], reverse=True)
        all_items = all_items[:20]

        max_mv = max(mv for _, mv, _ in all_items) if all_items else 1

        bars = []
        for name, mv, bar_type in all_items:
            width_pct = (mv / max_mv * 100) if max_mv > 0 else 0
            mv_text = self._format_market_cap(mv)
            bars.append(
                f"""      <div class="bar-row">
        <span class="bar-label">{html.escape(str(name))}</span>
        <div class="bar-track">
          <div class="bar-fill {bar_type}" style="width:{width_pct:.1f}%;">{mv_text}</div>
        </div>
      </div>"""
            )

        return f"""
<div class="section">
  <div class="section-title">考察期内日均A股总市值对比 <span class="count">前{len(all_items)}只</span></div>
  <div class="chart-container">
    <div class="bar-chart">
{chr(10).join(bars)}
    </div>
  </div>
</div>"""

    def _build_reasoning_section(self, reasoning: str) -> str:
        return f"""
<div class="section">
  <div class="section-title">分析说明</div>
  <div class="reasoning-box">{html.escape(reasoning)}</div>
</div>"""

    def _build_risks_section(self, details: list) -> str:
        items = []
        for d in details:
            items.append(f"      <li>{html.escape(str(d))}</li>")
        return f"""
<div class="section">
  <div class="section-title">风险因素</div>
  <div class="risk-list">
    <ul>
{chr(10).join(items)}
    </ul>
  </div>
</div>"""

    def _build_footer(self) -> str:
        return """
<div class="footer">
  本报告由A股指数成分股调整预测系统自动生成 ｜ 仅供参考，不构成投资建议
</div>
</body>
</html>"""
