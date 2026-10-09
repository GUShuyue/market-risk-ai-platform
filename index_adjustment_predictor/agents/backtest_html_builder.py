"""
Backtest HTML Report Builder - 回测结果可视化HTML报告生成器

生成自包含的HTML页面，包含：
    - 回测摘要（准确率指标卡片）
    - 命中明细（绿色标识）
    - 遗漏分析（橙色标识，附原因分析）
    - 误报分析（红色标识，附原因分析）
    - 偏差原因总结
    - LLM深层分析（如有）
"""

from datetime import datetime
import html


class BacktestHtmlBuilder:
    """回测结果HTML可视化报告生成器"""

    def build(
        self,
        backtest_result: dict,
        index_name: str = "",
        target_date: str = "",
        use_llm: bool = False,
    ) -> str:
        """构建完整的HTML报告"""
        accuracy = backtest_result.get("accuracy", {})
        matches = backtest_result.get("matches", {})
        misses = backtest_result.get("misses", {})
        false_positives = backtest_result.get("false_positives", {})
        deviation = backtest_result.get("deviation_analysis", {})
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        parts = []
        parts.append(self._build_head(index_name, target_date))
        parts.append(self._build_header(
            index_name, target_date, now, accuracy, use_llm
        ))

        # 准确率指标卡片
        parts.append(self._build_metrics_section(accuracy))

        # 命中明细
        parts.append(self._build_match_section(
            "调入命中", matches.get("add", []), "add"
        ))
        parts.append(self._build_match_section(
            "调出命中", matches.get("remove", []), "remove"
        ))

        # 遗漏分析（附原因）
        missed_adds = deviation.get("missed_adds", [])
        missed_removes = deviation.get("missed_removes", [])
        if missed_adds:
            parts.append(self._build_deviation_section(
                "遗漏调入", missed_adds, "miss", "add"
            ))
        if missed_removes:
            parts.append(self._build_deviation_section(
                "遗漏调出", missed_removes, "miss", "remove"
            ))

        # 误报分析（附原因）
        false_adds = deviation.get("false_adds", [])
        false_removes = deviation.get("false_removes", [])
        if false_adds:
            parts.append(self._build_deviation_section(
                "误报调入", false_adds, "false", "add"
            ))
        if false_removes:
            parts.append(self._build_deviation_section(
                "误报调出", false_removes, "false", "remove"
            ))

        # 偏差总结
        if deviation.get("summary"):
            parts.append(self._build_summary_section(deviation))

        # LLM深层分析
        llm_text = deviation.get("llm_analysis", "")
        if llm_text and "不可用" not in llm_text and "失败" not in llm_text:
            parts.append(self._build_llm_section(llm_text))

        parts.append(self._build_footer())

        return "\n".join(parts)

    def _build_head(self, index_name: str, target_date: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(index_name)} 回测报告 {target_date}</title>
<style>
:root {{
  --bg: #f0f2f5;
  --bg2: #ffffff;
  --ink: #1a1a2e;
  --muted: #6b7280;
  --rule: #e5e7eb;
  --accent: #2563eb;
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
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
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
.summary-item .value.orange {{ color: var(--orange); }}
.summary-item .label {{
  font-size: 13px;
  color: var(--muted);
  margin-top: 4px;
}}
/* 指标表格 */
.metrics-table {{
  background: var(--bg2);
  border-radius: 10px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  overflow-x: auto;
}}
.metrics-table table {{
  width: 100%;
  border-collapse: collapse;
}}
.metrics-table th, .metrics-table td {{
  padding: 10px 16px;
  text-align: center;
  border-bottom: 1px solid var(--rule);
}}
.metrics-table th {{
  font-size: 14px;
  color: var(--muted);
  font-weight: 600;
}}
.metrics-table td {{
  font-size: 16px;
  font-weight: 700;
}}
.metrics-table td.metric-good {{ color: var(--green); }}
.metrics-table td.metric-mid {{ color: var(--orange); }}
.metrics-table td.metric-bad {{ color: var(--red); }}
/* 股票列表 */
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
}}
.stock-card:hover {{
  box-shadow: 0 3px 10px rgba(0,0,0,0.10);
  transform: translateX(2px);
}}
.stock-card.type-match {{ border-left-color: var(--green); }}
.stock-card.type-miss {{ border-left-color: var(--orange); }}
.stock-card.type-false {{ border-left-color: var(--red); }}
.card-header {{
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
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
.card-meta {{
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  margin-top: 6px;
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
.card-reason {{
  font-size: 13px;
  color: var(--ink);
  line-height: 1.6;
  margin-top: 8px;
  padding: 8px 12px;
  background: var(--bg);
  border-radius: 6px;
  border-left: 3px solid var(--accent);
}}
.card-reason.miss {{ border-left-color: var(--orange); }}
.card-reason.false {{ border-left-color: var(--red); }}
.fin-tag {{
  display: inline-block;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 11px;
  font-weight: 500;
  margin-left: 4px;
}}
.fin-tag.loss {{ background: var(--red-bg); color: var(--red); }}
.fin-tag.warn {{ background: var(--orange-bg); color: var(--orange); }}
.fin-tag.ok {{ background: var(--green-bg); color: var(--green); }}
/* 总结框 */
.summary-box {{
  background: var(--bg2);
  border-radius: 10px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  font-size: 15px;
  line-height: 1.8;
}}
/* LLM分析框 */
.llm-box {{
  background: var(--bg2);
  border-radius: 10px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  font-size: 15px;
  line-height: 1.8;
  white-space: pre-wrap;
  border-left: 4px solid var(--accent);
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
  .metrics-table {{ padding: 12px; }}
}}
</style>
</head>"""

    def _build_header(
        self, index_name, target_date, now, accuracy, use_llm
    ) -> str:
        mode_class = "badge-llm" if use_llm else "badge-quant"
        mode_text = "量化+LLM" if use_llm else "纯量化"
        overall_f1 = accuracy.get("overall_f1", 0)
        f1_color = "green" if overall_f1 >= 0.8 else "orange" if overall_f1 >= 0.5 else "red"

        return f"""
<body>
<div class="summary">
  <div class="summary-header">
    <div>
      <div class="summary-title">{html.escape(str(index_name))} 回测评估报告</div>
      <div class="summary-subtitle">
        预测目标: {html.escape(str(target_date))} ｜
        生成时间: {html.escape(now)}
      </div>
    </div>
    <span class="summary-badge {mode_class}">{mode_text}</span>
  </div>
  <div class="summary-grid">
    <div class="summary-item">
      <div class="value green">{accuracy.get('add_match_count', 0)}/{accuracy.get('add_actual_count', 0)}</div>
      <div class="label">调入命中</div>
    </div>
    <div class="summary-item">
      <div class="value red">{accuracy.get('remove_match_count', 0)}/{accuracy.get('remove_actual_count', 0)}</div>
      <div class="label">调出命中</div>
    </div>
    <div class="summary-item">
      <div class="value {f1_color}">{overall_f1:.0%}</div>
      <div class="label">综合F1</div>
    </div>
  </div>
</div>"""

    def _build_metrics_section(self, accuracy: dict) -> str:
        """构建准确率指标表格"""
        def _metric_class(val):
            if val >= 0.8:
                return "metric-good"
            elif val >= 0.5:
                return "metric-mid"
            else:
                return "metric-bad"

        return f"""
<div class="section">
  <div class="section-title">准确率指标</div>
  <div class="metrics-table">
    <table>
      <tr>
        <th>指标</th>
        <th>调入</th>
        <th>调出</th>
        <th>综合</th>
      </tr>
      <tr>
        <td>精确率</td>
        <td class="{_metric_class(accuracy.get('add_precision', 0))}">{accuracy.get('add_precision', 0):.1%}</td>
        <td class="{_metric_class(accuracy.get('remove_precision', 0))}">{accuracy.get('remove_precision', 0):.1%}</td>
        <td class="{_metric_class(accuracy.get('overall_precision', 0))}">{accuracy.get('overall_precision', 0):.1%}</td>
      </tr>
      <tr>
        <td>召回率</td>
        <td class="{_metric_class(accuracy.get('add_recall', 0))}">{accuracy.get('add_recall', 0):.1%}</td>
        <td class="{_metric_class(accuracy.get('remove_recall', 0))}">{accuracy.get('remove_recall', 0):.1%}</td>
        <td class="{_metric_class(accuracy.get('overall_recall', 0))}">{accuracy.get('overall_recall', 0):.1%}</td>
      </tr>
      <tr>
        <td>F1</td>
        <td class="{_metric_class(accuracy.get('add_f1', 0))}">{accuracy.get('add_f1', 0):.1%}</td>
        <td class="{_metric_class(accuracy.get('remove_f1', 0))}">{accuracy.get('remove_f1', 0):.1%}</td>
        <td class="{_metric_class(accuracy.get('overall_f1', 0))}">{accuracy.get('overall_f1', 0):.1%}</td>
      </tr>
    </table>
  </div>
</div>"""

    def _build_match_section(self, title, stocks, direction) -> str:
        """构建命中明细"""
        if not stocks:
            return f"""
<div class="section">
  <div class="section-title">{title} <span class="count">0只</span></div>
  <div class="empty-state">无{title}</div>
</div>"""

        cards = []
        for s in stocks:
            code = s.get("stock_code", s.get("code", s.get("ts_code", "")))
            name = s.get("stock_name", s.get("name", ""))
            cards.append(
                f"""    <div class="stock-card type-match">
      <div class="card-header">
        <span class="card-code">{html.escape(str(code))}</span>
        <span class="card-name">{html.escape(str(name))}</span>
      </div>
    </div>"""
            )

        return f"""
<div class="section">
  <div class="section-title">{title} <span class="count">{len(stocks)}只</span></div>
  <div class="stock-list">
{chr(10).join(cards)}
  </div>
</div>"""

    def _build_deviation_section(
        self, title, items, dev_type, direction
    ) -> str:
        """构建偏差分析区域（遗漏或误报）"""
        if not items:
            return ""

        cards = []
        for item in items:
            cards.append(self._build_deviation_card(item, dev_type))

        return f"""
<div class="section">
  <div class="section-title">{title} <span class="count">{len(items)}只</span></div>
  <div class="stock-list">
{chr(10).join(cards)}
  </div>
</div>"""

    def _build_deviation_card(self, item, dev_type) -> str:
        """构建单只偏差股票卡片"""
        code = item.get("code", "")
        name = item.get("name", "")
        rank = item.get("rank", 0)
        mv = item.get("market_cap", 0)
        reason = item.get("reason", "")
        fin = item.get("financial", {})
        is_constituent = item.get("is_constituent", False)

        # 格式化市值
        mv_text = "-"
        if mv and isinstance(mv, (int, float)):
            if mv >= 1e8:
                mv_text = f"{mv / 1e8:.0f}亿"
            elif mv >= 1e4:
                mv_text = f"{mv / 1e4:.0f}万"
            else:
                mv_text = str(mv)

        # 财务标签
        fin_tags = []
        if fin.get("available"):
            if fin.get("is_loss"):
                np = fin.get("net_profit", 0)
                fin_tags.append(
                    f'<span class="fin-tag loss">亏损{np/1e8:.1f}亿</span>'
                )
            audit = fin.get("audit_opinion", "")
            if audit and audit.strip() not in ("", "标准无保留意见", "无保留意见", "未知"):
                fin_tags.append(
                    f'<span class="fin-tag warn">审计:{html.escape(audit)}</span>'
                )
            roe = fin.get("roe", 0)
            if roe and isinstance(roe, (int, float)) and roe > 15:
                fin_tags.append(
                    f'<span class="fin-tag ok">ROE {roe:.1f}%</span>'
                )

        # 成分股标记
        constituent_tag = (
            '<span class="fin-tag warn">老样本</span>' if is_constituent
            else '<span class="fin-tag ok">新样本</span>'
        )

        reason_class = "miss" if dev_type == "miss" else "false"
        card_type = "type-miss" if dev_type == "miss" else "type-false"

        return f"""    <div class="stock-card {card_type}">
      <div class="card-header">
        <span class="card-code">{html.escape(str(code))}</span>
        <span class="card-name">{html.escape(str(name))}</span>
        {constituent_tag}{''.join(fin_tags)}
      </div>
      <div class="card-meta">
        <div class="meta-item"><span class="meta-label">排名</span> <span class="meta-value">{rank if rank else '-'}</span></div>
        <div class="meta-item"><span class="meta-label">日均市值</span> <span class="meta-value">{mv_text}</span></div>
      </div>
      <div class="card-reason {reason_class}">{html.escape(reason)}</div>
    </div>"""

    def _build_summary_section(self, deviation: dict) -> str:
        """构建偏差总结"""
        summary = deviation.get("summary", "")
        return f"""
<div class="section">
  <div class="section-title">偏差总结</div>
  <div class="summary-box">{html.escape(summary)}</div>
</div>"""

    def _build_llm_section(self, llm_text: str) -> str:
        """构建LLM深层分析"""
        return f"""
<div class="section">
  <div class="section-title">LLM深层分析</div>
  <div class="llm-box">{html.escape(llm_text)}</div>
</div>"""

    def _build_footer(self) -> str:
        return """
<div class="footer">
  本报告由A股指数成分股调整预测系统自动生成 ｜ 仅供参考，不构成投资建议
</div>
</body>
</html>"""
