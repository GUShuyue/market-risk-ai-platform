"""
港股通标的池监控系统 - 报告生成器

支持两种输出：
1. ConsoleReporter - 控制台格式化输出
2. HtmlReporter - HTML报告文件
"""

import os
from datetime import date
from typing import List, Optional

from mode_a.models import AssessmentResult, RiskLevel, RiskType, format_hkd
from mode_a.core.monitor import ScanSummary
from mode_a.core.change_detector import ChangeReport, ChangeItem, AlertLevel, ChangeType


class ConsoleReporter:
    """控制台报告生成器"""

    def report(self, summary: ScanSummary) -> None:
        """打印扫描结果到控制台"""
        print()
        print("=" * 64)
        print("  港股通标的池监控系统 · 模式A 持续预警扫描")
        print("=" * 64)
        print(f"扫描时间: {summary.scan_date}")
        print(f"扫描股票数: {summary.total_stocks}")
        print(f"当前标的池: {summary.pool_size} 只")

        # 风险清单
        risk_list = summary.risk_list
        if risk_list:
            print()
            print("━" * 3 + " 风险清单 " + "━" * 52)
            for r in risk_list:
                self._print_risk_item(r)
        else:
            print("\n未发现风险股票")

        # 安全股票
        safe_list = summary.safe_list
        if safe_list:
            print()
            print("━" * 3 + " 安全股票 " + "━" * 51)
            for r in safe_list:
                self._print_safe_item(r)

        # 汇总
        print()
        print("━" * 64)
        print(f"高风险: {summary.high_risk_count} 只 | "
              f"中风险: {summary.medium_risk_count} 只 | "
              f"低风险: {summary.low_risk_count} 只 | "
              f"安全: {summary.safe_count} 只")

        # 按类型统计
        type_counts = summary.count_by_risk_type()
        if type_counts:
            type_str = " | ".join(f"{rt.value}: {cnt}" for rt, cnt in type_counts.items())
            print(f"风险类型: {type_str}")

        print("=" * 64)

    def _print_risk_item(self, r: AssessmentResult) -> None:
        level_icon = {"高": "■", "中": "◆", "低": "○"}
        icon = level_icon.get(r.risk_level.value, "·")
        pool_tag = "在池" if r.in_pool else "不在池"
        print(f"\n{icon} [{r.risk_level.value}] {r.code} {r.name} "
              f"({r.index_type.value}·{pool_tag})")
        print(f"  类型: {r.risk_type.value}")
        print(f"  详情: {r.details}")

        # 打印关键量化指标
        if r.metrics:
            metrics_str = self._format_metrics(r)
            if metrics_str:
                print(f"  指标: {metrics_str}")

    def _print_safe_item(self, r: AssessmentResult) -> None:
        pool_tag = "在池" if r.in_pool else "不在池"
        print(f"  [安全] {r.code} {r.name} ({r.index_type.value}·{pool_tag})")

    def _format_metrics(self, r: AssessmentResult) -> str:
        m = r.metrics
        parts = []

        # 小型股市值指标
        if "avg_monthly_market_cap" in m:
            parts.append(f"平均月末市值 {format_hkd(m['avg_monthly_market_cap'])}")
        if "distance_to_removal" in m:
            parts.append(f"距调出线 {m['distance_to_removal']:+.1%}")
        if "distance_to_inclusion" in m:
            parts.append(f"距调入线 {m['distance_to_inclusion']:+.1%}")

        # WVR指标
        if "avg_daily_market_cap_183d" in m:
            parts.append(f"183日日均市值 {format_hkd(m['avg_daily_market_cap_183d'])}")
        if "total_trading_value_183d" in m:
            parts.append(f"183日成交总额 {format_hkd(m['total_trading_value_183d'])}")
        if "listing_months_actual" in m:
            parts.append(f"上市{m['listing_months_actual']}个月")

        return " | ".join(parts)


class HtmlReporter:
    """HTML报告生成器"""

    def report(self, summary: ScanSummary, output_path: str) -> str:
        """生成HTML报告文件，返回文件路径"""
        html = self._build_html(summary)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    def _build_html(self, summary: ScanSummary) -> str:
        risk_rows = "\n".join(
            self._build_risk_row(r) for r in summary.risk_list
        )
        safe_rows = "\n".join(
            self._build_safe_row(r) for r in summary.safe_list
        )
        type_counts = summary.count_by_risk_type()
        type_summary = " · ".join(
            f"{rt.value} {cnt}" for rt, cnt in type_counts.items()
        ) if type_counts else "无"

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>港股通标的池监控报告 - {summary.scan_date}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif;
          background: #f5f5f8; color: #1a1a1a; padding: 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  .header {{ background: #4B3FE3; color: #fff; padding: 24px 32px; border-radius: 12px;
             margin-bottom: 24px; }}
  .header h1 {{ font-size: 22px; margin-bottom: 8px; }}
  .header .meta {{ font-size: 14px; opacity: 0.9; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat-card {{ background: #fff; padding: 16px 24px; border-radius: 10px; flex: 1;
                min-width: 140px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .stat-card .label {{ font-size: 13px; color: #71717a; margin-bottom: 4px; }}
  .stat-card .value {{ font-size: 24px; font-weight: 600; }}
  .stat-card.high .value {{ color: #E8463A; }}
  .stat-card.medium .value {{ color: #EFAA17; }}
  .stat-card.low .value {{ color: #3B82F6; }}
  .stat-card.safe .value {{ color: #1DC981; }}
  .section {{ background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 24px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .section h2 {{ font-size: 16px; margin-bottom: 16px; padding-bottom: 12px;
                 border-bottom: 1px solid #e4e4e7; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: left; padding: 10px 12px; background: #f4f4f5; font-weight: 500;
        color: #52525b; font-size: 13px; white-space: nowrap; }}
  td {{ padding: 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }}
  tr:hover td {{ background: #fafafa; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 12px; font-weight: 500; }}
  .badge-high {{ background: #fee2e2; color: #dc2626; }}
  .badge-medium {{ background: #fef3c7; color: #d97706; }}
  .badge-low {{ background: #dbeafe; color: #2563eb; }}
  .badge-safe {{ background: #d1fae5; color: #059669; }}
  .badge-pool {{ background: #ede9fe; color: #6d28d9; }}
  .badge-nopool {{ background: #f4f4f5; color: #71717a; }}
  .badge-type {{ background: #f0f9ff; color: #0369a1; }}
  .details {{ color: #52525b; line-height: 1.6; font-size: 13px; }}
  .metrics {{ margin-top: 6px; font-size: 12px; color: #71717a; font-family: monospace; }}
  .footer {{ text-align: center; padding: 16px; color: #a1a1aa; font-size: 12px; }}
  .type-summary {{ margin-top: 8px; font-size: 13px; color: #52525b; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>港股通标的池监控报告 · 模式A 持续预警扫描</h1>
    <div class="meta">扫描时间: {summary.scan_date} · 扫描股票: {summary.total_stocks} 只 · 当前标的池: {summary.pool_size} 只</div>
  </div>

  <div class="stats">
    <div class="stat-card high"><div class="label">高风险</div><div class="value">{summary.high_risk_count}</div></div>
    <div class="stat-card medium"><div class="label">中风险</div><div class="value">{summary.medium_risk_count}</div></div>
    <div class="stat-card low"><div class="label">低风险</div><div class="value">{summary.low_risk_count}</div></div>
    <div class="stat-card safe"><div class="label">安全</div><div class="value">{summary.safe_count}</div></div>
  </div>

  <div class="section">
    <h2>风险清单</h2>
    <div class="type-summary">风险类型分布: {type_summary}</div>
    <table style="margin-top:12px">
      <thead>
        <tr>
          <th>风险等级</th><th>代码</th><th>名称</th><th>指数类型</th>
          <th>池内</th><th>风险类型</th><th>详情</th>
        </tr>
      </thead>
      <tbody>
        {risk_rows if risk_rows else '<tr><td colspan="7" style="text-align:center;color:#a1a1aa;padding:24px">未发现风险股票</td></tr>'}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>安全股票 ({len(summary.safe_list)})</h2>
    <table>
      <thead>
        <tr><th>代码</th><th>名称</th><th>指数类型</th><th>池内</th><th>说明</th></tr>
      </thead>
      <tbody>
        {safe_rows if safe_rows else '<tr><td colspan="5" style="text-align:center;color:#a1a1aa;padding:12px">无</td></tr>'}
      </tbody>
    </table>
  </div>

  <div class="footer">
    本报告由港股通标的池监控系统自动生成 · 数据来源：akshare + 东方财富API
  </div>
</div>
</body>
</html>"""

    def _build_risk_row(self, r: AssessmentResult) -> str:
        level_class = {
            "高": "badge-high", "中": "badge-medium", "低": "badge-low"
        }.get(r.risk_level.value, "badge-low")
        pool_class = "badge-pool" if r.in_pool else "badge-nopool"
        pool_text = "在池" if r.in_pool else "不在池"

        metrics_html = self._build_metrics_html(r)

        return f"""        <tr>
          <td><span class="badge {level_class}">{r.risk_level.value}</span></td>
          <td><code>{r.code}</code></td>
          <td>{r.name}</td>
          <td>{r.index_type.value}</td>
          <td><span class="badge {pool_class}">{pool_text}</span></td>
          <td><span class="badge badge-type">{r.risk_type.value}</span></td>
          <td><div class="details">{r.details}</div>{metrics_html}</td>
        </tr>"""

    def _build_safe_row(self, r: AssessmentResult) -> str:
        pool_class = "badge-pool" if r.in_pool else "badge-nopool"
        pool_text = "在池" if r.in_pool else "不在池"
        return f"""        <tr>
          <td><code>{r.code}</code></td>
          <td>{r.name}</td>
          <td>{r.index_type.value}</td>
          <td><span class="badge {pool_class}">{pool_text}</span></td>
          <td><span class="badge badge-safe">安全</span></td>
        </tr>"""

    def _build_metrics_html(self, r: AssessmentResult) -> str:
        m = r.metrics
        parts = []

        if "avg_monthly_market_cap" in m:
            parts.append(f"平均月末市值: {format_hkd(m['avg_monthly_market_cap'])}")
        if "distance_to_removal" in m:
            parts.append(f"距调出线: {m['distance_to_removal']:+.1%}")
        if "distance_to_inclusion" in m:
            parts.append(f"距调入线: {m['distance_to_inclusion']:+.1%}")
        if "avg_daily_market_cap_183d" in m:
            parts.append(f"183日日均市值: {format_hkd(m['avg_daily_market_cap_183d'])}")
        if "total_trading_value_183d" in m:
            parts.append(f"183日成交总额: {format_hkd(m['total_trading_value_183d'])}")
        if "listing_months_actual" in m:
            parts.append(f"上市: {m['listing_months_actual']}个月")
        if "all_conditions_met" in m:
            parts.append(f"条件全满足: {'是' if m['all_conditions_met'] else '否'}")

        if not parts:
            return ""

        return f'<div class="metrics">{" | ".join(parts)}</div>'


class ChangeReporter:
    """变更检测报告生成器（控制台+HTML）"""

    def console_report(self, report: ChangeReport) -> None:
        """打印变更报告到控制台"""
        print()
        print("=" * 64)
        print("  港股通标的池监控 · 变更检测报告")
        print("=" * 64)
        print(f"当前扫描: {report.current_scan_date}")
        if report.previous_scan_date:
            print(f"上次扫描: {report.previous_scan_date}")
        else:
            print(f"上次扫描: 无（首次运行）")

        if not report.has_changes:
            print("\n无变更")
            print("=" * 64)
            return

        print(f"\n变更总数: {len(report.changes)}")

        # 紧急告警
        critical = report.critical_items
        if critical:
            print()
            print("!" * 3 + f" 紧急告警 ({len(critical)}) " + "!" * 44)
            for item in critical:
                self._print_item(item)

        # 预警
        warnings = report.warning_items
        if warnings:
            print()
            print("~" * 3 + f" 预警 ({len(warnings)}) " + "~" * 46)
            for item in warnings:
                self._print_item(item)

        # 通知
        infos = report.info_items
        if infos:
            print()
            print("-" * 3 + f" 通知 ({len(infos)}) " + "-" * 46)
            for item in infos:
                self._print_item(item)

        # 解除
        resolved = report.resolved_items
        if resolved:
            print()
            print("+" * 3 + f" 风险解除 ({len(resolved)}) " + "+" * 42)
            for item in resolved:
                self._print_item(item)

        print()
        print("=" * 64)

    def _print_item(self, item: ChangeItem) -> None:
        icons = {
            AlertLevel.CRITICAL: "!!!",
            AlertLevel.WARNING: " ! ",
            AlertLevel.INFO: "   ",
            AlertLevel.RESOLVED: " + ",
        }
        icon = icons.get(item.alert_level, "   ")
        pool_tag = ""
        if item.curr_in_pool is not None:
            pool_tag = " [在池]" if item.curr_in_pool else " [不在池]"
        elif item.prev_in_pool is not None:
            pool_tag = " [在池]" if item.prev_in_pool else " [不在池]"

        print(f"\n{icon} [{item.alert_level.value}] {item.code} {item.name}{pool_tag}")
        print(f"  变更: {item.change_type.value} - {item.summary}")
        if item.curr_details:
            print(f"  详情: {item.curr_details}")
        if item.metrics_change:
            for k, v in item.metrics_change.items():
                print(f"  指标变化 [{k}]: {v}")

    def html_report(self, report: ChangeReport, output_path: str) -> str:
        """生成HTML变更报告"""
        html = self._build_html(report)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    def _build_html(self, report: ChangeReport) -> str:
        critical_rows = "\n".join(
            self._build_change_row(c) for c in report.critical_items
        )
        warning_rows = "\n".join(
            self._build_change_row(c) for c in report.warning_items
        )
        info_rows = "\n".join(
            self._build_change_row(c) for c in report.info_items
        )
        resolved_rows = "\n".join(
            self._build_change_row(c) for c in report.resolved_items
        )

        prev_date_str = report.previous_scan_date.isoformat() if report.previous_scan_date else "无（首次运行）"

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>港股通变更检测报告 - {report.current_scan_date}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif;
          background: #f5f5f8; color: #1a1a1a; padding: 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  .header {{ background: #DC2626; color: #fff; padding: 24px 32px; border-radius: 12px;
             margin-bottom: 24px; }}
  .header.no-alerts {{ background: #059669; }}
  .header h1 {{ font-size: 22px; margin-bottom: 8px; }}
  .header .meta {{ font-size: 14px; opacity: 0.9; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat-card {{ background: #fff; padding: 16px 24px; border-radius: 10px; flex: 1;
                min-width: 120px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
  .stat-card .label {{ font-size: 13px; color: #71717a; margin-bottom: 4px; }}
  .stat-card .value {{ font-size: 28px; font-weight: 700; }}
  .stat-card.critical .value {{ color: #DC2626; }}
  .stat-card.warning .value {{ color: #EFAA17; }}
  .stat-card.info .value {{ color: #3B82F6; }}
  .stat-card.resolved .value {{ color: #059669; }}
  .section {{ background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 24px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .section h2 {{ font-size: 16px; margin-bottom: 16px; padding-bottom: 12px;
                 border-bottom: 1px solid #e4e4e7; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: left; padding: 10px 12px; background: #f4f4f5; font-weight: 500;
        color: #52525b; font-size: 13px; white-space: nowrap; }}
  td {{ padding: 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }}
  tr:hover td {{ background: #fafafa; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 12px; font-weight: 500; }}
  .badge-critical {{ background: #fee2e2; color: #dc2626; }}
  .badge-warning {{ background: #fef3c7; color: #d97706; }}
  .badge-info {{ background: #dbeafe; color: #2563eb; }}
  .badge-resolved {{ background: #d1fae5; color: #059669; }}
  .details {{ color: #52525b; line-height: 1.6; font-size: 13px; }}
  .metrics-change {{ margin-top: 6px; font-size: 12px; color: #71717a; font-family: monospace; }}
  .footer {{ text-align: center; padding: 16px; color: #a1a1aa; font-size: 12px; }}
  .empty {{ text-align: center; color: #a1a1aa; padding: 24px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header {'no-alerts' if not report.has_alerts else ''}">
    <h1>港股通标的池监控 · 变更检测报告</h1>
    <div class="meta">
      当前扫描: {report.current_scan_date} ·
      上次扫描: {prev_date_str} ·
      变更总数: {len(report.changes)}
    </div>
  </div>

  <div class="stats">
    <div class="stat-card critical"><div class="label">紧急</div><div class="value">{len(report.critical_items)}</div></div>
    <div class="stat-card warning"><div class="label">预警</div><div class="value">{len(report.warning_items)}</div></div>
    <div class="stat-card info"><div class="label">通知</div><div class="value">{len(report.info_items)}</div></div>
    <div class="stat-card resolved"><div class="label">解除</div><div class="value">{len(report.resolved_items)}</div></div>
  </div>

  {self._build_section("紧急告警", critical_rows, "badge-critical", "新出现的高风险排除事件（ST/退市/非港币报价）")}
  {self._build_section("预警", warning_rows, "badge-warning", "新出现的其他高风险 或 风险升级到高")}
  {self._build_section("通知", info_rows, "badge-info", "风险升级/降级/类型变化/新进入扫描")}
  {self._build_section("风险解除", resolved_rows, "badge-resolved", "从有风险变为安全的股票")}

  <div class="footer">
    本报告由港股通标的池监控系统自动生成 · 变更检测基于历史扫描状态对比
  </div>
</div>
</body>
</html>"""

    def _build_section(self, title: str, rows: str, badge_class: str, desc: str) -> str:
        if not rows:
            return f"""  <div class="section">
    <h2>{title} (0)</h2>
    <p style="color:#a1a1aa;font-size:13px;padding:8px 0">{desc}</p>
    <div class="empty">无</div>
  </div>"""
        count = rows.count("<tr>")
        return f"""  <div class="section">
    <h2>{title} ({count})</h2>
    <p style="color:#a1a1aa;font-size:13px;padding:8px 0">{desc}</p>
    <table>
      <thead>
        <tr><th>告警等级</th><th>代码</th><th>名称</th><th>变更类型</th><th>变更摘要</th><th>详情</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>"""

    def _build_change_row(self, item: ChangeItem) -> str:
        level_class = {
            AlertLevel.CRITICAL.value: "badge-critical",
            AlertLevel.WARNING.value: "badge-warning",
            AlertLevel.INFO.value: "badge-info",
            AlertLevel.RESOLVED.value: "badge-resolved",
        }.get(item.alert_level.value, "badge-info")

        metrics_html = ""
        if item.metrics_change:
            parts = [f"{k}: {v}" for k, v in item.metrics_change.items()]
            metrics_html = f'<div class="metrics-change">{" | ".join(parts)}</div>'

        details = item.curr_details or item.prev_details or ""

        return f"""        <tr>
          <td><span class="badge {level_class}">{item.alert_level.value}</span></td>
          <td><code>{item.code}</code></td>
          <td>{item.name}</td>
          <td><span class="badge {level_class}">{item.change_type.value}</span></td>
          <td>{item.summary}</td>
          <td><div class="details">{details}</div>{metrics_html}</td>
        </tr>"""
