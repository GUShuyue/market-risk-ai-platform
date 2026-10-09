#!/usr/bin/env python3
"""
港股通标的池监控系统 - 模式A 持续预警扫描

入口文件。使用真实市场数据执行全市场扫描，
输出控制台报告和HTML报告。

每次扫描自动保存状态，并与上次扫描结果对比，
识别新出现的风险并生成变更检测报告。

数据源：
  - 港股通标的池.xls：港股通标的池清单（上交所官网下载）
  - hsci_classification.csv：HSCI指数分类（恒生官网PDF转换，convert_hsci_pdf.py生成）
  - akshare(新浪)：全港股实时行情、港股历史日K线、A+H股清单
  - 百度估值：港股历史总市值
  - 东方财富：港股上市日期、A股ST/*ST列表

用法：
    python main_a.py              # 仅扫描（默认）
    python main_a.py --date 2025-12-31  # 指定基准日
    python main_a.py --change     # 扫描 + 变更检测
"""

import os
import sys
import argparse
from datetime import date

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mode_a.data.akshare_data_provider import AkShareDataProvider
from mode_a.core.rules import RuleEngine
from mode_a.core.monitor import HkConnectMonitor
from reports.reporter import ConsoleReporter, HtmlReporter, ChangeReporter
from mode_a.state.state_store import StateStore
from mode_a.core.change_detector import ChangeDetector

# 报告输出目录
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "mode_a")


def main():
    parser = argparse.ArgumentParser(description="港股通标的池监控系统 - 模式A")
    parser.add_argument("--date", type=str, default=None,
                        help="扫描基准日 YYYY-MM-DD（默认今天）")
    parser.add_argument("--change", action="store_true",
                        help="启用变更检测（对比上次扫描结果，默认关闭）")
    args = parser.parse_args()

    # 设定扫描基准日
    if args.date:
        scan_date = date.fromisoformat(args.date)
    else:
        scan_date = date.today()

    print("正在初始化港股通标的池监控系统...")

    # 1. 初始化数据提供器
    print("  [模式] 使用真实数据 (akshare + 百度估值 + HSCI分类CSV + ST检测)")
    data_provider = AkShareDataProvider(reference_date=scan_date, verbose=True)

    # 2. 初始化规则引擎
    rule_engine = RuleEngine(reference_date=scan_date)

    # 3. 初始化监控协调器
    monitor = HkConnectMonitor(
        data_provider=data_provider,
        rule_engine=rule_engine,
        reference_date=scan_date,
    )

    # 4. 加载上次扫描状态（用于变更检测）
    state_store = StateStore()
    prev_state = None
    if args.change:
        prev_state = state_store.load_latest()
        if prev_state:
            prev_date = state_store.get_previous_scan_date(prev_state)
            print(f"  [状态] 上次扫描: {prev_date}")
        else:
            print("  [状态] 无历史扫描记录（首次运行）")

    # 5. 执行扫描
    print(f"开始扫描（基准日: {scan_date}）...")
    summary = monitor.scan()
    print(f"扫描完成，共评估 {summary.total_stocks} 只股票\n")

    # 6. 输出控制台报告
    console_reporter = ConsoleReporter()
    console_reporter.report(summary)

    # 7. 输出HTML扫描报告
    html_reporter = HtmlReporter()
    scan_html_path = os.path.join(REPORTS_DIR, "report.html")
    html_reporter.report(summary, scan_html_path)
    print(f"\nHTML扫描报告已生成: {scan_html_path}")

    # 8. 保存当前扫描状态
    state_path = state_store.save(summary)
    print(f"扫描状态已保存: {state_path}")

    # 9. 变更检测
    if args.change and prev_state:
        print("\n正在执行变更检测...")
        prev_map = state_store.deserialize_to_map(prev_state)
        curr_map = state_store.deserialize_to_map(
            state_store.load_latest()
        )

        prev_date = state_store.get_previous_scan_date(prev_state)
        detector = ChangeDetector()
        change_report = detector.detect(
            curr_map, prev_map,
            current_date=scan_date,
            previous_date=prev_date,
        )

        # 控制台变更报告
        change_reporter = ChangeReporter()
        change_reporter.console_report(change_report)

        # HTML变更报告
        change_html_path = os.path.join(REPORTS_DIR, "change_report.html")
        change_reporter.html_report(change_report, change_html_path)
        print(f"\n变更检测报告已生成: {change_html_path}")

        if change_report.has_alerts:
            print("\n>>> 检测到需要关注的变更，请查看变更检测报告 <<<")
    elif args.change and not prev_state:
        print("\n首次运行，无上次扫描数据可比对。下次运行将自动生成变更检测报告。")


if __name__ == "__main__":
    main()
