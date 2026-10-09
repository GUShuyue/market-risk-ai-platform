"""
港股通调入调出批量预测系统 (Mode B) - 入口

用法:
    python main_b.py predict 2026-03           # 预测 + 生成LLM分析报告(MD+HTML)
    python main_b.py predict 2026-03 --no-report  # 仅预测，跳过LLM报告
    python main_b.py backtest 2026-03          # 回测2026年3月生效的调整

两条命令都会自动生成 JSON + HTML 报告。
predict 默认额外调用大模型生成 Markdown + HTML 格式的分析报告，
使用 --no-report 可跳过LLM报告生成。

日期推导规则:
    调整月份 → 数据截止日
    3月     → 前一年12月31日
    9月     → 当年6月30日
    6月     → 前一年12月31日 (快速纳入)
    12月    → 当年6月30日 (快速纳入)
"""

import os
import sys
import json
import argparse
import logging
from datetime import date, datetime
from typing import Tuple, Optional

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mode_b")


def resolve_dates(adjustment: str, command: str) -> Tuple[date, Optional[date]]:
    """
    从调整生效月份推导出 (cutoff_date, target_effective_date)

    参数:
        adjustment: 调整生效月份 (如 "2026-03")
        command: 当前命令 ("predict" 或 "backtest")

    返回:
        (cutoff_date, target_effective_date)
        target_effective_date在predict时为None, backtest时从历史数据查找
    """
    from mode_b.utils.date_utils import get_cutoff_date, get_effective_date

    cutoff = get_cutoff_date(adjustment)

    if command == "backtest":
        from mode_b.data.hsci_history_loader import HSCIHistoryLoader
        loader = HSCIHistoryLoader()
        loader.load()
        effective = get_effective_date(adjustment, loader)
    else:
        effective = None

    return cutoff, effective


def _run_predict_core(cutoff: date, min_mc: float) -> dict:
    """
    预测核心逻辑 (供cmd_predict和cmd_backtest调用)

    参数:
        cutoff: 数据截止日
        min_mc: 最低市值阈值 (元)

    返回:
        预测结果字典
    """
    # 加载股票信息 (含上市日期, 用于历史回测时过滤未上市股票)
    stock_info_map = {}
    connect_pool_codes = set()
    try:
        from mode_a.data.akshare_data_provider import AkShareDataProvider
        provider = AkShareDataProvider()
        all_stocks = provider.get_all_stocks()
        connect_pool_codes = provider.get_current_pool()
        connect_pool_codes = {c.strip().zfill(5) for c in connect_pool_codes}

        for s in all_stocks:
            code = s.code.strip().zfill(5)
            stock_info_map[code] = {
                "name": s.name,
                "is_secondary_listing": s.is_secondary_listing,
                "is_wvr": s.is_wvr,
                "is_a_plus_h": s.is_a_plus_h,
                "currency": s.currency,
                "listing_date": s.listing_date.isoformat() if s.listing_date else None,
            }
        logger.info(f"已加载 {len(stock_info_map)} 只股票信息, 港股通池 {len(connect_pool_codes)} 只")
    except Exception as e:
        logger.warning(f"无法加载港股通池信息: {e}")

    # 加载历史成分数据 (用于重建cutoff时点港股通池和获取当前成分股)
    from mode_b.data.hsci_history_loader import HSCIHistoryLoader
    hist_loader = HSCIHistoryLoader()
    hist_loader.load()

    # 重建cutoff日期时的港股通池 (从当前池逆推, 撤销cutoff之后的调整)
    if connect_pool_codes:
        connect_pool_codes = hist_loader.get_connect_pool_at(cutoff, connect_pool_codes)
        logger.info(f"使用cutoff时点港股通池: {len(connect_pool_codes)} 只")

    # === 计算选股范畴市值 (cutoff前12个月日均市值) ===
    # UniverseBuilder内部自动完成: HDR过滤、外国公司第二上市过滤
    from mode_b.data.universe_builder import UniverseBuilder

    builder = UniverseBuilder(cutoff_date=cutoff)
    universe_data = builder.build()

    market_cap_data = []
    skipped = 0
    for u in universe_data:
        code = u["code"]

        if code.startswith("8") and len(code) == 5:
            continue

        listing_date_str = u.get("listing_date")
        if listing_date_str:
            try:
                listing_date = datetime.strptime(
                    listing_date_str[:10], "%Y-%m-%d"
                ).date()
                if (cutoff - listing_date).days < 90:
                    skipped += 1
                    continue
            except (ValueError, TypeError):
                pass

        mc = u.get("avg_market_cap", 0)
        if mc > 0:
            market_cap_data.append({
                "code": code,
                "name": u.get("name", ""),
                "avg_market_cap": mc,
                "source": u.get("source", ""),
            })

    logger.info(
        f"选股范畴市值计算完成: {len(market_cap_data)} 只, 过滤 {skipped} 只 (HDR/未上市)"
    )

    # === 换手率测试 (编算总则5.6.2, 必做) ===
    from mode_b.data.turnover_tester import TurnoverTester
    turnover_failed_codes = set()

    turnover_file = os.path.join(
        PROJECT_ROOT, "mode_b", "data",
        f"turnover_test_{cutoff.strftime('%Y%m%d')}.json"
    )

    turnover_results = None
    if os.path.exists(turnover_file):
        with open(turnover_file, "r", encoding="utf-8") as f:
            turnover_results = json.load(f)
        logger.info(f"加载换手率测试结果: {len(turnover_results)} 只")

    constituent_codes = {c["code"].strip().zfill(5) for c in hist_loader.current_constituents}

    if turnover_results is None:
        stocks_to_test = [
            {"code": s["code"], "avg_market_cap": s["avg_market_cap"], "name": s.get("name", "")}
            for s in market_cap_data
            if s["code"] not in constituent_codes and s["avg_market_cap"] >= min_mc
        ]
        logger.info(
            f"开始换手率测试: {len(stocks_to_test)} 只非成份股 "
            f"(总{len(market_cap_data)}只, 跳过{len(market_cap_data)-len(stocks_to_test)}只成份股/低市值)"
        )
        tester = TurnoverTester(cutoff_date=cutoff)
        turnover_results = tester.test_batch(stocks_to_test)
        tester.save_results(turnover_results, turnover_file)

    for code, tr in turnover_results.items():
        if not tr.get("passed", True):
            turnover_failed_codes.add(code)

    passed_count = len(turnover_results) - len(turnover_failed_codes)
    logger.info(
        f"换手率测试: 通过{passed_count} 未通过{len(turnover_failed_codes)} "
        f"(仅过滤调入候选, 不影响覆盖率计算)"
    )

    # === 月末市值计算 ===
    from mode_b.data.month_end_market_cap import MonthEndMarketCapCalculator
    month_end_file = os.path.join(
        PROJECT_ROOT, "mode_b", "data",
        f"month_end_mc_{cutoff.strftime('%Y%m%d')}.json"
    )

    month_end_mc_map = None
    if os.path.exists(month_end_file):
        with open(month_end_file, "r", encoding="utf-8") as f:
            month_end_mc_map = json.load(f)
        logger.info(f"加载月末市值数据: {len(month_end_mc_map)} 只")

    if month_end_mc_map is None:
        me_calc = MonthEndMarketCapCalculator(cutoff_date=cutoff)
        month_end_mc_map = me_calc.calc_batch(market_cap_data, stock_info_map)
        me_calc.save_results(month_end_mc_map, month_end_file)

    logger.info(f"月末市值: {len(month_end_mc_map)} 只")

    # === 运行预测 ===
    from mode_b.core.orchestrator_b import OrchestratorB
    orch = OrchestratorB()
    result = orch.run_predict(
        market_cap_data=market_cap_data,
        cutoff_date=cutoff,
        stock_info_map=stock_info_map,
        connect_pool_codes=connect_pool_codes,
        use_current_constituents=False,  # 始终用cutoff时点的历史快照
        min_market_cap=min_mc,
        month_end_market_cap_map=month_end_mc_map,
    )

    # === 换手率后过滤 ===
    if turnover_failed_codes:
        hsci_pred = result["hsci_prediction"]
        before_adds = len(hsci_pred["additions"])
        removed_adds = []
        kept_adds = []
        for s in hsci_pred["additions"]:
            if s["code"] in turnover_failed_codes:
                removed_adds.append(s)
            else:
                kept_adds.append(s)
        hsci_pred["additions"] = kept_adds

        connect_pred = result["connect_prediction"]
        removed_codes = {s["code"] for s in removed_adds}
        connect_pred["connect_additions"] = [
            s for s in connect_pred["connect_additions"] if s["code"] not in removed_codes
        ]

        rd = result["report_data"]
        rd["hsci_additions"] = len(hsci_pred["additions"])
        rd["connect_additions"] = len(connect_pred["connect_additions"])
        rd["predicted_hsci_count"] = rd["current_hsci_count"] + rd["hsci_additions"] - rd["hsci_removals"]

        logger.info(
            f"换手率后过滤: HSCI调入 {before_adds}→{len(hsci_pred['additions'])} "
            f"(排除{len(removed_adds)}只低换手率), 港股通调入→{len(connect_pred['connect_additions'])}"
        )

    return result


def cmd_predict(args):
    """运行预测"""
    adjustment = args.date
    cutoff, _ = resolve_dates(adjustment, "predict")
    min_mc = float(getattr(args, "min_mc", 5)) * 1e8

    from mode_b.utils.date_utils import format_adjustment_label, parse_adjustment_date
    year, month = parse_adjustment_date(adjustment)
    adj_str = f"{year:04d}-{month:02d}"
    adjustment_label = format_adjustment_label(adjustment)
    print(f"\n{'='*60}")
    print(f"HSCI/港股通调入调出预测")
    print(f"调整生效: {adjustment_label}  |  数据截止: {cutoff}")
    print(f"{'='*60}")

    result = _run_predict_core(cutoff, min_mc)

    # 保存结果
    output_file = os.path.join(
        PROJECT_ROOT, "output", "mode_b",
        f"prediction_{adj_str}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    # 打印摘要
    rd = result["report_data"]
    print(f"\n选股范畴: {rd['total_universe']} 只")
    print(f"当前HSCI成份股: {rd['current_hsci_count']} 只")
    print(f"预测HSCI成份股: {rd['predicted_hsci_count']} 只")
    print(f"\nHSCI预测: 调入 {rd['hsci_additions']} 只, 调出 {rd['hsci_removals']} 只")
    print(f"港股通预测: 调入 {rd['connect_additions']} 只, 调出 {rd['connect_removals']} 只")
    print(f"不合资格: {rd['ineligible']} 只")

    # 打印调入详情
    hsci_pred = result["hsci_prediction"]
    if hsci_pred["additions"]:
        print("\n--- HSCI预测调入 ---")
        for s in hsci_pred["additions"][:20]:
            print(f"  {s['code']} {s['name']:30s} "
                  f"排名{s['rank']:4d} 覆盖率{s['coverage']:.1f}% "
                  f"市值{s['avg_market_cap_hkd_yi']:.1f}亿 "
                  f"→ {s['predicted_sub_index']}")
        if len(hsci_pred["additions"]) > 20:
            print(f"  ... 共 {len(hsci_pred['additions'])} 只")

    if hsci_pred["removals"]:
        print("\n--- HSCI预测调出 ---")
        for s in hsci_pred["removals"][:20]:
            print(f"  {s['code']} {s['name']:30s} "
                  f"排名{s['rank']:4d} 覆盖率{s['coverage']:.1f}% "
                  f"市值{s['avg_market_cap_hkd_yi']:.1f}亿 "
                  f"({s['current_sub_index']})")
        if len(hsci_pred["removals"]) > 20:
            print(f"  ... 共 {len(hsci_pred['removals'])} 只")

    # 打印港股通预测
    connect_pred = result["connect_prediction"]
    if connect_pred["connect_additions"]:
        print("\n--- 港股通预测调入 ---")
        for s in connect_pred["connect_additions"][:20]:
            print(f"  {s['code']} {s.get('name', ''):30s} "
                  f"→ {s.get('predicted_sub_index', '')} "
                  f"({s.get('eligibility', '')})")
        if len(connect_pred["connect_additions"]) > 20:
            print(f"  ... 共 {len(connect_pred['connect_additions'])} 只")

    if connect_pred["connect_removals"]:
        print("\n--- 港股通预测调出 ---")
        for s in connect_pred["connect_removals"][:20]:
            print(f"  {s['code']} {s.get('name', ''):30s} "
                  f"({s.get('reason', '')})")
        if len(connect_pred["connect_removals"]) > 20:
            print(f"  ... 共 {len(connect_pred['connect_removals'])} 只")

    # 生成HTML报告
    from reports.html_report_builder import build_prediction_report
    html_file = output_file.replace(".json", ".html")
    result["adjustment_label"] = adjustment
    build_prediction_report(result, html_file)
    print(f"\n预测结果: {output_file}")
    print(f"HTML报告: {html_file}")

    # 生成LLM分析报告 (默认开启, --no-report 可跳过)
    if not getattr(args, "no_report", False):
        print("\n[LLM报告] 正在生成大模型分析报告(MD+HTML)...")
        try:
            from reports.llm_report_generator import LLMReportGenerator
            gen = LLMReportGenerator()
            paths = gen.generate(
                prediction=result,
                adjustment_label=adj_str,
            )
            if paths:
                print(f"LLM报告(MD):   {paths['md']}")
                print(f"LLM报告(HTML): {paths['html']}")
            else:
                print("[LLM报告] 生成失败，请检查 .env 中的 LLM_API_KEY 配置")
        except Exception as e:
            print(f"[LLM报告] 生成失败: {e}")


def cmd_backtest(args):
    """运行回测"""
    adjustment = args.date
    cutoff, target = resolve_dates(adjustment, "backtest")
    min_mc = float(getattr(args, "min_mc", 5)) * 1e8

    from mode_b.utils.date_utils import format_adjustment_label, parse_adjustment_date
    year, month = parse_adjustment_date(adjustment)
    adj_str = f"{year:04d}-{month:02d}"
    adjustment_label = format_adjustment_label(adjustment)
    print(f"\n{'='*60}")
    print(f"HSCI/港股通回测验证")
    print(f"调整生效: {adjustment_label}  |  数据截止: {cutoff}")
    print(f"{'='*60}")

    # 1. 检查预测结果是否存在, 不存在则自动运行预测
    pred_file = os.path.join(
        PROJECT_ROOT, "output", "mode_b",
        f"prediction_{adj_str}.json"
    )

    if not os.path.exists(pred_file):
        print(f"\n[步骤1] 预测结果不存在, 自动运行预测...")
        prediction = _run_predict_core(cutoff, min_mc)
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(prediction, f, ensure_ascii=False, indent=2, default=str)

        # 生成HTML报告
        from reports.html_report_builder import build_prediction_report
        prediction["adjustment_label"] = adjustment
        build_prediction_report(prediction, pred_file.replace(".json", ".html"))
        print(f"  预测完成, 结果已保存")
    else:
        print(f"\n[步骤1] 加载已有预测结果: {pred_file}")
        with open(pred_file, "r", encoding="utf-8") as f:
            prediction = json.load(f)

    print(f"\n[步骤2] 回测: cutoff={cutoff}, target_adjustment={target}")

    # 3. 执行回测
    from mode_b.core.orchestrator_b import OrchestratorB
    orch = OrchestratorB()
    result = orch.run_backtest(
        prediction=prediction,
        target_adjustment_date=target,
    )

    # 保存结果
    output_file = os.path.join(
        PROJECT_ROOT, "output", "mode_b",
        f"backtest_{adj_str}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    # 打印结果
    print(f"\n[步骤3] 回测结果 (target: {target})")

    l1 = result["layer1_hsci"]
    l2 = result["layer2_connect"]

    print("\n--- Layer 1: HSCI预测准确率 ---")
    print(f"  调入: P={l1['add']['precision']:.2%} R={l1['add']['recall']:.2%} F1={l1['add']['f1']:.2%} "
          f"(命中{l1['add']['matches']}/实际{l1['add']['actual']}/预测{l1['add']['predicted']})")
    print(f"  调出: P={l1['remove']['precision']:.2%} R={l1['remove']['recall']:.2%} F1={l1['remove']['f1']:.2%} "
          f"(命中{l1['remove']['matches']}/实际{l1['remove']['actual']}/预测{l1['remove']['predicted']})")
    print(f"  综合: P={l1['overall']['precision']:.2%} R={l1['overall']['recall']:.2%} F1={l1['overall']['f1']:.2%}")

    print("\n--- Layer 2: 港股通映射准确率 ---")
    print(f"  调入: P={l2['add']['precision']:.2%} R={l2['add']['recall']:.2%} F1={l2['add']['f1']:.2%} "
          f"(命中{l2['add']['matches']}/实际{l2['add']['actual']}/预测{l2['add']['predicted']})")
    print(f"  调出: P={l2['remove']['precision']:.2%} R={l2['remove']['recall']:.2%} F1={l2['remove']['f1']:.2%} "
          f"(命中{l2['remove']['matches']}/实际{l2['remove']['actual']}/预测{l2['remove']['predicted']})")
    print(f"  综合: P={l2['overall']['precision']:.2%} R={l2['overall']['recall']:.2%} F1={l2['overall']['f1']:.2%}")

    print(f"\n偏差分析: {result['deviation_analysis']['summary']}")

    # 生成HTML报告
    from reports.html_report_builder import build_backtest_report
    html_file = output_file.replace(".json", ".html")
    result["adjustment_label"] = adjustment
    result["cutoff_date_str"] = cutoff.isoformat()
    build_backtest_report(result, html_file)
    print(f"\n回测结果: {output_file}")
    print(f"HTML报告: {html_file}")


def main():
    parser = argparse.ArgumentParser(
        description="港股通调入调出批量预测系统 (Mode B)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # predict
    sp_predict = subparsers.add_parser("predict", help="运行预测")
    sp_predict.add_argument("date", help="调整生效月份, 如 2026-03, 2026年9月")
    sp_predict.add_argument("--min-mc", type=float, default=5,
                            help="最低市值过滤阈值(亿港币), 近似流动性测试, 默认5亿")
    sp_predict.add_argument("--no-report", action="store_true", default=False,
                            help="跳过LLM分析报告生成(默认会生成MD+HTML分析报告)")

    # backtest
    sp_backtest = subparsers.add_parser("backtest", help="运行回测")
    sp_backtest.add_argument("date", help="调整生效月份, 如 2026-03, 2026年9月")
    sp_backtest.add_argument("--min-mc", type=float, default=5,
                             help="最低市值过滤阈值(亿港币), 与predict一致, 默认5亿")

    args = parser.parse_args()

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
