"""
项目入口 - 命令行交互

用法:
    python main.py                              # 启动交互式对话
    python main.py --api                        # 启动API服务
    python main.py --predict 2024-06 csi300     # 直接预测（量化+LLM）
    python main.py --predict 2024-06 csi300 --no-llm  # 仅量化预测
    python main.py --backtest 2024-06 csi300    # 回测验证
"""

import sys
import argparse
import yaml
import json
import os
import glob
import re
from interface.chat_handler import ChatHandler


def load_env():
    """加载 .env 文件中的环境变量"""
    env_file = ".env"
    if not os.path.exists(env_file):
        return

    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # 移除引号
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                os.environ[key] = value


def expand_env_vars(config):
    """递归替换配置中的环境变量引用 ${VAR_NAME}"""
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        # 匹配 ${VAR_NAME} 模式
        pattern = r'\$\{([^}]+)\}'
        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, "")
        return re.sub(pattern, replacer, config)
    else:
        return config


def load_config():
    """加载配置"""
    # 先加载环境变量
    load_env()

    config = {}
    config_dir = "config"
    for f in os.listdir(config_dir):
        if f.endswith(".yaml"):
            with open(os.path.join(config_dir, f), "r", encoding="utf-8") as fp:
                config.update(yaml.safe_load(fp) or {})

    # 替换环境变量引用
    config = expand_env_vars(config)

    return config


def run_interactive():
    """运行交互式对话"""
    config = load_config()
    handler = ChatHandler(config)

    print("=" * 60)
    print("  A股指数成分股调整预测系统")
    print("  支持指数: 沪深300")
    print("=" * 60)
    print()

    while True:
        user_input = input("请输入您的需求> ").strip()
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("再见!")
            break
        if not user_input:
            continue

        response = handler.process(user_input)
        print()
        print(response)
        print()


def run_api():
    """启动API服务"""
    import uvicorn
    print("启动API服务: http://localhost:8000")
    # reload=False: 热重载会 spawn 子进程，Windows 下 Ctrl+C 只杀父进程，子进程成为孤儿继续运行
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)


def run_predict(target_date: str, index_name: str, use_llm: bool = True):
    """直接执行预测"""
    from utils.date_utils import get_cutoff_date
    config = load_config()
    handler = ChatHandler(config)
    cutoff_date = get_cutoff_date(target_date)
    result = handler.orchestrator.run_workflow(
        "predict_adjustment",
        {"target_date": target_date, "index_name": index_name,
         "cutoff_date": cutoff_date, "use_llm": use_llm}
    )

    # 只打印完成提示，详细结果见报告文件
    report_path = result.get("report_path", "")
    analysis_report_path = result.get("analysis_report_path", "")
    analysis_report_html_path = result.get("analysis_report_html_path", "")
    mode = "量化+LLM" if use_llm else "纯量化"
    print()
    print(f"预测完成（{mode}模式），完整报告: {report_path}")
    if analysis_report_path:
        print(f"大模型分析报告: {analysis_report_path}")
    if analysis_report_html_path:
        print(f"大模型分析报告(HTML): {analysis_report_html_path}")


def find_latest_prediction_json(
    index_name: str, target_date: str, use_llm: bool = True
) -> str:
    """
    查找最新的预测结果 JSON 文件

    文件名格式:
        - LLM模式: prediction_{index_name}_{target_date}.json
        - 纯量化模式: prediction_{index_name}_{target_date}_no_llm.json

    返回:
        最新文件路径，如果没找到返回 None
    """
    report_dir = "reports"
    suffix = "" if use_llm else "_no_llm"
    pattern = os.path.join(
        report_dir, f"prediction_{index_name}_{target_date}{suffix}.json"
    )

    files = glob.glob(pattern)
    if not files:
        return None

    # 按修改时间排序，取最新的
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_prediction_from_json(filepath: str) -> dict:
    """从 JSON 文件加载预测结果"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_backtest(target_date: str, index_name: str, use_llm: bool = True):
    """
    执行回测验证

    流程:
        1. 查找是否有对应日期的预测结果 JSON
        2. 如果有，直接加载；如果没有，先运行预测
        3. 获取实际调整数据
        4. 对比预测与实际
    """
    from utils.date_utils import get_cutoff_date
    config = load_config()
    mode = "量化+LLM" if use_llm else "纯量化"

    print(f"=== 回测验证: {target_date} {index_name}（{mode}模式）===")
    print()

    # 1. 查找并加载预测结果
    print("[步骤1] 加载预测结果...")
    prediction_file = find_latest_prediction_json(
        index_name, target_date, use_llm=use_llm
    )

    if prediction_file:
        print(f"  找到预测文件: {prediction_file}")
        prediction_data = load_prediction_from_json(prediction_file)
        prediction = prediction_data.get("prediction", {})
        rule_result = prediction_data.get("rule_result")
        market_data = prediction_data.get("market_data")
        print(f"  预测调入: {len(prediction.get('predicted_add', []))} 只")
        alt_count = len(prediction.get('alternative_add', []))
        if alt_count:
            print(f"  备选调入: {alt_count} 只")
        print(f"  预测调出: {len(prediction.get('predicted_remove', []))} 只")
    else:
        print(f"  未找到{mode}预测文件，开始预测...")
        cutoff_date = get_cutoff_date(target_date)
        handler = ChatHandler(config)
        predict_result = handler.orchestrator.run_workflow(
            "predict_adjustment",
            {"target_date": target_date, "index_name": index_name,
             "cutoff_date": cutoff_date, "use_llm": use_llm}
        )
        prediction = predict_result.get("prediction", {})
        rule_result = predict_result.get("rule_result")
        market_data = predict_result.get("market_data")
        print(f"  预测完成: 调入 {len(prediction.get('predicted_add', []))} 只, "
              f"调出 {len(prediction.get('predicted_remove', []))} 只")
        alt_count = len(prediction.get('alternative_add', []))
        if alt_count:
            print(f"  备选调入: {alt_count} 只")

    print()

    # 2. 获取实际调整数据
    print("[步骤2] 获取实际调整数据...")
    from agents.data_agent import DataAgent
    data_agent = DataAgent(config)

    actual_add, actual_remove = data_agent.index_fetcher.get_adjustment_history_by_date(
        index_name=index_name,
        target_date=target_date
    )

    print(f"  实际调入: {len(actual_add)} 只")
    print(f"  实际调出: {len(actual_remove)} 只")
    print()

    if not actual_add and not actual_remove:
        print("  警告: 未获取到实际调整数据，可能日期尚未发生调整或接口返回为空")
        print()

    # 3. 执行回测
    print("[步骤3] 对比预测与实际...")
    handler = ChatHandler(config)
    backtest_result = handler.orchestrator.backtest_agent.backtest(
        prediction=prediction,
        actual_adjustment={
            "actual_add": actual_add,
            "actual_remove": actual_remove
        },
        rule_result=rule_result,
        market_data=market_data,
        use_llm=use_llm,
    )

    # 4. 生成回测报告
    report_path = handler.orchestrator.report_agent.generate_backtest_report(
        backtest_result, index_name=index_name, target_date=target_date,
        use_llm=use_llm
    )

    # 5. 打印结果摘要
    accuracy = backtest_result.get("accuracy", {})
    print()
    print("=== 回测结果 ===")
    print(f"调入精确率: {accuracy.get('add_precision', 0):.1%}")
    print(f"调入召回率: {accuracy.get('add_recall', 0):.1%}")
    print(f"调出精确率: {accuracy.get('remove_precision', 0):.1%}")
    print(f"调出召回率: {accuracy.get('remove_recall', 0):.1%}")
    print(f"综合F1: {accuracy.get('overall_f1', 0):.1%}")
    print()

    # 打印匹配详情
    matches = backtest_result.get("matches", {})
    print("调入命中:")
    for stock in matches.get("add", []):
        print(f"  - {stock.get('stock_code')} {stock.get('stock_name')}")
    if not matches.get("add"):
        print("  无")

    print("调出命中:")
    for stock in matches.get("remove", []):
        print(f"  - {stock.get('stock_code')} {stock.get('stock_name')}")
    if not matches.get("remove"):
        print("  无")

    print()
    print(f"回测报告: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股指数成分股调整预测")
    parser.add_argument("--api", action="store_true", help="启动API服务")
    parser.add_argument("--predict", nargs=2, metavar=("DATE", "INDEX"),
                        help="直接预测, 如: --predict 2024-06 csi300")
    parser.add_argument("--no-llm", action="store_true",
                        help="仅使用量化排名，跳过LLM分析（与--predict或--backtest配合使用）")
    parser.add_argument("--backtest", nargs=2, metavar=("DATE", "INDEX"),
                        help="回测验证, 如: --backtest 2024-06 csi300")

    args = parser.parse_args()

    if args.api:
        run_api()
    elif args.predict:
        run_predict(args.predict[0], args.predict[1], use_llm=not args.no_llm)
    elif args.backtest:
        run_backtest(args.backtest[0], args.backtest[1], use_llm=not args.no_llm)
    else:
        run_interactive()
