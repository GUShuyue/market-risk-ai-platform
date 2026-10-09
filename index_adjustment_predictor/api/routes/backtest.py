"""回测接口路由"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from core.orchestrator import Orchestrator

router = APIRouter()


class BacktestRequest(BaseModel):
    target_date: str        # 预测目标时间 (如 "2024-06")
    index_name: str
    use_llm: bool = True    # 是否回测LLM预测结果（False则回测纯量化结果）


class BacktestResponse(BaseModel):
    accuracy: dict
    matches: dict
    misses: dict
    false_positives: dict
    deviation_analysis: dict
    report_path: str


@router.post("/", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, http_request: Request):
    """执行回测验证"""
    import os, glob, json
    from utils.date_utils import get_cutoff_date

    config = http_request.app.state.config
    orchestrator = Orchestrator(config=config)

    target_date = request.target_date
    index_name = request.index_name
    use_llm = request.use_llm

    # 1. 加载已有预测结果（含rule_result和market_data）
    report_dir = "reports"
    suffix = "" if use_llm else "_no_llm"
    pattern = os.path.join(
        report_dir, f"prediction_{index_name}_{target_date}{suffix}.json"
    )
    files = glob.glob(pattern)
    if not files:
        return {
            "accuracy": {}, "matches": {}, "misses": {},
            "false_positives": {}, "deviation_analysis": {},
            "report_path": f"未找到 {target_date} {index_name} 的预测结果"
        }

    files.sort(key=os.path.getmtime, reverse=True)
    with open(files[0], "r", encoding="utf-8") as f:
        prediction_data = json.load(f)

    # 2. 获取实际调整数据
    actual_add, actual_remove = \
        orchestrator.data_agent.index_fetcher.get_adjustment_history_by_date(
            index_name=index_name, target_date=target_date
        )

    # 3. 执行回测
    result = orchestrator.run_workflow(
        "backtest",
        {
            "prediction": prediction_data.get("prediction", {}),
            "rule_result": prediction_data.get("rule_result"),
            "market_data": prediction_data.get("market_data"),
            "actual_adjustment": {
                "actual_add": actual_add,
                "actual_remove": actual_remove
            },
            "use_llm": use_llm,
        }
    )

    backtest = result.get("backtest_result", {})
    return {
        "accuracy": backtest.get("accuracy", {}),
        "matches": backtest.get("matches", {}),
        "misses": backtest.get("misses", {}),
        "false_positives": backtest.get("false_positives", {}),
        "deviation_analysis": backtest.get("deviation_analysis", {}),
        "report_path": result.get("report_path", "")
    }
