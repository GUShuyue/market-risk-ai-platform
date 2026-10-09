"""预测接口路由"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
from core.orchestrator import Orchestrator

router = APIRouter()


class PredictRequest(BaseModel):
    target_date: str        # 如 "2024-06"
    index_name: str         # csi300
    cutoff_date: Optional[str] = None
    use_llm: bool = True    # 是否启用LLM增强分析，False时仅使用量化排名


class PredictResponse(BaseModel):
    predicted_add: list
    alternative_add: list = []
    predicted_remove: list
    confidence: float
    reasoning: str
    report_path: str


@router.post("/", response_model=PredictResponse)
async def predict_adjustment(request: PredictRequest, http_request: Request):
    """预测指数成分股调整"""
    config = http_request.app.state.config
    orchestrator = Orchestrator(config=config)
    result = orchestrator.run_workflow(
        "predict_adjustment",
        {
            "target_date": request.target_date,
            "index_name": request.index_name,
            "cutoff_date": request.cutoff_date,
            "use_llm": request.use_llm
        }
    )
    return result
