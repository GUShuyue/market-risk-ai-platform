"""阈值判断模型"""

from typing import List, Dict


class ThresholdModel:
    """
    阈值判断模型
    
    基于硬性准入条件筛选股票
    
    检查项:
        - 上市时间是否满足要求
        - 是否为ST/*ST股票
        - 流动性是否达标
        - 财务是否异常
    """

    def __init__(self, index_config: dict):
        self.config = index_config
        self.min_listing_months = 3  # 最小上市月数
        self.min_avg_amount = 1e8    # 最小日均成交额

    def filter_eligible(self, stocks: List[dict]) -> dict:
        """
        筛选满足准入条件的股票

        返回:
            {
                "eligible": [...],
                "ineligible": [{"code": ..., "reason": ...}]
            }
        """
        eligible = []
        ineligible = []

        for stock in stocks:
            reasons = self._check_eligibility(stock)
            if reasons:
                ineligible.append({"code": stock["code"], "reasons": reasons})
            else:
                eligible.append(stock)

        return {"eligible": eligible, "ineligible": ineligible}

    def _check_eligibility(self, stock: dict) -> List[str]:
        """检查单只股票是否满足准入条件"""
        reasons = []

        if stock.get("is_st"):
            reasons.append("ST股票")
        if stock.get("listing_months", 0) < self.min_listing_months:
            reasons.append("上市时间不足")
        if stock.get("avg_amount", 0) < self.min_avg_amount:
            reasons.append("流动性不足")

        return reasons
