"""特征构建器"""

import pandas as pd
from typing import List, Dict


class FeatureBuilder:
    """
    特征构建器
    
    构建用于预测的特征:
        - 市值排名
        - 流动性排名
        - 综合排名
        - 上市时长
        - 行业分类
    """

    def build_features(self, market_data: List[dict]) -> List[dict]:
        """构建预测特征"""
        df = pd.DataFrame(market_data)

        # 计算市值排名
        df["market_cap_rank"] = df["total_mv"].rank(ascending=False)

        # 计算流动性排名
        df["liquidity_rank"] = df["amount"].rank(ascending=False)

        # 计算综合排名 (市值70% + 流动性30%)
        df["composite_rank"] = (
            df["market_cap_rank"] * 0.7 + df["liquidity_rank"] * 0.3
        )
        df["composite_rank"] = df["composite_rank"].rank()

        return df.to_dict("records")
