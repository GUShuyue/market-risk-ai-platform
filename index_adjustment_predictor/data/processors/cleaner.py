"""数据清洗器"""

import pandas as pd
from typing import List, Dict


class DataCleaner:
    """
    数据清洗器
    
    功能:
        - 去除重复数据
        - 处理缺失值
        - 数据类型转换
        - 异常值处理
    """

    def clean_market_data(self, raw_data: List[dict]) -> List[dict]:
        """清洗市场数据"""
        df = pd.DataFrame(raw_data)
        df = df.drop_duplicates(subset=["ts_code"])
        df = df.dropna(subset=["total_mv", "amount"])
        df["total_mv"] = pd.to_numeric(df["total_mv"], errors="coerce")
        return df.to_dict("records")

    def clean_constituent_data(self, raw_data: List[dict]) -> List[dict]:
        """清洗成分股数据"""
        df = pd.DataFrame(raw_data)
        df = df.drop_duplicates(subset=["con_code"])
        return df.to_dict("records")
