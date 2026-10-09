"""数据校验工具"""

from typing import List, Dict
from datetime import datetime


class Validator:
    """
    数据校验器
    
    确保数据完整性和时间点一致性
    """

    @staticmethod
    def validate_no_future_data(data: dict, cutoff_date: str) -> bool:
        """
        验证数据中没有超过截止日期的未来数据

        检查数据字典中的日期字段是否都 <= cutoff_date

        参数:
            data: 待验证数据 (DataAgent采集的结果)
            cutoff_date: 数据截止日 (格式: YYYY-MM-DD)

        返回:
            是否通过验证
        """
        cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")

        # 检查数据中常见的日期字段
        date_fields = ["trade_date", "ann_date", "list_date", "suspend_date"]

        def check_date_value(value):
            """检查单个日期值是否超过截止日"""
            if not value or not isinstance(value, str):
                return True
            for fmt in ["%Y-%m-%d", "%Y%m%d"]:
                try:
                    dt = datetime.strptime(value, fmt)
                    return dt <= cutoff
                except ValueError:
                    continue
            return True  # 无法解析的日期，放行

        def scan_dict(obj):
            """递归扫描字典/列表中的日期字段"""
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if key in date_fields and isinstance(val, str):
                        if not check_date_value(val):
                            return False
                    if isinstance(val, (dict, list)):
                        if not scan_dict(val):
                            return False
            elif isinstance(obj, list):
                for item in obj:
                    if not scan_dict(item):
                        return False
            return True

        return scan_dict(data)

    @staticmethod
    def validate_constituent_count(
        constituents: List[dict], expected_count: int
    ) -> bool:
        """验证成分股数量是否正确"""
        return len(constituents) == expected_count

    @staticmethod
    def validate_rankings(rankings: List[dict]) -> bool:
        """验证排名数据完整性"""
        if not rankings:
            return False
        required_fields = ["code", "rank", "total_mv"]
        return all(
            all(f in r for f in required_fields) for r in rankings
        )
