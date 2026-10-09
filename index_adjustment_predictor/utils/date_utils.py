"""
Date Utils - 日期处理工具

处理指数调整相关的日期计算
"""

from datetime import datetime, timedelta
from typing import Tuple


def get_cutoff_date(target_adjustment_date: str) -> str:
    """
    根据调整公告日返回数据截止日

    规则:
        - 6月调整  -> 截止日为4月30日
        - 12月调整 -> 截止日为10月31日

    参数:
        target_adjustment_date: 目标调整月份 (如 "2024-06")

    返回:
        数据截止日 (如 "2024-04-30")
    """
    year, month = _parse_year_month(target_adjustment_date)

    if month == 6:
        return f"{year}-04-30"
    elif month == 12:
        return f"{year}-10-31"
    else:
        raise ValueError(
            f"不支持的调整月份: {month}, 仅支持6月或12月"
        )


def get_announcement_date(target_adjustment_date: str) -> str:
    """
    获取指数调整公告发布日期

    规则:
        - 6月调整  -> 公告日约为6月第1-2周
        - 12月调整 -> 公告日约为12月第1-2周

    参数:
        target_adjustment_date: 目标调整月份

    返回:
        公告日期
    """
    year, month = _parse_year_month(target_adjustment_date)

    if month == 6:
        return f"{year}-06-10"
    elif month == 12:
        return f"{year}-12-10"
    else:
        raise ValueError(
            f"不支持的调整月份: {month}, 仅支持6月或12月"
        )


def get_effective_date(target_adjustment_date: str) -> str:
    """
    获取调整生效日期

    规则:
        - 6月调整  -> 生效日为6月第2个星期五后的下一交易日
        - 12月调整 -> 生效日为12月第2个星期五后的下一交易日
    """
    year, month = _parse_year_month(target_adjustment_date)

    if month == 6:
        return f"{year}-06-17"
    elif month == 12:
        return f"{year}-12-17"
    else:
        raise ValueError(
            f"不支持的调整月份: {month}, 仅支持6月或12月"
        )


def get_review_period(cutoff_date: str) -> Tuple[str, str]:
    """
    获取考察期

    规则:
        - 6月调整  -> 考察期为前一年5月1日至4月30日
        - 12月调整 -> 考察期为前一年11月1日至10月31日
    """
    dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
    year = dt.year

    if dt.month == 4:
        start = f"{year-1}-05-01"
        end = cutoff_date
    elif dt.month == 10:
        start = f"{year-1}-11-01"
        end = cutoff_date
    else:
        raise ValueError(f"无效的截止日期: {cutoff_date}")

    return (start, end)


def _parse_year_month(date_str: str) -> Tuple[int, int]:
    """解析 'YYYY-MM' 格式的日期"""
    parts = date_str.split("-")
    return int(parts[0]), int(parts[1])
