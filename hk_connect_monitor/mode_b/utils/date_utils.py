"""
Mode B 日期处理工具

处理恒生综合指数(HSCI)定期检讨相关的日期计算。

HSCI定期检讨周期:
    - 半年度检讨: 每年两次
    - 数据截止日: 6月30日 和 12月31日
    - 生效日期: 通常在3月和9月的第二个星期一附近的交易日
    - 公告日期: 通常在生效日前1-2周

推导规则:
    - 3月调整 → 数据截止日 = 前一年12月31日
    - 9月调整 → 数据截止日 = 当年6月30日
"""

import re
from datetime import date, datetime
from typing import Tuple, Optional


def parse_adjustment_date(date_str: str) -> Tuple[int, int]:
    """
    解析调整日期字符串, 支持多种格式:

        "2026-03"     → (2026, 3)
        "2026-3"      → (2026, 3)
        "2026年3月"    → (2026, 3)
        "2026年03月"   → (2026, 3)
        "202603"      → (2026, 3)

    参数:
        date_str: 调整日期字符串

    返回:
        (year, month) 元组

    异常:
        ValueError: 无法解析的日期格式
    """
    date_str = date_str.strip()

    # "2026年3月" 或 "2026年03月"
    m = re.match(r'^(\d{4})年(\d{1,2})月?$', date_str)
    if m:
        return int(m.group(1)), int(m.group(2))

    # "2026-03" 或 "2026-3"
    m = re.match(r'^(\d{4})-(\d{1,2})$', date_str)
    if m:
        return int(m.group(1)), int(m.group(2))

    # "202603" (6位数字)
    m = re.match(r'^(\d{4})(\d{2})$', date_str)
    if m:
        return int(m.group(1)), int(m.group(2))

    raise ValueError(
        f"无法解析调整日期: {date_str}. "
        f"支持的格式: '2026-03', '2026年3月', '202603'"
    )


def get_cutoff_date(adjustment_date: str) -> date:
    """
    根据调整生效月份推导数据截止日

    规则:
        - 3月调整 → 截止日 = 前一年12月31日
        - 9月调整 → 截止日 = 当年6月30日
        - 6月调整 → 截止日 = 前一年12月31日 (快速纳入, 小批量)
        - 12月调整 → 截止日 = 当年6月30日 (快速纳入, 小批量)

    参数:
        adjustment_date: 调整日期字符串 (如 "2026-03")

    返回:
        数据截止日期
    """
    year, month = parse_adjustment_date(adjustment_date)

    if month == 3:
        # 3月定期检讨 → 前一年12月31日
        return date(year - 1, 12, 31)
    elif month == 9:
        # 9月定期检讨 → 当年6月30日
        return date(year, 6, 30)
    elif month == 6:
        # 6月快速纳入 → 前一年12月31日 (与3月检讨同一截止日)
        return date(year - 1, 12, 31)
    elif month == 12:
        # 12月快速纳入 → 当年6月30日 (与9月检讨同一截止日)
        return date(year, 6, 30)
    else:
        raise ValueError(
            f"不支持的调整月份: {month}. "
            f"HSCI调整仅在3月/6月/9月/12月发生"
        )


def get_effective_date(adjustment_date: str, history_loader=None) -> date:
    """
    获取调整的实际生效日期

    优先从历史数据中查找该月份的实际生效日期,
    如果历史数据中没有(未来调整), 则返回该月第9天作为近似值。

    参数:
        adjustment_date: 调整日期字符串 (如 "2026-03")
        history_loader: HSCIHistoryLoader实例 (可选, 用于查找实际生效日)

    返回:
        实际生效日期
    """
    year, month = parse_adjustment_date(adjustment_date)

    if history_loader is not None:
        # 从历史数据中查找该月份的调整日期
        all_dates = history_loader.get_adjustment_dates()
        for d in all_dates:
            if d.year == year and d.month == month:
                return d

    # 未找到历史记录, 使用近似日期
    # HSCI定期检讨通常在3月/9月的第9天左右生效
    if month == 3:
        return date(year, 3, 9)
    elif month == 9:
        return date(year, 9, 5)
    elif month == 6:
        return date(year, 6, 8)
    elif month == 12:
        return date(year, 12, 8)
    else:
        return date(year, month, 1)


def get_review_period(cutoff_date: date) -> Tuple[date, date]:
    """
    获取检讨考察期

    规则:
        - 截止日12月31日 → 考察期为当年1月1日至12月31日
        - 截止日6月30日  → 考察期为前一年7月1日至6月30日

    参数:
        cutoff_date: 数据截止日期

    返回:
        (考察期开始, 考察期结束)
    """
    if cutoff_date.month == 12:
        return date(cutoff_date.year, 1, 1), cutoff_date
    elif cutoff_date.month == 6:
        return date(cutoff_date.year - 1, 7, 1), cutoff_date
    else:
        raise ValueError(f"无效的截止日期: {cutoff_date}")


def format_adjustment_label(adjustment_date: str) -> str:
    """
    格式化调整日期为中文标签

    "2026-03" → "2026年3月"
    """
    year, month = parse_adjustment_date(adjustment_date)
    return f"{year}年{month}月"
