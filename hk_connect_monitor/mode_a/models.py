"""
港股通标的池监控系统 - 数据模型
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional, Dict, Any


class IndexType(Enum):
    """恒生综合指数成份股类型"""
    HSCI_LARGE = "恒生综合大型股指数"
    HSCI_MID = "恒生综合中型股指数"
    HSCI_SMALL = "恒生综合小型股指数"
    NONE = "非指数成份股"


class RiskType(Enum):
    """风险类型"""
    REMOVAL_RISK = "调出风险"
    INCLUSION_CANDIDATE = "调入候选"
    WVR_PENDING = "WVR待纳入"
    EXCLUSION_EVENT = "排除事件"
    GREY_ZONE = "灰色区间"
    SAFE = "安全"


class RiskLevel(Enum):
    """风险等级"""
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"
    NONE = "无"


@dataclass
class StockInfo:
    """股票基础信息"""
    code: str                          # 股票代码，如 "00700"
    name: str                          # 股票名称
    currency: str                      # 报价货币，如 "HKD", "USD"
    listing_date: date                 # 上市日期
    is_wvr: bool = False               # 是否为不同投票权架构(WVR)公司
    is_secondary_listing: bool = False  # 是否为第二上市（-S/-SW后缀，不可纳入港股通）
    is_a_plus_h: bool = False          # 是否为A+H股上市公司
    a_share_code: Optional[str] = None # 对应A股代码
    a_share_st: bool = False           # A股是否为风险警示(ST/*ST)
    a_share_delisting: bool = False    # A股是否处于退市整理期
    index_type: IndexType = IndexType.NONE  # 指数成份股类型
    in_pool: bool = False              # 是否在港股通标的池中


@dataclass
class MonthlyMarketCap:
    """月末市值数据"""
    date: date          # 月末日期
    market_cap: float   # 市值（港币元）


@dataclass
class DailyMarketData:
    """日度行情数据"""
    date: date                  # 日期
    market_cap: float           # 市值（港币元）
    trading_value: float        # 成交额（港币元）


@dataclass
class AssessmentResult:
    """单只股票的评估结果"""
    code: str
    name: str
    index_type: IndexType
    in_pool: bool
    risk_type: RiskType
    risk_level: RiskLevel
    details: str                              # 人类可读的详情说明
    metrics: Dict[str, Any] = field(default_factory=dict)  # 关键量化指标

    @property
    def sort_key(self) -> int:
        """用于排序的优先级（高风险在前）"""
        level_order = {
            RiskLevel.HIGH: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.LOW: 2,
            RiskLevel.NONE: 3,
        }
        return level_order.get(self.risk_level, 99)


def format_hkd(amount: float) -> str:
    """将港币金额格式化为中文习惯表示"""
    if amount >= 100_000_000:
        return f"{amount / 100_000_000:.1f}亿"
    elif amount >= 10_000:
        return f"{amount / 10_000:.1f}万"
    else:
        return f"{amount:.0f}"
