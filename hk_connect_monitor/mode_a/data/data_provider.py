"""
港股通标的池监控系统 - 数据提供器抽象接口

定义统一的数据访问接口。接入真实数据源时，只需实现 DataProvider 接口即可。
"""

from datetime import date
from typing import List, Set

from mode_a.models import StockInfo, MonthlyMarketCap, DailyMarketData, IndexType


class DataProvider:
    """数据提供器抽象接口"""

    def get_all_stocks(self) -> List[StockInfo]:
        """获取所有待扫描的股票"""
        raise NotImplementedError

    def get_current_pool(self) -> Set[str]:
        """获取当前港股通标的池（股票代码集合）"""
        raise NotImplementedError

    def get_monthly_market_caps(self, code: str, months: int = 12) -> List[MonthlyMarketCap]:
        """获取指定股票的月末市值数据"""
        raise NotImplementedError

    def get_daily_market_data(self, code: str, days: int = 183) -> List[DailyMarketData]:
        """获取指定股票的日度行情数据"""
        raise NotImplementedError

    def classify_index_type(self, code: str) -> IndexType:
        """按需分类HSCI指数类型（默认NONE，子类可覆盖）"""
        return IndexType.NONE

    def get_listing_date(self, code: str) -> date:
        """获取上市日期（默认返回2020-01-01，子类可覆盖）"""
        return date(2020, 1, 1)
