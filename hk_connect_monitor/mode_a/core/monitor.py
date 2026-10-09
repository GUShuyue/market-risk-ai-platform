"""
港股通标的池监控系统 - 监控协调器

负责协调数据提供器和规则引擎，执行全市场扫描，
收集评估结果并提供汇总统计。
"""

from datetime import date
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from mode_a.data.data_provider import DataProvider
from mode_a.core.rules import RuleEngine
from mode_a.models import AssessmentResult, RiskLevel, RiskType, IndexType
from mode_a import config


class ScanSummary:
    """扫描结果汇总统计"""

    def __init__(self, scan_date: date, total_stocks: int, pool_size: int):
        self.scan_date = scan_date
        self.total_stocks = total_stocks
        self.pool_size = pool_size
        self.results: List[AssessmentResult] = []

    @property
    def high_risk_count(self) -> int:
        return sum(1 for r in self.results if r.risk_level == RiskLevel.HIGH)

    @property
    def medium_risk_count(self) -> int:
        return sum(1 for r in self.results if r.risk_level == RiskLevel.MEDIUM)

    @property
    def low_risk_count(self) -> int:
        return sum(1 for r in self.results if r.risk_level == RiskLevel.LOW)

    @property
    def safe_count(self) -> int:
        return sum(1 for r in self.results if r.risk_level == RiskLevel.NONE)

    @property
    def risk_list(self) -> List[AssessmentResult]:
        """返回有风险的股票（按风险等级排序）"""
        return sorted(
            [r for r in self.results if r.risk_level != RiskLevel.NONE],
            key=lambda r: r.sort_key,
        )

    @property
    def safe_list(self) -> List[AssessmentResult]:
        """返回安全的股票"""
        return [r for r in self.results if r.risk_level == RiskLevel.NONE]

    def count_by_risk_type(self) -> Dict[RiskType, int]:
        """按风险类型统计"""
        counts: Dict[RiskType, int] = {}
        for r in self.risk_list:
            counts[r.risk_type] = counts.get(r.risk_type, 0) + 1
        return counts


class HkConnectMonitor:
    """港股通标的池监控协调器"""

    def __init__(
        self,
        data_provider: DataProvider,
        rule_engine: Optional[RuleEngine] = None,
        reference_date: Optional[date] = None,
        max_workers: int = 8,
    ):
        self.data_provider = data_provider
        self.reference_date = reference_date or date.today()
        self.rule_engine = rule_engine or RuleEngine(self.reference_date)
        # 并发扫描线程数。设为 1 时退回串行（原始行为）。
        self.max_workers = max_workers

    def scan(self) -> ScanSummary:
        """
        执行全市场扫描。

        流程：
        1. 获取所有待扫描股票
        2. 对每只股票获取市场数据
        3. 运行规则引擎评估
        4. 返回汇总结果

        数据获取与规则评估均为"每只股票相互独立"的纯计算 + 网络IO，
        因此当 max_workers > 1 时使用线程池并发，重叠网络等待时间。
        """
        stocks = self.data_provider.get_all_stocks()
        current_pool = self.data_provider.get_current_pool()

        summary = ScanSummary(
            scan_date=self.reference_date,
            total_stocks=len(stocks),
            pool_size=len(current_pool),
        )

        total = len(stocks)

        if self.max_workers and self.max_workers > 1 and total > 1:
            self._scan_parallel(stocks, summary, total)
        else:
            self._scan_serial(stocks, summary, total)

        return summary

    def _assess_one(self, stock) -> AssessmentResult:
        """
        对单只股票执行完整评估（原 scan 循环体）。

        每只股票是独立对象，各线程处理不同 stock，互不共享可变状态，
        因此本方法可在多线程中安全并发执行。
        """
        # HSCI分类已在初始化时从CSV加载，此处为兜底
        if stock.index_type == IndexType.NONE:
            stock.index_type = self.data_provider.classify_index_type(stock.code)

        # WVR不在池时按需获取真实上市日期（初始化时为默认值）
        if stock.is_wvr and not stock.in_pool:
            stock.listing_date = self.data_provider.get_listing_date(stock.code)

        # 根据股票类型获取不同粒度的数据
        # 小型股需要月度市值数据（用于市值门槛检查）
        monthly_data = []
        if stock.index_type == IndexType.HSCI_SMALL:
            monthly_data = self.data_provider.get_monthly_market_caps(
                stock.code, config.SMALL_CAP_ASSESSMENT_MONTHS
            )

        # WVR不在池时需要日度行情数据（用于183日市值和成交额检查）
        daily_data = []
        if stock.is_wvr and not stock.in_pool:
            daily_data = self.data_provider.get_daily_market_data(
                stock.code, config.WVR_ASSESSMENT_DAYS
            )

        return self.rule_engine.assess_stock(stock, monthly_data, daily_data)

    def _scan_serial(self, stocks, summary: ScanSummary, total: int):
        """串行扫描（原始行为）"""
        for idx, stock in enumerate(stocks, 1):
            if idx % 50 == 0 or idx == total:
                print(f"  [扫描进度] {idx}/{total} ({idx*100//total}%)")
            summary.results.append(self._assess_one(stock))

    def _scan_parallel(self, stocks, summary: ScanSummary, total: int):
        """并发扫描（线程池）"""
        # 预分配结果槽位，按原顺序回填，保证 results 顺序与输入一致
        results: List[Optional[AssessmentResult]] = [None] * total
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._assess_one, stock): idx
                for idx, stock in enumerate(stocks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                if completed % 50 == 0 or completed == total:
                    print(f"  [扫描进度] {completed}/{total} ({completed*100//total}%)")

        summary.results = [r for r in results if r is not None]
