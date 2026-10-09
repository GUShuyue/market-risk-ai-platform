"""
Data Agent - 数据采集Agent

职责：在指定时间点采集所需数据，严格遵守"不看未来数据"原则
确保所有采集的数据均不超过 cutoff_date

采集两类数据：
1. 量化数据：市值、流动性、上市时间（供 rank_model 和 threshold_model 使用）
2. 定性数据：行业分类、停牌状态、重大公告、财务异常（供 qualitative_model 使用）
"""

from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from utils.date_utils import get_review_period


class DataAgent:
    """
    数据采集Agent
    
    核心原则：时间点数据隔离，防止未来函数
    
    根据 config 中的 data_source.provider 自动选择数据源:
        - akshare: 使用AKShare版fetcher (免费，无需API Key)
        - tushare: 使用Tushare版fetcher (付费，需API Key)
    
    参数:
        target_date: 预测目标时间点 (如 "2024-06")
        cutoff_date: 数据截止日 (如 "2024-04-30")
    """

    def __init__(self, config: dict):
        self.config = config
        data_source_config = config.get("data_source", config)
        provider = data_source_config.get("provider", "akshare")

        if provider == "akshare":
            from data.scripts.akshare.market_data_ak import MarketDataFetcherAK
            from data.scripts.akshare.index_constituent_ak import IndexConstituentFetcherAK
            from data.scripts.akshare.financial_data_ak import FinancialDataFetcherAK
            from data.scripts.akshare.market_events_ak import MarketEventsFetcherAK

            self.market_fetcher = MarketDataFetcherAK(data_source_config)
            self.index_fetcher = IndexConstituentFetcherAK(data_source_config)
            self.financial_fetcher = FinancialDataFetcherAK(data_source_config)
            self.events_fetcher = MarketEventsFetcherAK(data_source_config)

        elif provider == "tushare":
            from data.scripts.market_data import MarketDataFetcher
            from data.scripts.index_constituent import IndexConstituentFetcher
            from data.scripts.financial_data import FinancialDataFetcher
            from data.scripts.market_events import MarketEventsFetcher

            self.market_fetcher = MarketDataFetcher(data_source_config)
            self.index_fetcher = IndexConstituentFetcher(data_source_config)
            self.financial_fetcher = FinancialDataFetcher(data_source_config)
            self.events_fetcher = MarketEventsFetcher(data_source_config)

        else:
            raise ValueError(f"不支持的数据源: {provider}")

    def run(self, inputs: dict) -> dict:
        """
        统一接口：执行数据采集

        inputs 必含: target_date, cutoff_date, index_name
        返回: collect 的完整数据字典
        """
        return self.collect(
            target_date=inputs["target_date"],
            cutoff_date=inputs["cutoff_date"],
            index_name=inputs["index_name"]
        )

    def collect(self, target_date: str, cutoff_date: str, index_name: str) -> dict:
        """
        采集指定时间点所需的所有数据（量化 + 定性）

        参数:
            target_date: 预测目标调整时间 (如 "2024-06")
            cutoff_date: 数据截止日 (如 "2024-04-30")
            index_name: 指数名称 (csi300)

        返回:
            包含所有采集数据的字典
        """
        # 验证时间点合理性
        self._validate_dates(target_date, cutoff_date)

        # 计算考察期
        review_start, review_end = get_review_period(cutoff_date)

        # 阶段A: 三个相互独立的数据源并发采集
        # 1. 指数成分股  2. 全市场行情  3. 上市信息
        with ThreadPoolExecutor(max_workers=3) as ex:
            fut_constituents = ex.submit(
                self.index_fetcher.get_constituents,
                index_name=index_name, date=cutoff_date
            )
            fut_market = ex.submit(
                self.market_fetcher.fetch_by_date,
                date=cutoff_date, stocks="all"
            )
            fut_listing = ex.submit(
                self.market_fetcher.fetch_listing_info,
                cutoff_date=cutoff_date
            )
            current_constituents = fut_constituents.result()
            market_data = fut_market.result()
            listing_info = fut_listing.result()

        # 4. 提取关注的股票列表（成分股 + 排名靠前的候选股）
        stock_codes = self._get_focus_stocks(market_data, current_constituents, index_name)

        # 阶段B: 依赖 stock_codes 的四个采集任务并发
        # 5. 财务数据  6. 事件数据  7. 价格变动  8. 行业分布
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut_financial = ex.submit(
                self.financial_fetcher.fetch_financial_flags,
                cutoff_date=cutoff_date, stock_codes=stock_codes
            )
            fut_events = ex.submit(
                self.events_fetcher.fetch_qualitative_data,
                cutoff_date=cutoff_date, review_start=review_start,
                stock_codes=stock_codes
            )
            fut_price = ex.submit(
                self.market_fetcher.fetch_price_changes,
                stock_codes=stock_codes, end_date=cutoff_date, days=20
            )
            fut_industry = ex.submit(
                self.market_fetcher.fetch_industry_distribution,
                cutoff_date=cutoff_date
            )
            financial_flags = fut_financial.result()
            events_data = fut_events.result()
            price_changes = fut_price.result()
            industry_distribution = fut_industry.result()

        return {
            "target_date": target_date,
            "cutoff_date": cutoff_date,
            "review_start": review_start,
            "index_name": index_name,
            # 量化数据
            "current_constituents": current_constituents,
            "market_data": market_data,
            "listing_info": listing_info,
            "financial_flags": financial_flags,
            # 定性数据
            "events_data": events_data,
            "price_changes": price_changes,
            "industry_distribution": industry_distribution,
            "collected_at": datetime.now().isoformat()
        }

    def _get_focus_stocks(
        self, market_data: list, current_constituents: list, index_name: str
    ) -> list:
        """
        提取需要关注的股票列表（成分股 + 候选股）
        用于缩小定性数据的查询范围，避免全市场查询太慢

        沪深300: 市值前600 (缓冲区360 + 余量)
        """
        codes = set()

        # 当前成分股
        for c in current_constituents:
            code = c.get("con_code") or c.get("ts_code", "")
            if code:
                codes.add(code)

        # 市值排名前600的候选股
        sorted_stocks = sorted(
            market_data,
            key=lambda x: x.get("total_mv", 0),
            reverse=True
        )
        for stock in sorted_stocks[:600]:
            code = stock.get("ts_code", "")
            if code:
                codes.add(code)

        return list(codes)

    def _validate_dates(self, target_date: str, cutoff_date: str):
        """验证时间点合理性，确保cutoff_date早于target_date"""
        target = datetime.strptime(target_date, "%Y-%m")
        cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")
        if cutoff >= target:
            raise ValueError(
                f"数据截止日 {cutoff_date} 必须早于目标调整时间 {target_date}"
            )
