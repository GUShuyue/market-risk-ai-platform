"""
Market Data Fetcher (AKShare) - 市场数据抓取脚本

使用AKShare获取股票市值、流动性、上市时间、行业分类等市场数据。
AKShare是免费开源库，无需API Key。

缓存策略:
    - 首次运行: 逐只从AKShare抓取，存入SQLite缓存
    - 后续运行: 直接从缓存读取，秒级返回
    - 历史数据不会变化，缓存永久有效
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import sys
import os

# 添加项目根目录到path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from data.storage.cache_manager import CacheManager


class MarketDataFetcherAK:
    """
    AKShare版市场数据抓取器（带持久化缓存）
    
    抓取内容:
        - 总市值 / 流通市值
        - 日均成交额 (近1年)
        - 上市日期 / 上市时长 / 行业分类
        - ST状态
        - 近20日涨跌幅
    
    缓存逻辑:
        - fetch_by_date(): 先查缓存，没有则抓取后存入
        - fetch_listing_info(): 先查缓存，没有则抓取后存入
        - fetch_price_changes(): 先查缓存，没有则抓取后存入
        - calculate_avg_amount(): 先查缓存，没有则计算后存入
    """

    def __init__(self, data_source_config: dict = None):
        self.config = data_source_config or {}
        self.cache = CacheManager(
            self.config.get("cache_db", "data/storage/database/cache.db")
        )
        self._stock_info_cache = None
        self._total_share_cache = None

    def fetch_by_date(
        self, date: str, stocks: str = "all"
    ) -> List[dict]:
        """
        获取指定日期的全市场股票数据

        缓存策略:
            1. 先查缓存，命中则直接返回
            2. 未命中则从AKShare抓取
            3. 抓取后存入缓存

        参数:
            date: 日期 (格式: YYYY-MM-DD)
            stocks: "all" 或股票代码列表

        返回:
            股票数据列表
        """
        # 1. 检查缓存
        if self.cache.has_market_data(date):
            print(f"  [缓存] 读取 {date} 市场数据...")
            data = self.cache.get_market_data(date)
            # 补充日均成交额
            avg_amounts = self.cache.get_avg_amount_batch(date)
            for row in data:
                if row.get("avg_amount", 0) == 0:
                    row["avg_amount"] = avg_amounts.get(row["ts_code"], 0)
            return data

        # 2. 缓存未命中，从AKShare抓取
        date_dt = datetime.strptime(date, "%Y-%m-%d")
        today = datetime.now()
        days_diff = (today - date_dt).days

        if days_diff <= 1:
            data = self._fetch_realtime(date)
        else:
            data = self._fetch_historical(date, stocks)

        # 3. 存入缓存
        print(f"  [缓存] 保存 {date} 市场数据 ({len(data)}条)...")
        self.cache.save_market_data(date, data)

        return data

    def _fetch_realtime(self, date: str) -> List[dict]:
        """获取实时全市场数据"""
        df = ak.stock_zh_a_spot_em()

        result = []
        for _, row in df.iterrows():
            code = str(row.get("代码", ""))
            if not code:
                continue

            ts_code = self._format_stock_code(code)

            result.append({
                "ts_code": ts_code,
                "name": row.get("名称", ""),
                "total_mv": float(row.get("总市值", 0) or 0),
                "circ_mv": float(row.get("流通市值", 0) or 0),
                "close": float(row.get("最新价", 0) or 0),
                "pct_chg": float(row.get("涨跌幅", 0) or 0),
                "amount": float(row.get("成交额", 0) or 0),
                "vol": float(row.get("成交量", 0) or 0),
                "pe": float(row.get("市盈率-动态", 0) or 0),
                "pb": float(row.get("市净率", 0) or 0),
                "turnover_rate": float(row.get("换手率", 0) or 0),
                "avg_amount": 0,
            })

        return result

    def _fetch_historical(self, date: str, stocks: str = "all") -> List[dict]:
        """获取历史某天的全市场数据（逐只查询，首次运行较慢）"""
        stock_list = self._get_all_stock_codes()

        if stocks != "all" and isinstance(stocks, list):
            stock_list = [s for s in stock_list if s in stocks]

        result = []
        total = len(stock_list)

        # 检查哪些股票已经有缓存的日均成交额
        cached_avg = self.cache.get_avg_amount_batch(date)
        missing_avg = [s for s in stock_list if s not in cached_avg]

        for i, code in enumerate(stock_list):
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{total}")

            stock_data = self._fetch_single_stock_history(code, date)

            # 如果有缓存的日均成交额，直接用
            if code in cached_avg:
                stock_data["avg_amount"] = cached_avg[code]

            if stock_data:
                result.append(stock_data)

            time.sleep(0.05)

        return result

    def _fetch_single_stock_history(self, code: str, date: str) -> Optional[dict]:
        """获取单只股票在指定日期的数据"""
        ak_code = code.split(".")[0]
        date_str = date.replace("-", "")

        try:
            df = ak.stock_zh_a_hist(
                symbol=ak_code,
                period="daily",
                start_date=date_str,
                end_date=date_str,
                adjust=""
            )
            if df is None or df.empty:
                return None

            row = df.iloc[0]

            total_shares = self._get_total_shares(code)
            circ_shares = self._get_circ_shares(code)

            close = float(row.get("收盘", 0) or 0)
            total_mv = close * total_shares if total_shares else 0
            circ_mv = close * circ_shares if circ_shares else 0

            name = self._get_stock_name(code)

            return {
                "ts_code": code,
                "name": name,
                "total_mv": total_mv,
                "circ_mv": circ_mv,
                "close": close,
                "pct_chg": float(row.get("涨跌幅", 0) or 0),
                "amount": float(row.get("成交额", 0) or 0),
                "vol": float(row.get("成交量", 0) or 0),
                "avg_amount": 0,
            }
        except Exception:
            return None

    def fetch_listing_info(self, cutoff_date: str) -> List[dict]:
        """
        获取股票上市信息（含行业分类）

        缓存策略:
            1. 先查缓存
            2. 未命中则逐只从AKShare抓取，存入缓存
        """
        # 1. 检查缓存
        if self.cache.has_listing_info():
            print(f"  [缓存] 读取上市信息...")
            return self.cache.get_listing_info(cutoff_date)

        # 2. 从AKShare抓取
        print(f"  [抓取] 首次获取上市信息，可能需要几分钟...")
        cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")

        code_name_df = ak.stock_info_a_code_name()

        result = []
        total = len(code_name_df)
        for i, (_, row) in enumerate(code_name_df.iterrows()):
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{total}")

            code = str(row.get("code", ""))
            name = str(row.get("name", ""))
            if not code:
                continue

            ts_code = self._format_stock_code(code)
            info = self._get_individual_info(code)

            list_date_str = info.get("list_date", "")
            listing_months = 0
            if list_date_str:
                try:
                    list_dt = datetime.strptime(list_date_str, "%Y%m%d")
                    if list_dt > cutoff_dt:
                        continue
                    listing_months = (cutoff_dt.year - list_dt.year) * 12 + (cutoff_dt.month - list_dt.month)
                except ValueError:
                    pass

            result.append({
                "ts_code": ts_code,
                "name": name,
                "list_date": list_date_str,
                "listing_months": listing_months,
                "is_st": "ST" in name or "*ST" in name,
                "industry": info.get("industry", ""),
                "exchange": info.get("exchange", ""),
            })

            time.sleep(0.05)

        # 3. 存入缓存
        print(f"  [缓存] 保存上市信息 ({len(result)}条)...")
        self.cache.save_listing_info(result)

        return result

    def fetch_industry_distribution(self, cutoff_date: str) -> Dict[str, int]:
        """获取全市场行业分布统计"""
        listing_info = self.fetch_listing_info(cutoff_date)
        distribution = {}
        for stock in listing_info:
            industry = stock.get("industry", "未知")
            distribution[industry] = distribution.get(industry, 0) + 1
        return distribution

    def fetch_price_changes(
        self, stock_codes: List[str], end_date: str, days: int = 20
    ) -> Dict[str, dict]:
        """
        获取股票近N日涨跌幅

        缓存策略:
            1. 先查缓存
            2. 未命中的股票才从AKShare抓取
        """
        # 1. 检查缓存
        cached = self.cache.get_price_changes(end_date)
        cached_codes = set(cached.keys())

        # 分离已缓存和未缓存
        need_fetch = [c for c in stock_codes if c not in cached_codes]
        result = {k: v for k, v in cached.items() if k in stock_codes}

        if not need_fetch:
            print(f"  [缓存] 全部 {len(stock_codes)} 只股票价格变动已缓存")
            return result

        # 2. 抓取未缓存的
        print(f"  [抓取] {len(need_fetch)} 只股票需从AKShare获取，已缓存 {len(result)} 只")

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=days * 2)
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        for i, code in enumerate(need_fetch):
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(need_fetch)}")

            ak_code = code.split(".")[0]
            try:
                df = ak.stock_zh_a_hist(
                    symbol=ak_code,
                    period="daily",
                    start_date=start_str,
                    end_date=end_str,
                    adjust=""
                )
                if df is None or df.empty:
                    continue

                df = df.tail(days)
                close_prices = df["收盘"].values
                pct_changes = df["涨跌幅"].values

                if len(close_prices) < 2:
                    continue

                pct_change_total = (close_prices[-1] / close_prices[0] - 1) * 100
                max_daily = float(pct_changes.max())
                min_daily = float(pct_changes.min())
                volatility = float(pct_changes.std())

                is_abnormal = (
                    abs(pct_change_total) > 50
                    or abs(max_daily) > 9.5
                    or abs(min_daily) > 9.5
                )

                result[code] = {
                    "pct_change_20d": round(pct_change_total, 2),
                    "max_daily_pct": round(max_daily, 2),
                    "min_daily_pct": round(min_daily, 2),
                    "volatility": round(volatility, 2),
                    "is_abnormal": is_abnormal
                }
            except Exception:
                continue

            time.sleep(0.05)

        # 3. 存入缓存
        new_data = {k: v for k, v in result.items() if k in need_fetch}
        if new_data:
            self.cache.save_price_changes(end_date, new_data)

        return result

    def calculate_avg_amount(self, stock_code: str, end_date: str, days: int = 250) -> float:
        """
        计算单只股票近一年日均成交额

        缓存策略: 先查缓存，没有则计算后存入
        """
        # 1. 检查缓存
        if self.cache.has_avg_amount(stock_code, end_date):
            return self.cache.get_avg_amount(stock_code, end_date)

        # 2. 计算
        ak_code = stock_code.split(".")[0]
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=days * 2)

        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        try:
            df = ak.stock_zh_a_hist(
                symbol=ak_code,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust=""
            )
            if df is None or df.empty:
                return 0.0

            df = df.tail(days)
            avg = float(df["成交额"].mean())

            # 3. 存入缓存
            self.cache.save_avg_amount(stock_code, end_date, avg)
            return avg
        except Exception:
            return 0.0

    def batch_calculate_avg_amount(
        self, stock_codes: List[str], end_date: str, days: int = 250
    ) -> Dict[str, float]:
        """
        批量计算日均成交额（带缓存）

        只对未缓存的股票做计算，已缓存的直接读取
        """
        # 1. 批量查缓存
        cached = self.cache.get_avg_amount_batch(end_date)

        result = {}
        need_calc = []
        for code in stock_codes:
            if code in cached:
                result[code] = cached[code]
            else:
                need_calc.append(code)

        if not need_calc:
            print(f"  [缓存] 全部 {len(stock_codes)} 只股票日均成交额已缓存")
            return result

        print(f"  [抓取] {len(need_calc)} 只股票需计算日均成交额，已缓存 {len(result)} 只")

        # 2. 逐只计算未缓存的
        total = len(need_calc)
        for i, code in enumerate(need_calc):
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{total}")

            avg = self.calculate_avg_amount(code, end_date, days)
            result[code] = avg
            time.sleep(0.05)

        # 3. 批量存入缓存
        new_data = {k: v for k, v in result.items() if k in need_calc}
        if new_data:
            self.cache.save_avg_amount_batch(end_date, new_data)

        return result

    # ===== 辅助方法 =====

    def _format_stock_code(self, code: str) -> str:
        """格式化股票代码为 ts_code 格式"""
        code = str(code).zfill(6)
        if code.startswith(("60", "68")):
            return f"{code}.SH"
        elif code.startswith(("00", "30")):
            return f"{code}.SZ"
        elif code.startswith(("8", "4")):
            return f"{code}.BJ"
        return f"{code}.SZ"

    def _get_all_stock_codes(self) -> List[str]:
        """获取全市场股票代码列表"""
        df = ak.stock_info_a_code_name()
        codes = []
        for _, row in df.iterrows():
            code = str(row.get("code", ""))
            if code:
                codes.append(self._format_stock_code(code))
        return codes

    def _get_individual_info(self, code: str) -> dict:
        """获取单只股票的详细信息"""
        try:
            df = ak.stock_individual_info_em(symbol=code)
            if df is None or df.empty:
                return {}

            info = {}
            for _, row in df.iterrows():
                item = str(row.get("item", ""))
                value = str(row.get("value", ""))

                if item == "上市时间":
                    info["list_date"] = value.replace("-", "")
                elif item == "行业":
                    info["industry"] = value
                elif item == "总股本":
                    try:
                        info["total_shares"] = float(value)
                    except ValueError:
                        info["total_shares"] = 0
                elif item == "流通股":
                    try:
                        info["circ_shares"] = float(value)
                    except ValueError:
                        info["circ_shares"] = 0
                elif item == "交易所":
                    info["exchange"] = value

            return info
        except Exception:
            return {}

    def _get_stock_name(self, ts_code: str) -> str:
        """从缓存中获取股票名称"""
        if self._stock_info_cache is None:
            self._stock_info_cache = ak.stock_info_a_code_name()

        code = ts_code.split(".")[0]
        matches = self._stock_info_cache[self._stock_info_cache["code"] == code]
        if not matches.empty:
            return str(matches.iloc[0].get("name", ""))
        return ""

    def _get_total_shares(self, ts_code: str) -> float:
        """获取总股本（带内存缓存）"""
        if self._total_share_cache is None:
            self._total_share_cache = {}

        if ts_code in self._total_share_cache:
            return self._total_share_cache[ts_code]

        code = ts_code.split(".")[0]
        info = self._get_individual_info(code)
        shares = info.get("total_shares", 0)

        self._total_share_cache[ts_code] = shares
        return shares

    def _get_circ_shares(self, ts_code: str) -> float:
        """获取流通股本（带内存缓存）"""
        if self._total_share_cache is None:
            self._total_share_cache = {}

        cache_key = f"{ts_code}_circ"
        if cache_key in self._total_share_cache:
            return self._total_share_cache[cache_key]

        code = ts_code.split(".")[0]
        info = self._get_individual_info(code)
        shares = info.get("circ_shares", 0)

        self._total_share_cache[cache_key] = shares
        return shares
