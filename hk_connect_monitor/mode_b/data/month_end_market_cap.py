"""
月末市值计算模块

实施办法第70条(三)和第73条(二)要求小型股市值门槛使用
"前12个月港股平均月末市值" (average month-end market cap)。

本模块计算每月最后一个交易日的市值,取12个月平均:
    - 纯港股: 百度接口的月末市值采样点
    - A+H股: H股股本 × 月末收盘价

数据源:
    - AKShare stock_hk_daily (新浪): 日K线, 取每月最后交易日收盘价
    - AKShare stock_hk_security_profile_em (东财): H股股本
    - AKShare stock_hk_valuation_baidu (百度): 历史市值采样 (降级方案)
"""

import os
import sys
import json
import time
import logging
import socket
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)
socket.setdefaulttimeout(30)

from mode_b.data.hk_daily_fetcher import fetch_daily

try:
    import akshare as ak
except ImportError:
    ak = None
    logger.error("AKShare未安装")

# 百度返回值单位: 亿港币
BAIDU_MARKET_CAP_UNIT = 100_000_000


class MonthEndMarketCapCalculator:
    """
    月末市值计算器

    用法:
        calc = MonthEndMarketCapCalculator(cutoff_date=date(2025, 12, 31))
        results = calc.calc_batch(stocks, stock_info_map)
    """

    def __init__(self, cutoff_date: date, verbose: bool = True):
        self.cutoff = cutoff_date
        self.verbose = verbose
        self.start_date = cutoff_date - pd.DateOffset(months=12)
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._h_share_cache: Dict[str, float] = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [月末市值] {msg}")

    def calc_stock(self, code: str, stock_info: Dict) -> Optional[float]:
        """
        计算单只股票的12个月平均月末市值

        返回: 月末市值(港币元) 或 None
        """
        is_a_plus_h = stock_info.get("is_a_plus_h", False)

        if is_a_plus_h:
            return self._calc_a_plus_h(code)
        else:
            return self._calc_pure_hk(code)

    def _calc_pure_hk(self, code: str) -> Optional[float]:
        """
        纯港股: 用百度接口取月末市值采样点

        百度接口返回约每13天一个市值采样点,
        取每月最后一个采样点的市值作为月末市值近似
        """
        try:
            df = ak.stock_hk_valuation_baidu(
                symbol=code, indicator="总市值", period="全部"
            )
            if df is None or df.empty:
                return None

            df["date"] = pd.to_datetime(df["date"])
            cutoff_ts = pd.Timestamp(self.cutoff)

            # 过滤cutoff前12个月
            mask = (df["date"] <= cutoff_ts) & (df["date"] >= self.start_date)
            filtered = df.loc[mask].copy()

            if filtered.empty:
                # cutoff前12个月无数据, 取cutoff前所有可用数据
                filtered = df.loc[df["date"] <= cutoff_ts].copy()
                if filtered.empty or len(filtered) < 3:
                    return None

            # 按月分组, 取每月最后一个采样点
            filtered["month"] = filtered["date"].dt.to_period("M")
            month_end_values = filtered.groupby("month")["value"].last()

            if month_end_values.empty:
                return None

            avg_mc_yi = float(month_end_values.mean())
            return avg_mc_yi * BAIDU_MARKET_CAP_UNIT

        except Exception as e:
            logger.debug(f"  {code} 百度月末市值失败: {e}")
            return None

    def _calc_a_plus_h(self, code: str) -> Optional[float]:
        """
        A+H股: H股股本 × 每月最后交易日收盘价, 取12个月平均

        H股股本来自 stock_hk_security_profile_em (当前快照)
        """
        # 1. 获取H股股本
        h_shares = self._get_h_shares(code)
        if h_shares <= 0:
            # 降级到百度
            return self._calc_pure_hk(code)

        # 2. 获取日K数据
        daily = self._fetch_daily(code)
        if daily is None or daily.empty:
            return self._calc_pure_hk(code)

        # 3. 过滤cutoff前12个月
        start_ts = pd.Timestamp(self.start_date)
        cutoff_ts = pd.Timestamp(self.cutoff)
        daily = daily[(daily["date"] >= start_ts) & (daily["date"] <= cutoff_ts)]
        if daily.empty:
            return None

        # 4. 按月分组, 取每月最后交易日的收盘价
        daily = daily.copy()
        daily["month"] = daily["date"].dt.to_period("M")
        month_end_closes = daily.groupby("month")["close"].last()

        if month_end_closes.empty:
            return None

        # 5. 月末市值 = H股股本 × 月末收盘价, 取平均
        month_end_mcs = month_end_closes * h_shares
        return float(month_end_mcs.mean())

    def _get_h_shares(self, code: str) -> float:
        """获取H股股本 (带缓存)

        使用 stock_hk_financial_indicator_em 的 已发行股本-H股(股) 字段,
        准确区分H股和总股本。
        """
        if code in self._h_share_cache:
            return self._h_share_cache[code]

        h_shares = 0.0
        try:
            df = ak.stock_hk_financial_indicator_em(symbol=code)
            if df is not None and not df.empty:
                h_val = df.iloc[0].get("已发行股本-H股(股)", 0)
                h_shares = float(h_val) if h_val else 0.0
        except Exception as e:
            logger.debug(f"  {code} H股股本获取失败: {e}")

        self._h_share_cache[code] = h_shares
        return h_shares

    def _fetch_daily(self, code: str) -> Optional[pd.DataFrame]:
        """获取日K数据 (带缓存)"""
        if code in self._daily_cache:
            return self._daily_cache[code]

        df = fetch_daily(code)
        if df is not None and not df.empty:
            df = df.copy()  # copy 后再清洗，避免污染共享缓存
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"])
            self._daily_cache[code] = df
            return df

        return None

    def calc_batch(
        self,
        stocks: List[Dict],
        stock_info_map: Dict[str, Dict],
        max_workers: int = 8,
    ) -> Dict[str, float]:
        """
        批量计算月末市值

        参数:
            stocks: [{code, ...}, ...] 需要计算的股票列表
            stock_info_map: {code: {is_a_plus_h, ...}}
            max_workers: 并发数

        返回:
            {code: month_end_avg_market_cap}
        """
        results = {}
        total = len(stocks)

        self._log(f"开始计算 {total} 只股票的月末市值...")

        def _calc_one(stock):
            code = stock["code"]
            info = stock_info_map.get(code, {})
            try:
                mc = self.calc_stock(code, info)
                return code, mc
            except Exception as e:
                logger.debug(f"  {code} 月末市值异常: {e}")
                return code, None

        start_time = time.time()
        success_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_calc_one, s): s for s in stocks}

            for i, future in enumerate(as_completed(futures)):
                code, mc = future.result()
                if mc is not None and mc > 0:
                    results[code] = mc
                    success_count += 1

                if (i + 1) % 200 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (total - i - 1) / rate
                    self._log(
                        f"  {i+1}/{total} (成功{success_count}) "
                        f"速率: {rate:.1f}/s 剩余: {remaining:.0f}s"
                    )

        elapsed = time.time() - start_time
        self._log(
            f"完成: {success_count}/{total} 成功, 耗时{elapsed:.0f}s"
        )

        return results

    def save_results(self, results: Dict[str, float], output_file: str):
        """保存结果"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self._log(f"结果已保存: {output_file}")
