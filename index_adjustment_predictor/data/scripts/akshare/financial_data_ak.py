"""
Financial Data Fetcher (AKShare) - 财务数据抓取

使用AKShare获取股票财务指标数据，并提取财务异常信号。

与Tushare版本的主要差异:
    - 财务指标: ak.stock_financial_analysis_indicator(symbol="000001") 按股票逐只查询
    - 审计意见: AKShare无直接接口，用财务异常指标替代判断
    - 优势: 完全免费，无需API Key

注意:
    AKShare的财务指标接口是按股票逐只查询的，不能像Tushare那样
    一次查全市场。因此批量查询较慢，建议配合缓存使用。
"""

import akshare as ak
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from data.storage.cache_manager import CacheManager


class FinancialDataFetcherAK:
    """
    AKShare版财务数据抓取器
    
    抓取内容:
        - 营业收入 / 净利润
        - ROE / ROA
        - 资产负债率
        - 现金流
        - 财务亏损状态
        - 审计意见（AKShare无直接接口，用替代判断）
    """

    # AKShare财务分析指标返回的列名映射
    COLUMN_MAP = {
        "日期": "period",
        "摊薄每股收益(元)": "eps",
        "加权每股收益(元)": "weighted_eps",
        "销售毛利率(%)": "gross_margin",
        "净资产收益率(%)": "roe",
        "总资产报酬率(%)": "roa",
        "资产负债率(%)": "debt_to_assets",
        "净利润(万元)": "net_profit",
        "营业收入(万元)": "revenue",
        "营业利润(万元)": "operating_profit",
        "经营活动产生的现金流量净额(万元)": "operating_cash_flow",
    }

    def __init__(self, data_source_config: dict = None):
        self.config = data_source_config or {}
        self.cache = CacheManager(self.config.get("cache_db", "data/storage/database/cache.db"))

    def fetch_financials(self, cutoff_date: str) -> List[dict]:
        """
        获取截止日期前最近一期的财务数据

        AKShare限制: stock_financial_analysis_indicator 按股票逐只查询
        这里只获取全市场代码列表，实际财务数据需要按需逐只获取

        参数:
            cutoff_date: 数据截止日

        返回:
            财务数据列表（可能为空，需要配合 fetch_financial_flags 使用）
        """
        # AKShare无法批量查询全市场财务数据
        # 返回空列表，实际使用时通过 fetch_financial_flags 逐只获取
        return []

    def fetch_financial_flags(
        self, cutoff_date: str, stock_codes: List[str] = None
    ) -> Dict[str, dict]:
        """
        获取财务异常信号（逐只查询）

        参数:
            cutoff_date: 数据截止日
            stock_codes: 指定股票列表

        返回:
            {
                "000001.SZ": {
                    "is_loss": False,
                    "net_profit": 12345678900,
                    "roe": 12.5,
                    "debt_ratio": 65.3,
                    "is_high_debt": False,
                    "audit_opinion": "未知",  # AKShare无法获取审计意见
                    "has_audit_issue": False,
                    "has_financial_risk": False
                }
            }
        """
        # 确定查询的年份
        year = int(cutoff_date[:4])
        if int(cutoff_date[5:7]) < 5:
            year -= 1  # 截止日在4月30日前，用上一年年报

        target_period = f"{year}-12-31"

        result = {}

        # 1. 先批量查缓存
        if stock_codes:
            cached = self.cache.get_financial_flags_batch(stock_codes, target_period)
            result.update(cached)
            # 分离已缓存和未缓存的，只抓取未缓存的
            uncached_codes = [c for c in stock_codes if c not in cached]
        else:
            uncached_codes = []

        total = len(uncached_codes)

        # 2. 未命中则从AKShare抓取
        for i, ts_code in enumerate(uncached_codes):
            if (i + 1) % 50 == 0:
                print(f"  财务数据获取进度: {i+1}/{total}")

            flags = self._fetch_single_financial_flags(ts_code, target_period, year)
            if flags:
                result[ts_code] = flags
                # 3. 抓取后存入缓存
                self.cache.save_financial(ts_code, target_period, flags)

            time.sleep(0.05)  # 避免频率限制

        return result

    def _fetch_single_financial_flags(
        self, ts_code: str, target_period: str, year: int
    ) -> Optional[dict]:
        """
        获取单只股票的财务异常信号

        使用 ak.stock_financial_analysis_indicator(symbol="000001")
        该接口返回该股票所有历史期的财务指标
        """
        ak_code = ts_code.split(".")[0]

        try:
            df = ak.stock_financial_analysis_indicator(symbol=ak_code)
            if df is None or df.empty:
                return None

            # 重命名列
            df = df.rename(columns=self.COLUMN_MAP)

            # 找到目标期的数据
            period_col = "period" if "period" in df.columns else df.columns[0]

            # 尝试匹配目标期
            target_row = None
            for _, row in df.iterrows():
                period_str = str(row.get(period_col, ""))
                # 匹配年份（AKShare的日期格式可能是 "2023-12-31" 或 "2023/12/31"）
                if str(year) in period_str and "12" in period_str:
                    target_row = row
                    break

            if target_row is None and len(df) > 0:
                # 取第一行（最新的）
                target_row = df.iloc[0]

            if target_row is None:
                return None

            # 提取关键指标
            net_profit = self._safe_float(target_row.get("net_profit", 0))
            roe = self._safe_float(target_row.get("roe", 0))
            debt_ratio = self._safe_float(target_row.get("debt_to_assets", 0))

            # 净利润单位转换: AKShare返回的是"万元"，转为元
            if net_profit != 0:
                net_profit_yuan = net_profit * 10000
            else:
                net_profit_yuan = 0

            # 判断异常
            is_loss = net_profit < 0
            is_high_debt = debt_ratio > 70

            # AKShare无法获取审计意见，用替代判断:
            # 如果连续两年亏损，标记为审计风险
            audit_opinion = "未知"
            has_audit_issue = False

            # 检查是否连续亏损（ST风险）
            if is_loss:
                # 尝试获取上一年数据看是否也亏损
                prev_year = year - 1
                prev_period = f"{prev_year}-12-31"
                prev_row = None
                for _, row in df.iterrows():
                    period_str = str(row.get(period_col, ""))
                    if str(prev_year) in period_str and "12" in period_str:
                        prev_row = row
                        break
                if prev_row is not None:
                    prev_profit = self._safe_float(prev_row.get("net_profit", 0))
                    if prev_profit < 0:
                        has_audit_issue = True
                        audit_opinion = "连续两年亏损，ST风险"

            has_financial_risk = is_loss or is_high_debt or has_audit_issue

            return {
                "is_loss": is_loss,
                "net_profit": net_profit_yuan,
                "roe": round(roe, 2),
                "debt_ratio": round(debt_ratio, 2),
                "is_high_debt": is_high_debt,
                "audit_opinion": audit_opinion,
                "has_audit_issue": has_audit_issue,
                "has_financial_risk": has_financial_risk,
                "period": str(target_row.get(period_col, ""))
            }
        except Exception:
            return None

    def fetch_revenue(self, stock_code: str, year: int) -> dict:
        """
        获取指定股票的营业收入数据

        参数:
            stock_code: 股票代码 (如 "000001.SZ")
            year: 年份

        返回:
            {"revenue": 123456, "revenue_yoy": 15.3}
        """
        ak_code = stock_code.split(".")[0]
        target_period = f"{year}-12-31"

        try:
            df = ak.stock_financial_analysis_indicator(symbol=ak_code)
            if df is None or df.empty:
                return {}

            df = df.rename(columns=self.COLUMN_MAP)

            period_col = "period" if "period" in df.columns else df.columns[0]

            for _, row in df.iterrows():
                period_str = str(row.get(period_col, ""))
                if str(year) in period_str and "12" in period_str:
                    revenue = self._safe_float(row.get("revenue", 0))
                    return {
                        "revenue": revenue * 10000,  # 万元转元
                        "revenue_yoy": 0  # 同比增长率需额外计算
                    }
        except Exception:
            pass
        return {}

    def fetch_roe(self, stock_code: str, year: int) -> float:
        """获取指定股票的ROE"""
        ak_code = stock_code.split(".")[0]

        try:
            df = ak.stock_financial_analysis_indicator(symbol=ak_code)
            if df is None or df.empty:
                return 0.0

            df = df.rename(columns=self.COLUMN_MAP)
            period_col = "period" if "period" in df.columns else df.columns[0]

            for _, row in df.iterrows():
                period_str = str(row.get(period_col, ""))
                if str(year) in period_str and "12" in period_str:
                    return self._safe_float(row.get("roe", 0))
        except Exception:
            pass
        return 0.0

    def batch_fetch_financials(
        self, stock_codes: List[str], year: int
    ) -> Dict[str, dict]:
        """
        批量获取财务数据（带进度显示）

        参数:
            stock_codes: 股票代码列表
            year: 年份

        返回:
            {ts_code: {revenue, net_profit, roe, debt_ratio, ...}}
        """
        result = {}
        total = len(stock_codes)

        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0:
                print(f"  批量财务数据进度: {i+1}/{total}")

            ak_code = code.split(".")[0]
            try:
                df = ak.stock_financial_analysis_indicator(symbol=ak_code)
                if df is None or df.empty:
                    continue

                df = df.rename(columns=self.COLUMN_MAP)
                period_col = "period" if "period" in df.columns else df.columns[0]

                for _, row in df.iterrows():
                    period_str = str(row.get(period_col, ""))
                    if str(year) in period_str and "12" in period_str:
                        result[code] = {
                            "revenue": self._safe_float(row.get("revenue", 0)) * 10000,
                            "net_profit": self._safe_float(row.get("net_profit", 0)) * 10000,
                            "roe": self._safe_float(row.get("roe", 0)),
                            "debt_ratio": self._safe_float(row.get("debt_to_assets", 0)),
                            "operating_cash_flow": self._safe_float(
                                row.get("operating_cash_flow", 0)
                            ) * 10000,
                        }
                        break
            except Exception:
                continue

            time.sleep(0.05)

        return result

    def _safe_float(self, value) -> float:
        """安全转换为float，处理各种异常值"""
        try:
            if value is None or value == "" or value == "--":
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
