"""
Market Events Fetcher (AKShare) - 市场事件数据抓取

使用AKShare获取定性分析需要的事件类数据:
    - 停牌信息: ak.stock_tfp_em(date)
    - 重大事项公告: AKShare无直接接口，用替代方案
    - 风险警示: 从股票名称中检测ST/*ST

与Tushare版本的主要差异:
    - 停牌数据: ak.stock_tfp_em(date) 可获取，但字段格式不同
    - 公告数据: AKShare无disclosure_date等接口，用 stock_notice_report 替代
    - 优势: 完全免费
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from data.storage.cache_manager import CacheManager


class MarketEventsFetcherAK:
    """
    AKShare版市场事件数据抓取器
    
    抓取内容:
        - 停牌状态及停牌天数
        - 重大事项公告（并购重组等，使用替代接口）
        - 风险警示股票列表
    """

    def __init__(self, data_source_config: dict = None):
        self.config = data_source_config or {}
        self.cache = CacheManager(self.config.get("cache_db", "data/storage/database/cache.db"))

    def fetch_suspension_status(
        self, cutoff_date: str, stock_codes: List[str] = None
    ) -> Dict[str, dict]:
        """
        查询截止日时的停牌状态

        使用 ak.stock_tfp_em(date="20240430")
        获取东方财富的停牌数据

        参数:
            cutoff_date: 数据截止日
            stock_codes: 指定查询的股票列表，None则查全市场

        返回:
            {
                "000001.SZ": {
                    "is_suspended": True,
                    "suspend_date": "20240401",
                    "suspend_days": 29,
                    "suspend_reason": "重大资产重组"
                }
            }
        """
        # 1. 先查缓存
        if self.cache.has_events(cutoff_date, "suspension"):
            cached = self.cache.get_events(cutoff_date, "suspension")
            if cached:
                result = {}
                for ts_code, events in cached.items():
                    if events and isinstance(events, list):
                        detail = events[0].get("event_detail", "{}")
                        if isinstance(detail, str):
                            try:
                                result[ts_code] = json.loads(detail)
                            except Exception:
                                result[ts_code] = {}
                        elif isinstance(detail, dict):
                            result[ts_code] = detail
                return result

        date_str = cutoff_date.replace("-", "")
        result = {}

        try:
            # stock_tfp_em 获取指定日期的停牌股票
            df = ak.stock_tfp_em(date=date_str)
            if df is None or df.empty:
                return result

            cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")

            for _, row in df.iterrows():
                code = str(row.get("代码", ""))
                if not code:
                    continue

                ts_code = self._format_stock_code(code)

                if stock_codes and ts_code not in stock_codes:
                    continue

                # 解析停牌日期
                suspend_date_str = ""
                suspend_days = 0
                suspend_date_raw = str(row.get("停牌日期", ""))
                if suspend_date_raw and suspend_date_raw != "nan":
                    for fmt in ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]:
                        try:
                            suspend_dt = datetime.strptime(suspend_date_raw.replace("-", "").replace("/", ""), "%Y%m%d")
                            suspend_date_str = suspend_dt.strftime("%Y%m%d")
                            suspend_days = (cutoff_dt - suspend_dt).days
                            break
                        except ValueError:
                            continue

                over_25_trading_days = suspend_days >= 35  # 约25个交易日

                result[ts_code] = {
                    "is_suspended": True,
                    "suspend_date": suspend_date_str,
                    "suspend_days": suspend_days,
                    "over_25_trading_days": over_25_trading_days,
                    "suspend_reason": str(row.get("停牌原因", "")),
                    "raw_name": str(row.get("名称", ""))
                }
        except Exception as e:
            print(f"获取停牌数据失败: {e}")

        # 3. 抓取后存入缓存
        if result:
            self.cache.save_events(cutoff_date, "suspension", result)

        return result

    def fetch_major_announcements(
        self,
        cutoff_date: str,
        review_start: str,
        stock_codes: List[str] = None,
        event_types: List[str] = None
    ) -> Dict[str, List[dict]]:
        """
        获取考察期内的重大事项公告

        AKShare替代方案:
            1. ak.stock_notice_report(symbol, date) - 个股公告报告
            2. ak.stock_em_jysdrhbcn_em() - 上市公司并购重组
            3. 从东方财富网页面抓取

        参数:
            cutoff_date: 数据截止日
            review_start: 考察期开始日
            stock_codes: 指定股票列表
            event_types: 事件类型筛选

        返回:
            {
                "000001.SZ": [
                    {
                        "title": "关于重大资产重组的公告",
                        "ann_date": "20240115",
                        "event_type": "重大资产重组"
                    }
                ]
            }
        """
        # 事件类型关键词
        keywords = event_types or [
            "重大资产重组", "收购", "合并", "分立", "分拆",
            "回购", "再融资", "增发", "配股", "可转债",
            "退市", "破产", "重整", "风险警示"
        ]

        result = {}

        # 方法1: 逐只查询个股公告（如果指定了股票列表）
        if stock_codes:
            for code in stock_codes:
                events = self._fetch_stock_notices(code, review_start, cutoff_date, keywords)
                if events:
                    result[code] = events
                time.sleep(0.05)

        # 方法2: 获取并购重组列表（全市场）
        merger_events = self._fetch_merger_acquisition_list(review_start, cutoff_date, keywords)
        for code, events in merger_events.items():
            if code not in result:
                result[code] = events
            else:
                result[code].extend(events)

        return result

    def _fetch_stock_notices(
        self, ts_code: str, start_date: str, end_date: str, keywords: List[str]
    ) -> List[dict]:
        """
        获取单只股票的公告

        使用 ak.stock_notice_report(symbol="000001", date="20240101")
        注意: AKShare的公告接口按日期查询，需要按月分批
        """
        ak_code = ts_code.split(".")[0]
        events = []

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        current = start_dt

        while current <= end_dt:
            date_str = current.strftime("%Y%m%d")

            try:
                # stock_notice_report 按日期查公告
                df = ak.stock_notice_report(symbol=ak_code, date=date_str)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        title = str(row.get("公告标题", row.get("title", "")))
                        ann_date = str(row.get("公告日期", row.get("date", "")))

                        for kw in keywords:
                            if kw in title:
                                events.append({
                                    "title": title,
                                    "ann_date": ann_date,
                                    "event_type": kw
                                })
                                break
            except Exception:
                pass

            current += timedelta(days=7)  # 每周查一次减少请求量
            time.sleep(0.05)

        return events

    def _fetch_merger_acquisition_list(
        self, start_date: str, end_date: str, keywords: List[str]
    ) -> Dict[str, List[dict]]:
        """
        获取并购重组列表（全市场）

        使用 ak.stock_em_jysdrhbcn_em() 或类似接口
        """
        result = {}

        try:
            # 尝试获取并购重组数据
            df = ak.stock_em_jysdrhbcn_em()
            if df is None or df.empty:
                return result

            start_str = start_date.replace("-", "")
            end_str = end_date.replace("-", "")

            for _, row in df.iterrows():
                code = str(row.get("股票代码", ""))
                if not code:
                    continue

                ts_code = self._format_stock_code(code)
                title = str(row.get("公告标题", row.get("标题", "")))
                ann_date = str(row.get("公告日期", row.get("日期", "")))

                # 筛选日期范围
                date_clean = ann_date.replace("-", "").replace("/", "")
                if date_clean and (date_clean < start_str or date_clean > end_str):
                    continue

                # 匹配事件类型
                for kw in keywords:
                    if kw in title:
                        if ts_code not in result:
                            result[ts_code] = []
                        result[ts_code].append({
                            "title": title,
                            "ann_date": ann_date,
                            "event_type": kw
                        })
                        break
        except Exception:
            pass

        return result

    def fetch_st_stocks(self, cutoff_date: str) -> Dict[str, dict]:
        """
        获取ST/*ST股票列表

        使用 ak.stock_zh_a_st_em() 获取ST股票
        或从全市场行情中筛选名称含ST的股票

        返回:
            {
                "000001.SZ": {
                    "is_st": True,
                    "st_type": "ST",  # ST 或 *ST
                    "name": "*STxxx"
                }
            }
        """
        result = {}

        try:
            # 获取ST股票列表
            df = ak.stock_zh_a_st_em()
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    code = str(row.get("代码", ""))
                    if not code:
                        continue

                    ts_code = self._format_stock_code(code)
                    name = str(row.get("名称", ""))

                    result[ts_code] = {
                        "is_st": True,
                        "st_type": "*ST" if "*ST" in name else "ST",
                        "name": name
                    }
        except Exception:
            # 如果接口失败，从实时行情中筛选
            try:
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        name = str(row.get("名称", ""))
                        if "ST" in name:
                            code = str(row.get("代码", ""))
                            ts_code = self._format_stock_code(code)
                            result[ts_code] = {
                                "is_st": True,
                                "st_type": "*ST" if "*ST" in name else "ST",
                                "name": name
                            }
            except Exception:
                pass

        return result

    def fetch_qualitative_data(
        self, cutoff_date: str, review_start: str,
        stock_codes: List[str] = None
    ) -> dict:
        """
        一次性采集所有定性分析需要的事件数据

        参数:
            cutoff_date: 数据截止日
            review_start: 考察期开始日
            stock_codes: 需要查询的股票列表

        返回:
            {
                "suspension_status": {...},      # 停牌状态
                "major_announcements": {...},    # 重大公告
                "st_stocks": {...},              # ST股票列表
            }
        """
        # 1. 先查缓存（整体查缓存）
        if self.cache.has_events(cutoff_date, "qualitative"):
            cached = self.cache.get_events(cutoff_date, "qualitative")
            if cached and "all" in cached and cached["all"]:
                detail = cached["all"][0].get("event_detail", "{}")
                try:
                    return json.loads(detail)
                except Exception:
                    pass

        # 2. 未命中则从AKShare抓取
        # 1. 停牌状态
        suspension = self.fetch_suspension_status(cutoff_date, stock_codes)

        # 2. 重大事项公告
        announcements = self.fetch_major_announcements(
            cutoff_date=cutoff_date,
            review_start=review_start,
            stock_codes=stock_codes
        )

        # 3. ST股票列表
        st_stocks = self.fetch_st_stocks(cutoff_date)

        result = {
            "suspension_status": suspension,
            "major_announcements": announcements,
            "st_stocks": st_stocks,
            "cutoff_date": cutoff_date,
            "review_start": review_start
        }

        # 3. 抓取后存入缓存（整体存入）
        self.cache.save_events(cutoff_date, "qualitative", {"all": result})

        return result

    def _format_stock_code(self, code: str) -> str:
        """格式化股票代码"""
        code = str(code).zfill(6)
        if code.startswith(("60", "68")):
            return f"{code}.SH"
        elif code.startswith(("00", "30")):
            return f"{code}.SZ"
        elif code.startswith(("8", "4")):
            return f"{code}.BJ"
        return f"{code}.SZ"
