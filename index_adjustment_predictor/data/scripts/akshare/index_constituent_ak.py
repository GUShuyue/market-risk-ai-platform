"""
Index Constituent Fetcher (AKShare) - 指数成分股历史数据抓取

使用AKShare获取指数的成分股列表及调整记录。
AKShare无需API Key，但接口能力与Tushare有差异。

与Tushare版本的主要差异:
    - 当前成分股: ak.index_stock_cons() 可直接获取
    - 历史成分股: ak.index_stock_hist() 可获取历史变更记录
    - 权重数据: ak.index_stock_cons_weight_csindex() 需从中证指数官网下载
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from data.storage.cache_manager import CacheManager


class IndexConstituentFetcherAK:
    """
    AKShare版指数成分股抓取器
    
    支持的指数:
        - CSI300  (沪深300, 代码: 000300)
    """

    INDEX_CODES = {
        "csi300": "000300",
    }

    # AKShare指数代码前缀映射
    INDEX_PREFIX = {
        "000300": "sh",
    }

    def __init__(self, data_source_config: dict = None):
        self.config = data_source_config or {}
        self.cache = CacheManager(self.config.get("cache_db", "data/storage/database/cache.db"))
        # 反向映射: index_code -> index_name，用于当前成分股缓存
        self._code_to_name = {v: k for k, v in self.INDEX_CODES.items()}

    def get_constituents(
        self, index_name: str, date: str
    ) -> List[dict]:
        """
        获取指定日期的指数成分股

        策略:
            1. 先尝试用 index_stock_hist 查历史成分股
            2. 如果查不到精确日期，用当前成分股做近似
            3. 与历史变更记录对比，还原目标日期时的成分股

        参数:
            index_name: 指数名称 (csi300)
            date: 日期 (格式: YYYY-MM-DD)

        返回:
            成分股列表，每项包含 ts_code, name, weight(如有)
        """
        index_code = self.INDEX_CODES.get(index_name)
        if not index_code:
            raise ValueError(f"不支持的指数: {index_name}")

        # 1. 先查缓存
        if self.cache.has_constituents(index_name, date):
            cached = self.cache.get_constituents(index_name, date)
            if cached:
                return cached

        # 2. 未命中则从AKShare抓取
        # 方法1: 尝试获取当前成分股
        current_constituents = self._get_current_constituents(index_code)

        # 方法2: 获取历史变更记录，推算目标日期的成分股
        date_dt = datetime.strptime(date, "%Y-%m-%d")
        today = datetime.now()

        if (today - date_dt).days <= 30:
            # 最近30天内的数据，当前成分股基本就是目标日期的成分股
            result = current_constituents
        else:
            # 历史日期：尝试从变更记录还原
            result = self._reconstruct_historical_constituents(
                index_code, date, current_constituents
            )

        # 3. 抓取后存入缓存
        if result:
            self.cache.save_constituents(index_name, date, result)

        return result

    def _get_current_constituents(self, index_code: str) -> List[dict]:
        """
        获取当前最新成分股列表

        使用 ak.index_stock_cons(symbol="000300")
        返回: 代码、名称、纳入日期
        """
        # 查缓存（当前成分股以当天日期缓存）
        index_name = self._code_to_name.get(index_code, index_code)
        today = datetime.now().strftime("%Y-%m-%d")
        if self.cache.has_constituents(index_name, today):
            cached = self.cache.get_constituents(index_name, today)
            if cached:
                return cached

        try:
            df = ak.index_stock_cons(symbol=index_code)
            if df is None or df.empty:
                return []

            result = []
            for _, row in df.iterrows():
                code = str(row.get("品种代码", ""))
                if not code:
                    continue

                ts_code = self._format_stock_code(code)
                result.append({
                    "ts_code": ts_code,
                    "con_code": ts_code,
                    "name": str(row.get("品种名称", "")),
                    "in_date": str(row.get("纳入日期", "")),
                    "weight": 0.0  # 权重需另外获取
                })

            # 抓取后存入缓存
            if result:
                self.cache.save_constituents(index_name, today, result)

            return result
        except Exception as e:
            print(f"获取成分股失败: {e}")
            return []

    def get_constituent_weights(self, index_name: str) -> List[dict]:
        """
        获取成分股权重数据

        使用 ak.index_stock_cons_weight_csindex(symbol="000300")
        从中证指数官网下载权重数据

        返回:
            包含权重信息的成分股列表
        """
        index_code = self.INDEX_CODES.get(index_name)
        if not index_code:
            raise ValueError(f"不支持的指数: {index_name}")

        try:
            df = ak.index_stock_cons_weight_csindex(symbol=index_code)
            if df is None or df.empty:
                return []

            result = []
            for _, row in df.iterrows():
                code = str(row.get("成份券代码", ""))
                if not code:
                    continue

                ts_code = self._format_stock_code(code)
                result.append({
                    "ts_code": ts_code,
                    "con_code": ts_code,
                    "name": str(row.get("成份券名称", "")),
                    "weight": float(row.get("权重(%)", 0) or 0) / 100,
                    "free_float_mv": float(row.get("自由流通市值", 0) or 0)
                })

            return result
        except Exception as e:
            print(f"获取权重失败: {e}")
            return []

    def get_adjustment_history(
        self, index_name: str, start_date: str, end_date: str
    ) -> List[dict]:
        """
        获取指数调整历史记录

        使用 ak.index_stock_hist(symbol="000300", start_date, end_date)
        返回每次调入调出的记录

        参数:
            index_name: 指数名称
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        返回:
            调整记录列表
        """
        index_code = self.INDEX_CODES.get(index_name)
        if not index_code:
            raise ValueError(f"不支持的指数: {index_name}")

        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")

        try:
            df = ak.index_stock_hist(
                symbol=index_code,
                start_date=start_str,
                end_date=end_str
            )
            if df is None or df.empty:
                return []

            result = []
            for _, row in df.iterrows():
                result.append({
                    "date": str(row.get("日期", "")),
                    "stock_code": str(row.get("品种代码", "")),
                    "stock_name": str(row.get("品种名称", "")),
                    "action": str(row.get("纳入/剔除", "")),
                    "raw": row.to_dict()
                })

            return result
        except Exception as e:
            print(f"获取调整历史失败: {e}")
            return []

    def get_all_historical_constituents(
        self, index_name: str
    ) -> Dict[str, List[dict]]:
        """
        获取指数的所有历史成分股数据

        返回:
            按日期组织的历史成分股字典
            {"2024-06-01": [...], "2024-12-01": [...]}
        """
        index_code = self.INDEX_CODES.get(index_name)
        if not index_code:
            raise ValueError(f"不支持的指数: {index_name}")

        # 获取全部历史变更记录
        adjustments = self.get_adjustment_history(
            index_name, "2010-01-01", datetime.now().strftime("%Y-%m-%d")
        )

        # 获取当前成分股
        current = self._get_current_constituents(index_code)
        current_codes = {c["ts_code"] for c in current}

        # 从当前成分股反向推导历史成分股
        # 如果某股票在date之后被调入，那么在date时不在指数里
        # 如果某股票在date之后被剔除，那么在date时还在指数里
        result = {}
        for adj in adjustments:
            adj_date = adj.get("date", "")
            if adj.get("action") == "剔除":
                # 这只股票在这个日期之前是在指数里的
                if adj_date not in result:
                    result[adj_date] = []
                result[adj_date].append(adj)

        return result

    def _reconstruct_historical_constituents(
        self,
        index_code: str,
        target_date: str,
        current_constituents: List[dict]
    ) -> List[dict]:
        """
        从当前成分股和历史变更记录还原目标日期的成分股

        逻辑:
            - 获取目标日期之后的所有变更记录
            - 对于"调入"的股票，在目标日期时还不在指数里 → 从列表中移除
            - 对于"剔除"的股票，在目标日期时还在指数里 → 添加到列表中

        参数:
            index_code: 指数代码
            target_date: 目标日期
            current_constituents: 当前成分股列表

        返回:
            目标日期时的成分股列表
        """
        # 获取目标日期之后的所有变更
        today = datetime.now().strftime("%Y-%m-%d")
        adjustments = self.get_adjustment_history_by_code(
            index_code, target_date, today
        )

        if not adjustments:
            # 没有变更记录，直接返回当前成分股
            return current_constituents

        # 从当前列表开始调整
        result_codes = {c["ts_code"] for c in current_constituents}
        result_list = list(current_constituents)

        for adj in adjustments:
            action = adj.get("action", "")
            code = adj.get("stock_code", "")
            ts_code = self._format_stock_code(code)

            if "纳入" in action or "调入" in action:
                # 目标日期之后才调入的，在目标日期时不在指数里
                result_codes.discard(ts_code)
                result_list = [c for c in result_list if c["ts_code"] != ts_code]
            elif "剔除" in action or "调出" in action:
                # 目标日期之后才剔除的，在目标日期时还在指数里
                if ts_code not in result_codes:
                    result_list.append({
                        "ts_code": ts_code,
                        "con_code": ts_code,
                        "name": adj.get("stock_name", ""),
                        "weight": 0.0
                    })
                    result_codes.add(ts_code)

        return result_list

    def get_adjustment_history_by_code(
        self, index_code: str, start_date: str = None, end_date: str = None
    ) -> List[dict]:
        """
        按指数代码获取调整历史

        注意: AKShare 的 index_stock_hist 接口已被移除，
        此方法目前无法从 AKShare 获取历史调整数据。
        请使用 data/historical_adjustments.json 本地文件维护数据。

        返回:
            空列表
        """
        print("  [警告] AKShare 的 index_stock_hist 接口已移除，无法获取历史调整数据")
        print("  [提示] 请使用 data/historical_adjustments.json 本地文件维护数据")
        return []

    def get_adjustment_history_by_date(
        self, index_name: str, target_date: str
    ) -> tuple:
        """
        获取指定调整日期的实际调入调出数据

        优先从本地文件 data/historical_adjustments.json 读取，
        如果本地没有数据，再尝试从 AKShare 接口获取。

        参数:
            index_name: 指数名称 (csi300)
            target_date: 调整目标月 (如 "2025-12")

        返回:
            (actual_add, actual_remove) 两个列表
        """
        # 1. 优先从本地文件读取
        local_data = self._load_local_adjustment_data(index_name, target_date)
        if local_data:
            return local_data

        # 2. 本地无数据，尝试从 AKShare 获取
        print(f"  [提示] 本地无 {index_name} {target_date} 调整数据，尝试 AKShare...")
        index_code = self.INDEX_CODES.get(index_name)
        if not index_code:
            raise ValueError(f"不支持的指数: {index_name}")

        year, month = target_date.split("-")
        start_date = f"{year}{month}01"
        end_date = f"{year}{month}31"

        history = self.get_adjustment_history_by_code(
            index_code, start_date, end_date
        )

        actual_add = []
        actual_remove = []

        for item in history:
            action = item.get("action", "")
            stock_info = {
                "stock_code": self._format_stock_code(item.get("stock_code", "")),
                "stock_name": item.get("stock_name", ""),
                "date": item.get("date", "")
            }

            if "纳入" in action or "调入" in action:
                actual_add.append(stock_info)
            elif "剔除" in action or "调出" in action:
                actual_remove.append(stock_info)

        return actual_add, actual_remove

    def _load_local_adjustment_data(self, index_name: str, target_date: str) -> tuple:
        """
        从本地 JSON 文件加载调整数据

        参数:
            index_name: 指数名称
            target_date: 调整目标月

        返回:
            (actual_add, actual_remove) 或 None
        """
        import json

        # 本地文件路径
        local_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "historical_adjustments.json"
        )
        local_file = os.path.abspath(local_file)

        if not os.path.exists(local_file):
            return None

        try:
            with open(local_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            index_data = data.get(index_name, {})
            period_data = index_data.get(target_date, {})

            if not period_data:
                return None

            # 检查是否有实际数据
            added = period_data.get("added", [])
            removed = period_data.get("removed", [])

            if not added and not removed:
                return None

            # 格式化返回
            actual_add = [
                {
                    "stock_code": item.get("stock_code", ""),
                    "stock_name": item.get("stock_name", ""),
                    "date": period_data.get("adjustment_date", "")
                }
                for item in added
            ]
            actual_remove = [
                {
                    "stock_code": item.get("stock_code", ""),
                    "stock_name": item.get("stock_name", ""),
                    "date": period_data.get("adjustment_date", "")
                }
                for item in removed
            ]

            print(f"  [本地数据] 从 historical_adjustments.json 加载")
            return (actual_add, actual_remove)

        except Exception as e:
            print(f"  [警告] 读取本地调整数据失败: {e}")
            return None

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
