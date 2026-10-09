"""
Index Constituent Fetcher - 指数成分股历史数据抓取

抓取指数的历史成分股列表及调整记录
"""

import pandas as pd
from typing import List, Dict


class IndexConstituentFetcher:
    """
    指数成分股抓取器
    
    支持的指数:
        - CSI300  (沪深300, 代码: 000300.SH)
    """

    INDEX_CODES = {
        "csi300": "000300.SH",
    }

    def __init__(self, data_source_config: dict):
        self.config = data_source_config
        self.api = self._init_api()

    def _init_api(self):
        """初始化数据源API"""
        source = self.config.get("provider", "tushare")
        if source == "tushare":
            import tushare as ts
            # api_key 在 tushare 子配置里
            api_key = self.config.get("tushare", {}).get("api_key", "")
            ts.set_token(api_key)
            return ts.pro_api()
        else:
            raise ValueError(f"不支持的数据源: {source}")

    def get_constituents(
        self, index_name: str, date: str
    ) -> List[dict]:
        """
        获取指定日期的指数成分股

        如果精确日期无数据（非交易日或数据未更新），
        自动向前回退到最近一个有数据的交易日，避免成分股列表为空。

        参数:
            index_name: 指数名称 (csi300)
            date: 日期 (格式: YYYY-MM-DD)

        返回:
            成分股列表
        """
        index_code = self.INDEX_CODES.get(index_name)
        if not index_code:
            raise ValueError(f"不支持的指数: {index_name}")

        trade_date = date.replace("-", "")

        # 优先用精确日期查询
        try:
            df = self.api.index_weight(
                index_code=index_code,
                trade_date=trade_date
            )
        except Exception:
            df = None

        # 精确日期无数据时，向前回退查找最近有数据的交易日
        if df is None or df.empty:
            df = self._get_nearest_constituents(index_code, trade_date)

        if df is None or df.empty:
            print(f"  [成分股] 警告: {index_name} 在 {date} 及前30天内均无成分股数据")
            return []

        return df.to_dict("records")

    def _get_nearest_constituents(
        self, index_code: str, trade_date: str, lookback_days: int = 30
    ):
        """
        向前回退查找最近有成分股数据的交易日

        参数:
            index_code: 指数代码
            trade_date: 目标日期 (YYYYMMDD)
            lookback_days: 最多向前查找的天数

        返回:
            最近交易日的成分股 DataFrame，或 None
        """
        from datetime import datetime, timedelta

        end_dt = datetime.strptime(trade_date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=lookback_days)

        try:
            df = self.api.index_weight(
                index_code=index_code,
                start_date=start_dt.strftime("%Y%m%d"),
                end_date=trade_date
            )
            if df is not None and not df.empty:
                # 取最近一个交易日的数据
                latest_date = df["trade_date"].max()
                df = df[df["trade_date"] == latest_date].copy()
                actual_date = f"{str(latest_date)[:4]}-{str(latest_date)[4:6]}-{str(latest_date)[6:8]}"
                print(f"  [成分股] 精确日期无数据，回退到最近交易日: {actual_date}")
                return df
        except Exception:
            pass

        return None

    def get_adjustment_history_by_date(
        self, index_name: str, target_date: str
    ) -> tuple:
        """
        获取指定调整日期的实际调入调出数据

        通过对比调整前后的成分股快照计算调入调出。
        使用范围查询获取月末最近交易日的成分股，避免精确查询
        非交易日返回空集导致全部误判为调入。

        参数:
            index_name: 指数名称 (csi300)
            target_date: 调整目标月 (如 "2025-12")

        返回:
            (actual_add, actual_remove) 两个列表
        """
        index_code = self.INDEX_CODES.get(index_name)
        if not index_code:
            raise ValueError(f"不支持的指数: {index_name}")

        # 解析目标月，计算调整前后日期
        year, month = int(target_date[:4]), int(target_date[5:7])

        # 调整生效日通常是6月/12月第二个周五后的下一交易日
        # before: 取调整月前一个月的月末成分股（调整尚未生效）
        # after:  取调整月月末的成分股（调整已生效）
        if month == 6:
            before_range_start = f"{year}0520"
            before_range_end = f"{year}0531"
            after_range_start = f"{year}0620"
            after_range_end = f"{year}0630"
        elif month == 12:
            before_range_start = f"{year}1020"
            before_range_end = f"{year}1031"
            after_range_start = f"{year}1220"
            after_range_end = f"{year}1231"
        else:
            # 非调整月，返回空
            return [], []

        # 获取调整前的成分股（用范围查询取最近交易日，避免非交易日返回空）
        before_constituents = self._get_constituents_in_range(
            index_code, before_range_start, before_range_end
        )

        # 获取调整后的成分股
        after_constituents = self._get_constituents_in_range(
            index_code, after_range_start, after_range_end
        )

        print(f"  [实际调整] 调整前成分股: {len(before_constituents)} 只")
        print(f"  [实际调整] 调整后成分股: {len(after_constituents)} 只")

        # 计算调入调出
        added_codes = after_constituents - before_constituents
        removed_codes = before_constituents - after_constituents

        # 格式化返回
        actual_add = [
            {"stock_code": code, "stock_name": "", "date": target_date}
            for code in added_codes
        ]
        actual_remove = [
            {"stock_code": code, "stock_name": "", "date": target_date}
            for code in removed_codes
        ]

        return actual_add, actual_remove

    def _get_constituents_in_range(
        self, index_code: str, start_date: str, end_date: str
    ) -> set:
        """
        在日期范围内查询成分股，取最近一个交易日的数据

        参数:
            index_code: 指数代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        返回:
            成分股代码集合
        """
        try:
            df = self.api.index_weight(
                index_code=index_code,
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and not df.empty:
                # 取最近一个交易日的数据
                latest_date = df["trade_date"].max()
                df_latest = df[df["trade_date"] == latest_date]
                actual_date = f"{str(latest_date)[:4]}-{str(latest_date)[4:6]}-{str(latest_date)[6:8]}"
                print(f"  [成分股] 范围查询 {start_date}-{end_date}，取最近交易日: {actual_date} ({len(df_latest)} 只)")
                return set(df_latest["con_code"].values)
        except Exception:
            pass
        return set()

    def get_adjustment_history(
        self, index_name: str, start_date: str, end_date: str
    ) -> List[dict]:
        """
        获取指数调整历史记录

        通过对比不同时间点的成分股列表，计算调入调出

        参数:
            index_name: 指数名称
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        返回:
            调整记录列表:
            [{"adjustment_date": "2024-06-14", "added": [...], "removed": [...]}]
        """
        index_code = self.INDEX_CODES.get(index_name)
        if not index_code:
            raise ValueError(f"不支持的指数: {index_name}")

        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")

        # 获取起止日期之间的所有成分股快照
        df = self.api.index_weight(
            index_code=index_code,
            start_date=start_str,
            end_date=end_str
        )

        if df is None or df.empty:
            return []

        # 按交易日分组，对比前后差异
        trade_dates = sorted(df["trade_date"].unique())
        result = []

        for i in range(1, len(trade_dates)):
            prev_date = trade_dates[i - 1]
            curr_date = trade_dates[i]

            prev_stocks = set(
                df[df["trade_date"] == prev_date]["con_code"].values
            )
            curr_stocks = set(
                df[df["trade_date"] == curr_date]["con_code"].values
            )

            added = list(curr_stocks - prev_stocks)
            removed = list(prev_stocks - curr_stocks)

            if added or removed:
                result.append({
                    "adjustment_date": f"{curr_date[:4]}-{curr_date[4:6]}-{curr_date[6:8]}",
                    "added": added,
                    "removed": removed
                })

        return result

    def get_all_historical_constituents(
        self, index_name: str
    ) -> Dict[str, List[dict]]:
        """
        获取指数的所有历史成分股数据

        返回:
            按日期组织的历史成分股字典:
            {"20240430": [{...}, ...], "20240531": [{...}, ...]}
        """
        index_code = self.INDEX_CODES.get(index_name)
        if not index_code:
            raise ValueError(f"不支持的指数: {index_name}")

        # 获取最近3年的成分股数据
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y%m%d")

        df = self.api.index_weight(
            index_code=index_code,
            start_date=start_date,
            end_date=end_date
        )

        if df is None or df.empty:
            return {}

        result = {}
        for trade_date in df["trade_date"].unique():
            date_df = df[df["trade_date"] == trade_date]
            result[str(trade_date)] = date_df.to_dict("records")

        return result
