"""
Market Events Fetcher - 市场事件数据抓取

抓取定性分析需要的事件类数据:
    - 停牌信息
    - 重大事项公告（并购重组、股份变动等）
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
from utils.concurrency import run_concurrently


class MarketEventsFetcher:
    """
    市场事件数据抓取器
    
    数据源: Tushare Pro
    
    抓取内容:
        - 停牌状态及停牌天数
        - 重大事项公告（并购重组、再融资等）
    """

    def __init__(self, data_source_config: dict):
        self.config = data_source_config
        self.api = self._init_api()
        self.max_workers = int(
            self.config.get("tushare", {}).get("max_workers", 8)
        )

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

    def fetch_suspension_status(
        self, cutoff_date: str, stock_codes: List[str] = None
    ) -> Dict[str, dict]:
        """
        查询截止日时的停牌状态

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
        trade_date = cutoff_date.replace("-", "")
        result = {}

        try:
            # Tushare suspend_d 接口查询停牌数据
            df = self.api.suspend_d(trade_date=trade_date)
            if df is None or df.empty:
                return result

            for _, row in df.iterrows():
                ts_code = row.get("ts_code", "")
                if stock_codes and ts_code not in stock_codes:
                    continue

                suspend_date_str = str(row.get("suspend_date", ""))
                suspend_days = 0
                if suspend_date_str:
                    try:
                        suspend_dt = datetime.strptime(suspend_date_str, "%Y%m%d")
                        cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
                        suspend_days = (cutoff_dt - suspend_dt).days
                    except ValueError:
                        pass

                # 超过25个交易日的判断（约35个自然日）
                over_25_trading_days = suspend_days >= 35

                result[ts_code] = {
                    "is_suspended": True,
                    "suspend_date": suspend_date_str,
                    "suspend_days": suspend_days,
                    "over_25_trading_days": over_25_trading_days,
                    "suspend_reason": row.get("suspend_reason", "")
                }
        except Exception:
            pass

        return result

    def fetch_major_announcements(
        self,
        cutoff_date: str,
        review_start: str,
        stock_codes: List[str] = None,
        event_types: List[str] = None
    ) -> Dict[str, List[dict]]:
        """
        查询考察期内的重大事项公告

        参数:
            cutoff_date: 数据截止日
            review_start: 考察期开始日（如 "2023-05-01"）
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
        start_str = review_start.replace("-", "")
        end_str = cutoff_date.replace("-", "")

        # 事件类型映射
        type_keywords = event_types or [
            "重大资产重组", "收购", "合并", "分立", "分拆",
            "回购", "再融资", "增发", "配股", "可转债",
            "退市", "破产", "重整", "风险警示"
        ]

        result = {}

        codes_to_query = stock_codes or []

        if codes_to_query:
            # 按股票查询 (并发)
            print(f"  [重大公告] 查询 {len(codes_to_query)} 只股票 "
                  f"(并发{self.max_workers})...")

            def _query_one(code):
                events = self._query_announcements(
                    code, start_str, end_str, type_keywords
                )
                if events:
                    return code, events
                return None

            results = run_concurrently(
                _query_one, codes_to_query,
                max_workers=self.max_workers, progress_every=100, desc="公告"
            )
            for r in results:
                if r:
                    result[r[0]] = r[1]
        else:
            # 全市场查询（按日期范围逐日查询）
            events_all = self._query_all_announcements(
                start_str, end_str, type_keywords
            )
            for code, events in events_all.items():
                result[code] = events

        return result

    def _query_announcements(
        self, ts_code: str, start_str: str, end_str: str, keywords: List[str]
    ) -> List[dict]:
        """查询单只股票的公告"""
        events = []
        try:
            df = self.api.disclosure_date(
                ts_code=ts_code,
                start_date=start_str,
                end_date=end_str
            )
            if df is None or df.empty:
                return events

            for _, row in df.iterrows():
                title = str(row.get("title", ""))
                ann_date = str(row.get("ann_date", ""))

                # 匹配事件类型
                matched_type = None
                for kw in keywords:
                    if kw in title:
                        matched_type = kw
                        break

                if matched_type:
                    events.append({
                        "title": title,
                        "ann_date": ann_date,
                        "event_type": matched_type
                    })
        except Exception:
            pass

        return events

    def _query_all_announcements(
        self, start_str: str, end_str: str, keywords: List[str]
    ) -> Dict[str, List[dict]]:
        """全市场查询公告（按月分批）"""
        result = {}

        start_dt = datetime.strptime(start_str, "%Y%m%d")
        end_dt = datetime.strptime(end_str, "%Y%m%d")
        current = start_dt

        while current <= end_dt:
            # 每次查询一个月
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=current.month + 1, day=1)
            if next_month > end_dt:
                next_month = end_dt + timedelta(days=1)

            period_start = current.strftime("%Y%m%d")
            period_end = (next_month - timedelta(days=1)).strftime("%Y%m%d")

            try:
                df = self.api.disclosure_date(
                    start_date=period_start,
                    end_date=period_end
                )
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        title = str(row.get("title", ""))
                        ts_code = row.get("ts_code", "")
                        ann_date = str(row.get("ann_date", ""))

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

            current = next_month

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
            stock_codes: 需要查询的股票列表（成分股+候选股）

        返回:
            {
                "suspension_status": {...},      # 停牌状态
                "major_announcements": {...},     # 重大公告
            }
        """
        # 1. 停牌状态
        suspension = self.fetch_suspension_status(cutoff_date, stock_codes)

        # 2. 重大事项公告
        announcements = self.fetch_major_announcements(
            cutoff_date=cutoff_date,
            review_start=review_start,
            stock_codes=stock_codes
        )

        return {
            "suspension_status": suspension,
            "major_announcements": announcements,
            "cutoff_date": cutoff_date,
            "review_start": review_start
        }
