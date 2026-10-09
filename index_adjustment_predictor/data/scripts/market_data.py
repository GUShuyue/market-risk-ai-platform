"""
Market Data Fetcher - 市场数据抓取脚本

抓取股票市值、流动性、上市时间、行业分类等市场数据
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from utils.concurrency import run_concurrently


class MarketDataFetcher:
    """
    市场数据抓取器
    
    数据源:
        - Tushare Pro (主要)
    
    抓取内容:
        - 总市值 / 自由流通市值
        - 日均成交额 (近1年)
        - 上市日期 / 上市时长
        - 行业分类 (供定性分析行业均衡性用)
        - ST状态
        - 近20日涨跌幅 (供定性分析异常波动用)
    """

    def __init__(self, data_source_config: dict):
        self.config = data_source_config
        self.api = self._init_api()
        # A+H 股 H 股股本缓存 (万股), 懒加载
        self._ah_h_shares = None
        # 数据采集并发线程数
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

    # ==================== A+H 股市值修正 ====================

    def _load_ah_mapping(self) -> dict:
        """
        加载 A+H 股映射清单

        返回:
            {"601318.SH": {"hk_code": "02318", "name": "中国平安"}, ...}
        """
        import json
        from pathlib import Path

        mapping_path = Path(__file__).parent.parent.parent / "config" / "ah_stock_mapping.json"
        if not mapping_path.exists():
            print(f"  [A+H修正] 映射清单不存在: {mapping_path}, 跳过市值修正")
            return {}

        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mapping = data.get("mapping", {})
            print(f"  [A+H修正] 加载映射清单: {len(mapping)} 条 A+H 股")
            return mapping
        except Exception as e:
            print(f"  [A+H修正] 加载映射清单失败: {e}")
            return {}

    def _fetch_h_shares(self) -> dict:
        """
        获取所有 A+H 股的 H 股股本 (万股)

        调用 AKShare stock_hk_financial_indicator_em 接口,
        返回 {a_code: h_share_wan} 映射, 懒加载, 只调一次

        返回:
            {"601318.SH": 744757.69, ...}  # 单位: 万股
        """
        if self._ah_h_shares is not None:
            return self._ah_h_shares

        mapping = self._load_ah_mapping()
        if not mapping:
            self._ah_h_shares = {}
            return self._ah_h_shares

        import akshare as ak

        h_shares = {}
        print(f"  [A+H修正] 获取 {len(mapping)} 只 H 股股本 (AKShare, "
              f"并发{self.max_workers})...")

        def _fetch_one(item):
            a_code, info = item
            hk_code = info["hk_code"]
            try:
                fi = ak.stock_hk_financial_indicator_em(symbol=hk_code)
                if fi is not None and not fi.empty:
                    h_share_str = fi.iloc[0].get("已发行股本-H股(股)", 0)
                    h_share = float(h_share_str) if h_share_str else 0
                    # 股 -> 万股
                    return a_code, h_share / 10000
            except Exception:
                pass
            return None

        results = run_concurrently(
            _fetch_one, mapping.items(),
            max_workers=self.max_workers, progress_every=50, desc="H股股本"
        )
        for r in results:
            if r:
                h_shares[r[0]] = r[1]

        print(f"  [A+H修正] 成功获取 {len(h_shares)}/{len(mapping)} 只 H 股股本")
        self._ah_h_shares = h_shares
        return self._ah_h_shares

    def fetch_by_date(
        self, date: str, stocks: str = "all"
    ) -> List[dict]:
        """
        抓取指定日期的全市场股票数据

        参数:
            date: 日期 (格式: YYYY-MM-DD)
            stocks: 股票列表或"all"

        返回:
            股票数据列表，每只股票包含:
            - ts_code: 股票代码
            - name: 股票名称
            - total_mv: 总市值 (截止日当天)
            - avg_total_mv: 近一年日均总市值 (用于排名)
            - circ_mv: 流通市值
            - avg_amount: 近一年日均成交额
            - avg_turnover: 近一年日均换手率
            - close: 收盘价
            - pct_chg: 当日涨跌幅
        """
        trade_date = date.replace("-", "")

        # 获取每日基本面数据 (市值、PE、换手率等)
        daily_basic = self.api.daily_basic(trade_date=trade_date)

        # 获取日线数据 (成交额、涨跌幅)
        daily = self.api.daily(trade_date=trade_date)

        # 获取股票名称映射 (daily_basic 和 daily 都不返回 name 字段)
        name_map = self._get_name_map()

        # 计算近一年日均成交额
        avg_amount = self._calculate_avg_amount(date)

        # 计算近一年日均总市值 (编制规则要求的排名依据)
        avg_total_mv = self._calculate_avg_total_mv(date)

        # 合并数据
        result = self._merge_data(daily_basic, daily, avg_amount, avg_total_mv, name_map)

        # A+H 股市值修正: total_mv 含 H 股股本, 需扣除
        result = self._correct_ah_market_value(result)

        return result

    def _correct_ah_market_value(self, result: List[dict]) -> List[dict]:
        """
        修正 A+H 股的总市值

        Tushare daily_basic 的 total_share 包含多地市场总股本,
        total_mv = total_share × close 会把 H 股部分按 A 股价格计算,
        导致 A+H 股市值虚高。

        修正公式: 修正后总市值 = (total_share - H股股本) × close
        单位: total_share (万股), H股股本 (万股), close (元), 结果 (万元)

        特殊情况: 三地上市公司 (如百济神州 美股+港股+A股) 的
        Tushare total_share 口径与 AKShare H股股本不一致,
        可能出现 H股 > total_share 导致减法为负。
        此时改用 circ_mv (A股流通市值) 替代, 因为这类科创板
        公司通常已全流通, circ_mv ≈ 真实A股市值。
        """
        h_shares = self._fetch_h_shares()
        if not h_shares:
            return result

        corrected_count = 0
        fallback_count = 0
        for stock in result:
            a_code = stock.get("ts_code", "")
            if a_code not in h_shares:
                continue

            h_share_wan = h_shares[a_code]  # H 股股本 (万股)
            total_share_wan = stock.get("total_share", 0)  # 总股本 (万股)
            close = stock.get("close", 0)

            if total_share_wan > 0 and close > 0 and h_share_wan > 0:
                a_share_wan = total_share_wan - h_share_wan
                original_mv = stock.get("total_mv", 0)

                if a_share_wan > 0:
                    # 正常修正: (total_share - H股) × close
                    corrected_mv = a_share_wan * close  # 万元
                    stock["total_mv"] = corrected_mv
                    stock["total_mv_original"] = original_mv
                    corrected_count += 1
                else:
                    # 三地上市等特殊情况: H股股本 > Tushare total_share,
                    # 两个数据源口径不一致, 改用 circ_mv 替代
                    circ_mv = stock.get("circ_mv", 0)
                    if circ_mv > 0:
                        stock["total_mv"] = circ_mv
                        stock["total_mv_original"] = original_mv
                        fallback_count += 1
                        print(f"  [A+H修正] {a_code} {stock.get('name','')}: "
                              f"H股({h_share_wan:.0f}万股) > total_share({total_share_wan:.0f}万股), "
                              f"改用circ_mv={circ_mv/10000:.0f}亿元 "
                              f"(原total_mv={original_mv/10000:.0f}亿元)")

        if corrected_count > 0 or fallback_count > 0:
            print(f"  [A+H修正] 正常修正 {corrected_count} 只, "
                  f"circ_mv兜底 {fallback_count} 只")

        return result

    def _get_name_map(self) -> dict:
        """
        获取全市场股票代码→名称映射

        daily_basic 和 daily 接口都不返回 name 字段，
        需要从 stock_basic 获取。

        返回:
            {"000001.SZ": "平安银行", ...}
        """
        try:
            df = self.api.stock_basic(exchange="", list_status="L")
            if df is not None and not df.empty:
                return dict(zip(df["ts_code"], df["name"]))
        except Exception:
            pass
        return {}

    def fetch_listing_info(self, cutoff_date: str) -> List[dict]:
        """
        抓取股票上市信息（含行业分类）

        返回:
            包含上市日期、是否ST、行业分类等信息的列表
        """
        # 获取所有A股基本信息
        stock_basic = self.api.stock_basic(
            exchange="",
            list_status="L"
        )

        # 筛选在截止日前上市的股票
        cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")
        result = []
        for _, row in stock_basic.iterrows():
            list_date_str = str(row["list_date"])
            try:
                list_date = datetime.strptime(list_date_str, "%Y%m%d")
            except ValueError:
                continue
            if list_date <= cutoff:
                # 计算上市时长（月）
                listing_months = (cutoff.year - list_date.year) * 12 + (cutoff.month - list_date.month)

                result.append({
                    "ts_code": row["ts_code"],
                    "name": row["name"],
                    "market": row["market"],
                    "list_date": list_date_str,
                    "listing_months": listing_months,
                    "is_st": "ST" in row["name"],
                    "industry": row.get("industry", ""),       # 行业分类
                    "cnspell": row.get("cnspell", ""),         # 拼音缩写
                    "exchange": row.get("exchange", ""),       # 交易所
                })

        return result

    def fetch_industry_distribution(self, cutoff_date: str) -> Dict[str, int]:
        """
        获取全市场行业分布统计

        返回:
            {"银行": 42, "房地产": 68, ...}
        """
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
        获取股票近N日涨跌幅（供异常波动判断）

        参数:
            stock_codes: 股票代码列表
            end_date: 截止日期
            days: 回看天数（默认20个交易日）

        返回:
            {
                "000001.SZ": {
                    "pct_change_20d": 15.3,      # 近20日累计涨跌幅
                    "max_daily_pct": 9.8,        # 近20日单日最大涨幅
                    "min_daily_pct": -5.2,       # 近20日单日最大跌幅
                    "volatility": 3.1,          # 近20日波动率
                    "is_abnormal": True          # 是否异常波动
                }
            }
        """
        end = datetime.strptime(end_date, "%Y-%m-%d")
        start = end - timedelta(days=days * 2)  # 多取一些确保有足够交易日
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        def _fetch_one(code):
            try:
                df = self.api.daily(
                    ts_code=code,
                    start_date=start_str,
                    end_date=end_str
                )
                if df.empty:
                    return None

                # 取最近 days 个交易日
                df = df.tail(days)
                pct_changes = df["pct_chg"].values
                close_prices = df["close"].values

                if len(pct_changes) == 0:
                    return None

                pct_change_total = (
                    (close_prices[-1] / close_prices[0] - 1) * 100
                    if close_prices[0] > 0 else 0
                )
                max_daily = float(pct_changes.max())
                min_daily = float(pct_changes.min())
                volatility = float(pct_changes.std())

                # 异常波动判断: 累计涨跌幅超50%或单日涨跌幅超9.5%
                is_abnormal = (
                    abs(pct_change_total) > 50
                    or abs(max_daily) > 9.5
                    or abs(min_daily) > 9.5
                )

                return code, {
                    "pct_change_20d": round(pct_change_total, 2),
                    "max_daily_pct": round(max_daily, 2),
                    "min_daily_pct": round(min_daily, 2),
                    "volatility": round(volatility, 2),
                    "is_abnormal": is_abnormal
                }
            except Exception:
                return None

        results = run_concurrently(
            _fetch_one, stock_codes,
            max_workers=self.max_workers, progress_every=50, desc="价格变动"
        )

        result = {}
        for r in results:
            if r:
                result[r[0]] = r[1]

        return result

    def _calculate_avg_amount(self, end_date: str) -> dict:
        """
        计算近一年日均成交额

        按编制规则要求，使用考察期内所有交易日的日均成交额。
        考察期由 get_review_period 确定:
          - 6月调整: 前一年5月1日至当年4月30日
          - 12月调整: 前一年11月1日至当年10月31日
        Tushare按日期范围查询会截断（单次最多6000行），因此必须
        逐交易日查询（用trade_date参数，每次返回约5360行），
        全年约243次API调用。

        返回:
            {"000001.SZ": 123456789.0, ...}
        """
        from utils.date_utils import get_review_period

        end = datetime.strptime(end_date, "%Y-%m-%d")
        review_start, _ = get_review_period(end_date)
        start = datetime.strptime(review_start, "%Y-%m-%d")

        trading_days = self._get_trading_days(start, end)
        print(f"  [成交额数据] 逐日查询 {len(trading_days)} 个交易日 "
              f"({review_start} ~ {end_date}, 并发{self.max_workers})...")

        def _fetch_one(day):
            try:
                df = self.api.daily(trade_date=day, fields="ts_code,amount")
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
            return None

        results = run_concurrently(
            _fetch_one, trading_days,
            max_workers=self.max_workers, progress_every=20, desc="成交额"
        )
        all_chunks = [r for r in results if r is not None]

        if not all_chunks:
            return {}

        full_df = pd.concat(all_chunks, ignore_index=True)
        print(f"  [成交额数据] 完成，{len(full_df)} 条记录，覆盖 {full_df['ts_code'].nunique()} 只股票")
        return full_df.groupby("ts_code")["amount"].mean().to_dict()

    def _calculate_avg_total_mv(self, end_date: str) -> dict:
        """
        计算近一年日均总市值

        编制规则要求按考察期内日均总市值排名，必须使用考察期内所有
        交易日的真实日均，不能采样。逐交易日查询daily_basic，
        全年约243次API调用。
        考察期由 get_review_period 确定:
          - 6月调整: 前一年5月1日至当年4月30日
          - 12月调整: 前一年11月1日至当年10月31日

        A+H 股修正:
          daily_basic 的 total_share 含 H 股股本, total_mv 虚高
          对 A+H 股逐日修正: corrected_mv = (total_share - H股股本) * close
          非A+H股直接用 total_mv

        三地上市特殊处理:
          部分公司 (如百济神州 美股+港股+A股) 的 Tushare total_share
          口径与 AKShare H股股本不一致, H股 > total_share 导致减法为负。
          此时改用 circ_mv (A股流通市值) 替代 total_mv。

        返回:
            {"000001.SZ": 1234567890.0, ...}  # 单位: 万元
        """
        from utils.date_utils import get_review_period

        end = datetime.strptime(end_date, "%Y-%m-%d")
        review_start, _ = get_review_period(end_date)
        start = datetime.strptime(review_start, "%Y-%m-%d")

        trading_days = self._get_trading_days(start, end)
        print(f"  [市值数据] 逐日查询 {len(trading_days)} 个交易日 "
              f"({review_start} ~ {end_date}, 并发{self.max_workers})...")

        # 获取 H 股股本 (万股), 用于 A+H 修正
        h_shares = self._fetch_h_shares()

        # 记录需要 circ_mv 兜底的股票, 只警告一次
        import threading
        fallback_warned = set()
        warn_lock = threading.Lock()

        def _fetch_one(day):
            try:
                # 获取 total_share, close, circ_mv, 供 A+H 修正用
                df = self.api.daily_basic(
                    trade_date=day,
                    fields="ts_code,total_mv,total_share,close,circ_mv"
                )
                if df is None or df.empty:
                    return None
                # 对 A+H 股逐日修正 total_mv
                if h_shares:
                    for idx, row in df.iterrows():
                        a_code = row["ts_code"]
                        if a_code in h_shares:
                            h_wan = h_shares[a_code]
                            ts_wan = row.get("total_share", 0)
                            close = row.get("close", 0)
                            if ts_wan > 0 and close > 0 and h_wan > 0:
                                a_wan = ts_wan - h_wan
                                if a_wan > 0:
                                    df.at[idx, "total_mv"] = a_wan * close
                                else:
                                    # 三地上市: H股 > total_share, 用 circ_mv
                                    circ_mv = row.get("circ_mv", 0)
                                    if circ_mv > 0:
                                        df.at[idx, "total_mv"] = circ_mv
                                        with warn_lock:
                                            if a_code not in fallback_warned:
                                                fallback_warned.add(a_code)
                                                print(
                                                    f"  [A+H修正] {a_code}: "
                                                    f"H股({h_wan:.0f}万股) > "
                                                    f"total_share({ts_wan:.0f}万股), "
                                                    f"日均市值改用circ_mv"
                                                )
                return df
            except Exception:
                pass
            return None

        results = run_concurrently(
            _fetch_one, trading_days,
            max_workers=self.max_workers, progress_every=20, desc="市值"
        )
        all_chunks = [r for r in results if r is not None]

        if not all_chunks:
            return {}

        full_df = pd.concat(all_chunks, ignore_index=True)
        print(f"  [市值数据] 完成，{len(full_df)} 条记录，覆盖 {full_df['ts_code'].nunique()} 只股票")
        return full_df.groupby("ts_code")["total_mv"].mean().to_dict()

    def _get_trading_days(self, start: datetime, end: datetime) -> list:
        """
        获取指定日期范围内的真实交易日列表

        参数:
            start: 开始日期
            end: 结束日期

        返回:
            交易日字符串列表 (YYYYMMDD格式)
        """
        try:
            df = self.api.trade_cal(
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                is_open=1
            )
            if df is not None and not df.empty:
                return sorted(df["cal_date"].astype(str).tolist())
        except Exception:
            pass

        # 回退方案：用工作日近似
        days = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                days.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return days

    def _merge_data(
        self, daily_basic, daily, avg_amount, avg_total_mv=None, name_map=None
    ) -> List[dict]:
        """合并多个数据源"""
        if daily_basic is None or daily_basic.empty:
            return []

        avg_total_mv = avg_total_mv or {}
        name_map = name_map or {}

        # 将 daily 和 avg_amount 转为 dict 方便查找
        daily_dict = {}
        if daily is not None and not daily.empty:
            for _, row in daily.iterrows():
                daily_dict[row["ts_code"]] = {
                    "amount": row.get("amount", 0),
                    "close": row.get("close", 0),
                    "pct_chg": row.get("pct_chg", 0),
                    "vol": row.get("vol", 0),
                }

        result = []
        for _, row in daily_basic.iterrows():
            ts_code = row["ts_code"]
            avg_amt = avg_amount.get(ts_code, 0)
            avg_mv = avg_total_mv.get(ts_code, 0)
            daily_info = daily_dict.get(ts_code, {})

            result.append({
                "ts_code": ts_code,
                "name": name_map.get(ts_code, row.get("name", "")),
                "total_mv": row.get("total_mv", 0),
                "total_share": row.get("total_share", 0),   # 总股本 (万股), 供 A+H 修正用
                "avg_total_mv": avg_mv,          # 近一年日均总市值
                "circ_mv": row.get("circ_mv", 0),
                "avg_amount": avg_amt,
                "avg_turnover": row.get("turnover_rate", 0),
                "close": daily_info.get("close", 0),
                "pct_chg": daily_info.get("pct_chg", 0),
                "amount": daily_info.get("amount", 0),
                "vol": daily_info.get("vol", 0),
                "pe": row.get("pe", 0),
                "pb": row.get("pb", 0),
            })

        return result
