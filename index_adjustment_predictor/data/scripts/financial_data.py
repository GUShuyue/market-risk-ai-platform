"""
Financial Data Fetcher - 财务数据抓取

抓取股票财务指标数据，并提取财务异常信号
"""

from typing import List, Dict
from datetime import datetime
import time
from utils.concurrency import run_concurrently


class FinancialDataFetcher:
    """
    财务数据抓取器

    数据源: Tushare Pro

    抓取内容:
        - 营业收入 / 净利润
        - ROE
        - 资产负债率
        - 现金流
        - 财务亏损状态（供定性分析用）
        - 审计意见类型
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

    def fetch_financials(self, cutoff_date: str, stock_codes: List[str] = None) -> List[dict]:
        """
        抓取截止日期前最近一期已披露的年报财务数据

        使用 fina_indicator 接口获取真实的财务指标（净利润、ROE、资产负债率等），
        而非 daily_basic（仅含PE/PB/市值，不含财务报表字段）。

        年报披露规则: 上一自然年的年报在次年4月30日前披露完毕。
        因此在数据截止日（cutoff_date）时，最新已全部披露的年报是:
        - cutoff_date在1-3月: 前年的年报 (cutoff年 - 2)
        - cutoff_date在4月及以后: 去年的年报 (cutoff年 - 1)

        4月30日是年报法定披露截止日，当天可认为去年年报已全部披露。
        指数审核在5月下旬/11月下旬进行，委员会届时一定能看到最新年报。

        例如 cutoff_date=2025-04-30，最新可用年报是2024年报(20241231)。
        cutoff_date=2025-10-31，最新可用年报也是2024年报(20241231)，
        2025年报要到2026年4月底才披露完毕，此时还不存在。

        参数:
            cutoff_date: 数据截止日
            stock_codes: 指定股票列表，None则返回空（避免全量查询太慢）

        返回:
            财务数据列表，每条包含 ts_code, net_profit, roe, debt_to_assets 等
        """
        if not stock_codes:
            return []  # 不全量查询，避免超时

        year = int(cutoff_date[:4])
        # 年报披露截止日是次年4月30日
        # cutoff_date在4月30日(含)之后，去年年报已全部披露
        if int(cutoff_date[5:7]) < 4:
            year -= 2  # 1-3月: 前年年报
        else:
            year -= 1  # 4月及以后: 去年年报

        period = f"{year}1231"
        print(f"  [财务数据] 获取 {len(stock_codes)} 只股票的 {period} 年报指标 "
              f"(并发{self.max_workers})...")

        def _fetch_one(code):
            try:
                df = self.api.fina_indicator(
                    ts_code=code,
                    period=period
                )
                if df is not None and not df.empty:
                    # 取最新一期
                    row = df.iloc[0]
                    return {
                        "ts_code": code,
                        # fina_indicator 没有 netprofit 字段，用 profit_dedt (扣非净利润) 判断亏损
                        "net_profit": row.get("profit_dedt", 0),
                        "roe": row.get("roe", 0),
                        "debt_to_assets": row.get("debt_to_assets", 0),
                        "revenue": row.get("total_revenue_ps", 0),
                        "revenue_yoy": row.get("tr_yoy", 0),
                        "period": period,
                    }
            except Exception:
                pass
            return None

        results = run_concurrently(
            _fetch_one, stock_codes,
            max_workers=self.max_workers, progress_every=100, desc="财务"
        )
        all_results = [r for r in results if r is not None]

        print(f"  [财务数据] 成功获取 {len(all_results)} 条记录")
        return all_results

    def fetch_financial_flags(
        self, cutoff_date: str, stock_codes: List[str] = None
    ) -> Dict[str, dict]:
        """
        提取财务异常信号（供定性分析用）

        参数:
            cutoff_date: 数据截止日
            stock_codes: 指定股票列表

        返回:
            {
                "000001.SZ": {
                    "is_loss": False,           # 是否亏损
                    "net_profit": 12345678900,  # 净利润
                    "roe": 12.5,               # ROE
                    "debt_ratio": 65.3,         # 资产负债率
                    "is_high_debt": False,     # 是否高负债
                    "audit_opinion": "标准无保留意见",
                    "has_financial_risk": False # 是否有财务风险
                }
            }
        """
        # 获取最近一期年报财务指标（传入 stock_codes）
        raw_financials = self.fetch_financials(cutoff_date, stock_codes=stock_codes)

        # 获取审计意见
        audit_opinions = self._fetch_audit_opinions(cutoff_date)

        result = {}
        for item in raw_financials:
            ts_code = item.get("ts_code", "")

            net_profit = item.get("net_profit", 0) or 0
            roe = item.get("roe", 0) or 0
            debt_ratio = item.get("debt_to_assets", 0) or 0

            # 判断异常
            is_loss = net_profit < 0
            is_high_debt = debt_ratio > 70

            # 审计意见
            audit = audit_opinions.get(ts_code, "未知")
            has_audit_issue = audit not in ["标准无保留意见", "未知"]

            # 综合财务风险
            has_financial_risk = is_loss or is_high_debt or has_audit_issue

            result[ts_code] = {
                "is_loss": is_loss,
                "net_profit": net_profit,
                "roe": round(roe, 2) if roe else 0,
                "debt_ratio": round(debt_ratio, 2) if debt_ratio else 0,
                "is_high_debt": is_high_debt,
                "audit_opinion": audit,
                "has_audit_issue": has_audit_issue,
                "has_financial_risk": has_financial_risk
            }

        return result

    def _fetch_audit_opinions(self, cutoff_date: str) -> Dict[str, str]:
        """
        获取审计意见

        年份逻辑与 fetch_financials 保持一致：
        - cutoff_date在1-3月: 前年年报 (cutoff年 - 2)
        - cutoff_date在4月及以后: 去年年报 (cutoff年 - 1)

        返回:
            {"000001.SZ": "标准无保留意见", ...}
        """
        year = int(cutoff_date[:4])
        if int(cutoff_date[5:7]) < 4:
            year -= 2  # 1-3月: 前年年报
        else:
            year -= 1  # 4月及以后: 去年年报

        result = {}
        try:
            df = self.api.fina_audit(
                period=f"{year}1231"
            )
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    ts_code = row.get("ts_code", "")
                    audit_result = row.get("audit_result", "未知")
                    result[ts_code] = audit_result
        except Exception:
            pass

        return result

    def fetch_revenue(self, stock_code: str, year: int) -> dict:
        """获取指定股票的营业收入数据"""
        try:
            df = self.api.fina_indicator(
                ts_code=stock_code,
                period=f"{year}1231"
            )
            if df is not None and not df.empty:
                row = df.iloc[0]
                return {
                    "revenue": row.get("revenue", 0),
                    "revenue_yoy": row.get("revenue_yoy", 0)
                }
        except Exception:
            pass
        return {}

    def fetch_roe(self, stock_code: str, year: int) -> float:
        """获取指定股票的ROE"""
        try:
            df = self.api.fina_indicator(
                ts_code=stock_code,
                period=f"{year}1231"
            )
            if df is not None and not df.empty:
                return float(df.iloc[0].get("roe", 0) or 0)
        except Exception:
            pass
        return 0.0
