"""
选股范畴构建器

为HSCI覆盖率模型构建cutoff日期时点的全量选股范畴数据。

核心功能:
    1. 获取全量港股代码列表 (stock_hk_spot)
    2. 按cutoff日期过滤未上市股票 (stock_hk_daily首日日期)
    3. 纯港股: 百度估值接口获取12个月日均市值 (总市值=H股市值)
    4. A+H股: H股股本(stock_hk_security_profile_em) × 日线收盘价(stock_hk_daily)

数据源:
    - AKShare stock_hk_spot (新浪): 全量港股代码列表
    - AKShare stock_hk_valuation_baidu (百度): 历史日度市值采样
    - AKShare stock_hk_daily (新浪): 日K线历史数据
    - AKShare stock_hk_security_profile_em (东财): H股股本和上市日期
    - ah_stock_mapping.json: A+H股映射清单

输出格式:
    [{code, name, avg_market_cap, source, listing_date}, ...]
    source: "baidu_12m_avg" (纯港股百度接口)
            "h_share_calc" (A+H股H股股本×价格)
"""

import os
import sys
import re
import json
import time
import socket
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# 添加项目根目录
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

BAIDU_MARKET_CAP_UNIT = 100_000_000  # 百度返回值单位: 亿港币

# ISIN前缀 → 外国公司判定
# 编算总则5.3: 选股范畴包括"主要业务位于大中华区的第一及第二上市公司"和
#              "第一上市一股一票的外国公司"
# → 外国公司第二上市不可纳入HSCI选股范畴
FOREIGN_ISIN_PREFIXES = {
    'CA',   # Canada (加拿大)  e.g. 00945 宏利金融
    'US',   # USA (美国)
    'GB',   # UK (英国)
    'FR',   # France (法国)
    'DE',   # Germany (德国)
    'JP',   # Japan (日本)
    'AU',   # Australia (澳大利亚)
    'SG',   # Singapore (新加坡)
    'KR',   # Korea (韩国)
    'IN',   # India (印度)
}


def detect_secondary_listing(name: str) -> bool:
    """
    检测第二上市后缀 (-S/-SW/-SS, 兼容全角和半角)

    港股第二上市股票名称结尾标记:
        -S  / -SS  : 第二上市 (SS = 第二上市+合订证券)
        -SW        : 第二上市+WVR
    全角: －(U+FF0D) Ｓ(U+FF33) Ｗ(U+FF37)
    半角: -(U+002D) S(U+0053) W(U+0057)
    """
    if not name:
        return False
    name = name.strip()
    patterns = [
        r'[-－]\s*[SＳ][WＷ]$',      # -SW
        r'[-－]\s*[SＳ][SＳ]$',       # -SS
        r'[-－]\s*[SＳ]$',            # -S
    ]
    return any(re.search(p, name) for p in patterns)


def is_foreign_by_isin(isin: str) -> bool:
    """通过ISIN前缀判断是否外国公司"""
    if not isin or len(isin) < 2:
        return False
    return isin[:2] in FOREIGN_ISIN_PREFIXES


class UniverseBuilder:
    """
    选股范畴构建器

    用法:
        builder = UniverseBuilder(cutoff_date=date(2025, 12, 31))
        data = builder.build(output_file="universe_market_cap_20251231.json")
    """

    def __init__(self, cutoff_date: date, verbose: bool = True):
        self.cutoff = cutoff_date
        self.verbose = verbose
        self.cutoff_str = cutoff_date.strftime("%Y%m%d")
        self.start_date = cutoff_date - timedelta(days=400)  # 12个月+余量

        # A+H映射: {H股代码(5位): A股代码}
        self.ah_codes: Set[str] = self._load_ah_mapping()

        # H股股本+上市日期缓存
        self._h_share_cache: Dict[str, Dict] = {}
        # 百度接口降级缓存
        self._baidu_failed: Set[str] = set()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [选股范畴] {msg}")

    def _load_ah_mapping(self) -> Set[str]:
        """加载A+H股映射，返回H股代码集合"""
        # 优先使用mode_b/data/下的本地副本 (不依赖index_adjustment_predictor)
        mapping_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ah_stock_mapping.json"
        )
        ah_codes = set()
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for a_code, info in data.get("mapping", {}).items():
                hk_code = info.get("hk_code", "").strip().zfill(5)
                if hk_code:
                    ah_codes.add(hk_code)
            self._log(f"加载A+H映射: {len(ah_codes)} 只")
        except Exception as e:
            logger.warning(f"加载A+H映射失败: {e}")
        return ah_codes

    # ================================================================
    # 主流程
    # ================================================================

    def build(self, output_file: str = None) -> List[Dict]:
        """
        构建cutoff日期的选股范畴

        参数:
            output_file: 输出文件路径

        返回:
            [{code, name, avg_market_cap, source, listing_date}, ...]
        """
        if not ak:
            raise RuntimeError("AKShare未安装")

        # 1. 获取全量港股代码
        all_stocks = self._get_all_hk_stocks()
        self._log(f"全量港股: {len(all_stocks)} 只")

        # 2. 分类
        ah_stocks = [s for s in all_stocks if s["code"] in self.ah_codes]
        pure_hk_stocks = [s for s in all_stocks if s["code"] not in self.ah_codes]
        self._log(f"A+H股: {len(ah_stocks)} 只, 纯港股: {len(pure_hk_stocks)} 只")

        results = []
        failed = []

        # 3. 采集纯港股市值 (百度接口, 多线程)
        self._log(f"=== 采集纯港股市值 ({len(pure_hk_stocks)} 只) ===")
        pure_results, pure_failed = self._collect_pure_hk(pure_hk_stocks)
        results.extend(pure_results)
        failed.extend(pure_failed)

        # 4. 采集A+H股市值 (H股股本 × 日线价格)
        self._log(f"=== 采集A+H股市值 ({len(ah_stocks)} 只) ===")
        ah_results, ah_failed = self._collect_ah_stocks(ah_stocks)
        results.extend(ah_results)
        failed.extend(ah_failed)

        # 5. 过滤外国公司第二上市 (编算总则5.3)
        results = self._filter_foreign_secondary(results)
        
        # 6. 统计
        total_mc = sum(r["avg_market_cap"] for r in results if r["avg_market_cap"] > 0)
        self._log(f"\n=== 采集完成 ===")
        self._log(f"成功: {len(results)} 只, 失败: {len(failed)} 只")
        self._log(f"总市值: {total_mc / 1e8:.0f} 亿港币")

        source_dist = {}
        for r in results:
            src = r.get("source", "unknown")
            source_dist[src] = source_dist.get(src, 0) + 1
        self._log(f"来源分布: {source_dist}")

        # 7. 保存
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results_sorted = sorted(results, key=lambda x: x["code"])
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results_sorted, f, ensure_ascii=False, indent=2)
            self._log(f"结果已保存: {output_file}")

        return results

    # ================================================================
    # 获取全量港股代码
    # ================================================================

    def _get_all_hk_stocks(self) -> List[Dict]:
        """
        获取全量港股代码列表

        过滤规则:
        - 排除8开头的5位代码 (预托证券HDR, 如80700=腾讯-R)
          原因: 编算总则6.7规定同公司只纳入成交额较高的证券,
          实践中原始代码(0开头)成交额远高于HDR(8开头)
        """
        df = ak.stock_hk_spot()
        stocks = []
        filtered_hdr = 0
        for _, row in df.iterrows():
            code = str(row["代码"]).strip().zfill(5)
            name = str(row["中文名称"]).strip()
            if not code or not code[0].isdigit():
                continue
            # 过滤预托证券(HDR): 8开头5位代码是原始股票的衍生品
            if code.startswith("8") and len(code) == 5:
                filtered_hdr += 1
                continue
            stocks.append({"code": code, "name": name})
        if filtered_hdr:
            self._log(f"过滤HDR预托证券: {filtered_hdr} 只")
        return stocks

    # ================================================================
    # 外国公司第二上市过滤
    # ================================================================

    def _filter_foreign_secondary(self, results: List[Dict]) -> List[Dict]:
        """
        过滤外国公司第二上市 (编算总则5.3)

        规则: HSCI选股范畴包括"主要业务位于大中华区的第一及第二上市公司"
              和"第一上市一股一票的外国公司"
              → 外国公司第二上市不可纳入

        检测方法:
            1. 通过名称检测-S/-SW/-SS后缀 (第二上市标记)
            2. 对-S股票查ISIN, 若ISIN前缀为CA/US/GB等 → 外国公司
            3. 大中华区第二上市(如京东-SW, 百度-SW)不受影响
        """
        # 找出-S/-SW/-SS后缀的股票
        secondary_stocks = [
            (i, r) for i, r in enumerate(results)
            if detect_secondary_listing(r.get("name", ""))
        ]

        if not secondary_stocks:
            return results

        # 对每只-S股票查ISIN
        foreign_codes = set()
        self._log(f"检测到 {len(secondary_stocks)} 只-S/-SW股票, 查询ISIN...")

        for _, stock in secondary_stocks:
            code = stock["code"]
            isin = self._get_isin(code)
            if isin and is_foreign_by_isin(isin):
                foreign_codes.add(code)
                self._log(f"  排除: {code} {stock.get('name', '')} ISIN={isin} (外国公司第二上市)")

        if not foreign_codes:
            self._log(f"  无外国公司第二上市需排除")
            return results

        # 过滤
        filtered = [r for r in results if r["code"] not in foreign_codes]
        self._log(f"  过滤外国公司第二上市: {len(foreign_codes)} 只, 剩余 {len(filtered)} 只")
        return filtered

    def _get_isin(self, code: str) -> Optional[str]:
        """获取股票ISIN代码"""
        try:
            df = ak.stock_hk_security_profile_em(symbol=code)
            if df is not None and not df.empty:
                isin = str(df.iloc[0].get("ISIN（国际证券识别编码）", ""))
                if isin and isin != "nan":
                    return isin
        except Exception as e:
            logger.debug(f"  {code} ISIN获取失败: {e}")
        return None

    # ================================================================
    # 纯港股: 百度估值接口
    # ================================================================

    def _collect_pure_hk(
        self, stocks: List[Dict], max_workers: int = 5
    ) -> Tuple[List[Dict], List[str]]:
        """采集纯港股的12个月日均市值 (百度接口)"""

        # 断点续采
        results = []
        failed = []
        done_codes = set()

        partial_file = None
        if hasattr(self, "_output_file"):
            partial_file = self._output_file + ".pure_partial"
            if os.path.exists(partial_file):
                try:
                    with open(partial_file, "r", encoding="utf-8") as f:
                        partial = json.load(f)
                        for item in partial:
                            results.append(item)
                            done_codes.add(item["code"])
                    self._log(f"断点续采: 已有 {len(done_codes)} 只")
                except Exception:
                    pass

        remaining = [s for s in stocks if s["code"] not in done_codes]
        self._log(f"待采集: {len(remaining)} 只")

        if not remaining:
            return results, failed

        start_time = time.time()

        def _fetch_one(stock):
            code = stock["code"]
            name = stock["name"]
            try:
                mc = self._fetch_baidu_avg_mc(code)
                if mc and mc > 0:
                    return {
                        "code": code,
                        "name": name,
                        "avg_market_cap": mc,
                        "source": "baidu_12m_avg",
                        "listing_date": None,
                    }
            except Exception as e:
                logger.debug(f"  {code} 采集失败: {e}")
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, s): s for s in remaining}

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed.append(futures[future]["code"])

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining_time = (len(remaining) - i - 1) / rate
                    self._log(
                        f"  进度: {i+1}/{len(remaining)} "
                        f"(成功 {len(results)}, 失败 {len(failed)}) "
                        f"速率: {rate:.1f}/s 预计剩余: {remaining_time:.0f}s"
                    )
                    # 保存中间结果
                    if partial_file:
                        self._save_partial(partial_file, results)

        elapsed = time.time() - start_time
        self._log(f"纯港股采集完成: 成功 {len(results)}, 失败 {len(failed)}, 耗时 {elapsed:.0f}s")

        # 清理partial
        if partial_file and os.path.exists(partial_file):
            os.remove(partial_file)

        return results, failed

    def _fetch_baidu_avg_mc(self, code: str, max_retries: int = 2) -> Optional[float]:
        """
        百度估值接口获取cutoff前12个月平均市值

        period="全部" 返回上市至今的历史市值采样 (约每13天一个点)
        过滤出cutoff前12个月区间取平均
        如果cutoff前12个月无数据 → 未上市, 返回None
        """
        for attempt in range(max_retries + 1):
            try:
                df = ak.stock_hk_valuation_baidu(
                    symbol=code, indicator="总市值", period="全部"
                )
                if df is not None and not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    cutoff_ts = pd.Timestamp(self.cutoff)
                    start_ts = cutoff_ts - pd.DateOffset(months=12)

                    mask = (df["date"] <= cutoff_ts) & (df["date"] >= start_ts)
                    filtered = df.loc[mask, "value"]

                    if filtered.empty:
                        # cutoff前12个月无数据 → 可能未上市或上市不足12个月
                        # 取cutoff之前所有可用数据
                        before_cutoff = df.loc[df["date"] <= cutoff_ts, "value"]
                        if before_cutoff.empty:
                            return None  # cutoff前完全无数据 → 未上市
                        if len(before_cutoff) < 3:
                            return None  # 数据太少
                        filtered = before_cutoff

                    avg_mc_yi = float(filtered.mean())
                    return avg_mc_yi * BAIDU_MARKET_CAP_UNIT
            except Exception as e:
                logger.debug(f"  百度估值 {code} 第{attempt+1}次失败: {e}")

            if attempt < max_retries:
                time.sleep(0.5)

        return None

    # ================================================================
    # A+H股: H股股本 × 日线收盘价
    # ================================================================

    def _collect_ah_stocks(
        self, stocks: List[Dict]
    ) -> Tuple[List[Dict], List[str]]:
        """采集A+H股的12个月H股市值 (H股股本 × 日线收盘价)"""
        results = []
        failed = []
        total = len(stocks)

        for i, stock in enumerate(stocks):
            code = stock["code"]
            name = stock["name"]

            if (i + 1) % 20 == 0:
                self._log(f"  A+H进度: {i+1}/{total} (成功 {len(results)}, 失败 {len(failed)})")

            try:
                result = self._fetch_ah_market_cap(code, name)
                if result:
                    results.append(result)
                else:
                    failed.append(code)
            except Exception as e:
                logger.debug(f"  {code} A+H采集失败: {e}")
                failed.append(code)

            time.sleep(0.15)

        self._log(f"A+H股采集完成: 成功 {len(results)}, 失败 {len(failed)}")
        return results, failed

    def _fetch_ah_market_cap(self, code: str, name: str) -> Optional[Dict]:
        """
        获取单只A+H股的12个月平均H股市值

        步骤:
            1. 查 stock_hk_security_profile_em 获取H股股本和上市日期
            2. 如果上市日期 > cutoff, 返回None (cutoff时未上市)
            3. 查 stock_hk_daily 获取cutoff前12个月日线收盘价
            4. H股股本 × 收盘价 = 每日H股市值, 取平均
        """
        # 1. 获取H股股本和上市日期
        h_shares, listing_date = self._get_h_shares_and_listing_date(code)
        if h_shares <= 0:
            # profile接口失败, 降级到百度接口
            self._baidu_failed.add(code)
            mc = self._fetch_baidu_avg_mc(code)
            if mc and mc > 0:
                return {
                    "code": code,
                    "name": name,
                    "avg_market_cap": mc,
                    "source": "baidu_12m_avg_fallback",
                    "listing_date": listing_date.isoformat() if listing_date else None,
                }
            return None

        # 2. 检查cutoff时是否已上市
        if listing_date and listing_date > self.cutoff:
            self._log(f"  {code} {name}: 上市日{listing_date} > cutoff, 跳过")
            return None

        # 3. 获取日线收盘价
        daily_closes = self._fetch_daily_close(code)
        if not daily_closes:
            # 日线获取失败, 降级到百度
            mc = self._fetch_baidu_avg_mc(code)
            if mc and mc > 0:
                return {
                    "code": code,
                    "name": name,
                    "avg_market_cap": mc,
                    "source": "baidu_12m_avg_fallback",
                    "listing_date": listing_date.isoformat() if listing_date else None,
                }
            return None

        # 4. 计算每日H股市值, 取12个月平均
        start_dt = self.cutoff - timedelta(days=400)
        recent_closes = [(d, p) for d, p in daily_closes if start_dt <= d <= self.cutoff]

        if not recent_closes:
            # cutoff前无数据 → 可能上市不足12个月, 取所有可用数据
            recent_closes = [(d, p) for d, p in daily_closes if d <= self.cutoff]

        if not recent_closes:
            return None

        # 如果上市不足12个月, 按HSCI规则用自上市起的日均市值
        daily_mcs = [p * h_shares for _, p in recent_closes]
        avg_mc = sum(daily_mcs) / len(daily_mcs)

        return {
            "code": code,
            "name": name,
            "avg_market_cap": avg_mc,
            "source": "h_share_calc",
            "listing_date": listing_date.isoformat() if listing_date else None,
            "h_shares": h_shares,
            "data_points": len(recent_closes),
        }

    def _get_h_shares_and_listing_date(self, code: str) -> Tuple[float, Optional[date]]:
        """
        获取H股股本和上市日期

        H股股本来自 stock_hk_financial_indicator_em:
            已发行股本-H股(股): H股股本 (准确区分H股和总股本)
        上市日期来自 stock_hk_security_profile_em:
            上市日期: H股上市日期
        """
        if code in self._h_share_cache:
            cached = self._h_share_cache[code]
            return cached["h_shares"], cached.get("listing_date")

        h_shares = 0.0
        listing_date = None

        # 1. 用财务指标接口取H股股本 (已发行股本-H股)
        try:
            df = ak.stock_hk_financial_indicator_em(symbol=code)
            if df is not None and not df.empty:
                h_val = df.iloc[0].get("已发行股本-H股(股)", 0)
                h_shares = float(h_val) if h_val else 0.0
        except Exception as e:
            logger.debug(f"  {code} 财务指标获取H股股本失败: {e}")

        # 2. 用profile接口取上市日期 (财务指标接口无此字段)
        try:
            df = ak.stock_hk_security_profile_em(symbol=code)
            if df is not None and not df.empty:
                list_date_raw = str(df.iloc[0].get("上市日期", ""))
                if list_date_raw and list_date_raw != "nan":
                    try:
                        listing_date = pd.to_datetime(list_date_raw).date()
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"  {code} profile获取上市日期失败: {e}")

        self._h_share_cache[code] = {"h_shares": h_shares, "listing_date": listing_date}
        return h_shares, listing_date

    def _fetch_daily_close(self, code: str) -> List[Tuple[date, float]]:
        """
        获取cutoff日期前的日K收盘价

        stock_hk_daily 返回完整历史日K, 第一条日期=上市日。
        通过共享缓存获取，避免与 turnover_tester / month_end_market_cap 重复拉取。
        """
        try:
            df = fetch_daily(code)
            if df is None or df.empty:
                return []

            result = []
            for _, row in df.iterrows():
                d = pd.to_datetime(row["date"]).date()
                close = float(row["close"])
                if close > 0:
                    result.append((d, close))

            return result
        except Exception as e:
            logger.debug(f"  {code} 日K获取失败: {e}")
            return []

    # ================================================================
    # 辅助
    # ================================================================

    @staticmethod
    def _save_partial(partial_file: str, results: List[Dict]):
        """保存中间结果"""
        with open(partial_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)


def main():
    """命令行入口"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="HSCI选股范畴构建器")
    parser.add_argument("--cutoff", required=True, help="数据截止日 (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="输出文件路径")
    args = parser.parse_args()

    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").date()

    output_file = args.output or os.path.join(
        PROJECT_ROOT, "mode_b", "data",
        f"universe_market_cap_{cutoff.strftime('%Y%m%d')}.json"
    )

    builder = UniverseBuilder(cutoff_date=cutoff)
    builder._output_file = output_file
    data = builder.build(output_file=output_file)

    print(f"\n{'='*60}")
    print(f"选股范畴构建完成")
    print(f"{'='*60}")
    print(f"cutoff日期: {cutoff}")
    print(f"股票总数: {len(data)}")

    # 市值排名前10
    sorted_data = sorted(data, key=lambda x: x["avg_market_cap"], reverse=True)
    print(f"\n市值排名前10:")
    for i, s in enumerate(sorted_data[:10]):
        print(f"  #{i+1} {s['code']} {s['name'][:20]:20s} "
              f"市值={s['avg_market_cap']/1e8:.1f}亿 [{s.get('source','')}]")

    # 来源分布
    source_dist = {}
    for s in data:
        src = s.get("source", "unknown")
        source_dist[src] = source_dist.get(src, 0) + 1
    print(f"\n来源分布: {source_dist}")


if __name__ == "__main__":
    main()
