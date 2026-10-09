"""
换手率测试模块

实现恒生指数编算总则 5.6.2 的基准指数换手率测试:
    - 月换手率 = 日成交量中位数 / 月底流通调整后发行股份数
    - 月换手率 ≥ 0.05% → 该月通过
    - Fallback: 未通过月份, 该月总成交额排名全市场累计前90%以内 → 视为通过
    - 最近12个月至少10个月通过
    - 最近6个月至少5个月通过
    - 上市<6个月须所有月份通过; 上市≥6个月最多1个月未通过

近似说明:
    - 总股本 ≈ 12个月平均市值 / 12个月平均收盘价 (无流通系数,保守估计)
    - 用总股本代替流通调整后股本会低估换手率(总股本≥流通股本),
      导致部分实际通过的股票被判为未通过(假阴性),但不会产生假阳性
    - Fallback测试可部分弥补此问题: 高成交额股票即使换手率不达标也能通过

数据源:
    - AKShare stock_hk_daily (新浪): 日K线含volume和amount字段
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

# 换手率门槛: 0.05%
TURNOVER_THRESHOLD = 0.0005

# 通过条件
MIN_PASS_MONTHS_12M = 10   # 12个月至少10个月通过
MIN_PASS_MONTHS_6M = 5     # 6个月至少5个月通过

# Fallback: 全市场累计成交额前90%以内视为通过
FALLBACK_CUMULATIVE_THRESHOLD = 0.90


class TurnoverTester:
    """
    换手率测试器

    用法:
        tester = TurnoverTester(cutoff_date=date(2025, 12, 31))
        passed, details = tester.test_stock("00700", avg_market_cap=3.5e12)
        results = tester.test_batch(stocks)
    """

    def __init__(self, cutoff_date: date, verbose: bool = True):
        self.cutoff = cutoff_date
        self.verbose = verbose
        self.start_date = cutoff_date - pd.DateOffset(months=12)  # 12个月, 与market_cap口径一致
        self._cache: Dict[str, pd.DataFrame] = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [换手率] {msg}")

    def test_stock(self, code: str, avg_market_cap: float) -> Tuple[bool, Dict]:
        """
        测试单只股票的换手率 (不含fallback, 仅基于0.05%门槛)

        参数:
            code: 股票代码 (5位)
            avg_market_cap: 12个月平均市值 (港币元)

        返回:
            (passed, details)
            passed: 是否通过换手率测试 (不含fallback)
            details: {total_months, passed_months, monthly_details, ...}
        """
        # 1. 获取日K数据
        daily = self._fetch_daily(code)
        if daily is None or daily.empty:
            return False, {"reason": "无日K数据", "code": code}

        # 2. 过滤cutoff前12个月 (date→Timestamp, 避免类型不匹配)
        start_ts = pd.Timestamp(self.start_date)
        cutoff_ts = pd.Timestamp(self.cutoff)
        daily = daily[(daily['date'] >= start_ts) & (daily['date'] <= cutoff_ts)]
        if daily.empty:
            return False, {"reason": "cutoff前无数据", "code": code}

        # 3. 计算总股本 ≈ avg_market_cap / avg_close
        avg_close = daily['close'].mean()
        if avg_close <= 0 or avg_market_cap <= 0:
            return False, {"reason": "无法计算股本", "code": code}

        total_shares = avg_market_cap / avg_close

        # 4. 按月分组,计算月换手率和月总成交额
        daily = daily.copy()
        daily['month'] = daily['date'].dt.to_period('M')

        monthly_results = []
        for month, group in daily.groupby('month'):
            median_vol = group['volume'].median()
            total_amount = group['amount'].sum() if 'amount' in group.columns else 0
            turnover = median_vol / total_shares if total_shares > 0 else 0
            passed = turnover >= TURNOVER_THRESHOLD
            monthly_results.append({
                'month': str(month),
                'median_volume': median_vol,
                'total_amount': total_amount,
                'turnover': turnover,
                'turnover_pct': f"{turnover * 100:.3f}%",
                'passed': passed,
                'passed_via_fallback': False,  # 初始为False, 由test_batch填充
            })

        total_months = len(monthly_results)
        passed_months = sum(1 for m in monthly_results if m['passed'])

        # 5. 检查通过条件 (编算总则 5.6.2, 不含fallback)
        if total_months >= 12:
            pass_12m = passed_months >= MIN_PASS_MONTHS_12M
        elif total_months >= 6:
            pass_12m = passed_months >= total_months - 1
        else:
            pass_12m = passed_months >= total_months

        # 最近6个月至少5个月通过
        last_6 = monthly_results[-6:] if total_months >= 6 else monthly_results
        passed_6m_count = sum(1 for m in last_6 if m['passed'])
        pass_6m = passed_6m_count >= min(MIN_PASS_MONTHS_6M, len(last_6))

        passed = pass_12m and pass_6m

        return passed, {
            'code': code,
            'total_shares': total_shares,
            'avg_close': avg_close,
            'total_months': total_months,
            'passed_months': passed_months,
            'passed_6m': passed_6m_count,
            'pass_12m': pass_12m,
            'pass_6m': pass_6m,
            'monthly_details': monthly_results,
        }

    def test_batch(
        self,
        stocks: List[Dict],
        max_workers: int = 8,
    ) -> Dict[str, Dict]:
        """
        批量测试换手率, 含fallback测试

        Fallback (编算总则5.6.2):
            未通过0.05%月换手率门槛的月份, 若该月总成交额排名全市场累计前90%以内,
            视为通过。全市场指HSCI选股范围内的所有证券。

        参数:
            stocks: [{code, avg_market_cap, name}, ...]
            max_workers: 并发数 (仅用于计算阶段, 数据获取始终单线程)

        返回:
            {code: {passed, details}, ...}
        """
        results = {}
        total = len(stocks)
        passed_count = 0
        failed_count = 0
        error_count = 0

        self._log(f"开始测试 {total} 只股票的换手率 (含fallback)...")

        # === Phase 0: 单线程预取日K数据 (避免py_mini_racer多线程崩溃) ===
        self._log(f"Phase 0: 单线程预取日K数据 ({total} 只)...")
        try:
            from tqdm import tqdm
            iterator = tqdm(stocks, desc="  [换手率] 预取K线", unit="只")
        except ImportError:
            iterator = stocks

        fetch_start = time.time()
        for stock in iterator:
            code = stock['code']
            self._fetch_daily(code)
        fetch_elapsed = time.time() - fetch_start
        self._log(f"Phase 0 完成: 预取 {len(self._cache)} 只, 耗时{fetch_elapsed:.0f}s")

        def _test_one(stock):
            code = stock['code']
            mc = stock.get('avg_market_cap', 0)
            try:
                return code, self.test_stock(code, mc)
            except Exception as e:
                logger.debug(f"  {code} 换手率测试异常: {e}")
                return code, (False, {"reason": f"异常: {e}", "code": code})

        start_time = time.time()

        # === Phase 1: 逐股测试0.05%门槛 ===
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_test_one, s): s for s in stocks}

            for i, future in enumerate(as_completed(futures)):
                code, (passed, details) = future.result()
                results[code] = {'passed': passed, 'details': details}

                if passed:
                    passed_count += 1
                elif details.get('reason', '').startswith(('无', 'cutoff', '无法', '异常')):
                    error_count += 1
                else:
                    failed_count += 1

                if (i + 1) % 200 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (total - i - 1) / rate
                    self._log(
                        f"  Phase 1 (0.05%门槛): {i+1}/{total} "
                        f"(通过{passed_count} 未通过{failed_count} 错误{error_count}) "
                        f"速率: {rate:.1f}/s 剩余: {remaining:.0f}s"
                    )

        phase1_passed = passed_count
        phase1_failed = failed_count
        elapsed_p1 = time.time() - start_time
        self._log(
            f"Phase 1 完成: 通过{passed_count} 未通过{failed_count} 错误{error_count}, "
            f"耗时{elapsed_p1:.0f}s"
        )

        # === Phase 2: Fallback测试 ===
        # 对未通过0.05%门槛的月份, 检查该月总成交额是否在全市场累计前90%以内
        results = self._apply_fallback(results)

        # 重新统计
        final_passed = sum(1 for r in results.values() if r['passed'])
        fallback_rescued = sum(
            1 for r in results.values()
            if r['passed'] and any(
                m.get('passed_via_fallback', False)
                for m in r['details'].get('monthly_details', [])
            )
        )
        final_failed = total - final_passed - error_count

        elapsed_total = time.time() - start_time
        self._log(
            f"Phase 2 (Fallback) 完成: 通过{final_passed} "
            f"(其中fallback救回{fallback_rescued}) 未通过{final_failed} 错误{error_count}, "
            f"总耗时{elapsed_total:.0f}s"
        )

        return results

    def _apply_fallback(self, results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        应用fallback测试 (编算总则5.6.2)

        对未通过0.05%门槛的月份, 若该月总成交额排名全市场累计前90%以内, 视为通过。
        """
        # 1. 收集所有股票每个月的总成交额
        # {month_str: [(code, total_amount), ...]}
        monthly_amounts: Dict[str, List[Tuple[str, float]]] = {}

        for code, item in results.items():
            details = item.get('details', {})
            monthly = details.get('monthly_details', [])
            if not monthly:
                continue
            for m in monthly:
                month_str = m['month']
                amount = m.get('total_amount', 0)
                if month_str not in monthly_amounts:
                    monthly_amounts[month_str] = []
                monthly_amounts[month_str].append((code, amount))

        if not monthly_amounts:
            return results

        self._log(
            f"  Fallback: 收集了 {len(monthly_amounts)} 个月的成交额数据, "
            f"覆盖 {len(results)} 只股票"
        )

        # 2. 对每个月, 计算全市场累计成交额前90%的门槛
        # {month_str: set(codes_in_top_90%)}
        monthly_top90: Dict[str, set] = {}

        for month_str, code_amounts in monthly_amounts.items():
            # 按成交额降序排列
            sorted_ca = sorted(code_amounts, key=lambda x: x[1], reverse=True)
            total_amount = sum(a for _, a in sorted_ca)
            if total_amount <= 0:
                continue

            # 累计前90%的股票集合
            cumulative = 0
            top90_codes = set()
            for code, amount in sorted_ca:
                cumulative += amount
                top90_codes.add(code)
                if cumulative / total_amount >= FALLBACK_CUMULATIVE_THRESHOLD:
                    break

            monthly_top90[month_str] = top90_codes

        # 3. 对未通过0.05%门槛的月份, 应用fallback
        fallback_rescued_count = 0

        for code, item in results.items():
            details = item.get('details', {})
            monthly = details.get('monthly_details', [])
            if not monthly:
                continue

            for m in monthly:
                if not m['passed']:
                    # 未通过0.05%门槛, 检查fallback
                    month_str = m['month']
                    top90 = monthly_top90.get(month_str, set())
                    if code in top90:
                        m['passed'] = True
                        m['passed_via_fallback'] = True
                        fallback_rescued_count += 1

            # 重新计算通过条件
            total_months = len(monthly)
            passed_months = sum(1 for m in monthly if m['passed'])

            if total_months >= 12:
                pass_12m = passed_months >= MIN_PASS_MONTHS_12M
            elif total_months >= 6:
                pass_12m = passed_months >= total_months - 1
            else:
                pass_12m = passed_months >= total_months

            last_6 = monthly[-6:] if total_months >= 6 else monthly
            passed_6m_count = sum(1 for m in last_6 if m['passed'])
            pass_6m = passed_6m_count >= min(MIN_PASS_MONTHS_6M, len(last_6))

            old_passed = item['passed']
            new_passed = pass_12m and pass_6m

            item['passed'] = new_passed
            details['passed_months'] = passed_months
            details['passed_6m'] = passed_6m_count
            details['pass_12m'] = pass_12m
            details['pass_6m'] = pass_6m

            if not old_passed and new_passed:
                # 这只股票被fallback救回
                pass

        self._log(
            f"  Fallback: 救回 {fallback_rescued_count} 个月份, "
            f"涉及 {sum(1 for r in results.values() if r['passed'] and any(m.get('passed_via_fallback') for m in r.get('details',{}).get('monthly_details',[])))} 只股票"
        )

        return results

    def _fetch_daily(self, code: str) -> Optional[pd.DataFrame]:
        """获取日K数据 (带缓存)"""
        if code in self._cache:
            return self._cache[code]

        df = fetch_daily(code)
        if df is not None and not df.empty:
            df = df.copy()  # copy 后再清洗，避免污染共享缓存
            df['date'] = pd.to_datetime(df['date'])
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            else:
                df['amount'] = 0
            df = df.dropna(subset=['volume', 'close'])
            self._cache[code] = df
            return df

        return None

    def save_results(self, results: Dict, output_file: str):
        """保存测试结果"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 精简结果 (去掉monthly_details的冗余)
        slim = {}
        for code, item in results.items():
            d = item['details']
            slim[code] = {
                'passed': item['passed'],
                'total_months': d.get('total_months', 0),
                'passed_months': d.get('passed_months', 0),
                'passed_6m': d.get('passed_6m', 0),
                'avg_turnover_pct': None,
                'min_turnover_pct': None,
                'fallback_rescued_months': 0,
            }
            # 计算平均和最小换手率, 统计fallback救回的月份数
            monthly = d.get('monthly_details', [])
            if monthly:
                turnovers = [m['turnover'] for m in monthly]
                slim[code]['avg_turnover_pct'] = f"{sum(turnovers)/len(turnovers)*100:.3f}%"
                slim[code]['min_turnover_pct'] = f"{min(turnovers)*100:.3f}%"
                slim[code]['fallback_rescued_months'] = sum(
                    1 for m in monthly if m.get('passed_via_fallback', False)
                )

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(slim, f, ensure_ascii=False, indent=2)
        self._log(f"结果已保存: {output_file}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="换手率测试")
    parser.add_argument("--cutoff", required=True, help="数据截止日 (YYYY-MM-DD)")
    parser.add_argument("--input", default=None, help="输入市值数据JSON文件")
    args = parser.parse_args()

    cutoff = pd.to_datetime(args.cutoff).date()

    # 加载市值数据
    input_file = args.input or os.path.join(
        PROJECT_ROOT, "mode_b", "data",
        f"universe_market_cap_{cutoff.strftime('%Y%m%d')}.json"
    )

    with open(input_file, 'r', encoding='utf-8') as f:
        universe = json.load(f)

    # 过滤HDR和未上市
    stocks = []
    for s in universe:
        code = s['code']
        if code.startswith('8') and len(code) == 5:
            continue
        mc = s.get('avg_market_cap', 0)
        if mc > 0:
            stocks.append({'code': code, 'avg_market_cap': mc, 'name': s.get('name', '')})

    print(f"待测试: {len(stocks)} 只")

    tester = TurnoverTester(cutoff_date=cutoff)
    results = tester.test_batch(stocks)

    output_file = os.path.join(
        PROJECT_ROOT, "mode_b", "data",
        f"turnover_test_{cutoff.strftime('%Y%m%d')}.json"
    )
    tester.save_results(results, output_file)

    # 打印摘要
    passed = sum(1 for r in results.values() if r['passed'])
    failed = sum(1 for r in results.values() if not r['passed'])
    print(f"\n通过: {passed}, 未通过: {failed}")
