"""
港股日K线共享获取器（进程级缓存）

背景：
    mode_b 的三个数据采集模块——universe_builder(选股范畴)、
    turnover_tester(换手率)、month_end_market_cap(月末市值)——
    在同一个预测流程中都会调用 akshare 的 stock_hk_daily 拉取日K线。

    它们对同一批股票（尤其是 A+H 股）会重复拉取 2~3 次，
    而 stock_hk_daily 是较慢的新浪接口，重复请求造成大量时间浪费。

    本模块提供一个进程级共享缓存，同一只股票只请求一次，
    后续模块直接命中缓存，从"重复拉取"变成"拉一次、多处复用"。

线程安全：
    - stock_hk_daily 底层依赖 py_mini_racer（JS 引擎），非线程安全，
      多线程并发调用会崩溃，因此真正发起网络请求时加锁串行化。
    - 缓存读（dict 的 get/in）在 CPython 下是原子的，无需加锁。

用法：
    from mode_b.data.hk_daily_fetcher import fetch_daily
    df = fetch_daily("00700")   # 返回原始日K DataFrame（未清洗），或 None

    调用方拿到 df 后应自行 copy 再做字段清洗（加列/转类型/dropna），
    避免污染共享缓存、影响其他模块。
"""

import threading
from typing import Optional

import pandas as pd

# 进程级缓存：code -> 原始日K DataFrame（None 表示拉取失败，同样缓存以避免重试）
_daily_cache: dict = {}

# 网络请求锁：串行化 stock_hk_daily 调用，避免 py_mini_racer 并发崩溃
_fetch_lock = threading.Lock()

# 惰性加载 akshare，避免 import 本模块时即触发重依赖
_ak = None


def _get_ak():
    global _ak
    if _ak is None:
        import akshare as ak
        _ak = ak
    return _ak


def fetch_daily(code: str) -> Optional[pd.DataFrame]:
    """
    获取指定股票的日K线（akshare 原始返回，未做字段清洗）。

    返回:
        pandas.DataFrame 或 None（拉取失败 / 无数据）。
        调用方需自行处理 date/close/volume/amount 等字段，并 copy 后再修改。
    """
    code = str(code).strip().zfill(5)
    if code in _daily_cache:
        return _daily_cache[code]

    with _fetch_lock:
        # double-check：拿到锁后再确认一次，避免多线程下重复请求
        if code in _daily_cache:
            return _daily_cache[code]
        try:
            df = _get_ak().stock_hk_daily(symbol=code, adjust="")
        except Exception:
            df = None
        # 空 DataFrame 也视为无数据
        if df is not None and getattr(df, "empty", False):
            df = None
        _daily_cache[code] = df
        return df


def clear():
    """清空缓存（测试或需要强制刷新时使用）"""
    _daily_cache.clear()


def cache_size() -> int:
    """当前缓存条数（含缓存为 None 的失败项）"""
    return len(_daily_cache)
