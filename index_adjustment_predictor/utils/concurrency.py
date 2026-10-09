"""
并发工具

为数据采集阶段的网络 I/O 密集型循环提供线程池并发能力。
所有串行 API 循环（逐交易日/逐股票查询）均可通过 run_concurrently
安全地并发执行，缩短数据采集耗时。

线程安全说明:
    Tushare pro_api 客户端与 AKShare 函数式接口均无共享可变状态，
    多个线程共享同一客户端是安全的。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List


def run_concurrently(
    func: Callable[[Any], Any],
    items: Iterable,
    max_workers: int = 8,
    progress_every: int = 50,
    desc: str = "",
) -> List[Any]:
    """
    对 items 并发执行 func(item)，返回与 items 同序的结果列表。

    单个任务抛异常时，对应位置返回 None（由调用方决定是否过滤）。
    max_workers <= 1 时退化为顺序执行，保持原行为。

    参数:
        func: 单个任务函数，接收一个 item 返回结果
        items: 待处理的可迭代对象
        max_workers: 并发线程数
        progress_every: 每完成多少个任务打印一次进度，0 表示不打印
        desc: 进度描述文本

    返回:
        与 items 同序的结果列表，失败项为 None
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return []

    # 单线程时保持原有顺序执行逻辑
    if max_workers <= 1:
        results = []
        for i, item in enumerate(items):
            try:
                results.append(func(item))
            except Exception:
                results.append(None)
            if progress_every and (i + 1) % progress_every == 0 and desc:
                print(f"    进度: {i + 1}/{n} {desc}")
        return results

    results: List[Any] = [None] * n
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(func, item): i for i, item in enumerate(items)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None
            done += 1
            if progress_every and done % progress_every == 0 and desc:
                print(f"    进度: {done}/{n} {desc}")

    return results
