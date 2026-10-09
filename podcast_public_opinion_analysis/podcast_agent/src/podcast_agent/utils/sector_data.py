"""
东财板块成分股数据获取 — 带浏览器 UA 直连，规避 akshare 新版裸请求被断连的问题

背景:
  akshare 新版 (request.py 2025/12/31) 的 request_with_retry 用裸 requests.Session().get()，
  不带 User-Agent / Referer，东财对这类请求会直接断连 (RemoteDisconnected)。
  而东财接口本身是通的，只要带上浏览器 UA + Referer 就能正常返回。

用途:
  - resolve_board_code(sector_name): 把 LLM 给出的板块名解析成东财板块代码
  - get_board_components(sector_name, top_n): 返回板块真实成分股（代码/名称/最新价/涨跌幅）
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# 东财 push2 有多个节点，主域名 push2.eastmoney.com 易被限流断连，
# 带数字的子节点通常正常。随机轮询 + 限速，降低被限流的概率。
_CLIST_HOSTS = [
    "7.push2.eastmoney.com",
    "17.push2.eastmoney.com",
    "29.push2.eastmoney.com",
    "37.push2.eastmoney.com",
    "47.push2.eastmoney.com",
    "57.push2.eastmoney.com",
    "67.push2.eastmoney.com",
    "77.push2.eastmoney.com",
    "79.push2.eastmoney.com",
    "87.push2.eastmoney.com",
]
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}
_UT = "bd1d9ddb04089700cf9c27f6f7426281"

# 板块名单缓存（名单基本稳定，本地文件缓存 7 天，内存缓存 1 小时）
_board_cache: Optional[Dict[str, str]] = None
_board_cache_time: float = 0.0
_BOARD_TTL = 3600.0
_BOARD_CACHE_FILE = Path(__file__).resolve().parents[3] / "data" / "sector_boards.json"
_BOARD_FILE_TTL = 7 * 24 * 3600.0

# 概念板块缓存（仅 t:3，供 retrieve-then-generate 检索候选用）
_concept_cache: Optional[Dict[str, str]] = None
_concept_cache_time: float = 0.0
_CONCEPT_CACHE_FILE = Path(__file__).resolve().parents[3] / "data" / "concept_boards.json"

# 概念板块 embedding 向量缓存（504 个板块名批量向量化一次，本地持久化）
_concept_emb_cache: Optional[Dict[str, List[float]]] = None
_CONCEPT_EMB_FILE = Path(__file__).resolve().parents[3] / "data" / "concept_embeddings.json"

# 成分股缓存（盘中实时，300 秒）
_comp_cache: Dict[str, List[Dict[str, Any]]] = {}
_comp_cache_time: Dict[str, float] = {}
_COMP_TTL = 300.0
# 成分股文件持久化缓存（东财限流严重，成功拿到一次就落盘，避免冷启动反复被限流）
_COMP_FILE = Path(__file__).resolve().parents[3] / "data" / "board_components.json"
_COMP_FILE_TTL = 24 * 3600.0  # 成分股一天内相对稳定，文件缓存 24 小时


# 全局请求限速：东财对短时间高频请求会断连，控制最小间隔
_last_request_time: float = 0.0
_MIN_INTERVAL = 2.0
_throttle_lock = threading.Lock()


def _throttle() -> None:
    """保证两次请求之间至少有 _MIN_INTERVAL 秒间隔（多线程安全）"""
    global _last_request_time
    with _throttle_lock:
        now = time.time()
        wait = _MIN_INTERVAL - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.time()


def _get_json(params: Dict[str, str], timeout: int = 15) -> Dict[str, Any]:
    """带限速 + 重试的直连请求（随机节点轮询），返回 JSON"""
    last_exc: Optional[Exception] = None
    for attempt in range(4):
        _throttle()
        host = random.choice(_CLIST_HOSTS)
        url = f"https://{host}/api/qt/clist/get"
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(1.0 + attempt * 1.5)  # 指数退避
    raise last_exc  # type: ignore[misc]


def _fetch_pages(params: Dict[str, str], page_size: int = 100) -> List[Dict[str, Any]]:
    """分页拉取 clist 接口全部数据；某页失败时返回已拿到的部分（容忍限流）"""
    # 东财 clist 单页 pz 上限 100，超出的请求仍只返回 100 条，
    # 若不限制会导致翻页判断错误（只拿到第一页）
    page_size = min(page_size, 100)
    all_items: List[Dict[str, Any]] = []
    pn = 1
    while pn <= 10:  # 上限 1000 只，足够覆盖任何板块
        p = dict(params)
        p["pn"] = str(pn)
        p["pz"] = str(page_size)
        try:
            data = _get_json(p)
        except Exception as e:  # noqa: BLE001
            # 东财限流偶发，降级返回已拿到的部分；用 debug 避免刷屏
            logger.debug("分页第 %d 页请求失败，返回已拿到的 %d 条: %s", pn, len(all_items), e)
            break
        d = data.get("data") or {}
        diff = d.get("diff") or []
        if not diff:
            break
        # diff 可能是 list（多条）或 dict（单条）
        all_items.extend(diff if isinstance(diff, list) else [diff])
        total = int(d.get("total") or 0)
        if total and pn * page_size >= total:
            break
        if len(all_items) < pn * page_size:  # 本页没拿满，说明到末尾了
            break
        pn += 1
    return all_items


def _load_boards() -> Dict[str, str]:
    """
    加载 概念板块(t:3) + 行业板块(t:2) 的「板块名 -> 代码」映射。

    策略: 内存缓存 → 本地文件缓存（7 天）→ 请求东财 → 回退过期文件缓存
    """
    global _board_cache, _board_cache_time
    now = time.time()
    if _board_cache is not None and now - _board_cache_time < _BOARD_TTL:
        return _board_cache

    # 读本地文件缓存（作为兜底）
    fallback: Dict[str, str] = {}
    try:
        if _BOARD_CACHE_FILE.exists():
            cached = json.loads(_BOARD_CACHE_FILE.read_text(encoding="utf-8"))
            boards = cached.get("boards", {})
            if boards and now - cached.get("_ts", 0) < _BOARD_FILE_TTL:
                _board_cache = boards
                _board_cache_time = now
                return boards
            fallback = boards
    except Exception:  # noqa: BLE001
        fallback = {}

    # 请求东财
    mapping: Dict[str, str] = {}
    for t in ("3", "2"):
        params = {
            "pn": "1", "pz": "100", "po": "1", "np": "1",
            "ut": _UT, "fltt": "2", "invt": "2", "fid": "f12",
            "fs": f"m:90 t:{t} f:!50",
            "fields": "f12,f14",
        }
        for it in _fetch_pages(params, page_size=100):
            name = it.get("f14")
            code = it.get("f12")
            if name and code:
                mapping[str(name)] = str(code)

    if mapping:
        _board_cache = mapping
        _board_cache_time = now
        # 写本地缓存（板块名单稳定，避免每次冷启动都请求东财）
        try:
            _BOARD_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _BOARD_CACHE_FILE.write_text(
                json.dumps({"_ts": now, "boards": mapping}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass
        return mapping

    # 请求失败，回退到过期文件缓存
    if fallback:
        logger.warning("板块列表请求失败，回退到本地过期缓存（%d 个板块）", len(fallback))
        _board_cache = fallback
        _board_cache_time = now
        return fallback

    _board_cache = {}
    _board_cache_time = now
    return {}


def _load_concept_boards() -> Dict[str, str]:
    """
    只加载东财概念板块 (t:3，约 504 个) 的「板块名 -> 代码」映射，带缓存。

    与 _load_boards 的区别: _load_boards 混入行业板块(t:2)，而概念板块才是
    retrieve-then-generate 里 LLM 选板块的候选池（题材/事件驱动，贴合播客内容）。
    """
    global _concept_cache, _concept_cache_time
    now = time.time()
    if _concept_cache is not None and now - _concept_cache_time < _BOARD_TTL:
        return _concept_cache

    fallback: Dict[str, str] = {}
    try:
        if _CONCEPT_CACHE_FILE.exists():
            cached = json.loads(_CONCEPT_CACHE_FILE.read_text(encoding="utf-8"))
            boards = cached.get("boards", {})
            if boards and now - cached.get("_ts", 0) < _BOARD_FILE_TTL:
                _concept_cache = boards
                _concept_cache_time = now
                return boards
            fallback = boards
    except Exception:  # noqa: BLE001
        fallback = {}

    mapping: Dict[str, str] = {}
    params = {
        "pn": "1", "pz": "100", "po": "1", "np": "1",
        "ut": _UT, "fltt": "2", "invt": "2", "fid": "f12",
        "fs": "m:90 t:3 f:!50",
        "fields": "f12,f14",
    }
    for it in _fetch_pages(params, page_size=100):
        name = it.get("f14")
        code = it.get("f12")
        if name and code:
            mapping[str(name)] = str(code)

    if mapping:
        _concept_cache = mapping
        _concept_cache_time = now
        try:
            _CONCEPT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CONCEPT_CACHE_FILE.write_text(
                json.dumps({"_ts": now, "boards": mapping}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass
        return mapping

    if fallback:
        _concept_cache = fallback
        _concept_cache_time = now
        return fallback

    _concept_cache = {}
    _concept_cache_time = now
    return {}


def _split_keywords(topic: str) -> List[str]:
    """把主题词拆成候选关键词：英文按单词拆，中文按连续串拆（长串滑窗 2-3 字）"""
    import re

    kws: List[str] = []
    # 英文单词（2 字母以上）
    kws.extend(w for w in re.findall(r"[a-zA-Z]{2,}", topic))
    # 中文连续串
    for seg in re.findall(r"[\u4e00-\u9fff]+", topic):
        if len(seg) <= 4:
            kws.append(seg)
        else:
            for n in (2, 3):
                for i in range(len(seg) - n + 1):
                    kws.append(seg[i:i + n])
    return kws


def _embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """用 OpenAI Compatible Embedding API（DashScope text-embedding-v4）向量化"""
    from openai import OpenAI

    from ..config import get_config

    cfg = get_config()
    if not cfg.embedding_available:
        return None
    try:
        client = OpenAI(api_key=cfg.embedding_api_key, base_url=cfg.embedding_base_url)
        resp = client.embeddings.create(model=cfg.embedding_model, input=texts)
        return [item.embedding for item in resp.data]
    except Exception as e:  # noqa: BLE001
        logger.warning("Embedding API 调用失败: %s", e)
        return None


def _get_concept_embeddings() -> Dict[str, List[float]]:
    """概念板块名的 embedding 向量（批量向量化一次，本地持久化缓存）"""
    global _concept_emb_cache
    if _concept_emb_cache is not None:
        return _concept_emb_cache

    try:
        if _CONCEPT_EMB_FILE.exists():
            cached = json.loads(_CONCEPT_EMB_FILE.read_text(encoding="utf-8"))
            embs = cached.get("embeddings", {})
            if embs:
                _concept_emb_cache = embs
                return embs
    except Exception:  # noqa: BLE001
        pass

    boards = _load_concept_boards()
    names = list(boards.keys())
    if not names:
        _concept_emb_cache = {}
        return {}

    # DashScope text-embedding-v4 单次 batch 上限 10，分批向量化
    mapping: Dict[str, List[float]] = {}
    batch_size = 10
    for i in range(0, len(names), batch_size):
        batch = names[i:i + batch_size]
        vecs = _embed_texts(batch)
        if not vecs or len(vecs) != len(batch):
            logger.warning("Embedding 批次 %d 向量化失败，跳过", i // batch_size)
            continue
        for name, v in zip(batch, vecs):
            mapping[name] = [round(x, 4) for x in v]
    if not mapping:
        _concept_emb_cache = {}
        return {}
    _concept_emb_cache = mapping
    try:
        _CONCEPT_EMB_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CONCEPT_EMB_FILE.write_text(
            json.dumps({"embeddings": mapping}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass
    return mapping


def _cosine(a: List[float], b: List[float]) -> float:
    """余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _substring_match(topic: str) -> List[Dict[str, Any]]:
    """子串双向匹配 + 关键词拆分匹配，返回所有命中（按质量排序，不截断）"""
    boards = _load_concept_boards()
    topic = (topic or "").strip()
    if not topic or not boards:
        return []

    t = topic.lower()
    scored: Dict[str, float] = {}   # code -> score
    name_of: Dict[str, str] = {}    # code -> name

    for name, code in boards.items():
        name_of[code] = name
        nl = name.lower()
        score = 0.0
        # 1. 主题词是板块名的子串（"英伟达" in "英伟达概念"）
        if t and t in nl:
            score = max(score, 1.0)
        # 2. 板块名是主题词的子串（"半导体概念" in "半导体概念股"）
        if nl and nl in t:
            score = max(score, 0.85)
        # 3. 关键词拆分匹配
        for kw in _split_keywords(topic):
            if kw and kw.lower() in nl:
                score = max(score, 0.6)
        if score > 0:
            scored[code] = score

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], len(name_of[kv[0]])))
    return [
        {"name": name_of[code], "code": code, "score": round(s, 2)}
        for code, s in ranked
    ]


def _embedding_match(topic: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """用 embedding 语义检索，处理英文主题词 → 中文板块名（字面不同但语义相关）"""
    boards = _load_concept_boards()
    embs = _get_concept_embeddings()
    if not boards or not embs:
        return []
    tvec = _embed_texts([topic])
    if not tvec:
        return []
    tv = tvec[0]
    scored = []
    for name, code in boards.items():
        vec = embs.get(name)
        if vec is None:
            continue
        s = _cosine(tv, vec)
        if s > 0.3:  # 相似度阈值，过滤弱相关
            scored.append((name, code, s))
    scored.sort(key=lambda x: -x[2])
    return [
        {"name": n, "code": c, "score": round(s, 2)}
        for n, c, s in scored[:top_k]
    ]


def retrieve_candidate_sectors(topic: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    给定主题词，从东财概念板块中检索最相关的候选板块（retrieve 阶段）。

    策略: 子串匹配优先（中文主题词直接命中），命中不足 top_k 时用 embedding
    语义检索补齐（覆盖英文主题词 → 中文板块名 的语义匹配）。

    返回: [{name, code, score}]，可能为空（无匹配时由调用方降级）。
    """
    hits = _substring_match(topic)
    if len(hits) >= top_k:
        return hits[:top_k]

    # 子串命中不足，用 embedding 语义检索补齐
    existing = {h["code"] for h in hits}
    need = top_k - len(hits)
    semantic = [
        h for h in _embedding_match(topic, top_k)
        if h["code"] not in existing
    ]
    return hits + semantic[:need]


def resolve_board_code(sector_name: str) -> Optional[str]:
    """把 LLM 给出的板块名解析成东财板块代码（精确 → 子串，都失败返回 None）"""
    boards = _load_boards()
    name = (sector_name or "").strip()
    if not name:
        return None

    # 1. 精确匹配
    if name in boards:
        return boards[name]

    # 2. 板块名包含输入（如"半导体"匹配"半导体概念"），取最短的（最接近）
    candidates = [n for n in boards if name in n]
    if candidates:
        return boards[min(candidates, key=len)]

    # 3. 输入包含板块名（如"半导体概念板块"匹配"半导体概念"）
    candidates = [n for n in boards if n in name]
    if candidates:
        return boards[min(candidates, key=len)]

    return None


def _to_float(v: Any) -> float:
    """安全转 float，空值/异常返回 0"""
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _fetch_components_page(code: str) -> List[Dict[str, Any]]:
    """
    拉取单个板块的成分股第一页（fid=f20 按总市值降序，第一页即市值 top 100）。

    东财成分股接口限流比板块列表更狠，用更长退避 + 更多重试。
    只要 top_n(默认5) 只，第一页足够，无需翻页。
    """
    params = {
        "pn": "1", "pz": "100", "po": "1", "np": "1",
        "ut": _UT, "fltt": "2", "invt": "2", "fid": "f20",
        "fs": f"b:{code} f:!50",
        "fields": "f12,f14,f2,f3,f20",
    }
    last_exc: Optional[Exception] = None
    for attempt in range(8):
        _throttle()
        host = random.choice(_CLIST_HOSTS)
        url = f"https://{host}/api/qt/clist/get"
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            d = data.get("data") or {}
            diff = d.get("diff") or []
            return list(diff) if isinstance(diff, list) else ([diff] if diff else [])
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(6.0 + attempt * 4.0)  # 更长退避，扛限流
    if last_exc:
        logger.warning("成分股请求失败 (%s): %s", code, last_exc)
    return []


def _load_components_file() -> Dict[str, Any]:
    """读成分股文件缓存 -> {code: {ts, stocks}}"""
    try:
        if _COMP_FILE.exists():
            return json.loads(_COMP_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return {}


def _save_components_file(data: Dict[str, Any]) -> None:
    """写成分股文件缓存"""
    try:
        _COMP_FILE.parent.mkdir(parents=True, exist_ok=True)
        _COMP_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def get_board_components(sector_name: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    给定板块名，返回真实成分股，按总市值降序取 top_n。

    返回: [{"code", "name", "price", "change_pct", "market_cap"}, ...]
    板块名无法解析或无数据时返回空列表。

    缓存策略: 内存(300s) → 文件(24h) → 实时请求（成功即落盘）。
    东财成分股接口限流严重，落盘缓存是关键——成功拿到一次就持久化。
    """
    code = resolve_board_code(sector_name)
    if not code:
        return []

    now = time.time()

    # 1. 内存缓存
    items: Optional[List[Dict[str, Any]]] = None
    if code in _comp_cache and now - _comp_cache_time.get(code, 0.0) < _COMP_TTL:
        items = _comp_cache[code]

    # 2. 文件缓存
    if items is None:
        try:
            entry = _load_components_file().get(code)
            if entry and now - entry.get("ts", 0) < _COMP_FILE_TTL:
                items = entry.get("stocks", [])
        except Exception:  # noqa: BLE001
            items = None

    # 3. 实时请求（成功即写内存 + 文件缓存）
    if items is None:
        raw_items = _fetch_components_page(code)
        items = []
        for it in raw_items:
            name = it.get("f14")
            s_code = it.get("f12")
            if not name or not s_code:
                continue
            items.append({
                "code": str(s_code),
                "name": str(name),
                "price": _to_float(it.get("f2")),
                "change_pct": _to_float(it.get("f3")),
                "market_cap": _to_float(it.get("f20")),
            })
        if items:
            _comp_cache[code] = items
            _comp_cache_time[code] = now
            try:
                fcache = _load_components_file()
                fcache[code] = {"ts": now, "stocks": items}
                _save_components_file(fcache)
            except Exception:  # noqa: BLE001
                pass
        else:
            return []

    stocks = [s for s in items if s.get("name") and s.get("code")]
    stocks.sort(key=lambda x: x.get("market_cap", 0.0), reverse=True)
    return stocks[:top_n]
