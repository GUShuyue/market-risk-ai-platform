"""
港股通标的池监控系统 - 真实数据提供器 (akshare 新浪源 + 百度估值 + HSCI分类CSV)

数据源：
  1. 港股通标的池.xls              → 港股通标的池清单 (本地Excel, 上交所官网下载)
  2. hsci_classification.csv        → HSCI指数分类 (本地CSV, 恒生官网PDF转换, convert_hsci_pdf.py生成)
  3. stock_hk_spot()               → 全港股实时行情 (akshare, 新浪)
  4. stock_hk_daily()              → 港股历史日K线 (akshare, 新浪)
  5. stock_hk_valuation_baidu()    → 港股历史总市值 (akshare, 百度)
  6. stock_hk_security_profile_em()→ 港股上市日期 (akshare, 东方财富profile)
  7. stock_zh_ah_spot()            → A+H股清单 (akshare, 新浪)
  8. stock_zh_a_st_em()            → A股ST/*ST/退市整理列表 (akshare, 东方财富)
  9. ah_stock_mapping.json         → H股→A股代码映射 (本地JSON, generate_ah_mapping.py生成)
  10. 英文名称 "-W" 后缀           → WVR(不同投票权架构)识别

注意：
  - HSCI(恒生综合指数)成份股分类来自恒生官网PDF，精确区分大型/中型/小型
  - 当前总市值来自百度估值接口(单位:亿港币)，总股本 = 总市值 / 最新价反算
  - 历史市值 = 历史日K收盘价 × 总股本（假设考察期内总股本不变）
  - A股ST状态通过 stock_zh_a_st_em() 实时获取，映射到对应H股
"""

import os
import re
import time
import socket
import threading
import pandas as pd
from datetime import date, timedelta, datetime
from typing import List, Set, Optional, Dict

import akshare as ak

# 设置全局socket超时（30秒），防止akshare底层请求挂死
socket.setdefaulttimeout(30)

from mode_a.models import StockInfo, MonthlyMarketCap, DailyMarketData, IndexType
from mode_a.data.data_provider import DataProvider
from mode_a import config

# 项目根目录（mode_a/data/ 向上两级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 标的池Excel文件名（位于项目根目录）
POOL_EXCEL_FILENAME = "港股通标的池.xls"

# HSCI分类CSV文件名（位于项目根目录，由 convert_hsci_pdf.py 从恒生官网PDF生成）
HSCI_CSV_FILENAME = "hsci_classification.csv"

# 预过滤阈值
SCAN_MIN_TURNOVER_NON_POOL = 5_000_000   # 500万：不在池股票当日成交额低于此值跳过
SCAN_MIN_CAP_NON_POOL = 3_000_000_000    # 30亿：当前市值低于此值的不在池非WVR股跳过
SCAN_MIN_CAP_WVR = 10_000_000_000        # 100亿：当前市值低于此值的不在池WVR股跳过

# 百度估值返回的市值单位：亿港币
BAIDU_MARKET_CAP_UNIT = 100_000_000


class AkShareDataProvider(DataProvider):
    """基于 akshare(新浪) + 百度估值 的真实数据提供器"""

    def __init__(self, reference_date: Optional[date] = None, verbose: bool = True):
        self.reference_date = reference_date or date.today()
        self.verbose = verbose

        # Excel路径（位于项目根目录）
        self._excel_path = os.path.join(PROJECT_ROOT, POOL_EXCEL_FILENAME)

        # HSCI分类CSV路径（位于项目根目录）
        self._hsci_csv_path = os.path.join(PROJECT_ROOT, HSCI_CSV_FILENAME)

        # 缓存
        self._stocks: Optional[List[StockInfo]] = None
        self._pool_codes: Optional[Set[str]] = None
        self._pool_names: Optional[Dict[str, str]] = None
        self._all_hk_spot: Optional[Dict[str, dict]] = None  # code -> {name_cn, name_en, price, turnover}
        self._ah_codes: Optional[Set[str]] = None
        self._ah_to_a_mapping: Dict[str, str] = {}          # H股代码(5位) -> A股代码(6位)
        self._st_a_codes: Set[str] = set()                  # ST/*ST A股代码集合(6位)
        self._delisting_a_codes: Set[str] = set()            # 退市整理 A股代码集合(6位)
        self._hsci_map: Optional[Dict[str, IndexType]] = None  # code -> 指数类型
        self._market_cap_cache: Dict[str, float] = {}       # code -> 当前总市值(港币元)
        self._total_shares_cache: Dict[str, float] = {}      # code -> 总股本
        self._listing_date_cache: Dict[str, date] = {}       # code -> 上市日期
        self._monthly_cache: Dict[str, List[MonthlyMarketCap]] = {}
        self._daily_cache: Dict[str, List[DailyMarketData]] = {}

        # 日K线请求锁：stock_hk_daily 底层依赖 py_mini_racer(JS引擎)，
        # 非线程安全，多线程并发调用会崩溃，因此用锁将日K请求串行化。
        # 其余接口（百度市值、东财profile）不受影响，仍可并发。
        self._daily_kline_lock = threading.Lock()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [数据] {msg}")

    # ============================================================
    # 初始化：批量获取基础数据
    # ============================================================

    def _ensure_initialized(self):
        if self._stocks is not None:
            return
        self._log("开始初始化数据...")

        # 1. 从本地Excel读取港股通标的池
        self._pool_codes, self._pool_names = self._fetch_pool_from_excel()
        self._log(f"港股通标的池(股票): {len(self._pool_codes)} 只")

        # 2. 从本地CSV读取HSCI指数分类
        self._hsci_map = self._load_hsci_classification()
        self._log(f"HSCI指数分类: {len(self._hsci_map)} 只")

        # 3. 获取全港股实时行情
        self._all_hk_spot = self._fetch_all_hk_spot()
        self._log(f"全港股行情: {len(self._all_hk_spot)} 只")

        # 4. 获取A+H股清单 + H股→A股代码映射
        self._ah_codes = self._fetch_ah_codes()
        self._ah_to_a_mapping = self._load_ah_mapping()
        self._log(f"A+H股清单: {len(self._ah_codes)} 只, H→A映射: {len(self._ah_to_a_mapping)} 条")

        # 5. 获取 ST/*ST 和退市整理 A股列表
        self._st_a_codes, self._delisting_a_codes = self._fetch_st_a_shares()
        self._log(f"ST/*ST A股: {len(self._st_a_codes)} 只, 退市整理 A股: {len(self._delisting_a_codes)} 只")

        # 6. 构建股票列表（含预过滤）
        self._stocks = self._build_stock_list()
        self._log(f"待扫描股票: {len(self._stocks)} 只")

    def _fetch_pool_from_excel(self) -> tuple[Set[str], Dict[str, str]]:
        """
        从本地Excel读取港股通标的池（仅股票，排除ETF）。

        Excel格式（上交所官网下载）：
          证券代码 | 英文简称 | 中文简称(参考) | 品种
        过滤 品种 == "股票"
        """
        try:
            df = pd.read_excel(self._excel_path, header=0)
            # 过滤：只取股票，排除ETF
            df = df[df["品种"] == "股票"]
            # 代码补零到5位
            codes = set(df["证券代码"].astype(str).str.zfill(5).tolist())
            # 代码 -> 中文简称 映射
            names = {}
            for _, row in df.iterrows():
                code = str(row["证券代码"]).zfill(5)
                name = str(row.get("中文简称(参考)", "")).strip()
                names[code] = name
            return codes, names
        except FileNotFoundError:
            self._log(f"标的池Excel未找到: {self._excel_path}")
            return set(), {}
        except Exception as e:
            self._log(f"读取标的池Excel失败: {e}")
            return set(), {}

    def _load_hsci_classification(self) -> Dict[str, IndexType]:
        """
        从本地CSV读取HSCI指数分类（由 convert_hsci_pdf.py 从恒生官网PDF生成）。

        CSV格式：
          code, name, index_type
          00001, 长和, 恒生综合大型股指数
        """
        import csv

        index_str_map = {
            "恒生综合大型股指数": IndexType.HSCI_LARGE,
            "恒生综合中型股指数": IndexType.HSCI_MID,
            "恒生综合小型股指数": IndexType.HSCI_SMALL,
        }

        result: Dict[str, IndexType] = {}
        try:
            with open(self._hsci_csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = str(row["code"]).strip().zfill(5)
                    idx_type_str = str(row["index_type"]).strip()
                    idx_type = index_str_map.get(idx_type_str, IndexType.NONE)
                    if idx_type != IndexType.NONE:
                        result[code] = idx_type
        except FileNotFoundError:
            self._log(f"HSCI分类CSV未找到: {self._hsci_csv_path}")
        except Exception as e:
            self._log(f"读取HSCI分类CSV失败: {e}")

        return result

    def _fetch_all_hk_spot(self) -> Dict[str, dict]:
        """从新浪获取全港股实时行情"""
        try:
            df = ak.stock_hk_spot()
            result = {}
            for _, row in df.iterrows():
                code = str(row["代码"]).strip().zfill(5)
                result[code] = {
                    "name_cn": str(row.get("中文名称", "")).strip(),
                    "name_en": str(row.get("英文名称", "")).strip(),
                    "price": float(row.get("最新价", 0) or 0),
                    "turnover": float(row.get("成交额", 0) or 0),
                }
            return result
        except Exception as e:
            self._log(f"获取全港股行情失败: {e}")
            return {}

    def _fetch_ah_codes(self) -> Set[str]:
        """
        获取A+H股的H股代码集合。
        用线程级硬超时防止akshare底层请求挂死。
        """
        import threading

        result: Set[str] = set()
        error: list = []

        def _do_fetch():
            try:
                df = ak.stock_zh_ah_spot()
                result.update(df["代码"].astype(str).str.zfill(5).tolist())
            except Exception as e:
                error.append(e)

        t = threading.Thread(target=_do_fetch, daemon=True)
        t.start()
        t.join(timeout=60)  # 硬超时60秒

        if t.is_alive():
            self._log("获取A+H股清单超时(60s)，跳过")
            return set()
        if error:
            self._log(f"获取A+H股清单失败: {error[0]}")
            return set()

        self._log(f"A+H股清单获取成功: {len(result)} 只")
        return result

    def _load_ah_mapping(self) -> Dict[str, str]:
        """
        从 ah_stock_mapping.json 加载 H股代码(5位) → A股代码(6位) 映射。

        映射文件位于 mode_b/data/ah_stock_mapping.json，
        由 generate_ah_mapping.py 脚本生成（AKShare stock_zh_ah_name + Tushare stock_basic 匹配）。
        文件格式: {"mapping": {"000002.SZ": {"hk_code": "02202", "name": "万科A"}, ...}}
        """
        import json

        mapping_path = os.path.join(
            PROJECT_ROOT, "mode_b", "data", "ah_stock_mapping.json",
        )
        result: Dict[str, str] = {}
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for a_code_full, info in data.get("mapping", {}).items():
                hk_code = str(info.get("hk_code", "")).strip().zfill(5)
                # A股代码去掉交易所后缀: "000002.SZ" → "000002"
                a_code = str(a_code_full).split(".")[0].strip().zfill(6)
                if hk_code and a_code:
                    result[hk_code] = a_code
        except FileNotFoundError:
            self._log(f"A+H映射文件未找到: {mapping_path}，A股ST检测将不可用")
        except Exception as e:
            self._log(f"加载A+H映射失败: {e}")
        return result

    def _fetch_st_a_shares(self) -> tuple:
        """
        获取 ST/*ST 和退市整理 A股代码集合。

        使用 ak.stock_zh_a_st_em() 获取风险警示股票列表（东方财富源）。
        ST/*ST 通过名称中包含 "ST" 识别；
        退市整理期股票通过名称中包含 "退" 识别。

        返回:
            (st_codes: Set[str], delisting_codes: Set[str])
            代码均为6位数字字符串（无交易所后缀）。
        """
        import threading

        st_codes: set = set()
        delisting_codes: set = set()
        error: list = []

        def _do_fetch():
            try:
                df = ak.stock_zh_a_st_em()
                if df is None or df.empty:
                    return
                for _, row in df.iterrows():
                    code = str(row.get("代码", "")).strip().zfill(6)
                    name = str(row.get("名称", "")).strip()
                    if not code:
                        continue
                    if "退" in name:
                        delisting_codes.add(code)
                    elif "ST" in name or "*ST" in name:
                        st_codes.add(code)
            except Exception as e:
                error.append(e)

        t = threading.Thread(target=_do_fetch, daemon=True)
        t.start()
        t.join(timeout=60)

        if t.is_alive():
            self._log("获取ST A股列表超时(60s)，跳过")
            return set(), set()
        if error:
            self._log(f"获取ST A股列表失败: {error[0]}")
            return set(), set()

        return st_codes, delisting_codes

    def _build_stock_list(self) -> List[StockInfo]:
        """
        合并所有数据源，构建待扫描股票列表（含预过滤）。

        在池股票全部纳入扫描。
        不在池股票用当日成交额做粗略预过滤，减少后续市值查询量。
        """
        stocks = []
        skipped = 0

        # 先收集在池股票（全部纳入）
        pool_stock_codes = set()
        for code in self._pool_codes:
            spot = self._all_hk_spot.get(code)
            if spot is None:
                # 标的池中有但行情中无数据，用Excel名称
                name = self._pool_names.get(code, code)
                spot = {"name_cn": name, "name_en": "", "price": 0.0, "turnover": 0.0}
            pool_stock_codes.add(code)
            stocks.append(self._make_stock_info(code, spot, in_pool=True))

        # 再收集不在池的股票（预过滤）
        for code, spot in self._all_hk_spot.items():
            if code in pool_stock_codes:
                continue

            # 预过滤：成交额太低的不在池股票跳过
            if spot["turnover"] < SCAN_MIN_TURNOVER_NON_POOL:
                skipped += 1
                continue

            stocks.append(self._make_stock_info(code, spot, in_pool=False))

        if skipped > 0:
            self._log(f"预过滤跳过 {skipped} 只低成交额不在池股票")
        return stocks

    def _make_stock_info(self, code: str, spot: dict, in_pool: bool) -> StockInfo:
        """从行情数据构建StockInfo（不含上市日期和总股本，按需获取）"""
        name_cn = spot.get("name_cn", "")
        name_en = spot.get("name_en", "")

        # 识别WVR（英文名称以 -W 或 -SW 结尾）
        is_wvr = bool(re.search(r"-S?W$", name_en))

        # 识别第二上市（英文名称以 -S 或 -SW 结尾，不含 -SS 合订证券）
        is_secondary = bool(re.search(r"-S[^S]?W?$", name_en))

        # 识别A+H股
        is_ah = code in self._ah_codes

        # 识别A股ST/退市状态（仅A+H股需要检查）
        a_share_st = False
        a_share_delisting = False
        a_share_code = None
        if is_ah:
            a_code = self._ah_to_a_mapping.get(code)
            if a_code:
                # 构建带交易所后缀的A股代码（用于展示）
                suffix = "SH" if a_code.startswith(("60", "68")) else "SZ"
                a_share_code = f"{a_code}.{suffix}"
                if a_code in self._st_a_codes:
                    a_share_st = True
                if a_code in self._delisting_a_codes:
                    a_share_delisting = True

        # 识别非港币报价（新浪返回的都是港币柜台）
        currency = "HKD"

        # HSCI分类：从预加载的CSV映射中直接查询
        index_type = self._hsci_map.get(code, IndexType.NONE) if self._hsci_map else IndexType.NONE

        return StockInfo(
            code=code,
            name=name_cn or name_en,
            currency=currency,
            listing_date=date(2020, 1, 1),  # 默认值，WVR评估时按需获取
            is_wvr=is_wvr,
            is_secondary_listing=is_secondary,
            is_a_plus_h=is_ah,
            a_share_code=a_share_code,
            a_share_st=a_share_st,
            a_share_delisting=a_share_delisting,
            index_type=index_type,
            in_pool=in_pool,
        )

    # ============================================================
    # 按需数据获取：市值、总股本、上市日期
    # ============================================================

    def _get_current_market_cap(self, code: str) -> float:
        """
        从百度估值接口获取最近一日总市值（港币元）。
        百度返回单位为"亿港币"，需 ×10^8 转换。
        """
        if code in self._market_cap_cache:
            return self._market_cap_cache[code]

        try:
            df = ak.stock_hk_valuation_baidu(
                symbol=code, indicator="总市值", period="近一年"
            )
            if df is not None and not df.empty:
                # 取最后一行（最近一日）
                last_val = float(df.iloc[-1]["value"])
                market_cap = last_val * BAIDU_MARKET_CAP_UNIT
                self._market_cap_cache[code] = market_cap
                time.sleep(0.2)
                return market_cap
        except Exception as e:
            if self.verbose:
                print(f"  [数据] 获取 {code} 市值失败: {type(e).__name__}")

        self._market_cap_cache[code] = 0.0
        return 0.0

    def _get_total_shares(self, code: str) -> float:
        """总股本 = 当前总市值 / 当前最新价"""
        if code in self._total_shares_cache:
            return self._total_shares_cache[code]

        market_cap = self._get_current_market_cap(code)
        spot = self._all_hk_spot.get(code, {})
        price = spot.get("price", 0)

        if market_cap > 0 and price > 0:
            shares = market_cap / price
            self._total_shares_cache[code] = shares
            return shares

        self._total_shares_cache[code] = 0.0
        return 0.0

    def get_listing_date(self, code: str) -> date:
        """从东方财富profile接口获取上市日期"""
        if code in self._listing_date_cache:
            return self._listing_date_cache[code]

        try:
            df = ak.stock_hk_security_profile_em(symbol=code)
            if df is not None and not df.empty:
                raw = str(df.iloc[0].get("上市日期", ""))
                # 格式: "2004-06-16 00:00:00"
                raw = raw.strip()[:10]
                parts = raw.split("-")
                if len(parts) >= 3:
                    d = date(int(parts[0]), int(parts[1]), int(parts[2]))
                    self._listing_date_cache[code] = d
                    time.sleep(0.15)
                    return d
        except Exception:
            pass

        self._listing_date_cache[code] = date(2020, 1, 1)
        return date(2020, 1, 1)

    def classify_index_type(self, code: str) -> IndexType:
        """
        从恒生官网PDF生成的CSV中查询HSCI指数分类。

        CSV由 convert_hsci_pdf.py 从恒生综合指数成分股PDF转换而来，
        包含大型/中型/小型股指数的精确成分股列表。
        不在CSV中的股票返回NONE（非指数成份股）。
        """
        self._ensure_initialized()
        return self._hsci_map.get(code, IndexType.NONE)

    # ============================================================
    # DataProvider 接口实现
    # ============================================================

    def get_all_stocks(self) -> List[StockInfo]:
        self._ensure_initialized()
        return self._stocks

    def get_current_pool(self) -> Set[str]:
        self._ensure_initialized()
        return self._pool_codes

    def get_monthly_market_caps(self, code: str, months: int = 12) -> List[MonthlyMarketCap]:
        """
        获取月末市值数据。

        从 stock_hk_daily 日K中提取每个月最后一个交易日的收盘价，
        乘以总股本得到月末市值。
        """
        if code in self._monthly_cache:
            return self._monthly_cache[code]

        self._ensure_initialized()

        # 获取总股本
        total_shares = self._get_total_shares(code)
        if total_shares <= 0:
            self._monthly_cache[code] = []
            return []

        # 获取日K线（多取2个月的数据确保覆盖）
        daily = self._fetch_daily_kline(code, days=months * 31 + 60)
        if not daily:
            self._monthly_cache[code] = []
            return []

        # 按月分组，取每月最后一条
        monthly_map: Dict[str, DailyMarketData] = {}
        for d in daily:
            month_key = f"{d.date.year}-{d.date.month:02d}"
            if month_key not in monthly_map or d.date > monthly_map[month_key].date:
                monthly_map[month_key] = d

        # 转换为月末市值
        data = []
        for month_key in sorted(monthly_map.keys()):
            d = monthly_map[month_key]
            market_cap = d.market_cap  # 已在_fetch_daily_kline中计算
            data.append(MonthlyMarketCap(date=d.date, market_cap=market_cap))

        # 取最近 months 个月
        data = data[-months:] if len(data) > months else data
        self._monthly_cache[code] = data
        return data

    def get_daily_market_data(self, code: str, days: int = 183) -> List[DailyMarketData]:
        """获取日度行情数据（收盘价、成交额、市值）"""
        if code in self._daily_cache:
            return self._daily_cache[code]

        self._ensure_initialized()

        data = self._fetch_daily_kline(code, days=days)
        self._daily_cache[code] = data
        return data

    # ============================================================
    # 内部方法：日K线获取
    # ============================================================

    def _fetch_daily_kline(self, code: str, days: int) -> List[DailyMarketData]:
        """
        从新浪获取日K线，计算市值和截取日期范围。

        stock_hk_daily 返回全历史日K，这里截取最近 days 个日历日。
        """
        total_shares = self._get_total_shares(code)

        try:
            # stock_hk_daily 底层依赖 py_mini_racer，非线程安全，加锁串行化
            with self._daily_kline_lock:
                df = ak.stock_hk_daily(symbol=code, adjust="")
            if df is None or df.empty:
                return []

            cutoff = self.reference_date - timedelta(days=days)

            data = []
            for _, row in df.iterrows():
                row_date = pd.to_datetime(row["date"]).date()
                if row_date < cutoff:
                    continue
                close = float(row["close"])
                trading_value = float(row["amount"]) if pd.notna(row.get("amount")) else 0.0
                market_cap = close * total_shares if total_shares > 0 else 0.0
                data.append(DailyMarketData(
                    date=row_date,
                    market_cap=market_cap,
                    trading_value=trading_value,
                ))

            time.sleep(0.15)  # 限速
            return data

        except Exception as e:
            if self.verbose:
                print(f"  [数据] 获取 {code} 日K失败: {type(e).__name__}")
            return []
