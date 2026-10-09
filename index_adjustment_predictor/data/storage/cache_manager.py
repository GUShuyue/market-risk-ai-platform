"""
Cache Manager - SQLite持久化缓存

为AKShare数据抓取提供持久化缓存。
历史数据不会变化（如2024年4月30日的市值），因此缓存永久有效。

缓存策略:
    - 市场数据: 按 date + ts_code 缓存
    - 成分股: 按 index_name + date 缓存
    - 财务数据: 按 ts_code + period 缓存
    - 事件数据: 按 date + event_type 缓存
    - 上市信息: 按 ts_code 缓存

表结构:
    - market_data:  市值、成交额、行情数据
    - listing_info: 上市信息、行业分类
    - constituents: 指数成分股
    - financials:   财务指标
    - events:       事件数据（停牌、公告等）
    - avg_amount:    日均成交额
    - cache_meta:    缓存元信息（记录某批数据是否已完整抓取）
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd


class CacheManager:
    """
    SQLite缓存管理器
    
    用法:
        cache = CacheManager("data/storage/database/cache.db")
        
        # 检查缓存是否存在
        if cache.has_market_data("2024-04-30"):
            data = cache.get_market_data("2024-04-30")
        else:
            data = fetcher.fetch_by_date("2024-04-30")
            cache.save_market_data("2024-04-30", data)
    """

    def __init__(self, db_path: str = "data/storage/database/cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 市场数据表（按日期+股票代码）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                date TEXT NOT NULL,
                ts_code TEXT NOT NULL,
                name TEXT,
                total_mv REAL,
                circ_mv REAL,
                close REAL,
                pct_chg REAL,
                amount REAL,
                vol REAL,
                pe REAL,
                pb REAL,
                turnover_rate REAL,
                avg_amount REAL,
                PRIMARY KEY (date, ts_code)
            )
        """)

        # 上市信息表（按股票代码，不随日期变化）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS listing_info (
                ts_code TEXT PRIMARY KEY,
                name TEXT,
                list_date TEXT,
                listing_months INTEGER,
                is_st INTEGER,
                industry TEXT,
                exchange TEXT,
                updated_at TEXT
            )
        """)

        # 指数成分股表（按指数+日期）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS constituents (
                index_name TEXT NOT NULL,
                date TEXT NOT NULL,
                ts_code TEXT NOT NULL,
                name TEXT,
                weight REAL,
                in_date TEXT,
                PRIMARY KEY (index_name, date, ts_code)
            )
        """)

        # 财务数据表（按股票+报告期）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financials (
                ts_code TEXT NOT NULL,
                period TEXT NOT NULL,
                revenue REAL,
                net_profit REAL,
                roe REAL,
                roa REAL,
                debt_to_assets REAL,
                gross_margin REAL,
                operating_cash_flow REAL,
                eps REAL,
                is_loss INTEGER,
                is_high_debt INTEGER,
                has_financial_risk INTEGER,
                PRIMARY KEY (ts_code, period)
            )
        """)

        # 事件数据表（停牌、公告等）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                date TEXT NOT NULL,
                ts_code TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_detail TEXT,
                severity TEXT,
                PRIMARY KEY (date, ts_code, event_type)
            )
        """)

        # 日均成交额表（按股票+截止日）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS avg_amount (
                ts_code TEXT NOT NULL,
                cutoff_date TEXT NOT NULL,
                avg_amount REAL,
                PRIMARY KEY (ts_code, cutoff_date)
            )
        """)

        # 价格变动表（按股票+截止日）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_changes (
                ts_code TEXT NOT NULL,
                cutoff_date TEXT NOT NULL,
                pct_change_20d REAL,
                max_daily_pct REAL,
                min_daily_pct REAL,
                volatility REAL,
                is_abnormal INTEGER,
                PRIMARY KEY (ts_code, cutoff_date)
            )
        """)

        # 缓存元信息表（记录某批数据是否已完整抓取）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_meta (
                cache_key TEXT PRIMARY KEY,
                cache_type TEXT NOT NULL,
                status TEXT NOT NULL,
                count INTEGER,
                created_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    # ===== 市场数据 =====

    def has_market_data(self, date: str) -> bool:
        """检查某个日期的市场数据是否已缓存"""
        return self._check_meta(f"market_data:{date}")

    def get_market_data(self, date: str) -> List[dict]:
        """读取某个日期的全市场数据"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            "SELECT * FROM market_data WHERE date = ?",
            conn,
            params=[date]
        )
        conn.close()
        if df.empty:
            return []
        return df.to_dict("records")

    def save_market_data(self, date: str, data: List[dict]):
        """保存某个日期的全市场数据"""
        if not data:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 先删除该日期的旧数据
        cursor.execute(
            "DELETE FROM market_data WHERE date = ?",
            [date]
        )

        for row in data:
            cursor.execute("""
                INSERT OR REPLACE INTO market_data
                (date, ts_code, name, total_mv, circ_mv, close, pct_chg,
                 amount, vol, pe, pb, turnover_rate, avg_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                date,
                row.get("ts_code", ""),
                row.get("name", ""),
                row.get("total_mv", 0),
                row.get("circ_mv", 0),
                row.get("close", 0),
                row.get("pct_chg", 0),
                row.get("amount", 0),
                row.get("vol", 0),
                row.get("pe", 0),
                row.get("pb", 0),
                row.get("turnover_rate", 0),
                row.get("avg_amount", 0),
            ])

        self._set_meta(cursor, f"market_data:{date}", "market_data", "complete", len(data))
        conn.commit()
        conn.close()

    # ===== 上市信息 =====

    def has_listing_info(self) -> bool:
        """检查上市信息是否已缓存"""
        return self._check_meta("listing_info:all")

    def get_listing_info(self, cutoff_date: str) -> List[dict]:
        """读取上市信息（筛选在截止日前上市的）"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            "SELECT * FROM listing_info WHERE list_date <= ?",
            conn,
            params=[cutoff_date.replace("-", "")]
        )
        conn.close()
        if df.empty:
            return []
        df["is_st"] = df["is_st"].astype(bool)
        return df.to_dict("records")

    def save_listing_info(self, data: List[dict]):
        """保存上市信息"""
        if not data:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for row in data:
            cursor.execute("""
                INSERT OR REPLACE INTO listing_info
                (ts_code, name, list_date, listing_months, is_st,
                 industry, exchange, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                row.get("ts_code", ""),
                row.get("name", ""),
                row.get("list_date", ""),
                row.get("listing_months", 0),
                1 if row.get("is_st") else 0,
                row.get("industry", ""),
                row.get("exchange", ""),
                datetime.now().isoformat()
            ])

        self._set_meta(cursor, "listing_info:all", "listing_info", "complete", len(data))
        conn.commit()
        conn.close()

    # ===== 指数成分股 =====

    def has_constituents(self, index_name: str, date: str) -> bool:
        """检查某指数某日期的成分股是否已缓存"""
        return self._check_meta(f"constituents:{index_name}:{date}")

    def get_constituents(self, index_name: str, date: str) -> List[dict]:
        """读取成分股"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            "SELECT * FROM constituents WHERE index_name = ? AND date = ?",
            conn,
            params=[index_name, date]
        )
        conn.close()
        if df.empty:
            return []
        return df.to_dict("records")

    def save_constituents(self, index_name: str, date: str, data: List[dict]):
        """保存成分股"""
        if not data:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM constituents WHERE index_name = ? AND date = ?",
            [index_name, date]
        )

        for row in data:
            cursor.execute("""
                INSERT OR REPLACE INTO constituents
                (index_name, date, ts_code, name, weight, in_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                index_name,
                date,
                row.get("ts_code", row.get("con_code", "")),
                row.get("name", ""),
                row.get("weight", 0),
                row.get("in_date", ""),
            ])

        self._set_meta(
            cursor,
            f"constituents:{index_name}:{date}",
            "constituents",
            "complete",
            len(data)
        )
        conn.commit()
        conn.close()

    # ===== 财务数据 =====

    def has_financial(self, ts_code: str, period: str) -> bool:
        """检查某股票某期的财务数据是否已缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM financials WHERE ts_code = ? AND period = ?",
            [ts_code, period]
        )
        result = cursor.fetchone() is not None
        conn.close()
        return result

    def get_financial(self, ts_code: str, period: str) -> Optional[dict]:
        """读取单只股票的财务数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM financials WHERE ts_code = ? AND period = ?",
            [ts_code, period]
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        columns = [d[0] for d in cursor.description]
        return dict(zip(columns, row))

    def get_financial_flags_batch(self, ts_codes: List[str], period: str) -> Dict[str, dict]:
        """批量读取财务异常标记"""
        if not ts_codes:
            return {}
        conn = sqlite3.connect(self.db_path)
        placeholders = ",".join(["?" for _ in ts_codes])
        df = pd.read_sql(
            f"SELECT * FROM financials WHERE period = ? AND ts_code IN ({placeholders})",
            conn,
            params=[period] + ts_codes
        )
        conn.close()
        if df.empty:
            return {}
        result = {}
        for _, row in df.iterrows():
            result[row["ts_code"]] = {
                "is_loss": bool(row.get("is_loss", 0)),
                "net_profit": row.get("net_profit", 0),
                "roe": row.get("roe", 0),
                "debt_ratio": row.get("debt_to_assets", 0),
                "is_high_debt": bool(row.get("is_high_debt", 0)),
                "audit_opinion": "未知",
                "has_audit_issue": False,
                "has_financial_risk": bool(row.get("has_financial_risk", 0)),
            }
        return result

    def save_financial(self, ts_code: str, period: str, data: dict):
        """保存单只股票的财务数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO financials
            (ts_code, period, revenue, net_profit, roe, roa,
             debt_to_assets, gross_margin, operating_cash_flow, eps,
             is_loss, is_high_debt, has_financial_risk)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            ts_code,
            period,
            data.get("revenue", 0),
            data.get("net_profit", 0),
            data.get("roe", 0),
            data.get("roa", 0),
            data.get("debt_ratio", data.get("debt_to_assets", 0)),
            data.get("gross_margin", 0),
            data.get("operating_cash_flow", 0),
            data.get("eps", 0),
            1 if data.get("is_loss") else 0,
            1 if data.get("is_high_debt") else 0,
            1 if data.get("has_financial_risk") else 0,
        ])
        conn.commit()
        conn.close()

    def save_financial_flags_batch(self, period: str, data: Dict[str, dict]):
        """批量保存财务异常标记"""
        if not data:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for ts_code, flags in data.items():
            cursor.execute("""
                INSERT OR REPLACE INTO financials
                (ts_code, period, net_profit, roe, debt_to_assets,
                 is_loss, is_high_debt, has_financial_risk)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                ts_code,
                period,
                flags.get("net_profit", 0),
                flags.get("roe", 0),
                flags.get("debt_ratio", 0),
                1 if flags.get("is_loss") else 0,
                1 if flags.get("is_high_debt") else 0,
                1 if flags.get("has_financial_risk") else 0,
            ])
        conn.commit()
        conn.close()

    # ===== 事件数据 =====

    def has_events(self, date: str, event_type: str) -> bool:
        """检查事件数据是否已缓存"""
        return self._check_meta(f"events:{date}:{event_type}")

    def get_events(self, date: str, event_type: str = None) -> Dict[str, dict]:
        """读取事件数据"""
        conn = sqlite3.connect(self.db_path)
        if event_type:
            df = pd.read_sql(
                "SELECT * FROM events WHERE date = ? AND event_type = ?",
                conn,
                params=[date, event_type]
            )
        else:
            df = pd.read_sql(
                "SELECT * FROM events WHERE date = ?",
                conn,
                params=[date]
            )
        conn.close()
        if df.empty:
            return {}
        result = {}
        for _, row in df.iterrows():
            ts_code = row["ts_code"]
            if ts_code not in result:
                result[ts_code] = []
            result[ts_code].append({
                "event_type": row["event_type"],
                "event_detail": row["event_detail"],
                "severity": row.get("severity", ""),
            })
        return result

    def save_events(self, date: str, event_type: str, data: Dict[str, dict]):
        """保存事件数据"""
        if not data:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM events WHERE date = ? AND event_type = ?",
            [date, event_type]
        )

        for ts_code, event_data in data.items():
            if isinstance(event_data, dict):
                cursor.execute("""
                    INSERT OR REPLACE INTO events
                    (date, ts_code, event_type, event_detail, severity)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    date,
                    ts_code,
                    event_type,
                    json.dumps(event_data, ensure_ascii=False),
                    event_data.get("severity", "")
                ])
            elif isinstance(event_data, list):
                for item in event_data:
                    cursor.execute("""
                        INSERT OR REPLACE INTO events
                        (date, ts_code, event_type, event_detail, severity)
                        VALUES (?, ?, ?, ?, ?)
                    """, [
                        date,
                        ts_code,
                        event_type,
                        json.dumps(item, ensure_ascii=False),
                        item.get("severity", "") if isinstance(item, dict) else ""
                    ])

        self._set_meta(
            cursor,
            f"events:{date}:{event_type}",
            "events",
            "complete",
            len(data)
        )
        conn.commit()
        conn.close()

    # ===== 日均成交额 =====

    def has_avg_amount(self, ts_code: str, cutoff_date: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM avg_amount WHERE ts_code = ? AND cutoff_date = ?",
            [ts_code, cutoff_date]
        )
        result = cursor.fetchone() is not None
        conn.close()
        return result

    def get_avg_amount(self, ts_code: str, cutoff_date: str) -> float:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT avg_amount FROM avg_amount WHERE ts_code = ? AND cutoff_date = ?",
            [ts_code, cutoff_date]
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else 0.0

    def save_avg_amount(self, ts_code: str, cutoff_date: str, avg_amount: float):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO avg_amount
            (ts_code, cutoff_date, avg_amount)
            VALUES (?, ?, ?)
        """, [ts_code, cutoff_date, avg_amount])
        conn.commit()
        conn.close()

    def get_avg_amount_batch(self, cutoff_date: str) -> Dict[str, float]:
        """批量读取日均成交额"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ts_code, avg_amount FROM avg_amount WHERE cutoff_date = ?",
            [cutoff_date]
        )
        result = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return result

    def save_avg_amount_batch(self, cutoff_date: str, data: Dict[str, float]):
        """批量保存日均成交额"""
        if not data:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for ts_code, amount in data.items():
            cursor.execute("""
                INSERT OR REPLACE INTO avg_amount
                (ts_code, cutoff_date, avg_amount)
                VALUES (?, ?, ?)
            """, [ts_code, cutoff_date, amount])
        conn.commit()
        conn.close()

    # ===== 价格变动 =====

    def has_price_changes(self, ts_code: str, cutoff_date: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM price_changes WHERE ts_code = ? AND cutoff_date = ?",
            [ts_code, cutoff_date]
        )
        result = cursor.fetchone() is not None
        conn.close()
        return result

    def get_price_changes(self, cutoff_date: str) -> Dict[str, dict]:
        """批量读取价格变动"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            "SELECT * FROM price_changes WHERE cutoff_date = ?",
            conn,
            params=[cutoff_date]
        )
        conn.close()
        if df.empty:
            return {}
        result = {}
        for _, row in df.iterrows():
            result[row["ts_code"]] = {
                "pct_change_20d": row["pct_change_20d"],
                "max_daily_pct": row["max_daily_pct"],
                "min_daily_pct": row["min_daily_pct"],
                "volatility": row["volatility"],
                "is_abnormal": bool(row["is_abnormal"]),
            }
        return result

    def save_price_changes(self, cutoff_date: str, data: Dict[str, dict]):
        """批量保存价格变动"""
        if not data:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for ts_code, info in data.items():
            cursor.execute("""
                INSERT OR REPLACE INTO price_changes
                (ts_code, cutoff_date, pct_change_20d, max_daily_pct,
                 min_daily_pct, volatility, is_abnormal)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                ts_code,
                cutoff_date,
                info.get("pct_change_20d", 0),
                info.get("max_daily_pct", 0),
                info.get("min_daily_pct", 0),
                info.get("volatility", 0),
                1 if info.get("is_abnormal") else 0,
            ])
        conn.commit()
        conn.close()

    # ===== 缓存元信息 =====

    def _check_meta(self, cache_key: str) -> bool:
        """检查某批数据是否已完整缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status FROM cache_meta WHERE cache_key = ?",
            [cache_key]
        )
        row = cursor.fetchone()
        conn.close()
        return row is not None and row[0] == "complete"

    def _set_meta(
        self, cursor, cache_key: str, cache_type: str,
        status: str, count: int = 0
    ):
        """设置缓存元信息"""
        cursor.execute("""
            INSERT OR REPLACE INTO cache_meta
            (cache_key, cache_type, status, count, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, [
            cache_key,
            cache_type,
            status,
            count,
            datetime.now().isoformat()
        ])

    # ===== 统计信息 =====

    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        tables = [
            "market_data", "listing_info", "constituents",
            "financials", "events", "avg_amount", "price_changes"
        ]
        stats = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        conn.close()
        return stats

    def clear_cache(self, cache_type: str = None):
        """清空缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if cache_type:
            cursor.execute(f"DELETE FROM {cache_type}")
        else:
            for table in ["market_data", "listing_info", "constituents",
                         "financials", "events", "avg_amount",
                         "price_changes", "cache_meta"]:
                cursor.execute(f"DELETE FROM {table}")
        conn.commit()
        conn.close()
