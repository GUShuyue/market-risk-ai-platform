"""
HSCI历史成分股重建器

从恒生综合指数历史变动CSV重建任意历史时点的成分股列表，
并加载港股通历史调整记录用于回测验证。

数据源:
    1. history_hsci(HSCI).csv  - HSCI历史成分股变动记录 (2008-2026)
    2. hsci_classification.csv - 当前HSCI成分股及分类 (534只)
    3. 港股通标的证券调整信息.xls - 港股通历史调入调出记录 (2014-2026)

重建算法:
    从当前成分股列表出发，按时间倒序回溯变动记录，
    将目标日期之后的"Add"操作撤销(删除)、"Delete"操作撤销(加回)，
    得到目标日期当天的完整成分股列表。
"""

import os
import csv
import logging
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class HSCIChange:
    """单条HSCI成分股变动记录"""
    effective_date: date        # 生效日期
    code: str                   # 股票代码 (5位零填充, 如 "02627")
    name: str                   # 英文名称
    action: str                 # "Add" 或 "Delete"
    count_after: int            # 变动后成分股总数
    batch_count: int            # 该批次变动股票数 (同一effective_date的变动属于一个批次)


@dataclass
class HKConnectChange:
    """单条港股通调整记录"""
    effective_date: date        # 生效日期
    code: str                   # 股票代码 (5位零填充)
    name_en: str                # 英文简称
    name_cn: str                # 中文简称
    action: str                 # "调进" 或 "调出"
    category: str               # 品种 (如 "股票")


@dataclass
class ConstituentSnapshot:
    """某一时点的HSCI成分股快照"""
    snapshot_date: date
    constituents: List[Dict]    # [{code, name, index_type}, ...]
    total_count: int
    official_count: Optional[int] = None  # 官方记录的成分股数量 (来自count_after)
    count_discrepancy: int = 0  # 重建数量与官方数量的差值 (>0表示重建偏多)


class HSCIHistoryLoader:
    """
    HSCI历史数据加载器

    用法:
        loader = HSCIHistoryLoader()
        loader.load()

        # 获取某历史时点的成分股列表
        snapshot = loader.get_constituents_at(date(2024, 6, 1))

        # 获取某时间段内的变动
        changes = loader.get_changes_between(date(2024, 1, 1), date(2024, 12, 31))

        # 获取港股通历史调整
        hk_changes = loader.get_hk_connect_changes_between(date(2024, 1, 1), date(2024, 12, 31))
    """

    def __init__(
        self,
        history_csv: str = None,
        classification_csv: str = None,
        hk_connect_xls: str = None,
    ):
        self.history_csv = history_csv or os.path.join(
            PROJECT_ROOT, "history_hsci(HSCI).csv"
        )
        self.classification_csv = classification_csv or os.path.join(
            PROJECT_ROOT, "hsci_classification.csv"
        )
        self.hk_connect_xls = hk_connect_xls or os.path.join(
            PROJECT_ROOT, "港股通标的证券调整信息.xls"
        )

        # 当前成分股 (从classification CSV加载)
        self._current_constituents: List[Dict] = []

        # 历史变动记录 (按effective_date分组)
        self._changes: List[HSCIChange] = []
        self._changes_by_date: Dict[date, List[HSCIChange]] = {}

        # 港股通历史调整记录
        self._hk_connect_changes: List[HKConnectChange] = []
        self._hk_changes_by_date: Dict[date, List[HKConnectChange]] = {}

        self._loaded = False

    def load(self):
        """加载所有数据源"""
        logger.info("开始加载HSCI历史数据...")

        self._current_constituents = self._load_current_constituents()
        logger.info(f"当前HSCI成分股: {len(self._current_constituents)} 只")

        self._changes = self._load_hsci_history()
        self._changes_by_date = defaultdict(list)
        for ch in self._changes:
            self._changes_by_date[ch.effective_date].append(ch)
        logger.info(
            f"HSCI历史变动: {len(self._changes)} 条, "
            f"覆盖 {len(self._changes_by_date)} 个调整批次"
        )

        # 加载港股通历史调整
        try:
            self._hk_connect_changes = self._load_hk_connect_history()
            self._hk_changes_by_date = defaultdict(list)
            for ch in self._hk_connect_changes:
                self._hk_changes_by_date[ch.effective_date].append(ch)
            logger.info(
                f"港股通历史调整: {len(self._hk_connect_changes)} 条, "
                f"覆盖 {len(self._hk_changes_by_date)} 个调整批次"
            )
        except Exception as e:
            logger.warning(f"港股通历史调整加载失败: {e}")
            self._hk_connect_changes = []

        self._loaded = True
        logger.info("HSCI历史数据加载完成")

    # ================================================================
    # 公开查询接口
    # ================================================================

    def get_constituents_at(self, target_date: date) -> ConstituentSnapshot:
        """
        重建目标日期的HSCI成分股列表

        算法: 从当前成分股出发，逆序回溯所有effective_date > target_date的变动:
            - 撤销Add: 将该股票从列表中移除
            - 撤销Delete: 将该股票加回列表

        数据一致性校验:
            重建后与count_after对比, 若不一致则记录warning.
            已知数据问题: 02910(摯達科技)/02911(希迪智駕)有Add记录但
            不在当前classification CSV中且无Delete记录, 导致重建数量偏多2只.

        参数:
            target_date: 目标日期

        返回:
            ConstituentSnapshot (含official_count和count_discrepancy)
        """
        if not self._loaded:
            self.load()

        # 从当前成分股开始
        constituents = {c["code"]: c for c in self._current_constituents}

        # 跟踪未能撤销的Add (数据不一致: 有Add记录但股票不在当前列表中)
        phantom_adds = []

        # 按时间倒序处理目标日期之后的所有变动
        all_dates = sorted(self._changes_by_date.keys(), reverse=True)
        for d in all_dates:
            if d <= target_date:
                break
            for ch in self._changes_by_date[d]:
                if ch.action == "Add":
                    # 撤销Add: 移除
                    if ch.code in constituents:
                        constituents.pop(ch.code, None)
                    else:
                        phantom_adds.append(ch)
                elif ch.action == "Delete":
                    # 撤销Delete: 加回
                    # 不再跟踪index_type, 由覆盖率模型基于覆盖率统一分配子指数
                    if ch.code not in constituents:
                        constituents[ch.code] = {
                            "code": ch.code,
                            "name": ch.name,
                            "index_type": "",
                        }

        constituent_list = sorted(constituents.values(), key=lambda x: x["code"])
        reconstructed_count = len(constituent_list)

        # 获取官方记录的数量 (来自count_after)
        official_count = self.get_official_count_at(target_date)

        # 数据一致性校验
        discrepancy = 0
        if official_count is not None:
            discrepancy = reconstructed_count - official_count
            if discrepancy != 0:
                logger.warning(
                    f"HSCI成分股重建数量不一致: 重建={reconstructed_count}, "
                    f"官方={official_count}, 差异={discrepancy:+d}. "
                    f"可能原因: {len(phantom_adds)}个Add记录无法撤销"
                    f"(有Add但不在当前列表且无Delete记录)"
                )
                if phantom_adds:
                    for pa in phantom_adds:
                        logger.warning(
                            f"  幽灵Add: {pa.code} {pa.name} "
                            f"(Add on {pa.effective_date}, 无对应Delete记录)"
                        )

        return ConstituentSnapshot(
            snapshot_date=target_date,
            constituents=constituent_list,
            total_count=reconstructed_count,
            official_count=official_count,
            count_discrepancy=discrepancy,
        )

    def get_official_count_at(self, target_date: date) -> Optional[int]:
        """
        获取目标日期的官方HSCI成分股数量 (来自历史变动表的count_after字段)

        查找逻辑: 找到effective_date <= target_date的最近一条变动记录,
        返回其count_after字段值.

        参数:
            target_date: 目标日期

        返回:
            官方成分股数量, 若无数据则返回None
        """
        if not self._loaded:
            self.load()

        # 找到 <= target_date 的最近变动日期
        candidate_dates = [d for d in self._changes_by_date.keys() if d <= target_date]
        if not candidate_dates:
            return None

        nearest_date = max(candidate_dates)
        batch = self._changes_by_date[nearest_date]
        if batch:
            return batch[0].count_after
        return None

    def get_changes_between(
        self, start_date: date, end_date: date
    ) -> List[HSCIChange]:
        """获取指定时间段内的HSCI变动记录"""
        if not self._loaded:
            self.load()
        return [
            ch for ch in self._changes
            if start_date < ch.effective_date <= end_date
        ]

    def get_adjustment_dates(self) -> List[date]:
        """获取所有HSCI调整生效日期 (已排序)"""
        if not self._loaded:
            self.load()
        return sorted(self._changes_by_date.keys())

    def get_hk_connect_changes_between(
        self, start_date: date, end_date: date
    ) -> List[HKConnectChange]:
        """获取指定时间段内的港股通调整记录"""
        if not self._loaded:
            self.load()
        return [
            ch for ch in self._hk_connect_changes
            if start_date < ch.effective_date <= end_date
        ]

    def get_hk_connect_adjustment_dates(self) -> List[date]:
        """获取所有港股通调整生效日期 (已排序)"""
        if not self._loaded:
            self.load()
        return sorted(self._hk_changes_by_date.keys())

    def get_connect_pool_at(
        self, target_date: date, current_pool: Set[str]
    ) -> Set[str]:
        """
        重建目标日期的港股通标的池

        算法: 从当前池出发，逆序回溯所有effective_date > target_date的调整:
            - 撤销调进: 将该股票从池中移除
            - 撤销调出: 将该股票加回池中

        参数:
            target_date: 目标日期
            current_pool: 当前港股通池代码集合

        返回:
            目标日期时的港股通池代码集合
        """
        if not self._loaded:
            self.load()

        pool = set(current_pool)

        # 按时间倒序处理目标日期之后的所有调整
        all_changes = sorted(
            self._hk_connect_changes,
            key=lambda x: x.effective_date,
            reverse=True,
        )
        reversed_count = 0
        for ch in all_changes:
            if ch.effective_date <= target_date:
                break
            code = ch.code.strip().zfill(5)
            if ch.action == "调进":
                # 撤销调进: 移除
                if code in pool:
                    pool.discard(code)
                    reversed_count += 1
            elif ch.action == "调出":
                # 撤销调出: 加回
                if code not in pool:
                    pool.add(code)
                    reversed_count += 1

        logger.info(
            f"历史港股通池重建: target={target_date}, "
            f"当前池={len(current_pool)}只 → 历史池={len(pool)}只, "
            f"逆向{reversed_count}条调整"
        )
        return pool

    @property
    def current_constituents(self) -> List[Dict]:
        """当前HSCI成分股列表"""
        if not self._loaded:
            self.load()
        return self._current_constituents

    # ================================================================
    # 数据加载: 当前成分股
    # ================================================================

    def _load_current_constituents(self) -> List[Dict]:
        """从hsci_classification.csv加载当前HSCI成分股"""
        constituents = []
        # 尝试多种编码
        for encoding in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
            try:
                with open(self.classification_csv, "r", encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        code = row.get("code", "").strip().zfill(5)
                        name = row.get("name", "").strip()
                        index_type = row.get("index_type", "").strip()
                        if code:
                            constituents.append({
                                "code": code,
                                "name": name,
                                "index_type": index_type,
                            })
                if constituents:
                    break
            except (UnicodeDecodeError, FileNotFoundError):
                continue

        if not constituents:
            raise FileNotFoundError(
                f"无法加载HSCI分类文件: {self.classification_csv}"
            )
        return constituents

    # ================================================================
    # 数据加载: HSCI历史变动
    # ================================================================

    def _load_hsci_history(self) -> List[HSCIChange]:
        """
        解析HSCI历史变动CSV

        CSV格式:
            Effective Date, No. of Constituents, Change, Count, Stock Code, ...
            2026-07-10, 534, Delete, -3, 2627, Hong Kong, AB&B BIO-TECH - B, ...

        注意:
            - 前几行是标题/空行，需要跳过
            - 最后几行是注释，需要过滤
            - Stock Code需要零填充为5位
            - 同一effective_date的多条记录属于一个批次
        """
        changes = []

        # 尝试多种编码
        rows = None
        for encoding in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                with open(self.history_csv, "r", encoding=encoding) as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                break
            except UnicodeDecodeError:
                continue

        if rows is None:
            raise FileNotFoundError(f"无法读取HSCI历史CSV: {self.history_csv}")

        # 找到数据起始行 (包含 "Effective Date" 的行)
        header_idx = -1
        for i, row in enumerate(rows):
            if row and "Effective Date" in str(row[0]):
                header_idx = i
                break

        if header_idx == -1:
            # 尝试第二列
            for i, row in enumerate(rows):
                if len(row) > 1 and "Effective Date" in str(row[1]):
                    header_idx = i
                    break

        if header_idx == -1:
            logger.error("无法在CSV中找到表头行")
            return []

        # 解析数据行 (header后的行)
        for row in rows[header_idx + 1:]:
            if not row or not row[0].strip():
                continue

            # 跳过注释/页脚行
            first_col = row[0].strip()
            if not first_col[0].isdigit() and not first_col[0] in ("2", "1"):
                # 非日期格式的行
                continue

            try:
                # 解析日期
                date_str = first_col
                effective_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                # 解析其他字段
                # 列顺序: Effective Date, No. of Constituents, Change, Count, Stock Code, ...
                count_after = int(row[1].strip()) if len(row) > 1 and row[1].strip() else 0
                action_raw = row[2].strip() if len(row) > 2 else ""
                batch_count = int(row[3].strip()) if len(row) > 3 and row[3].strip() else 0
                code_raw = row[4].strip() if len(row) > 4 else ""
                name = row[6].strip() if len(row) > 6 else ""

                # 解析action (可能包含中文乱码, 只取英文部分)
                if "Add" in action_raw:
                    action = "Add"
                elif "Delete" in action_raw:
                    action = "Delete"
                else:
                    continue

                # 零填充股票代码
                code = code_raw.zfill(5)

                changes.append(HSCIChange(
                    effective_date=effective_date,
                    code=code,
                    name=name,
                    action=action,
                    count_after=count_after,
                    batch_count=batch_count,
                ))

            except (ValueError, IndexError) as e:
                # 跳过无法解析的行
                continue

        # 按日期排序
        changes.sort(key=lambda x: (x.effective_date, x.action))
        return changes

    # ================================================================
    # 数据加载: 港股通历史调整
    # ================================================================

    def _load_hk_connect_history(self) -> List[HKConnectChange]:
        """
        解析港股通历史调整Excel

        Excel格式:
            证券代码, 英文简称, 中文简称(参考), 品种, 调整内容, 生效日期
        """
        import xlrd

        changes = []
        workbook = xlrd.open_workbook(self.hk_connect_xls)
        sheet = workbook.sheet_by_index(0)

        # 找到表头行
        header_idx = -1
        col_map = {}
        for row_idx in range(min(10, sheet.nrows)):
            row_values = [str(sheet.cell_value(row_idx, col_idx)).strip()
                          for col_idx in range(sheet.ncols)]
            # 查找包含关键列名的行
            if any("证券代码" in v for v in row_values) or any("生效日期" in v for v in row_values):
                header_idx = row_idx
                for col_idx, val in enumerate(row_values):
                    col_map[val] = col_idx
                break

        if header_idx == -1:
            logger.error("无法在Excel中找到表头行")
            return []

        # 查找列索引
        code_col = self._find_col(col_map, ["证券代码", "代码"])
        name_en_col = self._find_col(col_map, ["英文简称", "英文名称"])
        name_cn_col = self._find_col(col_map, ["中文简称", "中文简称(参考)", "中文名称"])
        category_col = self._find_col(col_map, ["品种"])
        action_col = self._find_col(col_map, ["调整内容", "调整"])
        date_col = self._find_col(col_map, ["生效日期", "日期"])

        if code_col is None or date_col is None or action_col is None:
            logger.error(f"无法找到必要列, col_map={col_map}")
            return []

        for row_idx in range(header_idx + 1, sheet.nrows):
            try:
                code_raw = str(sheet.cell_value(row_idx, code_col)).strip()
                if not code_raw or not code_raw[0].isdigit():
                    continue

                code = code_raw.split(".")[0].zfill(5)

                # 解析日期 (可能是字符串或xlrd日期对象)
                date_val = sheet.cell_value(row_idx, date_col)
                if isinstance(date_val, float):
                    # xlrd日期数字
                    date_tuple = xlrd.xldate_as_tuple(date_val, workbook.datemode)
                    effective_date = date(date_tuple[0], date_tuple[1], date_tuple[2])
                else:
                    date_str = str(date_val).strip()
                    # 尝试多种日期格式
                    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y年%m月%d日"]:
                        try:
                            effective_date = datetime.strptime(date_str, fmt).date()
                            break
                        except ValueError:
                            continue
                    else:
                        continue

                name_en = str(sheet.cell_value(row_idx, name_en_col)).strip() if name_en_col is not None else ""
                name_cn = str(sheet.cell_value(row_idx, name_cn_col)).strip() if name_cn_col is not None else ""
                category = str(sheet.cell_value(row_idx, category_col)).strip() if category_col is not None else ""
                action_raw = str(sheet.cell_value(row_idx, action_col)).strip()

                # 解析action
                if "调进" in action_raw or "调入" in action_raw:
                    action = "调进"
                elif "调出" in action_raw:
                    action = "调出"
                else:
                    continue

                # 过滤非股票品种
                if category and "ETF" in category.upper():
                    continue

                changes.append(HKConnectChange(
                    effective_date=effective_date,
                    code=code,
                    name_en=name_en,
                    name_cn=name_cn,
                    action=action,
                    category=category,
                ))

            except (ValueError, IndexError):
                continue

        changes.sort(key=lambda x: x.effective_date)
        return changes

    @staticmethod
    def _find_col(col_map: Dict[str, int], candidates: List[str]) -> Optional[int]:
        """在列映射中查找匹配的列索引"""
        for key, idx in col_map.items():
            for candidate in candidates:
                if candidate in key:
                    return idx
        return None

    # ================================================================
    # 验证方法
    # ================================================================

    def verify_consistency(self) -> Dict:
        """
        验证数据一致性

        检查:
            1. 从当前成分股逆推到某历史日期，成分股数量是否与CSV记录一致
            2. 港股通调整日期与HSCI调整日期的关联性
        """
        if not self._loaded:
            self.load()

        results = {
            "current_count": len(self._current_constituents),
            "total_changes": len(self._changes),
            "adjustment_batches": len(self._changes_by_date),
            "date_range": {
                "earliest": min(self._changes_by_date.keys()).isoformat() if self._changes_by_date else None,
                "latest": max(self._changes_by_date.keys()).isoformat() if self._changes_by_date else None,
            },
            "sample_snapshots": [],
            "hk_connect_count": len(self._hk_connect_changes),
            "hk_connect_batches": len(self._hk_changes_by_date),
        }

        # 验证几个历史快照的成分股数量
        sample_dates = []
        if self._changes_by_date:
            all_dates = sorted(self._changes_by_date.keys())
            # 取最早、中间、最近的几个日期
            sample_dates = [all_dates[0], all_dates[len(all_dates) // 2], all_dates[-1]]

        for d in sample_dates:
            snapshot = self.get_constituents_at(d)
            # 找到该日期的count_after
            changes_on_date = self._changes_by_date.get(d, [])
            expected_count = changes_on_date[0].count_after if changes_on_date else None
            results["sample_snapshots"].append({
                "date": d.isoformat(),
                "reconstructed_count": snapshot.total_count,
                "csv_count_after": expected_count,
                "match": snapshot.total_count == expected_count if expected_count else "N/A",
            })

        return results
