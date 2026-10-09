"""
港股通标的池监控系统 - 状态持久化

将每次扫描的结果保存为JSON文件，支持加载上一次扫描结果用于变更检测。

状态文件存储在 mode_a/state/data/ 目录下，文件名格式: scan_YYYY-MM-DD.json
同时维护一个 latest.json 指向最近一次扫描。
"""

import os
import json
from datetime import date, datetime
from typing import Optional, Dict, Any, List

from mode_a.models import AssessmentResult, RiskLevel, RiskType, IndexType
from mode_a.core.monitor import ScanSummary


# 状态文件目录（与本文件同目录下的 data/）
STATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)


class StateStore:
    """扫描状态持久化"""

    def __init__(self, state_dir: Optional[str] = None):
        self.state_dir = state_dir or STATE_DIR
        os.makedirs(self.state_dir, exist_ok=True)

    @property
    def latest_path(self) -> str:
        """最新状态文件路径"""
        return os.path.join(self.state_dir, "latest.json")

    def save(self, summary: ScanSummary) -> str:
        """
        保存扫描结果到JSON文件。

        保存两个文件:
        1. scan_YYYY-MM-DD.json - 带日期的历史归档
        2. latest.json - 最新状态（覆盖写）

        返回: latest.json 的路径
        """
        data = self._serialize(summary)

        # 保存历史归档
        archive_path = os.path.join(
            self.state_dir,
            f"scan_{summary.scan_date.isoformat()}.json"
        )
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 保存最新状态
        with open(self.latest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return self.latest_path

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        加载最近一次扫描结果。

        返回: 序列化的状态字典，若无则返回None
        """
        if not os.path.exists(self.latest_path):
            return None
        try:
            with open(self.latest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def load(self, path: str) -> Optional[Dict[str, Any]]:
        """加载指定路径的状态文件"""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # ============================================================
    # 序列化 / 反序列化
    # ============================================================

    def _serialize(self, summary: ScanSummary) -> Dict[str, Any]:
        """将 ScanSummary 序列化为可JSON化的字典"""
        return {
            "scan_date": summary.scan_date.isoformat(),
            "saved_at": datetime.now().isoformat(),
            "total_stocks": summary.total_stocks,
            "pool_size": summary.pool_size,
            "summary_stats": {
                "high_risk": summary.high_risk_count,
                "medium_risk": summary.medium_risk_count,
                "low_risk": summary.low_risk_count,
                "safe": summary.safe_count,
            },
            "results": [
                self._serialize_result(r) for r in summary.results
            ],
        }

    def _serialize_result(self, r: AssessmentResult) -> Dict[str, Any]:
        """将单条 AssessmentResult 序列化"""
        # metrics 中可能有不可JSON化的值，做安全转换
        safe_metrics = {}
        for k, v in r.metrics.items():
            if isinstance(v, (str, int, float, bool)):
                safe_metrics[k] = v
            elif isinstance(v, date):
                safe_metrics[k] = v.isoformat()
            else:
                safe_metrics[k] = str(v)

        return {
            "code": r.code,
            "name": r.name,
            "index_type": r.index_type.value,
            "in_pool": r.in_pool,
            "risk_type": r.risk_type.value,
            "risk_level": r.risk_level.value,
            "details": r.details,
            "metrics": safe_metrics,
        }

    def deserialize_to_map(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        将序列化数据反序列化为 code -> result_dict 的映射。

        用于变更检测时快速查找。
        """
        results = {}
        for r in data.get("results", []):
            code = r["code"]
            results[code] = r
        return results

    def get_previous_scan_date(self, data: Dict[str, Any]) -> Optional[date]:
        """从序列化数据中提取扫描日期"""
        scan_date_str = data.get("scan_date")
        if scan_date_str:
            try:
                return date.fromisoformat(scan_date_str)
            except ValueError:
                return None
        return None
