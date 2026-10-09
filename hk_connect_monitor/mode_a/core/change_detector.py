"""
港股通标的池监控系统 - 变更检测

对比当前扫描与上次扫描的结果，识别状态变化：
  - 新增风险（安全 → 有风险）
  - 风险升级（低→中→高）
  - 风险降级（高→中→低）
  - 风险解除（有风险 → 安全）
  - 风险类型变化（如 灰色区间 → 调出风险）
  - 新进入扫描范围 / 不再被扫描
"""

from datetime import date
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from mode_a.models import RiskLevel, RiskType


class ChangeType(Enum):
    """变更类型"""
    NEW_RISK = "新增风险"           # 安全 → 有风险
    RISK_ESCALATED = "风险升级"     # 风险等级升高
    RISK_DEESCALATED = "风险降级"   # 风险等级降低
    RISK_RESOLVED = "风险解除"      # 有风险 → 安全
    RISK_TYPE_CHANGED = "风险类型变化"  # 同等级但类型不同
    NEWLY_SCANNED = "新进入扫描"    # 上次没有，这次有
    NO_LONGER_SCANNED = "不再扫描"  # 上次有，这次没有


class AlertLevel(Enum):
    """告警等级"""
    CRITICAL = "紧急"    # 新增高风险排除事件（ST/退市/非港币）
    WARNING = "预警"     # 新增风险 或 风险升级到高
    INFO = "通知"        # 风险升级/降级/类型变化
    RESOLVED = "解除"    # 风险解除


# 风险等级数值映射（用于比较）
_RISK_LEVEL_ORDER = {
    RiskLevel.NONE.value: 0,
    RiskLevel.LOW.value: 1,
    RiskLevel.MEDIUM.value: 2,
    RiskLevel.HIGH.value: 3,
}


@dataclass
class ChangeItem:
    """单条变更记录"""
    code: str
    name: str
    change_type: ChangeType
    alert_level: AlertLevel

    # 变更前状态
    prev_risk_type: Optional[str] = None
    prev_risk_level: Optional[str] = None
    prev_details: Optional[str] = None
    prev_in_pool: Optional[bool] = None

    # 变更后状态
    curr_risk_type: Optional[str] = None
    curr_risk_level: Optional[str] = None
    curr_details: Optional[str] = None
    curr_in_pool: Optional[bool] = None

    # 关键指标变化
    metrics_change: Dict[str, str] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """变更摘要"""
        if self.change_type == ChangeType.NEW_RISK:
            return f"从安全变为 {self.curr_risk_type}({self.curr_risk_level})"
        elif self.change_type == ChangeType.RISK_ESCALATED:
            return f"{self.prev_risk_type}({self.prev_risk_level}→{self.curr_risk_level})"
        elif self.change_type == ChangeType.RISK_DEESCALATED:
            return f"{self.prev_risk_type}({self.prev_risk_level}→{self.curr_risk_level})"
        elif self.change_type == ChangeType.RISK_RESOLVED:
            return f"{self.prev_risk_type}({self.prev_risk_level}) → 安全"
        elif self.change_type == ChangeType.RISK_TYPE_CHANGED:
            return f"{self.prev_risk_type} → {self.curr_risk_type}(等级不变{self.curr_risk_level})"
        elif self.change_type == ChangeType.NEWLY_SCANNED:
            pool_tag = "在池" if self.curr_in_pool else "不在池"
            return f"首次扫描: {self.curr_risk_type}({self.curr_risk_level}), {pool_tag}"
        elif self.change_type == ChangeType.NO_LONGER_SCANNED:
            pool_tag = "在池" if self.prev_in_pool else "不在池"
            return f"上次状态: {self.prev_risk_type}({self.prev_risk_level}), {pool_tag}"
        return str(self.change_type)


@dataclass
class ChangeReport:
    """变更检测报告"""
    current_scan_date: date
    previous_scan_date: Optional[date]
    changes: List[ChangeItem] = field(default_factory=list)

    @property
    def critical_items(self) -> List[ChangeItem]:
        return [c for c in self.changes if c.alert_level == AlertLevel.CRITICAL]

    @property
    def warning_items(self) -> List[ChangeItem]:
        return [c for c in self.changes if c.alert_level == AlertLevel.WARNING]

    @property
    def info_items(self) -> List[ChangeItem]:
        return [c for c in self.changes if c.alert_level == AlertLevel.INFO]

    @property
    def resolved_items(self) -> List[ChangeItem]:
        return [c for c in self.changes if c.alert_level == AlertLevel.RESOLVED]

    @property
    def has_changes(self) -> bool:
        return len(self.changes) > 0

    @property
    def has_alerts(self) -> bool:
        """是否有需要关注的告警（紧急+预警）"""
        return len(self.critical_items) > 0 or len(self.warning_items) > 0


class ChangeDetector:
    """变更检测器"""

    def detect(
        self,
        current_map: Dict[str, Dict[str, Any]],
        previous_map: Dict[str, Dict[str, Any]],
        current_date: date,
        previous_date: Optional[date] = None,
    ) -> ChangeReport:
        """
        对比当前与上次的扫描结果，检测变更。

        参数:
            current_map: 当前扫描 code -> result_dict
            previous_map: 上次扫描 code -> result_dict
            current_date: 当前扫描日期
            previous_date: 上次扫描日期

        返回: ChangeReport
        """
        report = ChangeReport(
            current_scan_date=current_date,
            previous_scan_date=previous_date,
        )

        curr_codes = set(current_map.keys())
        prev_codes = set(previous_map.keys())

        # 1. 两份都有的 → 检查状态变化
        common_codes = curr_codes & prev_codes
        for code in common_codes:
            change = self._compare_one(
                current_map[code], previous_map[code]
            )
            if change:
                report.changes.append(change)

        # 2. 当前有、上次没有 → 新进入扫描
        new_codes = curr_codes - prev_codes
        for code in new_codes:
            curr = current_map[code]
            # 只关注有风险的（安全的新进入不算告警）
            if curr["risk_level"] != RiskLevel.NONE.value:
                change = ChangeItem(
                    code=code,
                    name=curr["name"],
                    change_type=ChangeType.NEWLY_SCANNED,
                    alert_level=self._alert_for_new_risk(curr),
                    curr_risk_type=curr["risk_type"],
                    curr_risk_level=curr["risk_level"],
                    curr_details=curr.get("details", ""),
                    curr_in_pool=curr.get("in_pool", False),
                )
                report.changes.append(change)

        # 3. 上次有、当前没有 → 不再扫描
        gone_codes = prev_codes - curr_codes
        for code in gone_codes:
            prev = previous_map[code]
            # 只关注上次有风险的
            if prev["risk_level"] != RiskLevel.NONE.value:
                change = ChangeItem(
                    code=code,
                    name=prev["name"],
                    change_type=ChangeType.NO_LONGER_SCANNED,
                    alert_level=AlertLevel.INFO,
                    prev_risk_type=prev["risk_type"],
                    prev_risk_level=prev["risk_level"],
                    prev_details=prev.get("details", ""),
                    prev_in_pool=prev.get("in_pool", False),
                )
                report.changes.append(change)

        # 按告警等级排序
        alert_order = {
            AlertLevel.CRITICAL: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.INFO: 2,
            AlertLevel.RESOLVED: 3,
        }
        report.changes.sort(key=lambda c: alert_order.get(c.alert_level, 99))

        return report

    def _compare_one(
        self,
        curr: Dict[str, Any],
        prev: Dict[str, Any],
    ) -> Optional[ChangeItem]:
        """对比单只股票的前后状态"""
        curr_level = curr["risk_level"]
        prev_level = prev["risk_level"]
        curr_type = curr["risk_type"]
        prev_type = prev["risk_type"]

        # 状态完全相同 → 无变化
        if curr_level == prev_level and curr_type == prev_type:
            return None

        prev_was_safe = (prev_level == RiskLevel.NONE.value)
        curr_is_safe = (curr_level == RiskLevel.NONE.value)

        code = curr["code"]
        name = curr["name"]

        if prev_was_safe and not curr_is_safe:
            # 安全 → 有风险
            return ChangeItem(
                code=code, name=name,
                change_type=ChangeType.NEW_RISK,
                alert_level=self._alert_for_new_risk(curr),
                prev_risk_type=prev_type, prev_risk_level=prev_level,
                prev_details=prev.get("details", ""),
                prev_in_pool=prev.get("in_pool"),
                curr_risk_type=curr_type, curr_risk_level=curr_level,
                curr_details=curr.get("details", ""),
                curr_in_pool=curr.get("in_pool"),
                metrics_change=self._diff_metrics(curr, prev),
            )

        elif not prev_was_safe and curr_is_safe:
            # 有风险 → 安全
            return ChangeItem(
                code=code, name=name,
                change_type=ChangeType.RISK_RESOLVED,
                alert_level=AlertLevel.RESOLVED,
                prev_risk_type=prev_type, prev_risk_level=prev_level,
                prev_details=prev.get("details", ""),
                prev_in_pool=prev.get("in_pool"),
                curr_risk_type=curr_type, curr_risk_level=curr_level,
                curr_details=curr.get("details", ""),
                curr_in_pool=curr.get("in_pool"),
            )

        elif not prev_was_safe and not curr_is_safe:
            # 有风险 → 有风险（等级或类型变化）
            prev_order = _RISK_LEVEL_ORDER.get(prev_level, 0)
            curr_order = _RISK_LEVEL_ORDER.get(curr_level, 0)

            if curr_order > prev_order:
                # 风险升级
                change_type = ChangeType.RISK_ESCALATED
                # 升级到高 → 预警；否则 → 通知
                alert = AlertLevel.WARNING if curr_level == RiskLevel.HIGH.value else AlertLevel.INFO
            elif curr_order < prev_order:
                # 风险降级
                change_type = ChangeType.RISK_DEESCALATED
                alert = AlertLevel.INFO
            else:
                # 等级不变，类型变化
                change_type = ChangeType.RISK_TYPE_CHANGED
                alert = AlertLevel.INFO

            return ChangeItem(
                code=code, name=name,
                change_type=change_type,
                alert_level=alert,
                prev_risk_type=prev_type, prev_risk_level=prev_level,
                prev_details=prev.get("details", ""),
                prev_in_pool=prev.get("in_pool"),
                curr_risk_type=curr_type, curr_risk_level=curr_level,
                curr_details=curr.get("details", ""),
                curr_in_pool=curr.get("in_pool"),
                metrics_change=self._diff_metrics(curr, prev),
            )

        return None

    def _alert_for_new_risk(self, curr: Dict[str, Any]) -> AlertLevel:
        """
        根据新风险的类型和等级确定告警级别。

        排除事件（ST/退市/非港币）+ 高 → 紧急
        其他高风险 → 预警
        中/低风险 → 通知
        """
        curr_type = curr["risk_type"]
        curr_level = curr["risk_level"]

        if curr_type == RiskType.EXCLUSION_EVENT.value and curr_level == RiskLevel.HIGH.value:
            return AlertLevel.CRITICAL
        elif curr_level == RiskLevel.HIGH.value:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO

    def _diff_metrics(
        self,
        curr: Dict[str, Any],
        prev: Dict[str, Any],
    ) -> Dict[str, str]:
        """对比关键量化指标的变化"""
        curr_m = curr.get("metrics", {})
        prev_m = prev.get("metrics", {})
        changes = {}

        # 关注的指标
        watch_keys = [
            "avg_monthly_market_cap",
            "distance_to_removal",
            "distance_to_inclusion",
            "avg_daily_market_cap_183d",
            "total_trading_value_183d",
        ]

        for key in watch_keys:
            if key in curr_m or key in prev_m:
                old_val = prev_m.get(key)
                new_val = curr_m.get(key)
                if old_val != new_val and new_val is not None:
                    if isinstance(new_val, (int, float)) and isinstance(old_val, (int, float)):
                        diff = new_val - old_val
                        pct = (diff / abs(old_val) * 100) if old_val != 0 else 0
                        changes[key] = f"{old_val:.2f} → {new_val:.2f} ({pct:+.1f}%)"
                    elif old_val is not None:
                        changes[key] = f"{old_val} → {new_val}"
                    else:
                        changes[key] = f"新增: {new_val}"

        return changes
