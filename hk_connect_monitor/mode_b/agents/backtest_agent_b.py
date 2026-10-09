"""
Mode B 回测Agent

双层回测验证:
    Layer 1: HSCI预测准确率 (对比预测HSCI调入调出与实际HSCI变动)
    Layer 2: 港股通映射准确率 (对比预测港股通调入调出与实际港股通调整)

指标:
    - 精确率 (Precision): 命中数 / 预测总数
    - 召回率 (Recall): 命中数 / 实际总数
    - F1: 精确率和召回率的调和平均
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class BacktestAgentB:
    """
    Mode B 回测Agent

    用法:
        agent = BacktestAgentB()
        result = agent.backtest(
            prediction=prediction_dict,
            actual_hsci_changes=[HSCIChange, ...],
            actual_connect_changes=[HKConnectChange, ...],
            target_date=date(2024, 6, 10),
        )
    """

    def backtest(
        self,
        prediction: Dict,
        actual_hsci_changes: List,
        actual_connect_changes: List,
        target_date: date,
    ) -> Dict:
        """
        执行双层回测

        参数:
            prediction: OrchestratorB.run_predict()的返回结果
            actual_hsci_changes: 实际HSCI变动记录列表 (HSCIChange对象)
            actual_connect_changes: 实际港股通变动记录列表 (HKConnectChange对象)
            target_date: 目标调整日期

        返回:
            {
                "layer1_hsci": {...},      # HSCI预测准确率
                "layer2_connect": {...},    # 港股通映射准确率
                "deviation_analysis": {...},# 偏差分析
                "backtest_date": str,
                "target_date": str,
            }
        """
        # Layer 1: HSCI预测准确率
        layer1 = self._backtest_hsci(prediction, actual_hsci_changes)

        # Layer 2: 港股通映射准确率
        layer2 = self._backtest_connect(prediction, actual_connect_changes)

        # 偏差分析
        deviation = self._analyze_deviations(
            layer1, layer2, prediction,
            actual_hsci_changes, actual_connect_changes
        )

        return {
            "layer1_hsci": layer1,
            "layer2_connect": layer2,
            "deviation_analysis": deviation,
            "backtest_date": datetime.now().isoformat(),
            "target_date": target_date.isoformat(),
        }

    # ================================================================
    # Layer 1: HSCI预测准确率
    # ================================================================

    def _backtest_hsci(self, prediction: Dict, actual_changes: List) -> Dict:
        """回测HSCI预测准确率"""
        # 提取预测的调入调出
        hsci_pred = prediction.get("hsci_prediction", {})
        predicted_adds = {
            s["code"].strip().zfill(5) for s in hsci_pred.get("additions", [])
        }
        predicted_rems = {
            s["code"].strip().zfill(5) for s in hsci_pred.get("removals", [])
        }

        # 提取实际的调入调出
        actual_adds = set()
        actual_rems = set()
        for ch in actual_changes:
            code = ch.code.strip().zfill(5)
            if ch.action == "Add":
                actual_adds.add(code)
            elif ch.action == "Delete":
                actual_rems.add(code)

        # 计算匹配
        add_matches = predicted_adds & actual_adds
        add_misses = actual_adds - predicted_adds  # 实际调入但未预测
        add_false_pos = predicted_adds - actual_adds  # 预测调入但实际未调入

        rem_matches = predicted_rems & actual_rems
        rem_misses = actual_rems - predicted_rems
        rem_false_pos = predicted_rems - actual_rems

        # 计算指标
        add_metrics = self._calc_prf(
            len(add_matches), len(add_misses), len(add_false_pos)
        )
        rem_metrics = self._calc_prf(
            len(rem_matches), len(rem_misses), len(rem_false_pos)
        )

        # 综合指标
        total_matches = len(add_matches) + len(rem_matches)
        total_actual = len(actual_adds) + len(actual_rems)
        total_predicted = len(predicted_adds) + len(predicted_rems)
        overall_precision = total_matches / total_predicted if total_predicted > 0 else 0
        overall_recall = total_matches / total_actual if total_actual > 0 else 0
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0 else 0
        )

        return {
            "add": {
                "precision": round(add_metrics["precision"], 4),
                "recall": round(add_metrics["recall"], 4),
                "f1": round(add_metrics["f1"], 4),
                "matches": len(add_matches),
                "actual": len(actual_adds),
                "predicted": len(predicted_adds),
                "missed": sorted(add_misses),
                "false_positive": sorted(add_false_pos),
            },
            "remove": {
                "precision": round(rem_metrics["precision"], 4),
                "recall": round(rem_metrics["recall"], 4),
                "f1": round(rem_metrics["f1"], 4),
                "matches": len(rem_matches),
                "actual": len(actual_rems),
                "predicted": len(predicted_rems),
                "missed": sorted(rem_misses),
                "false_positive": sorted(rem_false_pos),
            },
            "overall": {
                "precision": round(overall_precision, 4),
                "recall": round(overall_recall, 4),
                "f1": round(overall_f1, 4),
                "total_matches": total_matches,
                "total_actual": total_actual,
                "total_predicted": total_predicted,
            },
        }

    # ================================================================
    # Layer 2: 港股通映射准确率
    # ================================================================

    def _backtest_connect(self, prediction: Dict, actual_changes: List) -> Dict:
        """回测港股通映射准确率"""
        # 提取预测的港股通调入调出
        connect_pred = prediction.get("connect_prediction", {})
        predicted_adds = {
            s["code"].strip().zfill(5) for s in connect_pred.get("connect_additions", [])
        }
        predicted_rems = {
            s["code"].strip().zfill(5) for s in connect_pred.get("connect_removals", [])
        }

        # 提取实际的港股通调入调出
        actual_adds = set()
        actual_rems = set()
        for ch in actual_changes:
            code = ch.code.strip().zfill(5)
            if ch.action == "调进":
                actual_adds.add(code)
            elif ch.action == "调出":
                actual_rems.add(code)

        # 计算匹配
        add_matches = predicted_adds & actual_adds
        add_misses = actual_adds - predicted_adds
        add_false_pos = predicted_adds - actual_adds

        rem_matches = predicted_rems & actual_rems
        rem_misses = actual_rems - predicted_rems
        rem_false_pos = predicted_rems - actual_rems

        # 计算指标
        add_metrics = self._calc_prf(
            len(add_matches), len(add_misses), len(add_false_pos)
        )
        rem_metrics = self._calc_prf(
            len(rem_matches), len(rem_misses), len(rem_false_pos)
        )

        total_matches = len(add_matches) + len(rem_matches)
        total_actual = len(actual_adds) + len(actual_rems)
        total_predicted = len(predicted_adds) + len(predicted_rems)
        overall_precision = total_matches / total_predicted if total_predicted > 0 else 0
        overall_recall = total_matches / total_actual if total_actual > 0 else 0
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0 else 0
        )

        return {
            "add": {
                "precision": round(add_metrics["precision"], 4),
                "recall": round(add_metrics["recall"], 4),
                "f1": round(add_metrics["f1"], 4),
                "matches": len(add_matches),
                "actual": len(actual_adds),
                "predicted": len(predicted_adds),
                "missed": sorted(add_misses),
                "false_positive": sorted(add_false_pos),
            },
            "remove": {
                "precision": round(rem_metrics["precision"], 4),
                "recall": round(rem_metrics["recall"], 4),
                "f1": round(rem_metrics["f1"], 4),
                "matches": len(rem_matches),
                "actual": len(actual_rems),
                "predicted": len(predicted_rems),
                "missed": sorted(rem_misses),
                "false_positive": sorted(rem_false_pos),
            },
            "overall": {
                "precision": round(overall_precision, 4),
                "recall": round(overall_recall, 4),
                "f1": round(overall_f1, 4),
                "total_matches": total_matches,
                "total_actual": total_actual,
                "total_predicted": total_predicted,
            },
        }

    # ================================================================
    # 偏差分析
    # ================================================================

    def _analyze_deviations(
        self,
        layer1: Dict,
        layer2: Dict,
        prediction: Dict,
        actual_hsci: List,
        actual_connect: List,
    ) -> Dict:
        """分析预测偏差原因"""
        analysis = {
            "hsci_add_misses": [],
            "hsci_add_false_positives": [],
            "hsci_remove_misses": [],
            "hsci_remove_false_positives": [],
            "connect_add_misses": [],
            "connect_add_false_positives": [],
            "connect_remove_misses": [],
            "connect_remove_false_positives": [],
            "summary": "",
        }

        # 构建查找映射
        hsci_pred_adds = {
            s["code"].strip().zfill(5): s
            for s in prediction.get("hsci_prediction", {}).get("additions", [])
        }
        hsci_pred_rems = {
            s["code"].strip().zfill(5): s
            for s in prediction.get("hsci_prediction", {}).get("removals", [])
        }
        connect_pred_adds = {
            s["code"].strip().zfill(5): s
            for s in prediction.get("connect_prediction", {}).get("connect_additions", [])
        }
        connect_pred_rems = {
            s["code"].strip().zfill(5): s
            for s in prediction.get("connect_prediction", {}).get("connect_removals", [])
        }

        # 实际变动查找映射
        actual_hsci_map = {}
        for ch in actual_hsci:
            code = ch.code.strip().zfill(5)
            actual_hsci_map[code] = ch
        actual_connect_map = {}
        for ch in actual_connect:
            code = ch.code.strip().zfill(5)
            actual_connect_map[code] = ch

        # 分析HSCI调入遗漏 (实际调入但未预测)
        for code in layer1["add"]["missed"]:
            ch = actual_hsci_map.get(code)
            analysis["hsci_add_misses"].append({
                "code": code,
                "name": ch.name if ch else "",
                "reason": self._infer_hsci_add_miss_reason(code, hsci_pred_adds, actual_hsci_map),
            })

        # 分析HSCI调入误报 (预测调入但实际未调入)
        for code in layer1["add"]["false_positive"]:
            pred = hsci_pred_adds.get(code, {})
            analysis["hsci_add_false_positives"].append({
                "code": code,
                "name": pred.get("name", ""),
                "coverage": pred.get("coverage", 0),
                "reason": self._infer_hsci_add_false_pos_reason(pred),
            })

        # 分析HSCI调出遗漏
        for code in layer1["remove"]["missed"]:
            ch = actual_hsci_map.get(code)
            analysis["hsci_remove_misses"].append({
                "code": code,
                "name": ch.name if ch else "",
                "reason": self._infer_hsci_remove_miss_reason(code, hsci_pred_rems, actual_hsci_map),
            })

        # 分析HSCI调出误报
        for code in layer1["remove"]["false_positive"]:
            pred = hsci_pred_rems.get(code, {})
            analysis["hsci_remove_false_positives"].append({
                "code": code,
                "name": pred.get("name", ""),
                "coverage": pred.get("coverage", 0),
                "reason": self._infer_hsci_remove_false_pos_reason(pred),
            })

        # 分析港股通调入遗漏
        for code in layer2["add"]["missed"]:
            ch = actual_connect_map.get(code)
            analysis["connect_add_misses"].append({
                "code": code,
                "name": ch.name_en if ch else "",
                "reason": self._infer_connect_add_miss_reason(
                    code, connect_pred_adds, actual_connect_map, hsci_pred_adds
                ),
            })

        # 分析港股通调入误报
        for code in layer2["add"]["false_positive"]:
            pred = connect_pred_adds.get(code, {})
            analysis["connect_add_false_positives"].append({
                "code": code,
                "name": pred.get("name", ""),
                "eligibility": pred.get("eligibility", ""),
                "reason": self._infer_connect_add_false_pos_reason(pred),
            })

        # 分析港股通调出遗漏
        for code in layer2["remove"]["missed"]:
            ch = actual_connect_map.get(code)
            analysis["connect_remove_misses"].append({
                "code": code,
                "name": ch.name_en if ch else "",
                "reason": self._infer_connect_remove_miss_reason(
                    code, connect_pred_rems, actual_connect_map, hsci_pred_rems
                ),
            })

        # 分析港股通调出误报
        for code in layer2["remove"]["false_positive"]:
            pred = connect_pred_rems.get(code, {})
            analysis["connect_remove_false_positives"].append({
                "code": code,
                "name": pred.get("name", ""),
                "reason": self._infer_connect_remove_false_pos_reason(pred),
            })

        # 汇总
        hsci_missed = len(analysis["hsci_add_misses"]) + len(analysis["hsci_remove_misses"])
        hsci_false = len(analysis["hsci_add_false_positives"]) + len(analysis["hsci_remove_false_positives"])
        connect_missed = len(analysis["connect_add_misses"]) + len(analysis["connect_remove_misses"])
        connect_false = len(analysis["connect_add_false_positives"]) + len(analysis["connect_remove_false_positives"])

        analysis["summary"] = (
            f"HSCI层: 漏报{hsci_missed}只, 误报{hsci_false}只; "
            f"港股通层: 漏报{connect_missed}只, 误报{connect_false}只"
        )

        return analysis

    # ================================================================
    # 偏差原因推断
    # ================================================================

    @staticmethod
    def _infer_hsci_add_miss_reason(code, pred_adds, actual_map):
        ch = actual_map.get(code)
        if not ch:
            return "实际变动记录中找不到该股票"
        return (
            f"实际于{ch.effective_date}调入HSCI, "
            f"但预测未命中 (可能因市值数据时点差异或覆盖率计算偏差)"
        )

    @staticmethod
    def _infer_hsci_add_false_pos_reason(pred):
        cov = pred.get("coverage", 0)
        if cov > 94:
            return f"覆盖率{cov:.1f}%接近94%边界, 属于灰色区间误判"
        return f"覆盖率{cov:.1f}%, 预测调入但实际未调入 (可能因缓冲区保护或数据时点差异)"

    @staticmethod
    def _infer_hsci_remove_miss_reason(code, pred_rems, actual_map):
        ch = actual_map.get(code)
        if not ch:
            return "实际变动记录中找不到该股票"
        return (
            f"实际于{ch.effective_date}从HSCI调出, "
            f"但预测未命中 (可能因市值数据时点差异或该股票在缓冲区内)"
        )

    @staticmethod
    def _infer_hsci_remove_false_pos_reason(pred):
        cov = pred.get("coverage", 0)
        if 95 < cov <= 96:
            return f"覆盖率{cov:.1f}%在95-96%灰色区间, 预测调出但实际被缓冲保留"
        return f"覆盖率{cov:.1f}%, 预测调出但实际未调出 (可能因数据时点差异)"

    @staticmethod
    def _infer_connect_add_miss_reason(code, pred_adds, actual_map, hsci_pred_adds):
        # 检查是否HSCI预测也遗漏了
        if code not in hsci_pred_adds:
            return "HSCI预测层即遗漏, 港股通层自然也遗漏"
        # HSCI预测了但港股通映射层遗漏
        return "HSCI预测调入但港股通资格检查未通过 (可能因第二上市/WVR/小型股市值门槛)"

    @staticmethod
    def _infer_connect_add_false_pos_reason(pred):
        elig = pred.get("eligibility", "")
        return f"预测港股通调入(资格:{elig})但实际未调入 (可能因港股通调整时点与HSCI不同步)"

    @staticmethod
    def _infer_connect_remove_miss_reason(code, pred_rems, actual_map, hsci_pred_rems):
        if code not in hsci_pred_rems:
            return "HSCI预测层即遗漏, 港股通层自然也遗漏"
        return "HSCI预测调出但港股通映射层未生成调出 (可能因该股票不在港股通池中)"

    @staticmethod
    def _infer_connect_remove_false_pos_reason(pred):
        return f"预测港股通调出但实际未调出 (可能因港股通调整规则与HSCI不完全同步)"

    # ================================================================
    # 辅助方法
    # ================================================================

    @staticmethod
    def _calc_prf(matches: int, misses: int, false_pos: int) -> Dict:
        """计算精确率、召回率、F1"""
        actual = matches + misses
        predicted = matches + false_pos

        precision = matches / predicted if predicted > 0 else 0.0
        recall = matches / actual if actual > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matches": matches,
            "actual": actual,
            "predicted": predicted,
        }
