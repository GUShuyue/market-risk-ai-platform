"""
Backtest Agent - 回测评估Agent

职责：对比预测结果与实际调整，计算准确率，分析偏差原因

偏差分析分两层:
    1. 量化原因分析（始终执行）: 基于排名位置、距缓冲区边界距离、财务数据推断原因
    2. LLM深层分析（可选）: 对量化无法解释的案例做定性推断
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class BacktestAgent:
    """
    回测Agent - 验证预测准确性

    关键指标:
        - 调入精确率: 命中数 / 预测调入总数
        - 调入召回率: 命中数 / 实际调入总数
        - 综合F1: 精确率和召回率的调和平均

    偏差分析:
        - 量化层: 排名位置、缓冲区状态、财务数据 → 自动归类原因
        - LLM层（可选）: 行业均衡、专家裁量等非量化因素
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._llm_client = None
        self._prompt_manager = None

    @property
    def llm_client(self):
        """懒加载LLM客户端（仅在use_llm=True时创建）"""
        if self._llm_client is None:
            from core.llm_client import LLMClient
            self._llm_client = LLMClient(self.config)
        return self._llm_client

    @property
    def prompt_manager(self):
        """懒加载Prompt管理器"""
        if self._prompt_manager is None:
            from prompts.prompt_manager import PromptManager
            self._prompt_manager = PromptManager()
        return self._prompt_manager

    def run(self, inputs: dict) -> dict:
        """
        统一接口：执行回测评估

        inputs 必含: prediction, actual_adjustment；可选: rule_result,
                    market_data, use_llm
        返回: {"backtest_result": {...}}
        """
        result = self.backtest(
            prediction=inputs["prediction"],
            actual_adjustment=inputs["actual_adjustment"],
            rule_result=inputs.get("rule_result"),
            market_data=inputs.get("market_data"),
            use_llm=inputs.get("use_llm", False),
        )
        return {"backtest_result": result}

    def backtest(
        self,
        prediction: dict,
        actual_adjustment: dict,
        rule_result: dict = None,
        market_data: dict = None,
        use_llm: bool = False,
    ) -> dict:
        """
        执行回测

        参数:
            prediction: PredictAgent的预测结果
            actual_adjustment: 实际调整数据
            rule_result: RuleAgent的计算结果（含rankings和rules），用于偏差分析
            market_data: DataAgent采集的市场数据（含financial_flags），用于偏差分析
            use_llm: 是否启用LLM深层分析

        返回:
            {
                "accuracy": {...},          # 准确率指标
                "matches": {...},           # 匹配详情
                "misses": {...},            # 遗漏分析
                "false_positives": {...},   # 误报分析
                "deviation_analysis": {...}, # 偏差原因分析
                "backtest_date": str
            }
        """
        # 1. 对比预测与实际调入
        add_match = self._compare_add(
            prediction.get("predicted_add", []),
            actual_adjustment.get("actual_add", [])
        )

        # 2. 对比预测与实际调出
        remove_match = self._compare_remove(
            prediction.get("predicted_remove", []),
            actual_adjustment.get("actual_remove", [])
        )

        # 3. 计算综合指标
        metrics = self._calculate_metrics(add_match, remove_match)

        # 4. 量化原因分析（始终执行）
        analysis = self._analyze_deviation_quantitative(
            add_match, remove_match, prediction, actual_adjustment,
            rule_result, market_data
        )

        # 5. LLM深层分析（可选）
        if use_llm:
            try:
                llm_analysis = self._analyze_deviation_llm(
                    analysis, prediction, actual_adjustment,
                    rule_result, market_data
                )
                analysis["llm_analysis"] = llm_analysis
            except Exception as e:
                logger.warning(f"LLM回测分析失败，使用纯量化结果: {e}")
                analysis["llm_analysis"] = f"LLM分析不可用: {e}"

        return {
            "accuracy": metrics,
            "matches": {
                "add": add_match["matches"],
                "remove": remove_match["matches"]
            },
            "misses": {
                "add": add_match["misses"],
                "remove": remove_match["misses"]
            },
            "false_positives": {
                "add": add_match["false_positives"],
                "remove": remove_match["false_positives"]
            },
            "deviation_analysis": analysis,
            "backtest_date": datetime.now().isoformat()
        }

    # ===== 对比逻辑 =====

    def _compare_add(
        self, predicted: List[dict], actual: List[dict]
    ) -> dict:
        """对比预测调入与实际调入"""
        pred_codes = self._extract_codes(predicted)
        actual_codes = self._extract_codes(actual)

        matches = [s for s in actual if self._get_code(s) in pred_codes]
        misses = [s for s in actual if self._get_code(s) not in pred_codes]
        false_positives = [s for s in predicted if self._get_code(s) not in actual_codes]

        return {"matches": matches, "misses": misses, "false_positives": false_positives}

    def _compare_remove(
        self, predicted: List[dict], actual: List[dict]
    ) -> dict:
        """对比预测调出与实际调出"""
        pred_codes = self._extract_codes(predicted)
        actual_codes = self._extract_codes(actual)

        matches = [s for s in actual if self._get_code(s) in pred_codes]
        misses = [s for s in actual if self._get_code(s) not in pred_codes]
        false_positives = [s for s in predicted if self._get_code(s) not in actual_codes]

        return {"matches": matches, "misses": misses, "false_positives": false_positives}

    def _calculate_metrics(self, add_match: dict, remove_match: dict) -> dict:
        """计算精确率、召回率、F1"""
        add_metrics = self._calc_prf(add_match)
        remove_metrics = self._calc_prf(remove_match)

        total_matches = add_metrics["matches"] + remove_metrics["matches"]
        total_actual = add_metrics["actual"] + remove_metrics["actual"]
        total_predicted = add_metrics["predicted"] + remove_metrics["predicted"]

        overall_precision = (
            total_matches / total_predicted if total_predicted > 0 else 0.0
        )
        overall_recall = (
            total_matches / total_actual if total_actual > 0 else 0.0
        )
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0 else 0.0
        )

        return {
            "add_precision": round(add_metrics["precision"], 4),
            "add_recall": round(add_metrics["recall"], 4),
            "add_f1": round(add_metrics["f1"], 4),
            "remove_precision": round(remove_metrics["precision"], 4),
            "remove_recall": round(remove_metrics["recall"], 4),
            "remove_f1": round(remove_metrics["f1"], 4),
            "overall_precision": round(overall_precision, 4),
            "overall_recall": round(overall_recall, 4),
            "overall_f1": round(overall_f1, 4),
            "overall_accuracy": round(overall_f1, 4),
            "add_match_count": add_metrics["matches"],
            "add_actual_count": add_metrics["actual"],
            "add_predicted_count": add_metrics["predicted"],
            "remove_match_count": remove_metrics["matches"],
            "remove_actual_count": remove_metrics["actual"],
            "remove_predicted_count": remove_metrics["predicted"],
        }

    # ===== 量化原因分析 =====

    def _analyze_deviation_quantitative(
        self, add_match, remove_match, prediction, actual,
        rule_result, market_data
    ) -> dict:
        """
        量化原因分析 - 基于排名位置和财务数据推断偏差原因

        对每只遗漏/误报的股票，结合以下数据生成具体原因:
            - 排名位置（在排名列表中的名次）
            - 距缓冲区边界的距离
            - 财务数据（净利润、ROE、资产负债率、亏损状态、审计意见）
            - 是否为老样本（成分股身份）
        """
        # 构建查找映射
        rankings = (rule_result or {}).get("rankings", [])
        rules = (rule_result or {}).get("rules", {})
        financial_flags = (market_data or {}).get("financial_flags", {})
        current_constituents = (market_data or {}).get("current_constituents", [])

        rank_map = {}
        short_to_full = {}
        for r in rankings:
            code = r.get("code", "")
            if code:
                rank_map[code] = r
                short_to_full[code.split(".")[0]] = code

        # 当前成分股代码集合
        constituent_codes = set()
        for c in current_constituents:
            if isinstance(c, dict):
                code = c.get("con_code") or c.get("ts_code", "")
            else:
                code = str(c)
            if code:
                constituent_codes.add(self._normalize_code(code))

        sample_count = rules.get("sample_count", 300)
        buffer = rules.get("buffer_zone", {})
        buffer_in = buffer.get("priority_in_rank", 240)
        buffer_out = buffer.get("priority_out_rank", 360)

        analysis = {
            "missed_adds": [],
            "missed_removes": [],
            "false_adds": [],
            "false_removes": [],
            "summary": ""
        }

        # 分析遗漏的调入（实际调入但预测未命中）
        for stock in add_match["misses"]:
            info = self._diagnose_missed_add(
                stock, rank_map, short_to_full, financial_flags,
                constituent_codes, buffer_in, sample_count, buffer_out
            )
            analysis["missed_adds"].append(info)

        # 分析遗漏的调出（实际调出但预测未命中）
        for stock in remove_match["misses"]:
            info = self._diagnose_missed_remove(
                stock, rank_map, short_to_full, financial_flags,
                constituent_codes, buffer_in, sample_count, buffer_out
            )
            analysis["missed_removes"].append(info)

        # 分析误报的调入（预测调入但实际未调入）
        for stock in add_match["false_positives"]:
            info = self._diagnose_false_add(
                stock, rank_map, short_to_full, financial_flags,
                constituent_codes, buffer_in, sample_count, buffer_out
            )
            analysis["false_adds"].append(info)

        # 分析误报的调出（预测调出但实际未调出）
        for stock in remove_match["false_positives"]:
            info = self._diagnose_false_remove(
                stock, rank_map, short_to_full, financial_flags,
                constituent_codes, buffer_in, sample_count, buffer_out
            )
            analysis["false_removes"].append(info)

        # 生成总结
        total_missed = len(analysis["missed_adds"]) + len(analysis["missed_removes"])
        total_false = len(analysis["false_adds"]) + len(analysis["false_removes"])
        analysis["summary"] = (
            f"漏报{total_missed}只 (调入{len(analysis['missed_adds'])}只, "
            f"调出{len(analysis['missed_removes'])}只), "
            f"误报{total_false}只 (调入{len(analysis['false_adds'])}只, "
            f"调出{len(analysis['false_removes'])}只)"
        )

        return analysis

    def _diagnose_missed_add(
        self, stock, rank_map, short_to_full,
        financial_flags, constituent_codes,
        buffer_in, sample_count, buffer_out
    ) -> dict:
        """诊断遗漏调入的原因（实际调入但预测未命中）"""
        code = self._get_code(stock)
        full_code = short_to_full.get(code, "")
        rank_info = rank_map.get(full_code, {})
        rank = rank_info.get("rank", 0)
        name = rank_info.get("name", stock.get("stock_name", stock.get("name", "")))
        mv = rank_info.get("avg_total_mv", 0)
        fin = financial_flags.get(full_code, {})
        is_constituent = code in constituent_codes

        reason = self._infer_missed_add_reason(
            rank, buffer_in, sample_count, buffer_out, fin, is_constituent
        )

        return {
            "code": code,
            "name": name,
            "rank": rank,
            "market_cap": mv,
            "is_constituent": is_constituent,
            "financial": self._summarize_financial(fin),
            "reason": reason
        }

    @staticmethod
    def _infer_missed_add_reason(
        rank, buffer_in, sample_count, buffer_out, fin, is_constituent
    ) -> str:
        """推断遗漏调入的具体原因"""
        if rank == 0:
            return "该股票不在排名列表中，可能因上市时间不足、ST状态或财务亏损被准入筛选排除，但实际被专家委员会调入"

        if rank <= buffer_in:
            # 排名在调入缓冲区内（≤240），量化模型应选入
            parts = [f"排名{rank}（≤{buffer_in}，第1优先级），量化模型应选入"]
            if fin.get("is_loss"):
                np = fin.get("net_profit", 0)
                parts.append(
                    f"但该股票财务亏损（净利润{np/1e8:.1f}亿），"
                    f"预测时可能被财务筛选排除，实际调入说明专家委员会未受此限制"
                )
            else:
                parts.append("预测遗漏，可能因LLM调整名单或数据时点差异")
            return "，".join(parts)

        elif rank <= sample_count:
            # 排名241-300，第4优先级，被老样本缓冲区挤占
            return (
                f"排名{rank}（{buffer_in+1}-{sample_count}区间，第4优先级），"
                f"量化模型中被老样本缓冲区（{sample_count+1}-{buffer_out}）挤占；"
                f"实际调整时老样本调出可能较少，释放名额使该股票补位入选"
            )

        elif rank <= buffer_out:
            # 排名301-360
            return (
                f"排名{rank}（{sample_count+1}-{buffer_out}区间），"
                f"非成分股此排名通常不被选入；"
                f"实际调入可能因本次老样本调出较多，该股票作为补位候选入选"
            )

        else:
            # 排名>360
            fin_hint = ""
            if fin.get("is_loss"):
                fin_hint = f"，且财务亏损（净利润{fin.get('net_profit',0)/1e8:.1f}亿）"
            return (
                f"排名{rank}（>{buffer_out}，远超缓冲区范围）{fin_hint}，"
                f"实际调入可能因特殊事件或专家委员会裁量"
            )

    def _diagnose_missed_remove(
        self, stock, rank_map, short_to_full,
        financial_flags, constituent_codes,
        buffer_in, sample_count, buffer_out
    ) -> dict:
        """诊断遗漏调出的原因（实际调出但预测未命中）"""
        code = self._get_code(stock)
        full_code = short_to_full.get(code, "")
        rank_info = rank_map.get(full_code, {})
        rank = rank_info.get("rank", 0)
        name = rank_info.get("name", stock.get("stock_name", stock.get("name", "")))
        mv = rank_info.get("avg_total_mv", 0)
        fin = financial_flags.get(full_code, {})
        is_constituent = code in constituent_codes

        reason = self._infer_missed_remove_reason(
            rank, buffer_in, sample_count, buffer_out, fin, is_constituent
        )

        return {
            "code": code,
            "name": name,
            "rank": rank,
            "market_cap": mv,
            "is_constituent": is_constituent,
            "financial": self._summarize_financial(fin),
            "reason": reason
        }

    @staticmethod
    def _infer_missed_remove_reason(
        rank, buffer_in, sample_count, buffer_out, fin, is_constituent
    ) -> str:
        """推断遗漏调出的具体原因"""
        if rank == 0:
            return "该股票不在排名列表中，可能已退市或数据缺失，实际被调出"

        if rank <= sample_count:
            # 排名≤300，量化模型保留
            parts = [f"排名{rank}（≤{sample_count}），量化模型判定保留"]
            if fin.get("is_loss"):
                np = fin.get("net_profit", 0)
                parts.append(
                    f"但该股票财务亏损（净利润{np/1e8:.1f}亿），"
                    f"专家委员会可能因财务风险将其调出"
                )
            elif fin.get("audit_opinion", "").strip() not in ("", "标准无保留意见", "无保留意见"):
                parts.append(
                    f"且审计意见为'{fin.get('audit_opinion')}'，"
                    f"专家委员会可能因审计风险将其调出"
                )
            else:
                parts.append("实际调出可能因行业均衡调整或专家委员会裁量")
            return "，".join(parts)

        elif rank <= buffer_out:
            # 排名301-360，在调出缓冲区内
            return (
                f"排名{rank}（{sample_count+1}-{buffer_out}，调出缓冲区内），"
                f"量化模型判定缓冲保留；实际调出说明专家委员会未行使缓冲保护"
            )

        else:
            # 排名>360，应该被调出
            return (
                f"排名{rank}（>{buffer_out}），量化模型应判定调出；"
                f"预测遗漏可能因数据时点差异或成分股身份判定问题"
            )

    def _diagnose_false_add(
        self, stock, rank_map, short_to_full,
        financial_flags, constituent_codes,
        buffer_in, sample_count, buffer_out
    ) -> dict:
        """诊断误报调入的原因（预测调入但实际未调入）"""
        code = self._get_code(stock)
        full_code = short_to_full.get(code, "")
        rank_info = rank_map.get(full_code, {})
        rank = rank_info.get("rank", 0)
        name = rank_info.get("name", stock.get("stock_name", stock.get("name", "")))
        mv = rank_info.get("avg_total_mv", 0)
        fin = financial_flags.get(full_code, {})
        is_constituent = code in constituent_codes

        reason = self._infer_false_add_reason(
            rank, buffer_in, sample_count, buffer_out, fin, is_constituent
        )

        return {
            "code": code,
            "name": name,
            "rank": rank,
            "market_cap": mv,
            "is_constituent": is_constituent,
            "financial": self._summarize_financial(fin),
            "reason": reason
        }

    @staticmethod
    def _infer_false_add_reason(
        rank, buffer_in, sample_count, buffer_out, fin, is_constituent
    ) -> str:
        """推断误报调入的具体原因"""
        if rank == 0:
            return "该股票不在排名列表中，预测数据可能存在误差"

        if rank <= buffer_in:
            # 排名≤240，应该被调入
            parts = [f"排名{rank}（≤{buffer_in}，第1优先级），量化模型判定调入"]
            if fin.get("is_loss"):
                np = fin.get("net_profit", 0)
                parts.append(
                    f"但该股票财务亏损（净利润{np/1e8:.1f}亿），"
                    f"专家委员会可能因财务风险拒绝调入"
                )
            elif fin.get("audit_opinion", "").strip() not in ("", "标准无保留意见", "无保留意见"):
                parts.append(
                    f"且审计意见为'{fin.get('audit_opinion')}'，"
                    f"专家委员会可能因审计问题拒绝调入"
                )
            else:
                parts.append("实际未调入可能因行业集中度限制或特殊事件")
            return "，".join(parts)

        elif rank <= sample_count:
            # 排名241-300，边界区域
            return (
                f"排名{rank}（{buffer_in+1}-{sample_count}，第4优先级），"
                f"量化模型预测可补位入选；"
                f"实际调整时老样本缓冲区保留了更多名额，该股票被挤占"
            )

        else:
            # 排名>300
            return (
                f"排名{rank}（>{sample_count}），不在选样范围；"
                f"预测调入可能有误或数据时点差异"
            )

    def _diagnose_false_remove(
        self, stock, rank_map, short_to_full,
        financial_flags, constituent_codes,
        buffer_in, sample_count, buffer_out
    ) -> dict:
        """诊断误报调出的原因（预测调出但实际未调出）"""
        code = self._get_code(stock)
        full_code = short_to_full.get(code, "")
        rank_info = rank_map.get(full_code, {})
        rank = rank_info.get("rank", 0)
        name = rank_info.get("name", stock.get("stock_name", stock.get("name", "")))
        mv = rank_info.get("avg_total_mv", 0)
        fin = financial_flags.get(full_code, {})
        is_constituent = code in constituent_codes

        reason = self._infer_false_remove_reason(
            rank, buffer_in, sample_count, buffer_out, fin, is_constituent
        )

        return {
            "code": code,
            "name": name,
            "rank": rank,
            "market_cap": mv,
            "is_constituent": is_constituent,
            "financial": self._summarize_financial(fin),
            "reason": reason
        }

    @staticmethod
    def _infer_false_remove_reason(
        rank, buffer_in, sample_count, buffer_out, fin, is_constituent
    ) -> str:
        """推断误报调出的具体原因"""
        if rank == 0:
            return "该股票不在排名列表中，数据可能存在误差"

        if rank <= sample_count:
            # 排名≤300，量化模型不应调出
            return (
                f"排名{rank}（≤{sample_count}），量化模型判定保留；"
                f"预测调出有误，可能因数据时点差异"
            )

        elif rank <= buffer_out:
            # 排名301-360，在调出缓冲区内
            return (
                f"排名{rank}（{sample_count+1}-{buffer_out}，调出缓冲区内），"
                f"量化模型判定缓冲保留但接近边界；"
                f"实际未调出说明专家委员会行使了缓冲保护权"
            )

        else:
            # 排名>360，应该被调出
            parts = [f"排名{rank}（>{buffer_out}），量化模型判定调出"]
            parts.append("实际未调出可能因专家委员会特别保留或数据时点差异")
            return "，".join(parts)

    # ===== LLM深层分析 =====

    def _analyze_deviation_llm(
        self, quant_analysis, prediction, actual_adjustment,
        rule_result, market_data
    ) -> str:
        """
        LLM深层分析 - 对量化无法解释的偏差做定性推断

        只把偏差案例（遗漏+误报）发给LLM，不发送全量数据，控制token消耗。
        """
        # 收集需要LLM分析的案例
        cases = []
        for item in quant_analysis.get("missed_adds", []):
            cases.append({
                "type": "遗漏调入",
                "code": item["code"],
                "name": item["name"],
                "rank": item["rank"],
                "financial": item["financial"],
                "quant_reason": item["reason"]
            })
        for item in quant_analysis.get("missed_removes", []):
            cases.append({
                "type": "遗漏调出",
                "code": item["code"],
                "name": item["name"],
                "rank": item["rank"],
                "financial": item["financial"],
                "quant_reason": item["reason"]
            })
        for item in quant_analysis.get("false_adds", []):
            cases.append({
                "type": "误报调入",
                "code": item["code"],
                "name": item["name"],
                "rank": item["rank"],
                "financial": item["financial"],
                "quant_reason": item["reason"]
            })
        for item in quant_analysis.get("false_removes", []):
            cases.append({
                "type": "误报调出",
                "code": item["code"],
                "name": item["name"],
                "rank": item["rank"],
                "financial": item["financial"],
                "quant_reason": item["reason"]
            })

        if not cases:
            return "无偏差案例，预测完全命中实际调整。"

        # 加载prompt模板
        prompt = self.prompt_manager.load("user/backtest_analysis.md")

        # 填充模板
        filled = self.prompt_manager.fill(
            prompt,
            index_name=(rule_result or {}).get("index_name", ""),
            target_date=prediction.get("target_date", ""),
            accuracy_summary=quant_analysis.get("summary", ""),
            deviation_cases=json.dumps(cases, ensure_ascii=False, indent=2)
        )

        # 调用LLM
        response = self.llm_client.chat(filled)
        return response.strip()

    # ===== 辅助方法 =====

    @staticmethod
    def _summarize_financial(fin: dict) -> dict:
        """提取财务数据摘要"""
        if not fin:
            return {"available": False}
        return {
            "available": True,
            "net_profit": fin.get("net_profit", 0),
            "roe": fin.get("roe", 0),
            "debt_ratio": fin.get("debt_ratio", 0),
            "is_loss": fin.get("is_loss", False),
            "audit_opinion": fin.get("audit_opinion", "未知")
        }

    @staticmethod
    def _normalize_code(code: str) -> str:
        """归一化股票代码，去掉交易所后缀"""
        if not code:
            return ""
        return code.split(".")[0]

    @classmethod
    def _extract_codes(cls, stocks: List[dict]) -> set:
        """从股票列表中提取归一化后的代码集合"""
        codes = set()
        for s in stocks:
            code = (
                s.get("stock_code") or s.get("ts_code")
                or s.get("con_code") or s.get("code", "")
            )
            if code:
                codes.add(cls._normalize_code(code))
        return codes

    @classmethod
    def _get_code(cls, stock: dict) -> str:
        """从股票字典中提取归一化后的代码"""
        code = (
            stock.get("stock_code") or stock.get("ts_code")
            or stock.get("con_code") or stock.get("code", "")
        )
        return cls._normalize_code(code)

    @staticmethod
    def _calc_prf(match_result: dict) -> dict:
        """计算精确率、召回率、F1"""
        matches = len(match_result["matches"])
        misses = len(match_result["misses"])
        false_positives = len(match_result["false_positives"])
        actual = matches + misses
        predicted = matches + false_positives

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
