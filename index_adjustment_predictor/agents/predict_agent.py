"""
Predict Agent - 预测分析Agent

职责：结合量化模型和LLM分析预测调入调出

两阶段预测:
    1. 量化模型: 基于排名规则计算调入调出候选
    2. LLM分析: 考虑非量化因素 (行业均衡、风险事件等)
"""

import json
import re
from datetime import datetime
from typing import Dict, List
from core.llm_client import LLMClient
from prompts.prompt_manager import PromptManager
from models.quantitative.rank_model import RankModel


class PredictAgent:
    """
    预测Agent - 量化模型 + LLM增强

    两阶段预测:
        1. 量化模型: 基于排名规则计算调入调出候选
        2. LLM分析: 考虑非量化因素 (行业均衡、ESG等)

    可通过 use_llm 参数控制是否启用LLM增强阶段。
    """

    def __init__(self, config: dict):
        self.config = config
        self.llm_client = LLMClient()
        self.prompt_manager = PromptManager()

    def run(self, inputs: dict) -> dict:
        """
        统一接口：执行预测

        inputs 必含: rule_result, market_data；可选: use_llm
        返回: {"prediction": {...}}
        """
        prediction = self.predict(
            inputs["rule_result"],
            inputs["market_data"],
            use_llm=inputs.get("use_llm", True)
        )
        return {"prediction": prediction}

    def predict(self, rule_result: dict, market_data: dict,
                use_llm: bool = True) -> dict:
        """
        执行预测

        参数:
            rule_result: RuleAgent的计算结果
            market_data: DataAgent采集的市场数据
            use_llm: 是否启用LLM增强分析（默认True）。
                     设为False时仅使用量化排名结果，输出格式不变。

        返回:
            {
                "predicted_add": [...],      # 预测调入（主候选，数量=调出数量）
                "alternative_add": [...],    # 备选调入（实际调入概率低）
                "predicted_remove": [...],   # 预测调出
                "confidence": float,         # 整体置信度
                "reasoning": str,            # 预测理由
                "quant_result": dict,        # 量化阶段结果
                "details": [...]             # 详细分析
            }
        """
        # 阶段1: 量化预测
        quant_result = self._quantitative_predict(rule_result, market_data)

        # 阶段2: LLM增强分析（可跳过）
        if use_llm:
            try:
                llm_analysis = self._llm_enhanced_predict(
                    quant_result, rule_result, market_data
                )
            except Exception as e:
                # LLM不可用时回退到纯量化结果
                llm_analysis = self._build_quant_only_result(quant_result, str(e))
        else:
            # 用户选择跳过LLM，直接使用量化结果
            llm_analysis = self._build_quant_only_result(quant_result)

        # 用排名数据回填正确的股票名称（LLM可能输出"待核实"等错误名称）
        rankings = rule_result.get("rankings", [])
        self._fill_stock_names(llm_analysis, rankings)

        return {
            "predicted_add": llm_analysis.get("predicted_add", []),
            "alternative_add": llm_analysis.get("alternative_add", []),
            "predicted_remove": llm_analysis.get("predicted_remove", []),
            "confidence": llm_analysis.get("confidence", 0.0),
            "reasoning": llm_analysis.get("reasoning", ""),
            "quant_result": quant_result,
            "details": llm_analysis.get("details", []),
            "index_name": rule_result.get("index_name", ""),
            "target_date": market_data.get("target_date", ""),
        }

    def _quantitative_predict(self, rule_result: dict, market_data: dict) -> dict:
        """
        量化模型预测

        使用RankModel基于排名和缓冲区规则识别调入调出候选:
        - 调入候选: 非成分股中排名进入buffer_in的
        - 调出候选: 成分股中排名跌出buffer_out的
        """
        rules = rule_result.get("rules", {})
        sample_count = rules.get("sample_count", 300)
        buffer_in = rules.get("buffer_zone", {}).get("priority_in_rank", 240)
        buffer_out = rules.get("buffer_zone", {}).get("priority_out_rank", 360)

        model = RankModel({
            "sample_count": sample_count,
            "buffer_in": buffer_in,
            "buffer_out": buffer_out
        })

        rankings = rule_result.get("rankings", [])

        # 使用完整的当前成分股列表（300只）
        # 注意: 不能用 current_rank.keys()，它只包含进入 rankings 的成分股，
        # 会漏掉被样本空间剔除的（如ST股），导致调入调出数量不平衡。
        raw_constituents = market_data.get("current_constituents", [])
        current_constituents = [
            c.get("con_code") or c.get("ts_code", "")
            for c in raw_constituents
            if isinstance(c, dict) and (c.get("con_code") or c.get("ts_code"))
        ]

        result = model.predict(rankings, current_constituents)

        # 补充缓冲区状态信息
        for stock in result["add_candidates"]:
            stock["buffer_status"] = "in_buffer"
        for stock in result["remove_candidates"]:
            stock["buffer_status"] = "out_buffer"

        return result

    def _llm_enhanced_predict(
        self, quant_result: dict, rule_result: dict, market_data: dict
    ) -> dict:
        """LLM增强预测，考虑非量化因素"""
        # 加载系统提示词
        try:
            system_prompt = self.prompt_manager.load("system/prediction.md")
        except Exception:
            system_prompt = None

        # 加载预测prompt模板
        prompt = self.prompt_manager.load("user/predict_template.md")

        # 填充所有模板变量
        filled_prompt = self.prompt_manager.fill(
            prompt,
            target_date=market_data.get("target_date", ""),
            cutoff_date=market_data.get("cutoff_date", ""),
            index_name=rule_result.get("index_name", ""),
            current_constituents=self._format_constituents(market_data),
            candidate_rankings=self._format_rankings(rule_result),
            eligibility_check=self._format_eligibility(rule_result),
            financial_data=self._format_financial_data(
                quant_result, market_data
            ),
            quant_result=json.dumps(
                quant_result, ensure_ascii=False, indent=2, default=str
            ),
            index_rules=self._format_rules(rule_result)
        )

        # 调用LLM
        response = self.llm_client.chat(filled_prompt, system_prompt)

        # 解析LLM返回
        return self._parse_llm_response(response)

    def _format_rules(self, rule_result: dict) -> str:
        """格式化规则信息为可读文本"""
        rules = rule_result.get("rules", {})
        lines = []

        lines.append(f"指数: {rules.get('index_name', '')}")
        lines.append(f"样本数量: {rules.get('sample_count', '')}")

        buffer = rules.get("buffer_zone", {})
        lines.append(
            f"缓冲区: 排名{buffer.get('priority_in_rank', '?')}以内优先调入，"
            f"排名{buffer.get('priority_out_rank', '?')}以外优先调出"
        )

        selection = rules.get("selection_method", {})
        if selection:
            lines.append(f"选样方法: {json.dumps(selection, ensure_ascii=False)}")

        return "\n".join(lines)

    def _format_constituents(self, market_data: dict) -> str:
        """格式化当前成分股列表"""
        constituents = market_data.get("current_constituents", [])
        lines = []
        for c in constituents:
            code = c.get("con_code") or c.get("ts_code", "")
            name = c.get("name", "")
            lines.append(f"  {code} {name}")
        return "\n".join(lines) if lines else "无"

    def _format_rankings(self, rule_result: dict) -> str:
        """格式化全部排名信息（紧凑格式，控制token消耗）"""
        rankings = rule_result.get("rankings", [])
        rules = rule_result.get("rules", {})
        current_rank = rule_result.get("current_rank", {})
        buffer_in = rules.get("buffer_zone", {}).get("priority_in_rank", 240)
        buffer_out = rules.get("buffer_zone", {}).get("priority_out_rank", 360)
        lines = []

        # 全部排名，紧凑格式：排名 代码 名称 市值(亿)
        # 用成分股标记区分成分股/非成分股，便于LLM判断调入调出
        constituent_codes = set(current_rank.keys())
        lines.append(
            f"全部排名 (共{len(rankings)}只, 缓冲区: 调入≤{buffer_in}, 调出>{buffer_out}, "
            f"★=当前成分股):"
        )
        for r in rankings:
            mv = r.get("avg_total_mv", 0) or r.get("total_mv", 0)
            is_constituent = r.get("code") in constituent_codes
            marker = "★" if is_constituent else ""
            lines.append(
                f"  {r.get('rank', '?')}. {r.get('code', '')} "
                f"{r.get('name', '')} {mv / 1e8:.0f}亿{marker}"
            )

        return "\n".join(lines)

    def _format_eligibility(self, rule_result: dict) -> str:
        """格式化准入条件检查结果"""
        eligible = rule_result.get("eligible", [])
        ineligible = rule_result.get("ineligible", [])
        lines = []
        lines.append(f"合格候选股票数: {len(eligible)}")
        lines.append(f"不合格股票数: {len(ineligible)}")
        if ineligible:
            lines.append("不合格原因 (前10条):")
            for item in ineligible[:10]:
                lines.append(
                    f"  {item.get('code', '')} {item.get('name', '')}: "
                    f"{item.get('reason', '')}"
                )
        return "\n".join(lines)

    def _format_financial_data(
        self, quant_result: dict, market_data: dict
    ) -> str:
        """
        格式化候选股票的财务数据，供LLM定性分析参考

        只提取调入候选、备选、调出候选的财务数据，避免全量输出导致token超限。
        每只股票展示: 净利润、ROE、资产负债率、是否亏损、审计意见。
        """
        financial_flags = market_data.get("financial_flags", {})
        if not financial_flags:
            return "无财务数据"

        # 收集所有候选股票代码（去重）
        candidate_codes = set()
        for key in ("add_candidates", "add_alternatives", "remove_candidates"):
            for s in quant_result.get(key, []):
                code = s.get("code", "")
                if code:
                    candidate_codes.add(code)

        if not candidate_codes:
            return "无候选股票"

        lines = [f"候选股票财务数据 (共{len(candidate_codes)}只):"]
        lines.append("  代码 | 名称 | 净利润(亿) | ROE(%) | 资产负债率(%) | 亏损 | 审计意见")
        for code in sorted(candidate_codes):
            flag = financial_flags.get(code)
            if not flag:
                lines.append(f"  {code} | - | - | - | - | - | 无数据")
                continue

            net_profit = flag.get("net_profit", 0) or 0
            roe = flag.get("roe", 0) or 0
            debt = flag.get("debt_ratio", 0) or 0
            is_loss = "是" if flag.get("is_loss") else "否"
            audit = flag.get("audit_opinion", "未知")

            lines.append(
                f"  {code} | {net_profit / 1e8:.2f} | {roe:.1f} | "
                f"{debt:.1f} | {is_loss} | {audit}"
            )

        return "\n".join(lines)

    def _format_market_data(self, market_data: dict) -> str:
        """格式化市场数据摘要"""
        lines = []
        market = market_data.get("market_data", [])
        lines.append(f"全市场股票数: {len(market)}")

        financial_flags = market_data.get("financial_flags", {})
        risk_count = sum(
            1 for v in financial_flags.values() if v.get("has_financial_risk")
        )
        lines.append(f"财务异常股票数: {risk_count}")

        events = market_data.get("events_data", {})
        susp = events.get("suspension_status", {})
        lines.append(f"停牌股票数: {len(susp)}")

        return "\n".join(lines)

    def _fill_stock_names(self, llm_analysis: dict, rankings: list):
        """
        用排名数据回填LLM输出的股票名称和代码

        LLM可能输出"待核实"或不准确的名称，用Tushare数据覆盖。
        同时LLM输出的代码可能缺少交易所后缀（如"688235"而非"688235.SH"），
        需要归一化为带后缀的标准格式，否则回测对比时会匹配失败。
        """
        # 构建代码→名称 和 短码→完整码 的映射
        code_to_name = {}
        short_to_full = {}
        for r in rankings:
            code = r.get("code", "")
            name = r.get("name", "")
            if code:
                if name:
                    code_to_name[code] = name
                # 短码 = 去掉 .SH/.SZ/.BJ 后缀
                short = code.split(".")[0]
                short_to_full[short] = code

        for key in ("predicted_add", "predicted_remove", "alternative_add"):
            for item in llm_analysis.get(key, []):
                if not isinstance(item, dict):
                    continue
                code = item.get("stock_code", item.get("code", ""))
                if not code:
                    continue

                # 归一化代码：如果LLM输出的是短码，补全为带后缀的标准码
                if "." not in code and code in short_to_full:
                    full_code = short_to_full[code]
                    item["stock_code"] = full_code
                    code = full_code  # 更新code用于后续名称查找

                # 回填名称
                correct_name = code_to_name.get(code)
                if correct_name:
                    item["stock_name"] = correct_name

    def _build_quant_only_result(
        self, quant_result: dict, error_msg: str = ""
    ) -> dict:
        """
        从量化结果构建与LLM输出格式一致的结果

        参数:
            quant_result: 量化预测结果
            error_msg: 若因LLM错误回退，传入错误信息；主动跳过时留空

        返回:
            与 _parse_llm_response 相同结构的dict
        """
        if error_msg:
            reasoning = f"LLM分析不可用，使用纯量化结果。原因: {error_msg}"
        else:
            add_count = len(quant_result.get("add_candidates", []))
            alt_count = len(quant_result.get("add_alternatives", []))
            remove_count = len(quant_result.get("remove_candidates", []))
            reasoning = (
                f"纯量化预测（未启用LLM分析）。"
                f"调入候选{add_count}只，备选{alt_count}只，调出候选{remove_count}只。"
                f"预测基于排名规则和缓冲区机制，未考虑行业均衡、风险事件等非量化因素。"
            )

        return {
            "predicted_add": self._stocks_to_prediction(
                quant_result.get("add_candidates", [])
            ),
            "alternative_add": self._stocks_to_prediction(
                quant_result.get("add_alternatives", [])
            ),
            "predicted_remove": self._stocks_to_prediction(
                quant_result.get("remove_candidates", [])
            ),
            "confidence": 0.5,
            "reasoning": reasoning,
            "details": []
        }

    def _stocks_to_prediction(self, stocks: list) -> list:
        """将量化候选转换为预测格式"""
        result = []
        for s in stocks:
            rank = s.get("rank", 0)
            buf_status = s.get("buffer_status", "")
            # 被样本空间剔除的（如ST/退市）不显示无意义的排名数字
            if buf_status == "excluded" or (isinstance(rank, int) and rank >= 999999):
                reason = "样本空间剔除（ST/退市/上市时间不足等），必然调出"
                rank = "-"
            else:
                reason = f"量化排名: {rank}, 缓冲区状态: {buf_status}"
            result.append({
                "stock_code": s.get("code", ""),
                "stock_name": s.get("name", ""),
                "rank": rank,
                "market_cap": s.get("total_mv", 0),
                "confidence": "medium",
                "reasoning": reason
            })
        return result

    def _parse_llm_response(self, response: str) -> dict:
        """解析LLM返回的JSON结果"""
        text = response.strip()

        # 尝试直接解析
        try:
            data = json.loads(text)
            return self._normalize_llm_result(data)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 块
        if "```json" in text:
            start = text.find("```json") + len("```json")
            end = text.find("```", start)
            if end > start:
                try:
                    data = json.loads(text[start:end].strip())
                    return self._normalize_llm_result(data)
                except json.JSONDecodeError:
                    pass

        # 尝试提取第一个 { 到最后一个 }
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                data = json.loads(text[first:last + 1])
                return self._normalize_llm_result(data)
            except json.JSONDecodeError:
                pass

        # 以上都失败，说明JSON被截断（常见于max_tokens不足）
        # 尝试从截断的JSON中恢复已完整的条目
        recovered = self._recover_truncated_json(text)
        if recovered and (recovered.get("predicted_add") or recovered.get("predicted_remove")):
            return self._normalize_llm_result(recovered)

        # 解析失败，返回原始文本作为推理
        return {
            "predicted_add": [],
            "alternative_add": [],
            "predicted_remove": [],
            "confidence": 0.0,
            "reasoning": text,
            "details": []
        }

    def _recover_truncated_json(self, text: str) -> dict:
        """
        从被截断的LLM输出中恢复已完整的条目

        当max_tokens不足导致JSON输出中途截断时，标准的json.loads会失败。
        本方法用正则匹配出每个完整的对象（花括号闭合），尽量恢复数据。

        参数:
            text: LLM返回的原始文本（可能含```json包裹）

        返回:
            恢复出的部分结果 dict，无恢复数据时返回空dict
        """
        result = {}

        for key in ("predicted_add", "predicted_remove", "alternative_add"):
            # 匹配 "key": [ 之后到数组结束或文本结束的部分
            pattern = rf'"{key}"\s*:\s*\['
            match = re.search(pattern, text)
            if not match:
                continue

            arr_start = match.end()
            # 从数组起始位置往后提取所有完整的 { ... } 对象
            items = []
            pos = arr_start
            depth = 0
            obj_start = -1

            while pos < len(text):
                ch = text[pos]
                if ch == "{":
                    if depth == 0:
                        obj_start = pos
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and obj_start != -1:
                        # 提取一个完整的对象
                        obj_str = text[obj_start:pos + 1]
                        try:
                            item = json.loads(obj_str)
                            items.append(item)
                        except json.JSONDecodeError:
                            pass
                        obj_start = -1
                elif ch == "]" and depth == 0:
                    # 数组正常结束
                    break
                pos += 1

            if items:
                result[key] = items

        return result

    def _normalize_llm_result(self, data: dict) -> dict:
        """标准化LLM返回结果，统一字段名和格式"""
        confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.3}

        def _normalize_items(raw_items: list) -> list:
            """将LLM输出的候选列表标准化"""
            result = []
            for item in raw_items:
                if isinstance(item, str):
                    result.append({
                        "stock_code": item, "confidence": 0.6
                    })
                elif isinstance(item, dict):
                    conf = item.get("confidence", "medium")
                    if isinstance(conf, str):
                        item["confidence"] = confidence_map.get(
                            conf.lower(), 0.6
                        )
                    result.append(item)
            return result

        predicted_add = _normalize_items(data.get("predicted_add", []))
        predicted_remove = _normalize_items(data.get("predicted_remove", []))
        alternative_add = _normalize_items(data.get("alternative_add", []))

        # 兜底: 如果LLM没有输出alternative_add，但predicted_add数量
        # 多于predicted_remove，则自动拆分多余的到alternative_add
        if (not alternative_add
                and len(predicted_remove) > 0
                and len(predicted_add) > len(predicted_remove)):
            split_at = len(predicted_remove)
            alternative_add = predicted_add[split_at:]
            predicted_add = predicted_add[:split_at]

        # 整体置信度（只统计主候选）
        all_confs = [
            p.get("confidence", 0.5)
            for p in predicted_add + predicted_remove
            if isinstance(p.get("confidence"), (int, float))
        ]
        overall = sum(all_confs) / len(all_confs) if all_confs else 0.5

        return {
            "predicted_add": predicted_add,
            "alternative_add": alternative_add,
            "predicted_remove": predicted_remove,
            "confidence": overall,
            "reasoning": data.get("summary", data.get("reasoning", "")),
            "details": data.get("risk_factors", [])
        }
