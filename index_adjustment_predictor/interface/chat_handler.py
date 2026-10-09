"""
自然语言交互处理器

混合意图识别：正则快速通道 + LLM兜底
支持多轮对话
"""

import re
import json
import os
import glob
from typing import Dict, Optional, List
from core.orchestrator import Orchestrator
from core.llm_client import LLMClient
from utils.date_utils import get_cutoff_date


class ChatHandler:
    """
    对话处理器 - 自然语言交互入口
    
    意图识别策略:
        1. 先用正则快速匹配（标准说法秒回）
        2. 匹配不到则调用LLM做意图识别（非标准说法也能理解）
    
    支持的意图:
        - predict: 预测调整
        - backtest: 回测验证
        - query_rules: 查询规则
        - help: 帮助
    
    支持多轮对话:
        - 记录上一轮的意图和参数
        - 用户说"沪深300呢"时继承上文
    """

    # 正则意图匹配模式（快速通道）
    INTENT_PATTERNS = {
        "predict": [
            r"预测.*调整",
            r"调入调出",
            r"成分股.*变化",
            r"哪些股票.*调入",
            r"预测.*沪深300",
        ],
        "backtest": [
            r"回测",
            r"验证.*准确",
            r"历史表现",
            r"预测.*准不准",
            r"对比.*实际"
        ],
        "query_rules": [
            r"查看.*规则",
            r"指数.*编制",
            r"调整.*时间",
            r".*编制规则"
        ],
        "help": [
            r"帮助",
            r"怎么用",
            r"能做什么"
        ]
    }

    # 指数名称映射
    INDEX_MAP = {
        "沪深300": "csi300",
        "沪深三百": "csi300",
        "300": "csi300",
    }

    # LLM意图识别 prompt
    LLM_INTENT_PROMPT = """你是一个意图识别助手。请分析用户的输入，返回JSON格式的意图和参数。

支持的四种意图：
1. predict - 预测指数调整（用户想知道某指数会调入/调出哪些股票）
2. backtest - 回测验证（用户想验证之前预测的准确性）
3. query_rules - 查询编制规则（用户想了解指数的编制规则或调整时间）
4. help - 用户需要帮助

支持的指数：
- csi300 (沪深300)

用户输入: "{user_input}"

如果用户的输入省略了某些信息（如没说指数名或时间），请用null表示。
如果用户说"沪深300呢"这类依赖上文的话，target_date和index_name用"inherit"表示继承上文。
如果用户要求纯量化模式（如"纯量化"、"不用大模型"、"仅量化"、"跳过大模型"等），use_llm设为false，否则为true。

请返回JSON:
```json
{{
    "intent": "predict|backtest|query_rules|help",
    "target_date": "YYYY-MM 或 null 或 inherit",
    "index_name": "csi300 或 null 或 inherit",
    "index_display": "中文显示名 或 null",
    "use_llm": "true 或 false"
}}
```"""

    def __init__(self, config: dict):
        self.config = config
        self.orchestrator = Orchestrator(config)
        self.llm_client = LLMClient()
        # 多轮对话上下文
        self._last_intent: Optional[str] = None
        self._last_params: Optional[dict] = None
        # 最近一次工作流产生的报告路径（供 process_with_meta 读取）
        self._last_report_path: str = ""
        # 最近一次预测产生的大模型分析报告 HTML 路径
        self._last_analysis_report_html_path: str = ""

    def process(self, user_input: str) -> str:
        """
        处理用户自然语言输入

        参数:
            user_input: 用户输入文本

        返回:
            自然语言回复
        """
        # 1. 正则快速匹配
        intent = self._regex_intent(user_input)
        
        # 2. 正则匹配不到，调用LLM
        if intent == "unknown":
            llm_result = self._llm_intent(user_input)
            intent = llm_result.get("intent", "unknown")
            params = self._extract_params_from_llm(llm_result, user_input)
        else:
            params = self._extract_params(user_input)

        # 3. 处理继承上文的情况
        params = self._handle_inheritance(params)

        # 4. 记录本轮上下文
        self._last_intent = intent
        self._last_params = params

        # 5. 执行对应工作流
        if intent == "predict":
            return self._handle_predict(params)
        elif intent == "backtest":
            return self._handle_backtest(params)
        elif intent == "query_rules":
            return self._handle_query_rules(params)
        elif intent == "help":
            return self._default_response()
        else:
            # LLM也识别不了，返回帮助
            return self._default_response()

    def process_with_meta(self, user_input: str) -> dict:
        """
        处理用户自然语言输入，返回带元信息的结果（供 Web 前端使用）

        返回:
            {
                "reply": str,          # 自然语言回复（markdown 文本）
                "report_path": str,    # 本轮生成的报告文件路径，无则为空串
                "analysis_report_html_path": str,  # 大模型分析报告 HTML 路径
                "intent": str,         # 识别出的意图: predict/backtest/query_rules/help/unknown
            }
        """
        # 每次处理前重置报告路径
        self._last_report_path = ""
        self._last_analysis_report_html_path = ""
        reply = self.process(user_input)
        return {
            "reply": reply,
            "report_path": self._last_report_path,
            "analysis_report_html_path": self._last_analysis_report_html_path,
            "intent": self._last_intent or "unknown",
        }

    def _regex_intent(self, text: str) -> str:
        """正则快速意图匹配"""
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent
        return "unknown"

    def _extract_params(self, text: str) -> dict:
        """从输入中提取参数（正则方式）"""
        params = {}

        # 提取指数名称
        for name, code in self.INDEX_MAP.items():
            if name in text:
                params["index_name"] = code
                params["index_display"] = name
                break

        # 提取时间 (如 "2024年6月" -> "2024-06")
        time_match = re.search(r"(\d{4})年(\d{1,2})月", text)
        if time_match:
            params["target_date"] = f"{time_match.group(1)}-{int(time_match.group(2)):02d}"

        # 检测是否要求纯量化模式（不用大模型/LLM）
        if re.search(r"纯量化|不用大模型|不用LLM|不用llm|仅量化|只用量化|跳过大模型|跳过LLM|跳过llm", text):
            params["use_llm"] = False

        return params

    def _llm_intent(self, user_input: str) -> dict:
        """调用LLM做意图识别"""
        prompt = self.LLM_INTENT_PROMPT.format(user_input=user_input)
        try:
            response = self.llm_client.chat(prompt, temperature=0.1)
            # 解析JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, Exception):
            pass
        return {"intent": "unknown"}

    def _extract_params_from_llm(self, llm_result: dict, user_input: str) -> dict:
        """从LLM识别结果中提取参数"""
        params = {}
        
        target_date = llm_result.get("target_date")
        if target_date and target_date not in ("null", "inherit"):
            params["target_date"] = target_date

        index_name = llm_result.get("index_name")
        if index_name and index_name not in ("null", "inherit"):
            params["index_name"] = index_name
            params["index_display"] = llm_result.get("index_display", index_name)

        # 如果LLM没提取到指数，尝试正则补充
        if not params.get("index_name"):
            for name, code in self.INDEX_MAP.items():
                if name in user_input:
                    params["index_name"] = code
                    params["index_display"] = name
                    break

        # 提取use_llm参数
        use_llm = llm_result.get("use_llm")
        if use_llm is not None and str(use_llm).lower() == "false":
            params["use_llm"] = False

        return params

    def _handle_inheritance(self, params: dict) -> dict:
        """处理继承上文的情况"""
        if not self._last_params:
            return params

        # 如果当前参数有inherit标记或缺失，从上文继承
        merged = dict(self._last_params)
        merged.update(params)
        return merged

    def _handle_predict(self, params: dict) -> str:
        """处理预测意图"""
        if not params.get("target_date"):
            return '请指定要预测的调整时间，如"2024年6月"'
        if not params.get("index_name"):
            return "请指定要预测的指数: 沪深300"

        # 自动计算cutoff_date
        if not params.get("cutoff_date"):
            try:
                params["cutoff_date"] = get_cutoff_date(params["target_date"])
            except ValueError:
                return f"不支持的调整时间: {params['target_date']}，仅支持6月或12月调整"

        index_display = params.get("index_display", params["index_name"])
        cutoff = params["cutoff_date"]

        result = self.orchestrator.run_workflow("predict_adjustment", params)
        self._last_report_path = result.get("report_path", "")
        self._last_analysis_report_html_path = result.get("analysis_report_html_path", "")
        return self._format_predict_response(result, params, index_display, cutoff)

    def _handle_backtest(self, params: dict) -> str:
        """处理回测意图"""
        if not params.get("target_date"):
            return '请指定要回测的预测时间，如"2024年6月"'
        if not params.get("index_name"):
            return "请指定要回测的指数: 沪深300"

        target_date = params["target_date"]
        index_name = params["index_name"]
        use_llm = params.get("use_llm", True)

        # 1. 加载已有预测结果（含rule_result和market_data用于偏差分析）
        prediction_data = self._load_prediction(
            index_name, target_date, use_llm=use_llm
        )
        if prediction_data is None:
            mode_hint = "纯量化" if not use_llm else ""
            return (
                f"未找到 {target_date} {index_name} 的{mode_hint}预测结果，请先运行预测。\n"
                f'例如：{"纯量化预测" if not use_llm else "预测"}{target_date} '
                f'{params.get("index_display", index_name)}的调整'
            )

        # 2. 获取实际调整数据
        try:
            actual_add, actual_remove = \
                self.orchestrator.data_agent.index_fetcher.get_adjustment_history_by_date(
                    index_name=index_name,
                    target_date=target_date
                )
        except Exception as e:
            return f"获取实际调整数据失败: {e}"

        if not actual_add and not actual_remove:
            return (
                f"未获取到 {target_date} 的实际调整数据，"
                "可能该日期尚未发生调整或数据不可用。"
            )

        # 3. 组装参数交给orchestrator执行回测
        params["prediction"] = prediction_data.get("prediction", {})
        params["rule_result"] = prediction_data.get("rule_result")
        params["market_data"] = prediction_data.get("market_data")
        params["actual_adjustment"] = {
            "actual_add": actual_add,
            "actual_remove": actual_remove
        }

        result = self.orchestrator.run_workflow("backtest", params)
        self._last_report_path = result.get("report_path", "")
        return self._format_backtest_response(result, params)

    def _load_prediction(
        self, index_name: str, target_date: str, use_llm: bool = True
    ) -> Optional[dict]:
        """
        从 reports 目录加载已有预测结果

        文件名格式:
            - LLM模式: prediction_{index_name}_{target_date}.json
            - 纯量化模式: prediction_{index_name}_{target_date}_no_llm.json
        返回完整预测数据（含prediction/rule_result/market_data），找不到返回 None
        """
        report_dir = "reports"
        suffix = "" if use_llm else "_no_llm"
        pattern = os.path.join(
            report_dir,
            f"prediction_{index_name}_{target_date}{suffix}.json"
        )
        files = glob.glob(pattern)
        if not files:
            return None

        # 取最新的
        files.sort(key=os.path.getmtime, reverse=True)
        try:
            with open(files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, OSError):
            return None

    def _handle_query_rules(self, params: dict) -> str:
        """处理规则查询意图"""
        if not params.get("index_name"):
            return (
                "请指定要查询的指数: 沪深300\n"
                '也可直接说"查看沪深300的编制规则"'
            )

        index_display = params.get("index_display", params["index_name"])
        
        # 读取规则文件的Markdown正文
        rule_files = {
            "csi300": "rules/csi300.md",
        }
        rule_file = rule_files.get(params["index_name"])
        if not rule_file:
            return f"不支持的指数: {params['index_name']}，目前仅支持: csi300"

        try:
            with open(rule_file, "r", encoding="utf-8") as f:
                content = f.read()
            # 去掉YAML frontmatter，只返回Markdown正文
            parts = content.split("---", 2)
            if len(parts) >= 3:
                rule_text = parts[2].strip()
            else:
                rule_text = content

            return f"## {index_display}编制规则\n\n{rule_text}"
        except FileNotFoundError:
            return f"规则文件不存在: {rule_file}"

    def _format_predict_response(
        self, result: dict, params: dict, index_display: str, cutoff: str
    ) -> str:
        """格式化预测结果为自然语言"""
        prediction = result.get("prediction", {})
        
        add_list = prediction.get("predicted_add", [])
        alt_list = prediction.get("alternative_add", [])
        remove_list = prediction.get("predicted_remove", [])
        confidence = prediction.get("confidence", 0)
        reasoning = prediction.get("reasoning", "")
        report_path = result.get("report_path", "")

        lines = []
        lines.append(f"## {params['target_date']} {index_display} 调整预测")
        lines.append(f"数据截止日: {cutoff}")
        use_llm = params.get("use_llm", True)
        lines.append(f"预测模式: {'量化+LLM' if use_llm else '纯量化'}")
        lines.append("")

        if add_list:
            lines.append(f"### 预测调入 ({len(add_list)}只)")
            for stock in add_list:
                code = stock.get("stock_code", "")
                name = stock.get("stock_name", "")
                rank = stock.get("rank", "")
                conf = stock.get("confidence", "")
                reason = stock.get("reasoning", "")
                lines.append(f"- {code} {name} (排名{rank}, 置信度: {conf})")
                if reason:
                    lines.append(f"  理由: {reason}")
        else:
            lines.append("### 预测调入: 无")

        if alt_list:
            lines.append("")
            lines.append(f"### 备选调入候选 ({len(alt_list)}只)")
            for stock in alt_list:
                code = stock.get("stock_code", "")
                name = stock.get("stock_name", "")
                rank = stock.get("rank", "")
                conf = stock.get("confidence", "")
                reason = stock.get("reasoning", "")
                lines.append(f"- {code} {name} (排名{rank}, 置信度: {conf})")
                if reason:
                    lines.append(f"  理由: {reason}")

        lines.append("")

        if remove_list:
            lines.append(f"### 预测调出 ({len(remove_list)}只)")
            for stock in remove_list:
                code = stock.get("stock_code", "")
                name = stock.get("stock_name", "")
                rank = stock.get("rank", "")
                conf = stock.get("confidence", "")
                reason = stock.get("reasoning", "")
                lines.append(f"- {code} {name} (排名{rank}, 置信度: {conf})")
                if reason:
                    lines.append(f"  理由: {reason}")
        else:
            lines.append("### 预测调出: 无")

        lines.append("")
        lines.append(f"整体置信度: {confidence}")

        if reasoning:
            lines.append("")
            lines.append(f"分析说明: {reasoning}")

        if report_path:
            lines.append("")
            lines.append(f"完整报告: {report_path}")

        return "\n".join(lines)

    def _format_backtest_response(self, result: dict, params: dict) -> str:
        """格式化回测结果为自然语言"""
        backtest = result.get("backtest_result", {})
        accuracy = backtest.get("accuracy", {})
        report_path = result.get("report_path", "")

        index_display = params.get("index_display", params.get("index_name", ""))
        use_llm = params.get("use_llm", True)
        mode_label = "量化+LLM" if use_llm else "纯量化"

        lines = []
        lines.append(f"## {params['target_date']} {index_display} 回测结果（{mode_label}）")
        lines.append("")

        # 准确率指标
        lines.append("### 准确率指标")
        lines.append(f"- 调入精确率: {accuracy.get('add_precision', 0):.1%}")
        lines.append(f"- 调入召回率: {accuracy.get('add_recall', 0):.1%}")
        lines.append(f"- 调出精确率: {accuracy.get('remove_precision', 0):.1%}")
        lines.append(f"- 调出召回率: {accuracy.get('remove_recall', 0):.1%}")
        lines.append(f"- 综合F1: {accuracy.get('overall_f1', 0):.1%}")
        lines.append("")

        # 匹配情况
        matches = backtest.get("matches", {})
        add_matches = matches.get("add", [])
        remove_matches = matches.get("remove", [])
        lines.append(f"### 匹配情况")
        lines.append(f"- 调入命中: {len(add_matches)}只")
        lines.append(f"- 调出命中: {len(remove_matches)}只")
        lines.append("")

        # 遗漏和误报
        misses = backtest.get("misses", {})
        false_positives = backtest.get("false_positives", {})
        lines.append(f"- 遗漏: 调入{len(misses.get('add', []))}只, 调出{len(misses.get('remove', []))}只")
        lines.append(f"- 误报: 调入{len(false_positives.get('add', []))}只, 调出{len(false_positives.get('remove', []))}只")

        # 偏差分析
        deviation = backtest.get("deviation_analysis", {})
        if deviation:
            lines.append("")
            lines.append("### 偏差分析")
            lines.append(deviation.get("summary", ""))

            # 列出遗漏和误报的具体原因
            for key, label in [
                ("missed_adds", "遗漏调入"),
                ("missed_removes", "遗漏调出"),
                ("false_adds", "误报调入"),
                ("false_removes", "误报调出"),
            ]:
                items = deviation.get(key, [])
                if items:
                    lines.append(f"\n**{label}** ({len(items)}只):")
                    for item in items:
                        name = item.get("name", "")
                        rank = item.get("rank", 0)
                        reason = item.get("reason", "")
                        lines.append(f"  - {name} (排名{rank}): {reason}")

            # LLM深层分析
            llm_text = deviation.get("llm_analysis", "")
            if llm_text and "不可用" not in llm_text and "失败" not in llm_text:
                lines.append("")
                lines.append("### LLM深层分析")
                lines.append(llm_text)

        if report_path:
            lines.append("")
            lines.append(f"完整报告: {report_path}")

        return "\n".join(lines)

    def _default_response(self) -> str:
        """默认回复"""
        return (
            "我可以帮您:\n"
            '1. 预测指数调整 - 如"预测2024年6月沪深300的调整"\n'
            '2. 回测验证 - 如"回测2024年6月预测的准确性"\n'
            '3. 查询规则 - 如"查看沪深300的编制规则"\n'
            "\n"
            "您也可以用自然语言描述，比如:\n"
            '- "明年6月那个300会不会换股票"\n'
            '- "上次预测准吗"\n'
            '- "纯量化预测2024年6月沪深300"（跳过LLM分析）\n'
            '- "纯量化回测2024年6月沪深300"（回测纯量化预测结果）'
        )
