"""
港股通监控系统 - 自然语言交互处理器

混合意图识别：正则快速通道 + LLM兜底
支持多轮对话，支持模式A(监控扫描)和模式B(预测/回测)

意图:
    - scan: 模式A 持续预警扫描
    - predict: 模式B 预测港股通调入调出
    - backtest: 模式B 回测验证
    - query_rules: 查询HSCI/港股通编制规则
    - help: 帮助
"""

import re
import os
import sys
import json
import glob
import logging
from datetime import date, datetime
from typing import Dict, Optional, List

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ChatHandler:
    """
    对话处理器 - 港股通监控系统自然语言交互入口

    意图识别策略:
        1. 先用正则快速匹配（标准说法秒回）
        2. 匹配不到则调用LLM做意图识别（非标准说法也能理解）

    支持的意图:
        - scan: 模式A 监控扫描
        - predict: 模式B 预测调整
        - backtest: 模式B 回测验证
        - query_rules: 查询编制规则
        - help: 帮助

    支持多轮对话:
        - 记录上一轮的意图和参数
        - 用户说"回测呢"时继承上文
    """

    # 正则意图匹配模式（快速通道）
    INTENT_PATTERNS = {
        "predict": [
            r"预测.*调整",
            r"预测.*港股通",
            r"预测.*HSCI",
            r"调入调出",
            r"成分股.*变化",
            r"哪些股票.*调入",
            r"哪些股票.*调出",
            r"预测.*调入",
        ],
        "backtest": [
            r"回测",
            r"验证.*准确",
            r"历史表现",
            r"预测.*准不准",
            r"对比.*实际",
        ],
        "scan": [
            r"扫描",
            r"监控",
            r"预警",
            r"风险.*检测",
            r"标的池.*扫描",
            r"模式\s*A",
            r"模式A",
        ],
        "query_rules": [
            r"查看.*规则",
            r"编制.*规则",
            r"调整.*时间",
            r"换手率.*要求",
            r"缓冲区",
            r"市值.*门槛",
            r"HSCI.*规则",
            r"港股通.*规则",
        ],
        "help": [
            r"帮助",
            r"怎么用",
            r"能做什么",
            r"功能",
        ],
    }

    # LLM意图识别 prompt
    LLM_INTENT_PROMPT = """你是一个港股通监控系统的意图识别助手。请分析用户的输入，返回JSON格式的意图和参数。

支持的五种意图：
1. scan - 模式A监控扫描（用户想扫描港股通标的池的风险状况）
2. predict - 模式B预测调整（用户想知道港股通会调入/调出哪些股票）
3. backtest - 模式B回测验证（用户想验证之前预测的准确性）
4. query_rules - 查询编制规则（用户想了解HSCI或港股通的编制规则、调整时间、换手率要求等）
5. help - 用户需要帮助

支持的调整月份格式: YYYY-MM (如 2026-03, 2026-09)
HSCI定期检讨调整生效月: 3月、9月 (数据截止分别为前一年12月31日、当年6月30日)
快速纳入调整生效月: 6月、12月

用户输入: "{user_input}"

如果用户的输入省略了某些信息（如没说时间），请用null表示。
如果用户说"回测呢"这类依赖上文的话，target_date和mode用"inherit"表示继承上文。

请返回JSON:
```json
{{
    "intent": "scan|predict|backtest|query_rules|help",
    "target_date": "YYYY-MM 或 null 或 inherit",
    "mode": "A 或 B 或 null 或 inherit",
    "scan_date": "YYYY-MM-DD 或 null"
}}
```"""

    def __init__(self):
        self.llm_client = LLMClient()
        # 多轮对话上下文
        self._last_intent: Optional[str] = None
        self._last_params: Optional[dict] = None
        # 最近一次工作流产生的报告路径（供 process_with_meta 读取）
        self._last_report_paths: List[str] = []
        # 进度回调（供SSE流式接口使用）
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """设置进度回调函数，在工作流各阶段被调用"""
        self._progress_callback = callback

    def _emit_progress(self, message: str):
        """发送进度消息（如果设置了回调）"""
        if self._progress_callback:
            self._progress_callback(message)

    def process(self, user_input: str) -> str:
        """
        处理用户自然语言输入

        参数:
            user_input: 用户输入文本

        返回:
            自然语言回复 (Markdown)
        """
        self._last_report_paths = []

        # 1. 正则快速匹配
        intent = self._regex_intent(user_input)

        # 2. 正则匹配不到，调用LLM
        if intent == "unknown":
            self._emit_progress("正在识别意图...")
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
        if intent == "scan":
            return self._handle_scan(params)
        elif intent == "predict":
            return self._handle_predict(params)
        elif intent == "backtest":
            return self._handle_backtest(params)
        elif intent == "query_rules":
            return self._handle_query_rules(params)
        elif intent == "help":
            return self._default_response()
        else:
            return self._default_response()

    def process_with_meta(self, user_input: str) -> dict:
        """
        处理用户自然语言输入，返回带元信息的结果（供 Web 前端使用）

        返回:
            {
                "reply": str,              # 自然语言回复（markdown 文本）
                "report_paths": list,      # 本轮生成的报告文件路径列表
                "report_html_urls": list,  # 可通过Web访问的HTML报告URL列表
                "intent": str,             # 识别出的意图
            }
        """
        self._last_report_paths = []
        reply = self.process(user_input)

        # 构建HTML报告URL列表
        report_html_urls = []
        for path in self._last_report_paths:
            if path.endswith(".html"):
                # 转换为相对于output目录的URL路径
                rel = os.path.relpath(path, PROJECT_ROOT)
                report_html_urls.append("/" + rel.replace("\\", "/"))

        return {
            "reply": reply,
            "report_paths": self._last_report_paths,
            "report_html_urls": report_html_urls,
            "intent": self._last_intent or "unknown",
        }

    # ========== 意图识别 ==========

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

        # 提取调整月份 (如 "2026年3月" -> "2026-03", "2026-03" -> "2026-03")
        time_match = re.search(r"(\d{4})年(\d{1,2})月", text)
        if time_match:
            params["target_date"] = f"{time_match.group(1)}-{int(time_match.group(2)):02d}"
        else:
            time_match = re.search(r"(\d{4})-(\d{1,2})\b", text)
            if time_match:
                params["target_date"] = f"{time_match.group(1)}-{int(time_match.group(2)):02d}"

        # 提取扫描日期 (如 "2025年12月31日" -> "2025-12-31")
        scan_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
        if scan_match:
            params["scan_date"] = (
                f"{scan_match.group(1)}-"
                f"{int(scan_match.group(2)):02d}-"
                f"{int(scan_match.group(3)):02d}"
            )
        else:
            scan_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
            if scan_match:
                params["scan_date"] = scan_match.group(0)

        # 检测模式
        if re.search(r"模式\s*A|模式A", text):
            params["mode"] = "A"
        elif re.search(r"模式\s*B|模式B", text):
            params["mode"] = "B"

        return params

    def _llm_intent(self, user_input: str) -> dict:
        """调用LLM做意图识别"""
        prompt = self.LLM_INTENT_PROMPT.format(user_input=user_input)
        try:
            response = self.llm_client.chat(prompt, temperature=0.1)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM意图识别失败: {e}")
        return {"intent": "unknown"}

    def _extract_params_from_llm(self, llm_result: dict, user_input: str) -> dict:
        """从LLM识别结果中提取参数"""
        params = {}

        target_date = llm_result.get("target_date")
        if target_date and target_date not in ("null", "inherit"):
            params["target_date"] = target_date

        mode = llm_result.get("mode")
        if mode and mode not in ("null", "inherit"):
            params["mode"] = mode

        scan_date = llm_result.get("scan_date")
        if scan_date and scan_date not in ("null", "inherit"):
            params["scan_date"] = scan_date

        # 如果LLM没提取到时间，尝试正则补充
        if not params.get("target_date"):
            time_match = re.search(r"(\d{4})年(\d{1,2})月", user_input)
            if time_match:
                params["target_date"] = (
                    f"{time_match.group(1)}-{int(time_match.group(2)):02d}"
                )
            else:
                time_match = re.search(r"(\d{4})-(\d{1,2})\b", user_input)
                if time_match:
                    params["target_date"] = (
                        f"{time_match.group(1)}-{int(time_match.group(2)):02d}"
                    )

        return params

    def _handle_inheritance(self, params: dict) -> dict:
        """处理继承上文的情况"""
        if not self._last_params:
            return params
        merged = dict(self._last_params)
        merged.update(params)
        return merged

    # ========== 工作流执行 ==========

    def _handle_scan(self, params: dict) -> str:
        """处理模式A监控扫描意图"""
        scan_date_str = params.get("scan_date")
        if scan_date_str:
            try:
                scan_date = date.fromisoformat(scan_date_str)
            except ValueError:
                return f"无效的扫描日期: {scan_date_str}，请使用 YYYY-MM-DD 格式"
        else:
            scan_date = date.today()

        try:
            self._emit_progress("正在初始化数据提供器...")
            from mode_a.data.akshare_data_provider import AkShareDataProvider
            from mode_a.core.rules import RuleEngine
            from mode_a.core.monitor import HkConnectMonitor
            from reports.reporter import HtmlReporter

            data_provider = AkShareDataProvider(reference_date=scan_date, verbose=True)
            rule_engine = RuleEngine(reference_date=scan_date)
            monitor = HkConnectMonitor(
                data_provider=data_provider,
                rule_engine=rule_engine,
                reference_date=scan_date,
            )

            self._emit_progress(f"正在扫描全市场 (基准日: {scan_date})，可能需要数分钟...")
            summary = monitor.scan()

            self._emit_progress("正在生成HTML报告...")
            html_reporter = HtmlReporter()
            report_dir = os.path.join(PROJECT_ROOT, "output", "mode_a")
            os.makedirs(report_dir, exist_ok=True)
            html_path = os.path.join(report_dir, "report.html")
            html_reporter.report(summary, html_path)
            self._last_report_paths.append(html_path)

            self._emit_progress("完成")
            return self._format_scan_response(summary, scan_date, html_path)

        except Exception as e:
            logger.error(f"扫描失败: {e}", exc_info=True)
            return f"扫描过程中出现错误: {e}"

    def _handle_predict(self, params: dict) -> str:
        """处理模式B预测意图"""
        target_date = params.get("target_date")
        if not target_date:
            return (
                "请指定要预测的调整生效月份，如 **2026-03** 或 **2026年9月**\n\n"
                "HSCI定期检讨调整生效月: 3月、9月\n"
                "快速纳入调整生效月: 6月、12月"
            )

        try:
            self._emit_progress("正在加载模式B模块...")
            from mode_b.utils.date_utils import get_cutoff_date, format_adjustment_label, parse_adjustment_date
            from mode_b.utils.date_utils import get_cutoff_date as _gcd
        except ImportError:
            return "模式B模块加载失败，请检查 mode_b 目录"

        try:
            cutoff = get_cutoff_date(target_date)
        except ValueError as e:
            return str(e)

        year, month = parse_adjustment_date(target_date)
        adj_str = f"{year:04d}-{month:02d}"
        adjustment_label = format_adjustment_label(target_date)
        min_mc = 5e8  # 默认5亿

        try:
            self._emit_progress(f"正在计算选股范畴市值 (截止日: {cutoff})，可能需要数分钟...")
            from main_b import _run_predict_core
            result = _run_predict_core(cutoff, min_mc)

            self._emit_progress("正在保存预测结果...")
            output_dir = os.path.join(PROJECT_ROOT, "output", "mode_b")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"prediction_{adj_str}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)

            self._emit_progress("正在生成HTML可视化报告...")
            from reports.html_report_builder import build_prediction_report
            html_file = output_file.replace(".json", ".html")
            result["adjustment_label"] = adj_str
            build_prediction_report(result, html_file)
            self._last_report_paths.append(html_file)

            self._emit_progress("正在调用大模型生成分析报告...")
            llm_report_paths = []
            try:
                from reports.llm_report_generator import LLMReportGenerator
                gen = LLMReportGenerator()
                paths = gen.generate(prediction=result, adjustment_label=adj_str)
                if paths:
                    self._last_report_paths.append(paths["html"])
                    llm_report_paths = [paths["md"], paths["html"]]
            except Exception as e:
                logger.warning(f"LLM报告生成失败: {e}")

            self._emit_progress("完成")
            return self._format_predict_response(
                result, target_date, adjustment_label, cutoff,
                html_file, llm_report_paths
            )

        except Exception as e:
            logger.error(f"预测失败: {e}", exc_info=True)
            return f"预测过程中出现错误: {e}"

    def _handle_backtest(self, params: dict) -> str:
        """处理模式B回测意图"""
        target_date = params.get("target_date")
        if not target_date:
            return (
                "请指定要回测的调整生效月份，如 **2026-03** 或 **2026年9月**\n\n"
                "HSCI定期检讨调整生效月: 3月、9月\n"
                "快速纳入调整生效月: 6月、12月"
            )

        try:
            self._emit_progress("正在加载模式B模块...")
            from mode_b.utils.date_utils import get_cutoff_date, get_effective_date, format_adjustment_label, parse_adjustment_date
        except ImportError:
            return "模式B模块加载失败"

        try:
            cutoff = get_cutoff_date(target_date)
        except ValueError as e:
            return str(e)

        year, month = parse_adjustment_date(target_date)
        adj_str = f"{year:04d}-{month:02d}"
        adjustment_label = format_adjustment_label(target_date)
        min_mc = 5e8

        try:
            self._emit_progress("正在加载历史成分数据...")
            from mode_b.data.hsci_history_loader import HSCIHistoryLoader
            loader = HSCIHistoryLoader()
            loader.load()
            target = get_effective_date(target_date, loader)
        except Exception as e:
            return f"获取生效日期失败: {e}"

        try:
            # 检查预测结果是否存在, 不存在则自动运行预测
            pred_file = os.path.join(
                PROJECT_ROOT, "output", "mode_b",
                f"prediction_{adj_str}.json"
            )
            if not os.path.exists(pred_file):
                self._emit_progress(f"未找到预测结果，正在运行预测 (截止日: {cutoff})，可能需要数分钟...")
                from main_b import _run_predict_core
                prediction = _run_predict_core(cutoff, min_mc)
                with open(pred_file, "w", encoding="utf-8") as f:
                    json.dump(prediction, f, ensure_ascii=False, indent=2, default=str)
            else:
                self._emit_progress("正在加载已有预测结果...")
                with open(pred_file, "r", encoding="utf-8") as f:
                    prediction = json.load(f)

            self._emit_progress("正在执行回测...")
            from mode_b.core.orchestrator_b import OrchestratorB
            orch = OrchestratorB()
            result = orch.run_backtest(
                prediction=prediction,
                target_adjustment_date=target,
            )

            self._emit_progress("正在保存结果...")
            output_dir = os.path.join(PROJECT_ROOT, "output", "mode_b")
            output_file = os.path.join(output_dir, f"backtest_{adj_str}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)

            self._emit_progress("正在生成HTML报告...")
            from reports.html_report_builder import build_backtest_report
            html_file = output_file.replace(".json", ".html")
            result["adjustment_label"] = adj_str
            result["cutoff_date_str"] = cutoff.isoformat()
            build_backtest_report(result, html_file)
            self._last_report_paths.append(html_file)

            self._emit_progress("完成")
            return self._format_backtest_response(
                result, target_date, adjustment_label, cutoff, html_file
            )

        except Exception as e:
            logger.error(f"回测失败: {e}", exc_info=True)
            return f"回测过程中出现错误: {e}"

    def _handle_query_rules(self, params: dict) -> str:
        """处理规则查询意图"""
        rule_file = os.path.join(PROJECT_ROOT, "rules", "hsci.md")
        try:
            with open(rule_file, "r", encoding="utf-8") as f:
                content = f.read()
            # 去掉YAML frontmatter，只返回Markdown正文
            parts = content.split("---", 2)
            if len(parts) >= 3:
                rule_text = parts[2].strip()
            else:
                rule_text = content
            return f"## 恒生综合指数(HSCI) / 港股通编制规则\n\n{rule_text}"
        except FileNotFoundError:
            return f"规则文件不存在: {rule_file}"

    # ========== 响应格式化 ==========

    def _format_scan_response(self, summary, scan_date, html_path) -> str:
        """格式化扫描结果为自然语言"""
        lines = []
        lines.append(f"## 港股通标的池监控扫描结果")
        lines.append(f"扫描日期: {scan_date}")
        lines.append(f"扫描股票: {summary.total_stocks} 只 | 当前标的池: {summary.pool_size} 只")
        lines.append("")

        # 风险统计
        lines.append("### 风险统计")
        lines.append(f"- 高风险: {summary.high_risk_count} 只")
        lines.append(f"- 中风险: {summary.medium_risk_count} 只")
        lines.append(f"- 低风险: {summary.low_risk_count} 只")
        lines.append(f"- 安全: {summary.safe_count} 只")
        lines.append("")

        # 风险类型分布
        type_counts = summary.count_by_risk_type()
        if type_counts:
            lines.append("### 风险类型分布")
            for rt, cnt in type_counts.items():
                lines.append(f"- {rt.value}: {cnt} 只")
            lines.append("")

        # 风险清单
        risk_list = summary.risk_list
        if risk_list:
            lines.append(f"### 风险清单 ({len(risk_list)}只)")
            for r in risk_list[:20]:
                pool_tag = "在池" if r.in_pool else "不在池"
                lines.append(
                    f"- [{r.risk_level.value}] {r.code} {r.name} "
                    f"({r.index_type.value}·{pool_tag}) - {r.risk_type.value}"
                )
            if len(risk_list) > 20:
                lines.append(f"- ... 共 {len(risk_list)} 只")
        else:
            lines.append("### 风险清单: 未发现风险股票")

        lines.append("")
        lines.append(f"HTML报告: {html_path}")

        return "\n".join(lines)

    def _format_predict_response(
        self, result, target_date, adjustment_label, cutoff,
        html_path, llm_report_paths
    ) -> str:
        """格式化预测结果为自然语言"""
        rd = result.get("report_data", {})
        hsci_pred = result.get("hsci_prediction", {})
        connect_pred = result.get("connect_prediction", {})

        lines = []
        lines.append(f"## {adjustment_label} HSCI/港股通调整预测")
        lines.append(f"数据截止日: {cutoff}")
        lines.append(f"选股范畴: {rd.get('total_universe', 0)} 只")
        lines.append("")

        # HSCI预测
        lines.append("### HSCI预测")
        lines.append(
            f"- 当前成份股: {rd.get('current_hsci_count', 0)} 只 → "
            f"预测: {rd.get('predicted_hsci_count', 0)} 只"
        )
        lines.append(
            f"- 调入: {rd.get('hsci_additions', 0)} 只 | "
            f"调出: {rd.get('hsci_removals', 0)} 只"
        )

        additions = hsci_pred.get("additions", [])
        if additions:
            lines.append(f"\n**预测调入 ({len(additions)}只):**")
            for s in additions[:15]:
                lines.append(
                    f"- {s['code']} {s.get('name', '')} "
                    f"(排名{s.get('rank', '')}, 覆盖率{s.get('coverage', '')}%, "
                    f"→ {s.get('predicted_sub_index', '')})"
                )
            if len(additions) > 15:
                lines.append(f"- ... 共 {len(additions)} 只")

        removals = hsci_pred.get("removals", [])
        if removals:
            lines.append(f"\n**预测调出 ({len(removals)}只):**")
            for s in removals[:15]:
                lines.append(
                    f"- {s['code']} {s.get('name', '')} "
                    f"(排名{s.get('rank', '')}, 覆盖率{s.get('coverage', '')}%, "
                    f"← {s.get('current_sub_index', '')})"
                )
            if len(removals) > 15:
                lines.append(f"- ... 共 {len(removals)} 只")

        lines.append("")

        # 港股通预测
        lines.append("### 港股通预测")
        lines.append(
            f"- 调入: {rd.get('connect_additions', 0)} 只 | "
            f"调出: {rd.get('connect_removals', 0)} 只 | "
            f"不合资格: {rd.get('ineligible', 0)} 只"
        )

        connect_adds = connect_pred.get("connect_additions", [])
        if connect_adds:
            lines.append(f"\n**港股通预测调入 ({len(connect_adds)}只):**")
            for s in connect_adds[:15]:
                lines.append(
                    f"- {s['code']} {s.get('name', '')} "
                    f"→ {s.get('predicted_sub_index', '')} "
                    f"({s.get('eligibility', '')})"
                )
            if len(connect_adds) > 15:
                lines.append(f"- ... 共 {len(connect_adds)} 只")

        connect_rems = connect_pred.get("connect_removals", [])
        if connect_rems:
            lines.append(f"\n**港股通预测调出 ({len(connect_rems)}只):**")
            for s in connect_rems[:15]:
                lines.append(
                    f"- {s['code']} {s.get('name', '')} ({s.get('reason', '')})"
                )
            if len(connect_rems) > 15:
                lines.append(f"- ... 共 {len(connect_rems)} 只")

        lines.append("")
        lines.append(f"HTML报告: {html_path}")
        if llm_report_paths:
            lines.append(f"LLM分析报告(MD): {llm_report_paths[0]}")
            lines.append(f"LLM分析报告(HTML): {llm_report_paths[1]}")

        return "\n".join(lines)

    def _format_backtest_response(
        self, result, target_date, adjustment_label, cutoff, html_path
    ) -> str:
        """格式化回测结果为自然语言"""
        l1 = result.get("layer1_hsci", {})
        l2 = result.get("layer2_connect", {})
        deviation = result.get("deviation_analysis", {})

        lines = []
        lines.append(f"## {adjustment_label} HSCI/港股通回测结果")
        lines.append(f"数据截止日: {cutoff}")
        lines.append("")

        # Layer 1: HSCI预测准确率
        lines.append("### Layer 1: HSCI预测准确率")
        l1_add = l1.get("add", {})
        l1_rem = l1.get("remove", {})
        l1_ov = l1.get("overall", {})
        lines.append(
            f"- 调入: P={l1_add.get('precision', 0):.1%} "
            f"R={l1_add.get('recall', 0):.1%} "
            f"F1={l1_add.get('f1', 0):.1%} "
            f"(命中{l1_add.get('matches', 0)}/实际{l1_add.get('actual', 0)}/预测{l1_add.get('predicted', 0)})"
        )
        lines.append(
            f"- 调出: P={l1_rem.get('precision', 0):.1%} "
            f"R={l1_rem.get('recall', 0):.1%} "
            f"F1={l1_rem.get('f1', 0):.1%} "
            f"(命中{l1_rem.get('matches', 0)}/实际{l1_rem.get('actual', 0)}/预测{l1_rem.get('predicted', 0)})"
        )
        lines.append(
            f"- 综合: P={l1_ov.get('precision', 0):.1%} "
            f"R={l1_ov.get('recall', 0):.1%} "
            f"F1={l1_ov.get('f1', 0):.1%}"
        )
        lines.append("")

        # Layer 2: 港股通映射准确率
        lines.append("### Layer 2: 港股通映射准确率")
        l2_add = l2.get("add", {})
        l2_rem = l2.get("remove", {})
        l2_ov = l2.get("overall", {})
        lines.append(
            f"- 调入: P={l2_add.get('precision', 0):.1%} "
            f"R={l2_add.get('recall', 0):.1%} "
            f"F1={l2_add.get('f1', 0):.1%} "
            f"(命中{l2_add.get('matches', 0)}/实际{l2_add.get('actual', 0)}/预测{l2_add.get('predicted', 0)})"
        )
        lines.append(
            f"- 调出: P={l2_rem.get('precision', 0):.1%} "
            f"R={l2_rem.get('recall', 0):.1%} "
            f"F1={l2_rem.get('f1', 0):.1%} "
            f"(命中{l2_rem.get('matches', 0)}/实际{l2_rem.get('actual', 0)}/预测{l2_rem.get('predicted', 0)})"
        )
        lines.append(
            f"- 综合: P={l2_ov.get('precision', 0):.1%} "
            f"R={l2_ov.get('recall', 0):.1%} "
            f"F1={l2_ov.get('f1', 0):.1%}"
        )
        lines.append("")

        # 偏差分析
        if deviation:
            lines.append("### 偏差分析")
            lines.append(deviation.get("summary", ""))

        lines.append("")
        lines.append(f"HTML报告: {html_path}")

        return "\n".join(lines)

    def _default_response(self) -> str:
        """默认回复"""
        return (
            "我可以帮您完成以下操作:\n\n"
            "**模式A - 监控扫描**\n"
            '- 扫描港股通标的池: "扫描港股通"\n'
            '- 指定日期扫描: "扫描2025年12月31日的港股通"\n\n'
            "**模式B - 预测/回测**\n"
            '- 预测调整: "预测2026年3月港股通的调整"\n'
            '- 回测验证: "回测2026年3月港股通"\n\n'
            "**规则查询**\n"
            '- 查看编制规则: "查看HSCI编制规则"\n'
            '- 查询换手率要求: "港股通的换手率要求是什么"\n\n'
            "您也可以用自然语言描述，比如:\n"
            '- "3月那个港股通会不会换股票"\n'
            '- "上次预测准吗"\n'
            '- "监控一下港股通的风险"'
        )
