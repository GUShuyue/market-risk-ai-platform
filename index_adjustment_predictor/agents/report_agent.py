"""
Report Agent - 报告生成Agent

职责：生成预测报告和回测报告
"""

from typing import Dict
from datetime import datetime
import json
import os
from agents.html_report_builder import HtmlReportBuilder
from agents.backtest_html_builder import BacktestHtmlBuilder


class ReportAgent:
    """
    报告Agent - 生成结构化分析报告
    
    支持的报告类型:
        - prediction_report: 预测结果报告
        - backtest_report: 回测评估报告
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config.get("report_output_dir", "reports")
        self.html_builder = HtmlReportBuilder()
        self.backtest_html_builder = BacktestHtmlBuilder()

    @staticmethod
    def _json_default(obj):
        """JSON 序列化兜底类型转换

        处理 numpy 标量（numpy.bool_ / numpy.int64 / numpy.float64 等，
        来自 tushare/pandas）以及 set 等非原生 JSON 类型，
        避免出现 "Object of type bool is not JSON serializable"。
        """
        try:
            import numpy as np
            if isinstance(obj, np.generic):
                return obj.item()
        except ImportError:
            pass
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    def run(self, inputs: dict) -> dict:
        """
        统一接口：生成报告（预测或回测）

        预测报告 inputs: prediction, rule_result, market_data, use_llm(可选)
        回测报告 inputs: backtest_result, index_name(可选), target_date(可选),
                         use_llm(可选), prediction(可选，用于回退取index/target)
        返回: {"report_path": "..."}
        """
        if "backtest_result" in inputs:
            prediction = inputs.get("prediction", {}) or {}
            path = self.generate_backtest_report(
                inputs["backtest_result"],
                index_name=inputs.get("index_name") or prediction.get("index_name"),
                target_date=inputs.get("target_date") or prediction.get("target_date"),
                use_llm=inputs.get("use_llm", True)
            )
        else:
            path = self.generate_prediction_report(
                inputs["prediction"],
                inputs["rule_result"],
                inputs["market_data"],
                use_llm=inputs.get("use_llm", True)
            )
        return {"report_path": path}

    def generate_prediction_report(
        self, prediction: dict, rule_result: dict, market_data: dict,
        use_llm: bool = True
    ) -> str:
        """
        生成预测报告 (Markdown + JSON + HTML)

        参数:
            use_llm: 是否使用了LLM增强分析。False时文件名中添加"_no_llm"后缀，
                     避免覆盖有LLM参与的预测结果。

        返回:
            报告文件路径
        """
        index_name = prediction.get("index_name", "unknown")
        target_date = prediction.get("target_date", "unknown")

        # 纯量化模式文件名添加后缀，避免覆盖LLM预测结果
        suffix = "" if use_llm else "_no_llm"
        name_part = f"{index_name}_{target_date}{suffix}"

        # 生成 Markdown 报告
        report = self._build_prediction_content(
            prediction, rule_result, market_data
        )

        filename = f"prediction_{name_part}.md"
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        # 同时保存 JSON 结果 (供回测使用)
        json_filename = f"prediction_{name_part}.json"
        json_filepath = os.path.join(self.output_dir, json_filename)

        prediction_data = {
            "prediction": prediction,
            "rule_result": rule_result,
            "market_data": {
                "target_date": market_data.get("target_date"),
                "cutoff_date": market_data.get("cutoff_date"),
                "index_name": market_data.get("index_name"),
                "current_constituents": market_data.get("current_constituents", []),
                "financial_flags": market_data.get("financial_flags", {}),
            },
            "use_llm": use_llm,
            "generated_at": datetime.now().isoformat()
        }

        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(prediction_data, f, ensure_ascii=False, indent=2,
                      default=self._json_default)

        # 生成可视化 HTML 报告
        html_report = self.html_builder.build(
            prediction, rule_result, market_data
        )
        html_filename = f"prediction_{name_part}.html"
        html_filepath = os.path.join(self.output_dir, html_filename)
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(html_report)

        return filepath

    def generate_backtest_report(
        self, backtest_result: dict, index_name: str = None,
        target_date: str = None, use_llm: bool = True
    ) -> str:
        """
        生成回测报告 (Markdown + JSON)

        参数:
            use_llm: 是否回测的是LLM预测结果。False时文件名添加_no_llm后缀。

        返回:
            报告文件路径
        """
        # 生成 Markdown 报告
        report = self._build_backtest_content(backtest_result)

        # 文件名包含指数和日期信息，纯量化模式添加后缀
        suffix = "" if use_llm else "_no_llm"
        name_part = f"{index_name}_{target_date}{suffix}" if index_name and target_date else "unknown"
        filename = f"backtest_{name_part}.md"
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        # 同时保存 JSON 结果
        json_filename = f"backtest_{name_part}.json"
        json_filepath = os.path.join(self.output_dir, json_filename)

        backtest_data = {
            "index_name": index_name,
            "target_date": target_date,
            "use_llm": use_llm,
            "backtest_result": backtest_result,
            "generated_at": datetime.now().isoformat()
        }

        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(backtest_data, f, ensure_ascii=False, indent=2,
                      default=self._json_default)

        # 生成可视化 HTML 报告
        html_report = self.backtest_html_builder.build(
            backtest_result, index_name=index_name or "",
            target_date=target_date or "", use_llm=use_llm
        )
        html_filename = f"backtest_{name_part}.html"
        html_filepath = os.path.join(self.output_dir, html_filename)
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(html_report)

        return filepath

    def _build_prediction_content(
        self, prediction: dict, rule_result: dict, market_data: dict
    ) -> str:
        """构建预测报告内容"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        index_name = prediction.get("index_name", rule_result.get("index_name", "未知"))
        target_date = prediction.get("target_date", market_data.get("target_date", "未知"))
        cutoff_date = market_data.get("cutoff_date", "未知")
        confidence = prediction.get("confidence", 0)
        reasoning = prediction.get("reasoning", "")
        predicted_add = prediction.get("predicted_add", [])
        alternative_add = prediction.get("alternative_add", [])
        predicted_remove = prediction.get("predicted_remove", [])

        # 按置信度从高到低排序
        predicted_add = self._sort_by_confidence(predicted_add)
        predicted_remove = self._sort_by_confidence(predicted_remove)
        alternative_add = self._sort_by_confidence(alternative_add)

        lines = [
            f"# 指数成分股调整预测报告",
            f"",
            f"| 项目 | 内容 |",
            f"|------|------|",
            f"| 指数 | {index_name} |",
            f"| 预测目标 | {target_date} |",
            f"| 数据截止日 | {cutoff_date} |",
            f"| 生成时间 | {now} |",
            f"| 置信度 | {confidence:.0%} |",
            f"",
            f"## 预测调入 ({len(predicted_add)}只)",
        ]

        if predicted_add:
            lines.append("| 代码 | 名称 | 排名 | 置信度 | 理由 |")
            lines.append("|------|------|------|--------|------|")
            for s in predicted_add:
                code = s.get("stock_code", s.get("code", ""))
                name = s.get("stock_name", s.get("name", ""))
                rank = s.get("rank", "-")
                conf = s.get("confidence", "-")
                reason = s.get("reasoning", "")
                lines.append(f"| {code} | {name} | {rank} | {conf} | {reason} |")
        else:
            lines.append("无调入候选")

        # 备选调入候选（单独展示，不计入回测指标）
        if alternative_add:
            lines.append(f"")
            lines.append(f"## 备选调入候选 ({len(alternative_add)}只)")
            lines.append("> 以下股票排名靠后或存在不确定性，实际调入概率较低，仅供参考。")
            lines.append("")
            lines.append("| 代码 | 名称 | 排名 | 置信度 | 理由 |")
            lines.append("|------|------|------|--------|------|")
            for s in alternative_add:
                code = s.get("stock_code", s.get("code", ""))
                name = s.get("stock_name", s.get("name", ""))
                rank = s.get("rank", "-")
                conf = s.get("confidence", "-")
                reason = s.get("reasoning", "")
                lines.append(f"| {code} | {name} | {rank} | {conf} | {reason} |")

        lines.append(f"")
        lines.append(f"## 预测调出 ({len(predicted_remove)}只)")

        if predicted_remove:
            lines.append("| 代码 | 名称 | 排名 | 置信度 | 理由 |")
            lines.append("|------|------|------|--------|------|")
            for s in predicted_remove:
                code = s.get("stock_code", s.get("code", ""))
                name = s.get("stock_name", s.get("name", ""))
                rank = s.get("rank", "-")
                conf = s.get("confidence", "-")
                reason = s.get("reasoning", "")
                lines.append(f"| {code} | {name} | {rank} | {conf} | {reason} |")
        else:
            lines.append("无调出候选")

        if reasoning:
            # 清洗reasoning：若包含JSON代码块或dict结构，说明是解析失败
            # 时塞入的原始LLM输出，不应作为分析说明展示给用户
            cleaned = reasoning.strip()
            is_raw_llm_output = (
                cleaned.startswith("{")
                or cleaned.startswith("```")
                or cleaned.startswith("[")
                or '"predicted_add"' in cleaned
                or '"predicted_remove"' in cleaned
                or '"stock_code"' in cleaned
                or "```json" in cleaned
                # 超长文本且含大量花括号，判定为原始JSON输出
                or (len(cleaned) > 2000 and cleaned.count("{") > 5)
            )
            if is_raw_llm_output:
                cleaned = ""  # 视为无效说明，丢弃
            if cleaned:
                lines.append("")
                lines.append("## 分析说明")
                lines.append(cleaned)

        details = prediction.get("details", [])
        if details:
            lines.append("")
            lines.append("## 风险因素")
            for d in details:
                lines.append(f"- {d}")

        return "\n".join(lines)

    @staticmethod
    def _sort_by_confidence(stocks: list) -> list:
        """按置信度从高到低排序"""
        def _conf_val(s):
            c = s.get("confidence", 0.5)
            if isinstance(c, str):
                return {"high": 0.9, "medium": 0.6, "low": 0.3}.get(c.lower(), 0.5)
            return c if isinstance(c, (int, float)) else 0.5
        return sorted(stocks, key=_conf_val, reverse=True)

    def _build_backtest_content(self, backtest_result: dict) -> str:
        """构建回测报告内容"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        accuracy = backtest_result.get("accuracy", {})
        matches = backtest_result.get("matches", {})
        misses = backtest_result.get("misses", {})
        false_positives = backtest_result.get("false_positives", {})
        deviation = backtest_result.get("deviation_analysis", {})

        lines = [
            "# 回测评估报告",
            "",
            f"| 项目 | 内容 |",
            f"|------|------|",
            f"| 生成时间 | {now} |",
            f"| 调入命中 | {accuracy.get('add_match_count', 0)}/{accuracy.get('add_actual_count', 0)} |",
            f"| 调出命中 | {accuracy.get('remove_match_count', 0)}/{accuracy.get('remove_actual_count', 0)} |",
            "",
            "## 准确率指标",
            "",
            "| 指标 | 调入 | 调出 | 综合 |",
            "|------|------|------|------|",
            f"| 精确率 | {accuracy.get('add_precision', 0):.1%} | {accuracy.get('remove_precision', 0):.1%} | {accuracy.get('overall_precision', 0):.1%} |",
            f"| 召回率 | {accuracy.get('add_recall', 0):.1%} | {accuracy.get('remove_recall', 0):.1%} | {accuracy.get('overall_recall', 0):.1%} |",
            f"| F1 | {accuracy.get('add_f1', 0):.1%} | {accuracy.get('remove_f1', 0):.1%} | {accuracy.get('overall_f1', 0):.1%} |",
        ]

        # 命中明细
        add_matches = matches.get("add", [])
        remove_matches = matches.get("remove", [])
        if add_matches or remove_matches:
            lines.append("")
            lines.append("## 命中明细")
            lines.append("")
            lines.append("### 调入命中")
            for s in add_matches:
                code = s.get("stock_code", s.get("code", s.get("ts_code", "")))
                name = s.get("stock_name", s.get("name", ""))
                lines.append(f"- {code} {name}")
            lines.append("")
            lines.append("### 调出命中")
            for s in remove_matches:
                code = s.get("stock_code", s.get("code", s.get("ts_code", "")))
                name = s.get("stock_name", s.get("name", ""))
                lines.append(f"- {code} {name}")

        # 遗漏明细（附原因分析）
        self._append_deviation_md(
            lines, deviation, "missed_adds", "遗漏调入",
            "实际调入但预测未命中"
        )
        self._append_deviation_md(
            lines, deviation, "missed_removes", "遗漏调出",
            "实际调出但预测未命中"
        )
        self._append_deviation_md(
            lines, deviation, "false_adds", "误报调入",
            "预测调入但实际未调入"
        )
        self._append_deviation_md(
            lines, deviation, "false_removes", "误报调出",
            "预测调出但实际未调出"
        )

        # 偏差总结
        if deviation:
            lines.append("")
            lines.append("## 偏差总结")
            lines.append(deviation.get("summary", ""))

            # LLM深层分析
            llm_text = deviation.get("llm_analysis", "")
            if llm_text and "不可用" not in llm_text and "失败" not in llm_text:
                lines.append("")
                lines.append("## LLM深层分析")
                lines.append(llm_text)

        return "\n".join(lines)

    @staticmethod
    def _append_deviation_md(
        lines: list, deviation: dict, key: str,
        title: str, subtitle: str
    ):
        """将偏差分析明细追加到Markdown报告"""
        items = deviation.get(key, [])
        if not items:
            return

        lines.append("")
        lines.append(f"## {title} ({len(items)}只)")
        lines.append(f"> {subtitle}")
        lines.append("")
        lines.append("| 代码 | 名称 | 排名 | 市值 | 原因分析 |")
        lines.append("|------|------|------|------|----------|")
        for item in items:
            code = item.get("code", "")
            name = item.get("name", "")
            rank = item.get("rank", 0)
            mv = item.get("market_cap", 0)
            mv_text = f"{mv/1e8:.0f}亿" if mv and isinstance(mv, (int, float)) and mv >= 1e8 else "-"
            reason = item.get("reason", "")
            lines.append(f"| {code} | {name} | {rank} | {mv_text} | {reason} |")
