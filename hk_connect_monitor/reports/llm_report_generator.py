"""
LLM报告生成器 (Mode B)

基于量化预测结果，调用大模型生成专业的分析报告（Markdown + HTML）。

工作流:
    预测JSON → 数据压缩/格式化 → Prompt构建 → LLM调用 → Markdown + HTML报告

用法:
    from reports.llm_report_generator import LLMReportGenerator
    gen = LLMReportGenerator()
    paths = gen.generate(prediction_data, adjustment_label="2026-03")
    # paths = {"md": "...md", "html": "...html"}
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.llm_client import LLMClient
from prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class LLMReportGenerator:
    """
    LLM报告生成器

    将量化预测结果转换为专业的大模型分析报告。
    """

    # 每类数据传入LLM的最大条数（控制token用量）
    MAX_ADDITIONS = 30
    MAX_REMOVALS = 30
    MAX_INELIGIBLE = 20

    def __init__(self, llm_config: Optional[dict] = None):
        """
        参数:
            llm_config: 可选的LLM配置覆盖
        """
        self.llm = LLMClient(llm_config)
        self.pm = PromptManager()

    def generate(
        self,
        prediction: dict,
        adjustment_label: str = "",
        output_dir: str = None,
    ) -> Optional[dict]:
        """
        生成LLM分析报告 (Markdown + HTML)

        参数:
            prediction: 预测结果字典 (main_b.py的_run_predict_core返回值)
            adjustment_label: 调整月份标签 (如 "2026-03")
            output_dir: 输出目录，默认为 mode_b/reports/

        返回:
            {"md": md文件路径, "html": html文件路径}，失败返回None
        """
        if not self.llm.is_available:
            logger.error("LLM API Key未配置，无法生成报告。请在 .env 中设置 LLM_API_KEY")
            return None

        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, "output", "mode_b")

        logger.info("开始生成LLM分析报告...")

        # 1. 压缩预测数据为文本
        data_text = self._build_data_text(prediction)

        # 2. 构建Prompt
        report_data = prediction.get("report_data", {})
        cutoff_date = prediction.get("cutoff_date", "未知")

        prompt = self.pm.get_user_prompt(
            "prediction_report_template",
            adjustment_label=adjustment_label or cutoff_date,
            cutoff_date=cutoff_date,
            total_universe=report_data.get("total_universe", 0),
            current_hsci_count=report_data.get("current_hsci_count", 0),
            predicted_hsci_count=report_data.get("predicted_hsci_count", 0),
            hsci_add_count=report_data.get("hsci_additions", 0),
            hsci_remove_count=report_data.get("hsci_removals", 0),
            connect_add_count=report_data.get("connect_additions", 0),
            connect_remove_count=report_data.get("connect_removals", 0),
            ineligible_count=report_data.get("ineligible", 0),
            hsci_additions=data_text["hsci_additions"],
            hsci_removals=data_text["hsci_removals"],
            connect_additions=data_text["connect_additions"],
            connect_removals=data_text["connect_removals"],
            ineligible_additions=data_text["ineligible_additions"],
        )

        system_prompt = self.pm.get_system_prompt("hk_connect_report")

        # 3. 调用LLM
        try:
            logger.info(f"调用大模型 ({self.llm.model})，生成分析报告...")
            report_md = self.llm.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=8192,
            )
            logger.info("LLM报告生成完成")
        except RuntimeError as e:
            logger.error(f"LLM报告生成失败: {e}")
            return None

        # 4. 添加报告头部元信息
        header = self._build_header(
            adjustment_label=adjustment_label,
            cutoff_date=cutoff_date,
            report_data=report_data,
        )
        full_md = header + "\n\n" + report_md.strip() + "\n"

        # 5. 保存Markdown报告
        os.makedirs(output_dir, exist_ok=True)
        suffix = adjustment_label.replace("-", "_") if adjustment_label else cutoff_date.replace("-", "_")
        md_filename = f"prediction_llm_report_{suffix}.md"
        md_filepath = os.path.join(output_dir, md_filename)

        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(full_md)
        logger.info(f"LLM Markdown报告已保存: {md_filepath}")

        # 6. 生成HTML报告
        html_filepath = md_filepath.replace(".md", ".html")
        html_content = self._build_html(full_md, adjustment_label, cutoff_date, report_data)
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"LLM HTML报告已保存: {html_filepath}")

        return {"md": md_filepath, "html": html_filepath}

    def _build_data_text(self, prediction: dict) -> dict:
        """
        将预测JSON压缩为LLM可理解的文本格式

        策略: 每类数据取前N条，用Markdown表格呈现，
        避免传入完整95KB JSON导致token浪费。
        """
        hsci_pred = prediction.get("hsci_prediction", {})
        connect_pred = prediction.get("connect_prediction", {})

        return {
            "hsci_additions": self._format_hsci_additions(
                hsci_pred.get("additions", [])
            ),
            "hsci_removals": self._format_hsci_removals(
                hsci_pred.get("removals", [])
            ),
            "connect_additions": self._format_connect_additions(
                connect_pred.get("connect_additions", [])
            ),
            "connect_removals": self._format_connect_removals(
                connect_pred.get("connect_removals", [])
            ),
            "ineligible_additions": self._format_ineligible(
                connect_pred.get("ineligible_additions", [])
            ),
        }

    def _format_hsci_additions(self, additions: List[dict]) -> str:
        """格式化HSCI调入数据"""
        if not additions:
            return "无调入标的"

        shown = additions[:self.MAX_ADDITIONS]
        lines = [
            "| 代码 | 名称 | 排名 | 累计覆盖率 | 平均市值(亿港币) | 预测分类 | 缓冲区状态 |",
            "|------|------|------|-----------|----------------|---------|-----------|",
        ]
        for s in shown:
            lines.append(
                f"| {s.get('code', '')} | {s.get('name', '')} | "
                f"{s.get('rank', '')} | {s.get('coverage', '')}% | "
                f"{s.get('avg_market_cap_hkd_yi', 0):.1f} | "
                f"{s.get('predicted_sub_index', '')} | "
                f"{s.get('buffer_status', '')} |"
            )
        if len(additions) > self.MAX_ADDITIONS:
            lines.append(f"\n*（共{len(additions)}只，仅展示前{self.MAX_ADDITIONS}只）*")
        return "\n".join(lines)

    def _format_hsci_removals(self, removals: List[dict]) -> str:
        """格式化HSCI调出数据"""
        if not removals:
            return "无调出标的"

        shown = removals[:self.MAX_REMOVALS]
        lines = [
            "| 代码 | 名称 | 排名 | 累计覆盖率 | 平均市值(亿港币) | 当前分类 | 缓冲区状态 |",
            "|------|------|------|-----------|----------------|---------|-----------|",
        ]
        for s in shown:
            lines.append(
                f"| {s.get('code', '')} | {s.get('name', '')} | "
                f"{s.get('rank', '')} | {s.get('coverage', '')}% | "
                f"{s.get('avg_market_cap_hkd_yi', 0):.1f} | "
                f"{s.get('current_sub_index', '')} | "
                f"{s.get('buffer_status', '')} |"
            )
        if len(removals) > self.MAX_REMOVALS:
            lines.append(f"\n*（共{len(removals)}只，仅展示前{self.MAX_REMOVALS}只）*")
        return "\n".join(lines)

    def _format_connect_additions(self, additions: List[dict]) -> str:
        """格式化港股通调入数据"""
        if not additions:
            return "无调入标的"

        shown = additions[:self.MAX_ADDITIONS]
        lines = [
            "| 代码 | 名称 | 排名 | 覆盖率 | 平均市值(亿港币) | 预测分类 | 资格 | 原因 |",
            "|------|------|------|--------|----------------|---------|------|------|",
        ]
        for s in shown:
            mc = s.get("avg_market_cap", 0)
            mc_yi = f"{mc / 1e8:.1f}" if isinstance(mc, (int, float)) and mc > 0 else "-"
            lines.append(
                f"| {s.get('code', '')} | {s.get('name', '')} | "
                f"{s.get('rank', '')} | {s.get('coverage', '')}% | "
                f"{mc_yi} | {s.get('predicted_sub_index', '')} | "
                f"{s.get('eligibility', '')} | {s.get('reason', '')} |"
            )
        if len(additions) > self.MAX_ADDITIONS:
            lines.append(f"\n*（共{len(additions)}只，仅展示前{self.MAX_ADDITIONS}只）*")
        return "\n".join(lines)

    def _format_connect_removals(self, removals: List[dict]) -> str:
        """格式化港股通调出数据"""
        if not removals:
            return "无调出标的"

        shown = removals[:self.MAX_REMOVALS]
        lines = [
            "| 代码 | 名称 | 排名 | 覆盖率 | 平均市值(亿港币) | 当前分类 | 调出原因 |",
            "|------|------|------|--------|----------------|---------|---------|",
        ]
        for s in shown:
            mc = s.get("avg_market_cap", 0)
            mc_yi = f"{mc / 1e8:.1f}" if isinstance(mc, (int, float)) and mc > 0 else "-"
            lines.append(
                f"| {s.get('code', '')} | {s.get('name', '')} | "
                f"{s.get('rank', '')} | {s.get('coverage', '')}% | "
                f"{mc_yi} | {s.get('current_sub_index', '')} | "
                f"{s.get('reason', '')} |"
            )
        if len(removals) > self.MAX_REMOVALS:
            lines.append(f"\n*（共{len(removals)}只，仅展示前{self.MAX_REMOVALS}只）*")
        return "\n".join(lines)

    def _format_ineligible(self, ineligible: List[dict]) -> str:
        """格式化不合资格调入数据"""
        if not ineligible:
            return "无不合资格标的"

        shown = ineligible[:self.MAX_INELIGIBLE]
        lines = [
            "| 代码 | 名称 | 排名 | 覆盖率 | 平均市值(亿港币) | 预测分类 | 资格状态 | 原因 |",
            "|------|------|------|--------|----------------|---------|---------|------|",
        ]
        for s in shown:
            mc = s.get("avg_market_cap", 0)
            mc_yi = f"{mc / 1e8:.1f}" if isinstance(mc, (int, float)) and mc > 0 else "-"
            lines.append(
                f"| {s.get('code', '')} | {s.get('name', '')} | "
                f"{s.get('rank', '')} | {s.get('coverage', '')}% | "
                f"{mc_yi} | {s.get('predicted_sub_index', '')} | "
                f"{s.get('eligibility', '')} | {s.get('reason', '')} |"
            )
        if len(ineligible) > self.MAX_INELIGIBLE:
            lines.append(f"\n*（共{len(ineligible)}只，仅展示前{self.MAX_INELIGIBLE}只）*")
        return "\n".join(lines)

    @staticmethod
    def _build_header(
        adjustment_label: str,
        cutoff_date: str,
        report_data: dict,
    ) -> str:
        """构建报告头部元信息"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"<!-- LLM生成报告 | 模型: 见.env LLM_MODEL | 生成时间: {now} -->\n"
            f"> 本报告由大模型基于量化预测数据自动生成，仅供参考分析，不构成投资建议。\n"
        )

    @staticmethod
    def _build_html(
        md_content: str,
        adjustment_label: str,
        cutoff_date: str,
        report_data: dict,
    ) -> str:
        """
        将Markdown报告转换为带样式的HTML页面

        使用 markdown 库渲染正文，包裹在专业的研究报告模板中。
        """
        import markdown as md_lib

        # 渲染Markdown为HTML（启用表格、代码高亮等扩展）
        html_body = md_lib.markdown(
            md_content,
            extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
        )

        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 提取关键数字用于信息栏
        hsci_add = report_data.get("hsci_additions", 0)
        hsci_remove = report_data.get("hsci_removals", 0)
        connect_add = report_data.get("connect_additions", 0)
        connect_remove = report_data.get("connect_removals", 0)
        total = report_data.get("total_universe", 0)
        current = report_data.get("current_hsci_count", 0)
        predicted = report_data.get("predicted_hsci_count", 0)

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>港股通调整预测LLM分析报告 - {adjustment_label or cutoff_date}</title>
<style>
  :root {{
    --primary: #1a56c4;
    --primary-light: #e8f0fe;
    --accent: #d4380d;
    --text: #1f1f1f;
    --text-secondary: #595959;
    --border: #e8e8e8;
    --bg: #fafafa;
    --card-bg: #ffffff;
    --green: #52c41a;
    --orange: #fa8c16;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC",
                 "WenQuanYi Micro Hei", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.8;
    font-size: 15px;
  }}
  .container {{
    max-width: 900px;
    margin: 0 auto;
    padding: 32px 24px 64px;
  }}
  .report-header {{
    background: linear-gradient(135deg, var(--primary) 0%, #2f7af0 100%);
    color: white;
    padding: 32px 40px;
    border-radius: 12px 12px 0 0;
  }}
  .report-header h1 {{
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 8px;
  }}
  .report-header .subtitle {{
    font-size: 14px;
    opacity: 0.85;
  }}
  .info-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    background: var(--primary-light);
    padding: 16px 40px;
    border-bottom: 1px solid var(--border);
    border-radius: 0 0 12px 12px;
    margin-bottom: 24px;
  }}
  .info-item {{
    font-size: 13px;
    color: var(--text-secondary);
  }}
  .info-item strong {{
    color: var(--primary);
    font-weight: 600;
  }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px;
    margin-bottom: 32px;
  }}
  .stat-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
  }}
  .stat-card .value {{
    font-size: 28px;
    font-weight: 700;
    line-height: 1.2;
  }}
  .stat-card .label {{
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
  }}
  .stat-card.add .value {{ color: var(--green); }}
  .stat-card.remove .value {{ color: var(--accent); }}
  .stat-card.neutral .value {{ color: var(--primary); }}
  .report-body {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 40px;
  }}
  .report-body h1 {{
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    margin: 32px 0 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--primary);
  }}
  .report-body h1:first-child {{ margin-top: 0; }}
  .report-body h2 {{
    font-size: 18px;
    font-weight: 600;
    color: var(--primary);
    margin: 28px 0 12px;
  }}
  .report-body h3 {{
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
    margin: 20px 0 8px;
  }}
  .report-body p {{
    margin-bottom: 12px;
    color: var(--text);
  }}
  .report-body blockquote {{
    border-left: 4px solid var(--orange);
    background: #fff7e6;
    padding: 12px 16px;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
    color: var(--text-secondary);
  }}
  .report-body table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 13px;
  }}
  .report-body th {{
    background: var(--primary);
    color: white;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
  }}
  .report-body td {{
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }}
  .report-body tr:nth-child(even) td {{
    background: #f9f9fb;
  }}
  .report-body tr:hover td {{
    background: var(--primary-light);
  }}
  .report-body ul, .report-body ol {{
    padding-left: 24px;
    margin-bottom: 12px;
  }}
  .report-body li {{
    margin-bottom: 6px;
  }}
  .report-body strong {{
    font-weight: 600;
    color: var(--text);
  }}
  .report-body em {{
    font-style: italic;
    color: var(--text-secondary);
  }}
  .report-body hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
  }}
  .report-footer {{
    text-align: center;
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 24px;
    padding: 16px;
  }}
</style>
</head>
<body>
<div class="container">
  <div class="report-header">
    <h1>港股通调入调出预测分析报告</h1>
    <div class="subtitle">
      预测目标: {adjustment_label or cutoff_date} ｜ 数据截止: {cutoff_date} ｜ 生成时间: {now}
    </div>
  </div>
  <div class="info-bar">
    <div class="info-item">选股范畴: <strong>{total}</strong> 只</div>
    <div class="info-item">当前HSCI: <strong>{current}</strong> 只</div>
    <div class="info-item">预测HSCI: <strong>{predicted}</strong> 只</div>
    <div class="info-item">HSCI调入: <strong>{hsci_add}</strong> 只</div>
    <div class="info-item">HSCI调出: <strong>{hsci_remove}</strong> 只</div>
    <div class="info-item">港股通调入: <strong>{connect_add}</strong> 只</div>
    <div class="info-item">港股通调出: <strong>{connect_remove}</strong> 只</div>
  </div>
  <div class="stats-grid">
    <div class="stat-card add"><div class="value">{hsci_add}</div><div class="label">HSCI调入</div></div>
    <div class="stat-card remove"><div class="value">{hsci_remove}</div><div class="label">HSCI调出</div></div>
    <div class="stat-card add"><div class="value">{connect_add}</div><div class="label">港股通调入</div></div>
    <div class="stat-card remove"><div class="value">{connect_remove}</div><div class="label">港股通调出</div></div>
    <div class="stat-card neutral"><div class="value">{predicted - current:+d}</div><div class="label">HSCI净变化</div></div>
  </div>
  <div class="report-body">
{html_body}
  </div>
  <div class="report-footer">
    本报告由大模型基于量化预测数据自动生成，仅供参考分析，不构成投资建议。
  </div>
</div>
</body>
</html>"""
