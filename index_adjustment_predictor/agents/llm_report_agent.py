"""
LLM Report Agent - 大模型报告生成Agent

职责：预测完成后，调用大模型基于预测名单、财务、行业分布，
撰写一篇叙述性的分析报告（区别于 ReportAgent 的结构化模板报告）。

通过统一接口 run(inputs) 接入可插拔框架：
    - 纯量化模式（use_llm=False）时跳过，不调用大模型
    - 输出 Markdown + HTML 两种格式：
        reports/prediction_{index}_{date}_analysis.md
        reports/prediction_{index}_{date}_analysis.html
"""

import os
import re
from datetime import datetime
from typing import Dict, List

from core.llm_client import LLMClient
from prompts.prompt_manager import PromptManager


class LLMReportAgent:
    """大模型报告生成Agent - 写叙述性分析报告"""

    name = "llm_report_agent"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.output_dir = self.config.get("report_output_dir", "reports")
        self.llm_client = LLMClient()
        self.prompt_manager = PromptManager()

    def run(self, inputs: dict) -> dict:
        """
        统一接口：生成 LLM 叙述性报告（Markdown + HTML）

        inputs 必含: prediction, market_data；可选: use_llm
        返回: {"analysis_report_path": "...", "analysis_report_html_path": "..."}
              纯量化模式返回空串
        """
        if not inputs.get("use_llm", True):
            return {"analysis_report_path": "", "analysis_report_html_path": ""}

        prediction = inputs.get("prediction", {}) or {}
        market_data = inputs.get("market_data", {}) or {}

        index_name = prediction.get("index_name", "")
        target_date = prediction.get("target_date", "")
        cutoff_date = market_data.get("cutoff_date", "")

        add_list = prediction.get("predicted_add", [])
        alternative_list = prediction.get("alternative_add", [])
        remove_list = prediction.get("predicted_remove", [])

        # 加载并填充模板
        template = self.prompt_manager.load("user/report_template.md")
        prompt = self.prompt_manager.fill(
            template,
            index_name=index_name,
            target_date=target_date,
            cutoff_date=cutoff_date,
            add_count=len(add_list),
            remove_count=len(remove_list),
            add_list=self._format_stocks(add_list),
            alternative_list=self._format_stocks(alternative_list),
            remove_list=self._format_stocks(remove_list),
            financial_data=self._format_financial(prediction, market_data),
            industry_distribution=self._format_industry(market_data),
        )

        # 调用大模型
        report = self.llm_client.chat(prompt)

        # 保存报告（Markdown + HTML）
        os.makedirs(self.output_dir, exist_ok=True)
        suffix = "" if inputs.get("use_llm", True) else "_no_llm"
        base = f"prediction_{index_name}_{target_date}_analysis{suffix}"
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        header = (
            f"# 指数成分股调整分析报告\n\n"
            f"> 由大模型基于量化预测结果自动生成 | 指数: {index_name} | "
            f"目标: {target_date} | 生成时间: {generated_at}\n\n"
        )

        md_path = os.path.join(self.output_dir, base + ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(header + report)

        html_path = os.path.join(self.output_dir, base + ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self._wrap_html(report, index_name, target_date, generated_at))

        return {
            "analysis_report_path": md_path,
            "analysis_report_html_path": html_path,
        }

    # ===== 格式化输入 =====

    @staticmethod
    def _format_stocks(stocks: List[dict]) -> str:
        """把预测名单格式化为可读文本"""
        if not stocks:
            return "无"
        lines = []
        for s in stocks:
            code = s.get("stock_code") or s.get("code", "")
            name = s.get("stock_name") or s.get("name", "")
            rank = s.get("rank", "-")
            reason = s.get("reasoning", "")
            lines.append(f"- {code} {name}（排名{rank}）: {reason}")
        return "\n".join(lines)

    @staticmethod
    def _format_financial(prediction: dict, market_data: dict) -> str:
        """提取候选股票的财务数据"""
        financial_flags = market_data.get("financial_flags", {})
        codes = set()
        for key in ("predicted_add", "alternative_add", "predicted_remove"):
            for s in prediction.get(key, []):
                code = s.get("stock_code") or s.get("code", "")
                if code:
                    codes.add(code)

        if not codes:
            return "无财务数据"

        lines = []
        for code in sorted(codes):
            flag = financial_flags.get(code)
            if not flag:
                continue
            net = flag.get("net_profit", 0) or 0
            roe = flag.get("roe", 0) or 0
            debt = flag.get("debt_ratio", 0) or 0
            loss = "亏损" if flag.get("is_loss") else "盈利"
            lines.append(
                f"- {code}: 净利润{net/1e8:.2f}亿, ROE {roe:.1f}%, "
                f"资产负债率{debt:.1f}%, {loss}"
            )
        return "\n".join(lines) if lines else "无财务数据"

    @staticmethod
    def _format_industry(market_data: dict) -> str:
        """统计当前成分股的行业分布"""
        listing_info = market_data.get("listing_info", [])
        constituents = market_data.get("current_constituents", [])

        code_to_industry = {
            i.get("ts_code", ""): i.get("industry", "未知")
            for i in listing_info
        }

        dist: Dict[str, int] = {}
        for c in constituents:
            if not isinstance(c, dict):
                continue
            code = c.get("con_code") or c.get("ts_code", "")
            industry = code_to_industry.get(code, "未知")
            dist[industry] = dist.get(industry, 0) + 1

        if not dist:
            return "无行业数据"

        items = sorted(dist.items(), key=lambda x: -x[1])
        return "、".join(f"{k}{v}只" for k, v in items[:15])

    # ===== Markdown → HTML =====

    @staticmethod
    def _markdown_to_html(md: str) -> str:
        """轻量 Markdown 转 HTML（标题/段落/列表/加粗/引用/行内代码）"""
        lines = md.split("\n")
        out = []
        in_ul = False
        in_ol = False

        def inline(text: str) -> str:
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
            text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
            return text

        def close_lists():
            nonlocal in_ul, in_ol
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if in_ol:
                out.append("</ol>")
                in_ol = False

        for line in lines:
            s = line.strip()
            if not s:
                close_lists()
                continue

            # 标题
            m = re.match(r"^(#{1,6})\s+(.*)", s)
            if m:
                close_lists()
                level = min(len(m.group(1)), 3)  # 限制到 h3，避免过深
                out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
                continue

            # 引用
            if s.startswith("> "):
                close_lists()
                out.append(f"<blockquote>{inline(s[2:])}</blockquote>")
                continue

            # 有序列表
            m = re.match(r"^(\d+)\.\s+(.*)", s)
            if m:
                if in_ul:
                    out.append("</ul>")
                    in_ul = False
                if not in_ol:
                    out.append("<ol>")
                    in_ol = True
                out.append(f"<li>{inline(m.group(2))}</li>")
                continue

            # 无序列表
            m = re.match(r"^[-*+]\s+(.*)", s)
            if m:
                if in_ol:
                    out.append("</ol>")
                    in_ol = False
                if not in_ul:
                    out.append("<ul>")
                    in_ul = True
                out.append(f"<li>{inline(m.group(1))}</li>")
                continue

            # 段落
            close_lists()
            out.append(f"<p>{inline(s)}</p>")

        close_lists()
        return "\n".join(out)

    @staticmethod
    def _wrap_html(body_md: str, index_name: str, target_date: str,
                   generated_at: str) -> str:
        """把报告正文包成完整 HTML 文档（亮色风格）"""
        body_html = LLMReportAgent._markdown_to_html(body_md)
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>指数成分股调整分析报告</title>
<style>
  :root {{
    --bg: #f6f8fb;
    --text: #1e293b;
    --dim: #64748b;
    --border: #e6ebf2;
    --primary: #2563eb;
    --primary-2: #6366f1;
    --accent: #0ea5e9;
    --orange: #d97706;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
    background: radial-gradient(1100px 500px at 80% -10%, #e6ecfb 0%, transparent 60%),
                radial-gradient(900px 500px at -10% 110%, #e0f2fe 0%, transparent 55%),
                var(--bg);
    color: var(--text);
    line-height: 1.8;
  }}
  .page {{ max-width: 840px; margin: 0 auto; padding: 40px 24px 72px; }}
  .meta-bar {{
    font-size: 13px; color: var(--dim);
    padding: 10px 16px; margin-bottom: 28px;
    background: #fff; border: 1px solid var(--border); border-radius: 10px;
    box-shadow: 0 1px 3px rgba(15,23,42,.05);
  }}
  .meta-bar b {{ color: var(--primary); font-weight: 600; }}
  main {{
    background: #fff; border: 1px solid var(--border); border-radius: 16px;
    padding: 32px 36px; box-shadow: 0 2px 12px rgba(15,23,42,.06);
  }}
  main h1 {{
    font-size: 24px; margin: 0 0 20px; line-height: 1.4;
    background: linear-gradient(135deg, var(--primary), var(--primary-2));
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
  }}
  main h2 {{
    font-size: 18px; margin: 30px 0 12px; padding-left: 12px;
    border-left: 4px solid var(--primary); line-height: 1.4;
  }}
  main h3 {{ font-size: 15px; margin: 20px 0 8px; color: var(--primary); }}
  main p {{ margin: 10px 0; }}
  main strong {{ color: #0f172a; }}
  main code {{
    background: #f1f5f9; border: 1px solid var(--border);
    padding: 1px 6px; border-radius: 5px; font-size: 13px; color: #be185d;
  }}
  main blockquote {{
    border-left: 3px solid var(--orange); margin: 12px 0; padding: 6px 14px;
    color: #92400e; background: #fef3c7; border-radius: 0 8px 8px 0;
  }}
  main ul, main ol {{ margin: 8px 0; padding-left: 26px; }}
  main li {{ margin: 6px 0; }}
  @media (max-width: 640px) {{ main {{ padding: 22px 18px; }} }}
</style>
</head>
<body>
  <div class="page">
    <div class="meta-bar">
      <b>指数成分股调整分析报告</b> · 指数: {index_name} · 目标: {target_date} · 生成时间: {generated_at}
    </div>
    <main>
{body_html}
    </main>
  </div>
</body>
</html>
"""
