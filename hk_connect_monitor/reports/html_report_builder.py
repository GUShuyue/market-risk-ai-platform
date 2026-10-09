"""
HTML报告生成器

为Mode B预测和回测结果生成带颜色编码的HTML可视化报告。

报告类型:
    1. 预测报告: HSCI调入调出预测 + 港股通映射
    2. 回测报告: 双层准确率指标 + 偏差分析
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict


def build_prediction_report(data: Dict, output_file: str):
    """生成预测HTML报告"""
    hsci_pred = data.get("hsci_prediction", {})
    connect_pred = data.get("connect_prediction", {})
    report_data = data.get("report_data", {})
    cutoff_str = data.get("cutoff_date", "")
    adjustment = data.get("adjustment_label", "")

    # 计算考察期 (cutoff前12个月)
    review_start = ""
    review_end = cutoff_str
    if cutoff_str:
        try:
            cutoff_dt = datetime.strptime(cutoff_str, "%Y-%m-%d")
            review_start_dt = cutoff_dt - timedelta(days=365)
            review_start = review_start_dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 构建信息栏
    info_parts = []
    if adjustment:
        info_parts.append(f"预测目标: {adjustment}")
    if cutoff_str:
        info_parts.append(f"数据截止日: {cutoff_str}")
    if review_start and review_end:
        info_parts.append(f"考察期: {review_start} ~ {review_end}")
    info_parts.append(f"生成时间: {gen_time}")
    info_bar = " ｜ ".join(info_parts)

    html = _build_html_template(
        title="港股通调入调出预测报告",
        info_bar=info_bar,
        body=_build_prediction_body(hsci_pred, connect_pred, report_data)
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)


def build_backtest_report(data: Dict, output_file: str):
    """生成回测HTML报告"""
    layer1 = data.get("layer1_hsci", {})
    layer2 = data.get("layer2_connect", {})
    deviation = data.get("deviation_analysis", {})
    target_date = data.get("target_date", "")
    cutoff_str = data.get("cutoff_date_str", "")
    adjustment = data.get("adjustment_label", "")

    # 计算考察期
    review_start = ""
    if cutoff_str:
        try:
            cutoff_dt = datetime.strptime(cutoff_str, "%Y-%m-%d")
            review_start_dt = cutoff_dt - timedelta(days=365)
            review_start = review_start_dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    info_parts = []
    if adjustment:
        info_parts.append(f"预测目标: {adjustment}")
    if cutoff_str:
        info_parts.append(f"数据截止日: {cutoff_str}")
    if review_start and cutoff_str:
        info_parts.append(f"考察期: {review_start} ~ {cutoff_str}")
    if target_date:
        info_parts.append(f"实际生效日: {target_date}")
    info_parts.append(f"生成时间: {gen_time}")
    info_bar = " ｜ ".join(info_parts)

    html = _build_html_template(
        title="港股通调入调出回测报告",
        info_bar=info_bar,
        body=_build_backtest_body(layer1, layer2, deviation)
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)


# ================================================================
# HTML构建
# ================================================================

def _build_html_template(title: str, info_bar: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei", sans-serif;
    background: #f0f4f8; color: #333; line-height: 1.6; padding: 20px;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; background: #fff;
    border-radius: 14px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); overflow: hidden; }}
  .header {{ background: linear-gradient(135deg, #48b6e0, #5b9ef5); color: #fff;
    padding: 28px 32px 20px; }}
  .header h1 {{ font-size: 22px; margin-bottom: 10px; font-weight: 600; }}
  .header .info-bar {{ font-size: 14px; opacity: 0.95;
    background: rgba(255,255,255,0.15); border-radius: 8px;
    padding: 8px 16px; display: inline-block; }}
  .content {{ padding: 28px 32px; }}
  .section {{ margin-bottom: 28px; }}
  .section h2 {{ font-size: 17px; color: #3182ce; border-left: 4px solid #48b6e0;
    padding-left: 12px; margin-bottom: 14px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px; margin-bottom: 20px; }}
  .metric-card {{ background: #f7fbfd; border-radius: 10px; padding: 16px; text-align: center;
    border: 1px solid #e2eef5; transition: box-shadow 0.2s; }}
  .metric-card:hover {{ box-shadow: 0 2px 8px rgba(72,182,224,0.15); }}
  .metric-card .value {{ font-size: 26px; font-weight: 700; color: #3182ce; }}
  .metric-card .label {{ font-size: 13px; color: #718096; margin-top: 4px; }}
  .metric-card.add {{ background: #f0faf4; border-color: #c6f6d5; }}
  .metric-card.add .value {{ color: #2f855a; }}
  .metric-card.remove {{ background: #fef5f5; border-color: #fed7d7; }}
  .metric-card.remove .value {{ color: #c53030; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 16px; }}
  th {{ background: #e8f4fb; color: #2c5282; padding: 10px 12px; text-align: left;
    font-weight: 600; border-bottom: 2px solid #bee3f8; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #edf2f7; }}
  tr:nth-child(even) {{ background: #f7fbfd; }}
  tr:hover {{ background: #e8f4fb; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 12px; font-weight: 500; }}
  .badge-add {{ background: #c6f6d5; color: #22543d; }}
  .badge-remove {{ background: #fed7d7; color: #742a2a; }}
  .badge-large {{ background: #bee3f8; color: #2a4365; }}
  .badge-mid {{ background: #fefcbf; color: #744210; }}
  .badge-small {{ background: #e9d8fd; color: #4a2385; }}
  .badge-grey {{ background: #e2e8f0; color: #4a5568; }}
  .badge-eligible {{ background: #c6f6d5; color: #22543d; }}
  .badge-ineligible {{ background: #fed7d7; color: #742a2a; }}
  .coverage-bar {{ display: inline-block; width: 60px; height: 8px;
    background: #e2e8f0; border-radius: 4px; margin-right: 8px; vertical-align: middle; }}
  .coverage-fill {{ height: 100%; border-radius: 4px; }}
  .metric-row {{ display: flex; gap: 24px; margin-bottom: 12px; align-items: center; }}
  .metric-row h3 {{ font-size: 15px; min-width: 60px; }}
  .metric-row .metrics {{ font-size: 14px; color: #4a5568; }}
  .footer {{ text-align: center; padding: 18px; color: #a0aec0; font-size: 12px;
    border-top: 1px solid #edf2f7; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>{title}</h1>
    <div class="info-bar">{info_bar}</div>
  </div>
  <div class="content">
    {body}
  </div>
  <div class="footer">港股通调入调出批量预测系统 (Mode B)</div>
</div>
</body>
</html>"""


def _build_prediction_body(hsci_pred, connect_pred, report_data):
    body = ""

    # 汇总指标 (只显示调入调出数量)
    body += '<div class="section">'
    body += '<h2>预测汇总</h2>'
    body += '<div class="summary-grid">'
    body += f'<div class="metric-card add"><div class="value">{report_data.get("hsci_additions", 0)}</div><div class="label">HSCI调入</div></div>'
    body += f'<div class="metric-card remove"><div class="value">{report_data.get("hsci_removals", 0)}</div><div class="label">HSCI调出</div></div>'
    body += f'<div class="metric-card add"><div class="value">{report_data.get("connect_additions", 0)}</div><div class="label">港股通调入</div></div>'
    body += f'<div class="metric-card remove"><div class="value">{report_data.get("connect_removals", 0)}</div><div class="label">港股通调出</div></div>'
    body += '</div>'
    body += '</div>'

    # HSCI预测调入
    additions = hsci_pred.get("additions", [])
    if additions:
        body += '<div class="section">'
        body += '<h2>HSCI预测调入</h2>'
        body += '<table><thead><tr>'
        body += '<th>代码</th><th>名称</th><th>排名</th><th>累计覆盖率</th>'
        body += '<th>平均市值(亿HKD)</th><th>预测子指数</th><th>缓冲区状态</th>'
        body += '</tr></thead><tbody>'
        for s in additions:
            cov = s.get("coverage", 0)
            cov_color = "#2f855a" if cov <= 90 else "#d69e2e" if cov <= 94 else "#718096"
            sub_badge = _sub_index_badge(s.get("predicted_sub_index", ""))
            body += f'<tr>'
            body += f'<td>{s["code"]}</td>'
            body += f'<td>{s.get("name", "")}</td>'
            body += f'<td>{s.get("rank", "")}</td>'
            body += f'<td><span class="coverage-bar"><span class="coverage-fill" style="width:{min(cov,100)}%;background:{cov_color}"></span></span>{cov:.1f}%</td>'
            body += f'<td>{s.get("avg_market_cap_hkd_yi", 0):.1f}</td>'
            body += f'<td>{sub_badge}</td>'
            body += f'<td>{s.get("buffer_status", "")}</td>'
            body += f'</tr>'
        body += '</tbody></table>'
        body += '</div>'

    # HSCI预测调出 (不显示当前子指数: 反推历史成分股时子类信息丢失)
    removals = hsci_pred.get("removals", [])
    if removals:
        body += '<div class="section">'
        body += '<h2>HSCI预测调出</h2>'
        body += '<table><thead><tr>'
        body += '<th>代码</th><th>名称</th><th>排名</th><th>累计覆盖率</th>'
        body += '<th>平均市值(亿HKD)</th><th>缓冲区状态</th>'
        body += '</tr></thead><tbody>'
        for s in removals:
            cov = s.get("coverage", 0)
            body += f'<tr>'
            body += f'<td>{s["code"]}</td>'
            body += f'<td>{s.get("name", "")}</td>'
            body += f'<td>{s.get("rank", "")}</td>'
            body += f'<td>{cov:.1f}%</td>'
            body += f'<td>{s.get("avg_market_cap_hkd_yi", 0):.1f}</td>'
            body += f'<td>{s.get("buffer_status", "")}</td>'
            body += f'</tr>'
        body += '</tbody></table>'
        body += '</div>'

    # 港股通预测
    conn_adds = connect_pred.get("connect_additions", [])
    conn_rems = connect_pred.get("connect_removals", [])
    if conn_adds or conn_rems:
        body += '<div class="section">'
        body += '<h2>港股通调入调出预测</h2>'

        if conn_adds:
            body += '<h3 style="margin: 16px 0 8px; color: #2f855a;">预测调入</h3>'
            body += '<table><thead><tr>'
            body += '<th>代码</th><th>名称</th><th>预测子指数</th><th>资格</th><th>原因</th>'
            body += '</tr></thead><tbody>'
            for s in conn_adds:
                elig_badge = _eligibility_badge(s.get("eligibility", ""))
                body += f'<tr>'
                body += f'<td>{s["code"]}</td>'
                body += f'<td>{s.get("name", "")}</td>'
                body += f'<td>{_sub_index_badge(s.get("predicted_sub_index", ""))}</td>'
                body += f'<td>{elig_badge}</td>'
                body += f'<td style="font-size:12px;">{s.get("reason", "")}</td>'
                body += f'</tr>'
            body += '</tbody></table>'

        if conn_rems:
            body += '<h3 style="margin: 16px 0 8px; color: #c53030;">预测调出</h3>'
            body += '<table><thead><tr>'
            body += '<th>代码</th><th>名称</th><th>原因</th>'
            body += '</tr></thead><tbody>'
            for s in conn_rems:
                body += f'<tr>'
                body += f'<td>{s["code"]}</td>'
                body += f'<td>{s.get("name", "")}</td>'
                body += f'<td style="font-size:12px;">{s.get("reason", "")}</td>'
                body += f'</tr>'
            body += '</tbody></table>'
        body += '</div>'

    # 不合资格的HSCI调入
    ineligible = connect_pred.get("ineligible_additions", [])
    if ineligible:
        body += '<div class="section">'
        body += '<h2>HSCI调入但港股通不合资格</h2>'
        body += '<table><thead><tr>'
        body += '<th>代码</th><th>名称</th><th>预测子指数</th><th>不合资格原因</th>'
        body += '</tr></thead><tbody>'
        for s in ineligible:
            body += f'<tr>'
            body += f'<td>{s["code"]}</td>'
            body += f'<td>{s.get("name", "")}</td>'
            body += f'<td>{_sub_index_badge(s.get("predicted_sub_index", ""))}</td>'
            body += f'<td style="font-size:12px;">{s.get("reason", "")}</td>'
            body += f'</tr>'
        body += '</tbody></table>'
        body += '</div>'

    return body


def _build_backtest_body(layer1, layer2, deviation):
    body = ""

    # Layer 1: HSCI准确率 (只显示调入调出分别指标)
    body += '<div class="section">'
    body += '<h2>Layer 1: HSCI预测准确率</h2>'
    for action, label in [("add", "调入"), ("remove", "调出")]:
        m = layer1.get(action, {})
        color = "#2f855a" if action == "add" else "#c53030"
        body += f'<div class="metric-row">'
        body += f'<h3 style="color:{color};">{label}</h3>'
        body += f'<div class="metrics">'
        body += f'精确率 <b>{m.get("precision",0):.1%}</b> ｜ '
        body += f'召回率 <b>{m.get("recall",0):.1%}</b> ｜ '
        body += f'F1 <b>{m.get("f1",0):.1%}</b>'
        body += f'</div></div>'
        body += f'<p style="font-size:13px;color:#a0aec0;margin-left:84px;margin-bottom:10px;">命中{m.get("matches",0)} / 实际{m.get("actual",0)} / 预测{m.get("predicted",0)}</p>'
    body += '</div>'

    # Layer 2: 港股通准确率 (只显示调入调出分别指标)
    body += '<div class="section">'
    body += '<h2>Layer 2: 港股通映射准确率</h2>'
    for action, label in [("add", "调入"), ("remove", "调出")]:
        m = layer2.get(action, {})
        color = "#2f855a" if action == "add" else "#c53030"
        body += f'<div class="metric-row">'
        body += f'<h3 style="color:{color};">{label}</h3>'
        body += f'<div class="metrics">'
        body += f'精确率 <b>{m.get("precision",0):.1%}</b> ｜ '
        body += f'召回率 <b>{m.get("recall",0):.1%}</b> ｜ '
        body += f'F1 <b>{m.get("f1",0):.1%}</b>'
        body += f'</div></div>'
        body += f'<p style="font-size:13px;color:#a0aec0;margin-left:84px;margin-bottom:10px;">命中{m.get("matches",0)} / 实际{m.get("actual",0)} / 预测{m.get("predicted",0)}</p>'
    body += '</div>'

    # 偏差分析
    body += '<div class="section">'
    body += '<h2>偏差分析</h2>'
    body += f'<p style="margin-bottom:16px;">{deviation.get("summary", "")}</p>'

    for category, label in [
        ("hsci_add_misses", "HSCI调入遗漏"),
        ("hsci_add_false_positives", "HSCI调入误报"),
        ("hsci_remove_misses", "HSCI调出遗漏"),
        ("hsci_remove_false_positives", "HSCI调出误报"),
        ("connect_add_misses", "港股通调入遗漏"),
        ("connect_add_false_positives", "港股通调入误报"),
        ("connect_remove_misses", "港股通调出遗漏"),
        ("connect_remove_false_positives", "港股通调出误报"),
    ]:
        items = deviation.get(category, [])
        if items:
            body += f'<h3 style="margin:12px 0 8px;">{label} ({len(items)}只)</h3>'
            body += '<table><thead><tr><th>代码</th><th>名称</th><th>原因</th></tr></thead><tbody>'
            for item in items:
                body += f'<tr><td>{item.get("code","")}</td><td>{item.get("name","")}</td><td style="font-size:12px;">{item.get("reason","")}</td></tr>'
            body += '</tbody></table>'
    body += '</div>'

    return body


def _sub_index_badge(sub_index: str) -> str:
    """生成子指数标签"""
    if "大型" in sub_index:
        return '<span class="badge badge-large">大型</span>'
    elif "中型" in sub_index:
        return '<span class="badge badge-mid">中型</span>'
    elif "小型" in sub_index:
        return '<span class="badge badge-small">小型</span>'
    return f'<span class="badge badge-grey">{sub_index}</span>'


def _eligibility_badge(eligibility: str) -> str:
    """生成资格标签"""
    if "合资格" in eligibility:
        return '<span class="badge badge-eligible">合资格</span>'
    else:
        return f'<span class="badge badge-ineligible">{eligibility}</span>'
