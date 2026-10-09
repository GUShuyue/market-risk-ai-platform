"""
定性分析模型 - LLM增强

使用LLM分析非量化因素，处理量化模型无法确定的灰色地带。
同时提供基于规则的风险标记，作为LLM分析的输入。
"""

from typing import Dict, List
from core.llm_client import LLMClient
from prompts.prompt_manager import PromptManager
import json


class QualitativeModel:
    """
    定性分析模型
    
    处理两类问题:
    1. 风险标记: 基于规则给候选股票打风险标签（不调用LLM）
    2. LLM分析: 把风险标记和量化结果一起发给LLM做最终判断
    
    分析维度:
        - 行业均衡性
        - 停牌风险
        - 重大事项（并购重组等）
        - 财务异常
        - 市场异常波动
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.prompt_manager = PromptManager()

    def analyze(
        self,
        quant_result: dict,
        market_data: dict,
        rule_text: str = ""
    ) -> dict:
        """
        执行定性分析

        参数:
            quant_result: 量化模型结果（add_candidates, remove_candidates）
            market_data: DataAgent采集的全部数据
            rule_text: 指数编制规则的Markdown正文（作为LLM context）

        返回:
            {
                "adjusted_add": [...],
                "adjusted_remove": [...],
                "risk_flags": {...},
                "industry_analysis": {...},
                "llm_reasoning": "..."
            }
        """
        # 1. 风险标记：给每只候选股票打风险标签
        risk_flags = self._flag_risks(quant_result, market_data)

        # 2. 行业均衡分析
        industry_analysis = self.analyze_industry_balance(
            market_data.get("current_constituents", []),
            quant_result.get("add_candidates", []),
            market_data.get("listing_info", [])
        )

        # 3. 构造LLM分析输入
        llm_input = self._build_llm_input(
            quant_result, risk_flags, industry_analysis, market_data, rule_text
        )

        # 4. 调用LLM做最终判断
        llm_response = self.llm.chat(llm_input)
        llm_result = self._parse_llm_response(llm_response)

        return {
            "adjusted_add": llm_result.get("adjusted_add", quant_result.get("add_candidates", [])),
            "adjusted_remove": llm_result.get("adjusted_remove", quant_result.get("remove_candidates", [])),
            "risk_flags": risk_flags,
            "industry_analysis": industry_analysis,
            "llm_reasoning": llm_result.get("reasoning", "")
        }

    def _flag_risks(
        self, quant_result: dict, market_data: dict
    ) -> Dict[str, dict]:
        """
        给候选股票打风险标签

        检查每只调入/调出候选股票是否有以下风险:
        - 停牌风险
        - 财务亏损
        - 高负债
        - 审计意见异常
        - 近期重大公告（并购重组等）
        - 股价异常波动
        """
        candidates = (
            quant_result.get("add_candidates", []) +
            quant_result.get("remove_candidates", [])
        )

        suspension = market_data.get("events_data", {}).get("suspension_status", {})
        announcements = market_data.get("events_data", {}).get("major_announcements", {})
        financial_flags = market_data.get("financial_flags", {})
        price_changes = market_data.get("price_changes", {})

        risk_flags = {}
        for stock in candidates:
            code = stock.get("code") or stock.get("ts_code", "")
            if not code:
                continue

            flags = []

            # 停牌风险
            susp = suspension.get(code, {})
            if susp.get("is_suspended"):
                flags.append({
                    "type": "suspension",
                    "severity": "high" if susp.get("over_25_trading_days") else "medium",
                    "detail": f"停牌{susp.get('suspend_days', 0)}天，原因: {susp.get('suspend_reason', '未知')}"
                })

            # 财务风险
            fin = financial_flags.get(code, {})
            if fin.get("is_loss"):
                flags.append({
                    "type": "financial_loss",
                    "severity": "high",
                    "detail": f"净利润为负: {fin.get('net_profit', 0)}"
                })
            if fin.get("is_high_debt"):
                flags.append({
                    "type": "high_debt",
                    "severity": "medium",
                    "detail": f"资产负债率: {fin.get('debt_ratio', 0)}%"
                })
            if fin.get("has_audit_issue"):
                flags.append({
                    "type": "audit_issue",
                    "severity": "high",
                    "detail": f"审计意见: {fin.get('audit_opinion', '未知')}"
                })

            # 重大公告
            events = announcements.get(code, [])
            for event in events:
                flags.append({
                    "type": "major_event",
                    "severity": "medium",
                    "detail": f"{event.get('event_type', '')}: {event.get('title', '')}"
                })

            # 股价异常波动
            price = price_changes.get(code, {})
            if price.get("is_abnormal"):
                flags.append({
                    "type": "price_abnormal",
                    "severity": "medium",
                    "detail": f"近20日涨跌幅: {price.get('pct_change_20d', 0)}%，波动率: {price.get('volatility', 0)}%"
                })

            if flags:
                risk_flags[code] = flags

        return risk_flags

    def analyze_industry_balance(
        self,
        current_constituents: List[dict],
        candidates: List[dict],
        listing_info: List[dict]
    ) -> dict:
        """
        分析行业均衡性

        计算当前指数的行业分布，以及调入/调出后行业分布的变化。

        返回:
            {
                "current_distribution": {"银行": 15, "房地产": 8, ...},
                "after_adjustment": {"银行": 14, "房地产": 9, ...},
                "imbalance": ["银行占比过高", "信息技术占比偏低"],
                "suggestion": "建议调入信息技术行业股票"
            }
        """
        # 构建股票代码到行业的映射
        code_to_industry = {}
        for info in listing_info:
            code = info.get("ts_code", "")
            industry = info.get("industry", "未知")
            code_to_industry[code] = industry

        # 当前行业分布
        current_dist = {}
        for c in current_constituents:
            code = c.get("con_code") or c.get("ts_code", "")
            industry = code_to_industry.get(code, "未知")
            current_dist[industry] = current_dist.get(industry, 0) + 1

        # 调入后行业分布变化
        after_dist = dict(current_dist)
        for stock in candidates:
            code = stock.get("code") or stock.get("ts_code", "")
            industry = code_to_industry.get(code, "未知")
            after_dist[industry] = after_dist.get(industry, 0) + 1

        # 检查是否失衡
        total = sum(after_dist.values()) or 1
        imbalance = []
        for industry, count in after_dist.items():
            ratio = count / total
            if ratio > 0.15:
                imbalance.append(f"{industry}占比{ratio:.0%}，偏高")
            elif ratio < 0.02 and count > 0:
                imbalance.append(f"{industry}占比{ratio:.0%}，偏低")

        suggestion = ""
        if imbalance:
            suggestion = "建议关注行业均衡，避免调入占比已偏高的行业"

        return {
            "current_distribution": current_dist,
            "after_adjustment": after_dist,
            "imbalance": imbalance,
            "suggestion": suggestion
        }

    def _build_llm_input(
        self,
        quant_result: dict,
        risk_flags: dict,
        industry_analysis: dict,
        market_data: dict,
        rule_text: str
    ) -> str:
        """构造发给LLM的分析输入"""
        prompt = f"""你是一位A股指数分析专家。请基于以下信息，对量化模型预测的调入调出候选做定性判断。

## 指数编制规则
{rule_text}

## 量化模型预测结果
调入候选: {json.dumps(quant_result.get('add_candidates', []), ensure_ascii=False, indent=2)}
调出候选: {json.dumps(quant_result.get('remove_candidates', []), ensure_ascii=False, indent=2)}

## 风险标记
{json.dumps(risk_flags, ensure_ascii=False, indent=2)}

## 行业均衡分析
当前行业分布: {json.dumps(industry_analysis.get('current_distribution', {}), ensure_ascii=False)}
调整后行业分布: {json.dumps(industry_analysis.get('after_adjustment', {}), ensure_ascii=False)}
行业失衡提示: {industry_analysis.get('imbalance', [])}

## 分析要求
1. 检查调入候选中是否有高风险股票（停牌、财务亏损、审计异常），如果有建议剔除
2. 检查调出候选中是否有误判（如因短期停牌导致排名下降但基本面良好）
3. 考虑行业均衡性，是否需要调整调入/调出名单
4. 对于缓冲区内的股票（排名在缓冲区上下限之间），给出倾向性判断

## 输出格式（JSON）
```json
{{
    "adjusted_add": ["代码1", "代码2"],
    "adjusted_remove": ["代码3", "代码4"],
    "removed_from_add": [{{"code": "代码", "reason": "剔除原因"}}],
    "added_to_add": [{{"code": "代码", "reason": "新增原因"}}],
    "reasoning": "整体分析说明"
}}
```
"""
        return prompt

    def _parse_llm_response(self, response: str) -> dict:
        """解析LLM返回的JSON结果"""
        try:
            # 尝试从响应中提取JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"reasoning": response}
