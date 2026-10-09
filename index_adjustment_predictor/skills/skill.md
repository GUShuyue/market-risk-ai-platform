---
name: index_adjustment_prediction
version: 1.2.0
workflows:
  predict_adjustment:
    steps:
      - id: collect
        agent: data_agent
        inputs:
          target_date: "{params.target_date}"
          cutoff_date: "{params.cutoff_date}"
          index_name: "{params.index_name}"
      - id: rank
        agent: rule_agent
        inputs:
          index_name: "{params.index_name}"
          stock_data: "{collect}"
      - id: predict
        agent: predict_agent
        inputs:
          rule_result: "{rank.rule_result}"
          market_data: "{collect}"
          use_llm: "{params.use_llm}"
      - id: report
        agent: report_agent
        inputs:
          prediction: "{predict.prediction}"
          rule_result: "{rank.rule_result}"
          market_data: "{collect}"
          use_llm: "{params.use_llm}"
      - id: analysis
        agent: llm_report_agent
        inputs:
          prediction: "{predict.prediction}"
          market_data: "{collect}"
          use_llm: "{params.use_llm}"
    returns:
      prediction: "{predict.prediction}"
      rule_result: "{rank.rule_result}"
      market_data: "{collect}"
      report_path: "{report.report_path}"
      analysis_report_path: "{analysis.analysis_report_path}"
      analysis_report_html_path: "{analysis.analysis_report_html_path}"
  backtest:
    steps:
      - id: eval
        agent: backtest_agent
        inputs:
          prediction: "{params.prediction}"
          actual_adjustment: "{params.actual_adjustment}"
          rule_result: "{params.rule_result}"
          market_data: "{params.market_data}"
          use_llm: "{params.use_llm}"
      - id: report
        agent: report_agent
        inputs:
          backtest_result: "{eval.backtest_result}"
          prediction: "{params.prediction}"
          index_name: "{params.index_name}"
          target_date: "{params.target_date}"
          use_llm: "{params.use_llm}"
    returns:
      backtest_result: "{eval.backtest_result}"
      report_path: "{report.report_path}"
---

# Skill: 指数成分股调整预测

## 名称
index_adjustment_prediction

## 描述
A股沪深300指数成分股调整预测技能。基于官方指数编制规则和历史数据，预测半年度调整方案，并支持回测验证。

## 版本
1.2.0

## 支持的指数
| 指数 | 代码 | 样本数 | 编制机构 | 规则文件 |
|------|------|--------|---------|---------|
| 沪深300 | 000300.SH | 300 | 中证指数公司 | rules/csi300.md |

### 预测顺序
沪深300（独立预测）

## 调整时间表
| 调整批次 | 考察截止日 | 审核时间 | 生效日 |
|---------|-----------|---------|--------|
| 6月调整 | 4月30日 | 5月下旬 | 6月第二个星期五后下一交易日 |
| 12月调整 | 10月31日 | 11月下旬 | 12月第二个星期五后下一交易日 |

## 工作流定义

工作流定义在本文件开头的 YAML frontmatter 中，由 `core/skill_loader.py` 解析、
`core/orchestrator.py` 执行。每个步骤通过 `agent` 字段引用 `core/agent_registry.py`
中注册的 Agent，`inputs` 用 `{params.xxx}` / `{步骤id}` / `{步骤id.key}` 引用上游数据。

**新增一个功能 = ① 实现一个 `run(inputs) -> dict` 的 Agent 并注册 ② 在 frontmatter 里加一步**，调度器无需改动。

## 规则文件格式
规则文件采用 **YAML frontmatter + Markdown** 格式：
- **YAML frontmatter**: 机器可解析的量化参数（样本数、缓冲区、剔除比例等）
- **Markdown正文**: 人可读的完整规则描述，作为LLM分析的context

RuleParser 解析后供 RuleAgent 使用，不再硬编码参数到代码中。

---

## 自然语言意图映射

### 预测意图
**触发模式**: "预测.*调整" / "调入调出" / "成分股变化" / "哪些股票会被调入" / "预测.*沪深300"
**映射工作流**: predict_adjustment

### 回测意图
**触发模式**: "回测" / "验证.*准确" / "历史表现" / "预测准不准" / "对比.*实际"
**映射工作流**: backtest

### 查询意图
**触发模式**: "查看.*规则" / "指数.*编制" / "调整.*时间"
**响应**: 返回对应指数的编制规则信息

---

## Agent依赖关系

```
data_agent ──┬──> rule_agent ──> predict_agent ──> report_agent
             │
             └──> (actual_adjustment) ──> backtest_agent ──> report_agent
```

## 配置依赖
- LLM API密钥
- 数据源API密钥 (Tushare等)
- 报告输出目录
- 规则文件: rules/csi300.md
