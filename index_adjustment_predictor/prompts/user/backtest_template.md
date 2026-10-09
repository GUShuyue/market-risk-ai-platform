# 回测分析Prompt模板

## 任务背景
- 预测日期: {{prediction_date}}
- 实际调整日期: {{actual_date}}
- 目标指数: {{index_name}}

## 预测结果
{{prediction_result}}

## 实际调整
{{actual_adjustment}}

## 分析要求
### 准确性分析
1. 调入预测准确率: 命中数/实际调入数
2. 调出预测准确率: 命中数/实际调出数
3. 整体覆盖率: 预测命中数/实际调整总数

### 偏差分析
1. 遗漏的调整: 实际发生但未预测到的调整
2. 误报的调整: 预测发生但实际未调整
3. 原因分析: 为什么出现偏差

### 模型改进建议
基于回测结果，提出改进建议

## 输出格式
```json
{
  "accuracy_metrics": {
    "add_precision": 0.8,
    "add_recall": 0.75,
    "remove_precision": 0.7,
    "remove_recall": 0.85,
    "overall_f1": 0.78
  },
  "deviation_analysis": {
    "missed_adjustments": [...],
    "false_predictions": [...],
    "root_causes": [...]
  },
  "improvement_suggestions": [...]
}
```
