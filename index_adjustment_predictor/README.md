# A股指数成分股调整预测系统

基于指数编制规则与大语言模型（LLM），预测沪深300指数半年度成分股调整方案，并支持历史回测验证。系统提供命令行交互、REST API、Web 页面三种入口，均支持自然语言输入。

## 核心功能

- **预测调整**：依据官方编制规则，量化预测某调整批次（如2025年6月）沪深300的调入调出名单
- **回测验证**：将历史预测结果与指数公司实际公告对比，计算精确率/召回率/F1，并对偏差逐只归因
- **自然语言交互**：中文对话式查询，支持多轮上下文与意图继承
- **双预测模式**：纯量化（可解释、稳定）与量化+LLM增强（结合财务、行业等定性因素），可按需切换

## 支持的指数

| 指数 | 代码 | 样本数 | 编制机构 | 选样规则 |
|------|------|--------|---------|---------|
| 沪深300 | 000300.SH | 300 | 中证指数公司 | 市值+流动性纯排名 |

沪深300编制规则为100%量化透明，适合用规则引擎预测。

## 技术架构

系统采用**配置驱动 + Agent 编排**架构：工作流由 `skills/skill.md` 的 YAML 定义驱动，编排器（orchestrator）按工作流逐步调用 Agent 注册表（agent_registry）中的六个独立 Agent；新增能力只需实现一个 Agent 并在 `skill.md` 增加一步，调度引擎零改动：

| Agent | 职责 |
|-------|------|
| DataAgent | 采集行情、成分股、财务、上市信息（Tushare 主数据源） |
| RuleAgent | 解析编制规则、筛选样本空间、计算日均总市值排名 |
| PredictAgent | 量化排名选样 + 可选 LLM 定性增强 |
| BacktestAgent | 对比预测与实际调整、偏差归因 |
| ReportAgent | 生成 Markdown / JSON / HTML 三格式报告 |
| LLMReportAgent | 基于大模型生成定性分析报告（自然语言叙述版） |

## 快速开始

### 环境要求

- Python 3.11+
- Tushare 账号（需 2000 积分以上权限，用于历史成分股快照、财务指标、交易日历）
- 阿里云百炼平台 API Key（用于通义千问大模型，可选——纯量化模式不需要）

### 安装

```bash
git clone <仓库地址>
cd index_adjustment_predictor
pip install -r requirements.txt
```

### 配置

在项目根目录创建 `.env` 文件，填入以下配置：

```ini
# ===== 大模型配置（纯量化模式可不配，但建议配置以启用LLM增强）=====
# 阿里云百炼平台 API Key: https://help.aliyun.com/zh/model-studio/get-api-key
LLM_API_KEY=your_dashscope_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
LLM_TEMPERATURE=0.1
# 沪深300候选集较大，token数不足会导致JSON截断，建议16384
LLM_MAX_TOKENS=16384
# 大模型响应较慢，建议至少120秒
LLM_TIMEOUT=360

# ===== 数据源配置 =====
# Tushare Token: https://tushare.pro/user/token
TUSHARE_API_TOKEN=your_tushare_token
```

> 如需切换到 OpenAI，只需改 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` 三行即可。

### 运行

#### 方式一：交互式对话（推荐首次体验）

```bash
python main.py
```

进入对话模式后直接输入自然语言：

```
请输入您的需求> 预测2025年6月沪深300的调整
请输入您的需求> 纯量化预测2025年6月沪深300      # 跳过LLM，仅用量化排名
请输入您的需求> 回测2025年6月沪深300            # 回测验证
请输入您的需求> 查看沪深300的编制规则
请输入您的需求> 退出
```

#### 方式二：命令行直接执行

```bash
# 预测（量化+LLM）
python main.py --predict 2025-06 csi300

# 纯量化预测（跳过LLM分析）
python main.py --predict 2025-06 csi300 --no-llm

# 回测
python main.py --backtest 2025-06 csi300
```

#### 方式三：启动 Web 服务

```bash
python main.py --api
```

浏览器打开 `http://localhost:8000/`，在页面输入框中输入自然语言即可。预测/回测完成后页面会显示"查看可视化报告"按钮，点击直达 HTML 可视化报告。

### 使用示例

| 输入 | 意图 | 说明 |
|------|------|------|
| 预测2025年6月沪深300的调整 | predict | 量化+LLM 预测 |
| 纯量化预测2025年6月沪深300 | predict | 跳过 LLM，仅量化 |
| 回测2025年6月沪深300 | backtest | 量化+LLM 回测 |
| 纯量化回测2025年6月沪深300 | backtest | 回测纯量化预测结果 |
| 查看沪深300的编制规则 | query_rules | 返回规则正文 |

支持自然语言省略与多轮继承，例如先问"预测2025年6月沪深300"，再问"上次准吗"会自动继承上下文进行回测。

## 项目结构

```
index_adjustment_predictor/
├── api/                    # FastAPI 接口层
│   ├── main.py             #   应用入口、路由注册、静态文件挂载
│   └── routes/            #   predict / backtest / chat 三个路由
├── core/
│   ├── orchestrator.py     #   工作流编排器（预测/回测两条工作流）
│   ├── skill_loader.py     #   从 skills/skill.md 加载工作流定义
│   ├── agent_registry.py   #   Agent 注册表（逻辑名→工厂）
│   ├── base_agent.py       #   Agent 基类
│   └── llm_client.py       #   大模型客户端（兼容 OpenAI 接口）
├── agents/                 # 业务 Agent 层
│   ├── data_agent.py       #   数据采集
│   ├── rule_agent.py       #   规则计算
│   ├── predict_agent.py    #   预测分析（量化+LLM）
│   ├── backtest_agent.py   #   回测评估
│   └── report_agent.py     #   报告生成（md/json/html）
├── interface/
│   └── chat_handler.py     #   自然语言意图识别（正则+LLM兜底）
├── models/
│   ├── quantitative/rank_model.py   # 量化排名选样模型
│   └── llm_enhanced/qualitative.py  # LLM 增强分析
├── data/                   # 数据采集脚本与缓存
│   ├── scripts/            #   Tushare 数据获取
│   └── storage/            #   SQLite 缓存
├── rules/
│   ├── csi300.md           # 沪深300编制规则（YAML frontmatter + Markdown）
│   └── rule_parser.py      # 规则解析器
├── prompts/                # LLM 提示词模板
├── config/                 # 配置文件（config.yaml / data_source.yaml）
├── static/index.html       # Web 前端单页应用
├── reports/                # 生成的预测/回测报告（.md/.json/.html）
├── main.py                 # 项目入口（CLI/API/交互）
├── requirements.txt
└── .env                    # 环境变量配置（不入库）
```

## 调整时间表

| 调整批次 | 考察截止日 | 审核时间 | 生效日 |
|---------|-----------|---------|--------|
| 6月调整 | 4月30日 | 5月下旬 | 6月第二个星期五后下一交易日 |
| 12月调整 | 10月31日 | 11月下旬 | 12月第二个星期五后下一交易日 |

预测时系统会根据目标月份自动计算 `cutoff_date`：6月调整用上一年5月1日至当年4月30日的考察期数据，12月调整用上一年11月1日至当年10月31日。

## 预测原理

### 量化排名选样

1. **样本空间筛选**：剔除 ST/*ST、北交所股票、上市时间不足（科创板/创业板满12个月，其他满3个月）、非成分股财务亏损等
2. **成交额排名**：按过去一年日均成交金额降序排名，剔除后50%（老样本若排名前60%可保留）
3. **市值排名**：对剩余证券按过去一年日均总市值降序排名，选取前300名作为新成分股
4. **缓冲区规则**：排名≤240的非成分股优先调入，排名>360的成分股优先调出，老样本在241-360区间享有保留优先权
5. **差集计算**：新成分股与原300只成分股求差集，新进为调入、消失为调出

### LLM 增强（可选）

在量化候选基础上，将排名数据、候选股票财务指标（净利润、ROE、资产负债率、审计意见）、编制规则正文一并送入大模型，由其结合行业均衡、风险事件等定性因素给出调整建议。`use_llm` 参数控制是否启用。

## 报告输出

每次预测/回测在 `reports/` 目录生成三种格式报告：

| 格式 | 文件名示例 | 用途 |
|------|-----------|------|
| Markdown | `prediction_csi300_2025-06.md` | 人可读，含调入调出表格与分析说明 |
| JSON | `prediction_csi300_2025-06.json` | 程序消费，含完整预测数据供回测加载 |
| HTML | `prediction_csi300_2025-06.html` | 可视化，含置信度色标、市值对比图 |

纯量化模式文件名带 `_no_llm` 后缀，如 `prediction_csi300_2025-06_no_llm.json`，避免覆盖 LLM 预测结果。

## 技术选型

| 类别 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 原生异步、自动生成 OpenAPI 文档 |
| 数据处理 | Pandas + NumPy | 金融时序数据处理 |
| 数据源 | Tushare（主）/ AKShare（备） | Tushare 提供历史成分股快照与财务指标 |
| 大模型 | 通义千问系列 | 中文理解强、支持长上下文，兼容 OpenAI 接口可切换 |
| 缓存 | SQLite | 历史数据永久缓存、实时数据24小时过期 |
| 前端 | 原生 HTML/CSS/JS | 无构建依赖、离线可用 |
| 配置 | YAML + .env | 结构化参数与密钥分离 |

## 常见问题

**Q: 启动 API 报 `WinError 10013`？**
A: 8000 端口被占用。执行 `netstat -ano | findstr :8000` 查到占用进程 PID，再 `taskkill /PID <PID> /F` 结束即可。

**Q: 预测很慢（几分钟）？**
A: 正常现象。预测需采集全市场行情、成分股、财务数据并调用大模型，单次约1-3分钟。纯量化模式（`--no-llm`）会快很多。

**Q: LLM 报超时错误？**
A: 调大 `.env` 中的 `LLM_TIMEOUT`（建议360秒），并确认 `LLM_MODEL` 选用响应较快的型号（如 `qwen-plus`）。

**Q: 报告里调入调出数量不一致？**
A: 已在新版本修复。原因是被样本空间剔除的成分股（如 ST 股）漏计入调出，现已确保遍历全部当前成分股计算调出。

## License

本项目仅用于学习与研究目的，不构成任何投资建议。
