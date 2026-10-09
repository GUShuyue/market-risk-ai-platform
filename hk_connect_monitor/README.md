# 港股通标的池监控系统

基于《上海证券交易所沪港通业务实施办法》和恒生综合指数（HSCI）编算规则，对港股通标的池的调入调出进行持续监控预警与定期调整预测。

系统分为两个工作模式和一个自然语言交互层：

- **Mode A** — 持续监控扫描，识别"接近调入/调出门槛"的个股并输出风险预警
- **Mode B** — 批量预测恒生综合指数定期检讨时的成分股变动，并映射到港股通调入调出
- **交互层** — 通过 CLI 或 Web 界面用自然语言调用上述功能，支持多轮对话和实时进度推送

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      自然语言交互层                          │
│   main_chat.py (CLI)          api_main.py (FastAPI)         │
│          │                          │  SSE 流式进度          │
│          └──────────┬───────────────┘                        │
│                     ▼                                        │
│          interface/chat_handler.py                           │
│     (正则快速匹配 + LLM 意图识别兜底)                         │
└─────────┬───────────────────────┬───────────────────────────┘
          ▼                       ▼
┌──────────────────┐   ┌──────────────────────────────────────┐
│     Mode A       │   │              Mode B                  │
│  持续监控扫描     │   │          预测 / 回测                  │
│                  │   │                                      │
│  规则引擎        │   │  HSCI覆盖率模型                       │
│  风险分级        │   │  港股通映射层                          │
│  变更检测        │   │  历史成分股重建                        │
│                  │   │  换手率测试                           │
│                  │   │  回测Agent                            │
└────────┬─────────┘   └──────────────┬───────────────────────┘
         │                            │
         ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      数据层                                  │
│  akshare (新浪行情)  百度估值  东方财富  上交所XLS  HSCI CSV  │
│  JSON 断点续采缓存（无 SQLite）                               │
└─────────────────────────────────────────────────────────────┘
```

## 环境要求

- Python >= 3.12
- 网络可访问新浪财经、百度估值、东方财富等数据源
- （可选）通义千问 API Key，用于 LLM 意图识别和分析报告生成

## 安装

```bash
pip install -r requirements.txt
```

依赖清单包括 fastapi、uvicorn、akshare、pandas、tqdm、pyyaml、pdfplumber、xlrd 等。

在项目根目录创建 `.env` 文件配置 LLM（用于自然语言交互和分析报告）：

```ini
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

不配置 LLM 时，Mode A 和 Mode B 的核心预测功能仍可正常运行，仅自然语言交互的意图兜底和 LLM 分析报告会跳过。

## 快速开始

### Mode A：监控扫描

```bash
# 扫描当前港股通标的风险状况
python main_a.py

# 指定基准日扫描
python main_a.py --date 2025-12-31

# 扫描并与上次结果对比，输出变更检测报告
python main_a.py --change
```

扫描完成后，控制台输出风险汇总，同时生成 HTML 报告到 `output/mode_a/report.html`，扫描状态保存到 `mode_a/state/data/` 供下次变更检测使用。

### Mode B：预测与回测

```bash
# 预测 2026 年 3 月生效的港股通调整（含 LLM 分析报告）
python main_b.py predict 2026-03

# 仅预测，跳过 LLM 分析报告
python main_b.py predict 2026-03 --no-report

# 回测 2026 年 3 月的预测准确率
python main_b.py backtest 2026-03

# 设置最低市值过滤阈值（亿港币，默认 5 亿）
python main_b.py predict 2026-03 --min-mc 10
```

调整月份支持多种写法：`2026-03`、`2026-3`、`2026年3月`、`202603`。

日期推导规则：3 月和 9 月为定期检讨生效月，对应数据截止日分别为前一年 12 月 31 日和当年 6 月 30 日；6 月和 12 月为快速纳入生效月。

预测结果保存到 `output/mode_b/prediction_{月份}.json` 和 `.html`，回测结果保存到 `output/mode_b/backtest_{月份}.json` 和 `.html`。

### 自然语言交互

```bash
# CLI 交互模式
python main_chat.py

# 启动 Web 服务（默认端口 8000）
python main_chat.py --api

# 指定端口
python main_chat.py --api --port 9000
```

Web 模式下浏览器访问 `http://localhost:8000`，支持以下自然语言指令：

| 说法示例 | 触发功能 |
|---------|---------|
| "扫描港股通" / "监控一下风险" | Mode A 扫描 |
| "预测2026年3月港股通调整" | Mode B 预测 |
| "回测2026年3月" / "上次预测准吗" | Mode B 回测 |
| "查看HSCI编制规则" / "换手率要求是什么" | 规则查询 |
| "帮助" / "能做什么" | 帮助信息 |

支持多轮上下文：先说"预测2026年3月"，再说"回测呢"，系统自动继承 `2026-03` 参数。

## Mode A：监控扫描

Mode A 的定位是持续扫描可量化的边界条件，识别"谁有风险"，而非预测何时调整。真正的调入调出发生在恒生综合指数定期调整生效日，由 Mode B 负责。

### 规则引擎

规则引擎按优先级顺序评估每只股票，对应《实施办法》的具体法条：

| 优先级 | 检查项 | 法条依据 |
|--------|--------|---------|
| 1 | 排除条件（A+H股A股 ST/退市、非港币报价） | 第七十条第二款 |
| 2 | 第二上市排除（-S/-SW 后缀） | 第六十五条/七十一条 |
| 3 | WVR 首次纳入条件（上市时长+市值+成交额） | 第七十一条 |
| 4 | 小型股市值门槛（调入 50 亿/调出 40 亿） | 第七十条第三项/七十三条 |
| 5 | 默认安全（大型/中型/A+H 股无量化门槛） | — |

### 风险分级

根据股票当前市值距门槛的百分比划分风险等级：

- **HIGH** — 距边界 10% 以内
- **MEDIUM** — 距边界 10%-20%
- **LOW** — 距边界 20% 以上

### 变更检测

使用 `--change` 参数时，系统对比前后两次扫描结果，识别 7 种变更类型：新增高风险排除事件（CRITICAL）、新增风险或风险升级（WARNING）、风险类型变化（INFO）、风险解除（RESOLVED）等。

## Mode B：预测与回测

### 预测流程

```
输入: 调整月份 (如 2026-03)
  │
  ├─ 1. 日期推导 → cutoff_date (如 2025-12-31)
  ├─ 2. 加载股票信息 + 当前港股通池 + HSCI历史数据
  ├─ 3. 重建 cutoff 时点的港股通池
  ├─ 4. 构建选股范畴市值 (百度估值 / H股股本×价格)
  ├─ 5. 换手率测试 (编算总则 5.6.2)
  ├─ 6. 月末市值计算 (实施办法第70/73条)
  ├─ 7. HSCI 覆盖率模型预测
  │     ├─ 按12个月日均市值降序排列
  │     ├─ 计算累计市值覆盖率
  │     ├─ 非成分股覆盖率 <= 94% → 调入
  │     ├─ 现有成分股覆盖率 > 96% → 调出
  │     └─ 94%-96% 灰色区间 → 维持现状
  ├─ 8. 港股通映射层
  │     ├─ HSCI调入 → 资格检查 (第二上市/WVR/小型股市值/货币)
  │     └─ HSCI调出 → 若在港股通池中 → 调出 (A+H股保留)
  ├─ 9. 换手率后过滤
  └─ 10. 输出: JSON + HTML + (可选)LLM分析报告
```

### HSCI 覆盖率模型

恒生综合指数的成分股挑选方法：将选股范畴内所有证券按前 12 个月日均市值降序排列，累计市值覆盖率前 95% 以内的证券被选为成分股。

缓冲区规则避免边缘股票频繁进出：

- 非成分股覆盖率 <= 94% → 加入指数
- 现有成分股覆盖率 > 96% → 从指数剔除
- 94%-96% 之间维持现状

子指数分级采用缓冲区矩阵，根据股票当前分类（大型/中型/小型/非成分股）查表决定升降级门槛：

| 目标分类 | 现有大型 | 现有中型 | 现有小型 | 非成分股 |
|---------|---------|---------|---------|---------|
| 大型 | <= 83% | <= 77% | <= 77% | <= 77% |
| 中型 | <= 97% | <= 97% | <= 93% | <= 93% |
| 小型 | 其余 | 其余 | 其余 | 其余 |

### 港股通映射层

HSCI 成分股变动映射到港股通调入调出时的资格检查：

| 检查项 | 规则 |
|--------|------|
| 第二上市排除 | -S/-SW 后缀股票不可纳入（除非已转为主要上市） |
| WVR 条件 | 上市满 6 个月 + 20 个交易日；183 日日均市值 >= 200 亿；183 日总成交额 >= 60 亿 |
| 小型股市值门槛 | 调入线 50 亿港币，调出线 40 亿港币（12 个月平均月末市值） |
| 货币检查 | 非港币报价排除 |
| A+H 股保留 | HSCI 调出但属于 A+H 股的，保留港股通资格（第七十条第四项） |

### 历史成分股重建

cutoff date 的 HSCI 成分股清单通过当前成分股和变动记录倒推得到：从当前列表出发，按时间倒序回溯 cutoff 之后的变动，撤销 Add（删除）和 Delete（加回）。重建后与官方 `count_after` 字段对比校验数量一致性。

### 回测

回测对预测结果与实际调整记录进行双层比对：

- **Layer 1（HSCI 层）** — 预测的 HSCI 调入调出 vs 实际 HSCI 变动
- **Layer 2（港股通层）** — 预测的港股通调入调出 vs 实际港股通调整

每层计算 Precision、Recall、F1，并对漏报和误报逐只推断偏差原因（覆盖率接近边界、缓冲区保护、数据时点差异、资格检查未通过等）。

## 数据源

| 数据 | 来源 | 用途 |
|------|------|------|
| 港股通标的池 | 上交所官网 XLS 文件 | 当前标的池 |
| HSCI 成分股分类 | 恒生官网 PDF → CSV 转换 | 大/中/小型分类 |
| HSCI 历史变动 | 恒生官网 CSV | 成分股重建、回测基准 |
| 港股通历史调整 | 上交所 XLS | 回测基准 |
| 实时行情 | 新浪财经（akshare） | 当前价格、成交量 |
| 历史市值 | 百度估值（akshare） | 12 个月日均市值 |
| 上市日期/股本 | 东方财富（akshare） | A+H 股 H 股股本、上市日期 |
| 日 K 线 | 新浪财经（akshare） | 换手率测试、月末市值 |

大规模数据采集（2800+ 港股的日 K 线）使用 JSON 文件断点续采缓存（`mode_b/data/` 下按 cutoff 日期命名），已采集部分直接跳过，避免重复请求。换手率测试结果、月末市值、选股范畴市值同样按 cutoff 日期缓存为 JSON 文件，存在则直接加载。

## 项目结构

```
hk_connect_monitor/
├── main_a.py                     # Mode A 入口
├── main_b.py                     # Mode B 入口
├── main_chat.py                  # 自然语言 CLI 入口
├── api_main.py                   # FastAPI Web 后端
├── requirements.txt              # 依赖清单
├── .env                          # LLM / Tushare 配置
│
├── mode_a/                       # Mode A — 监控扫描
│   ├── config.py                 # 规则阈值常量
│   ├── models.py                 # 数据模型 (StockInfo, AssessmentResult)
│   ├── core/
│   │   ├── rules.py              # 规则引擎
│   │   ├── monitor.py            # 监控协调器
│   │   └── change_detector.py    # 变更检测器
│   ├── data/
│   │   ├── data_provider.py      # 数据提供器抽象接口
│   │   └── akshare_data_provider.py  # akshare 实现
│   ├── state/
│   │   ├── state_store.py        # 状态持久化
│   │   └── data/                 # 扫描历史存档
│   └── scripts/
│       └── convert_hsci_pdf.py   # HSCI PDF → CSV 转换工具
│
├── mode_b/                       # Mode B — 预测/回测
│   ├── core/
│   │   └── orchestrator_b.py     # 编排器
│   ├── models/
│   │   ├── hsci_coverage_model.py    # HSCI 覆盖率模型
│   │   └── stock_connect_mapper.py   # 港股通映射层
│   ├── data/
│   │   ├── hsci_history_loader.py    # 历史成分股重建
│   │   ├── universe_builder.py       # 选股范畴构建
│   │   ├── month_end_market_cap.py   # 月末市值计算
│   │   ├── turnover_tester.py        # 换手率测试
│   │   └── ah_stock_mapping.json     # A+H 股映射
│   ├── agents/
│   │   └── backtest_agent_b.py       # 回测 Agent
│   └── utils/
│       └── date_utils.py             # 日期推导
│
├── interface/
│   └── chat_handler.py           # 核心对话处理器
│
├── core/
│   └── llm_client.py             # LLM 客户端封装
│
├── reports/                      # 报告生成（共享）
│   ├── reporter.py               # Console/HTML/Change 报告器
│   ├── html_report_builder.py    # 预测/回测 HTML 报告
│   └── llm_report_generator.py   # LLM 分析报告
│
├── rules/
│   └── hsci.md                   # HSCI 编算规则 (YAML + Markdown)
│
├── static/
│   └── index.html                # 前端单页（浅色主题，华泰蓝品牌色）
│
├── output/                       # 报告输出
│   ├── mode_a/
│   └── mode_b/
│
└── 数据文件/
    ├── 港股通标的池.xls
    ├── 港股通标的证券调整信息.xls
    ├── hsci_classification.csv
    ├── history_hsci(HSCI).csv
    ├── 恒生综合指数编算细则.pdf
    ├── 恒生指数系列编算总则.pdf
    └── 上海证券交易所沪港通业务实施办法.docx
```

## 配置

### 规则参数

HSCI 编算规则集中在 `rules/hsci.md`，采用 YAML frontmatter + Markdown 结构，包含选股范畴、流动性要求、成分股挑选方法、缓冲区规则、子指数分级矩阵、快速纳入机制等。覆盖率模型启动时自动加载此文件，修改参数后无需改代码。

Mode A 的阈值常量（市值门槛、WVR 条件、风险等级划分等）定义在 `mode_a/config.py`，来源于《实施办法》的具体法条。

### LLM 配置

`.env` 文件管理以下配置：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API 密钥 | — |
| `LLM_BASE_URL` | API 地址 | 通义千问 |
| `LLM_MODEL` | 模型名称 | qwen-plus |
| `LLM_TEMPERATURE` | 采样温度 | 0.3 |
| `LLM_MAX_TOKENS` | 最大 token 数 | 8192 |
| `LLM_TIMEOUT` | 超时秒数 | 120 |

切换到 OpenAI 只需修改 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 三行。

## 输出

### Mode A 输出

- `output/mode_a/report.html` — HTML 扫描报告，按风险等级分组展示
- `mode_a/state/data/scan_{date}.json` — 扫描状态存档，供变更检测使用

### Mode B 输出

- `output/mode_b/prediction_{月份}.json` — 预测结果（HSCI 调入调出 + 港股通映射）
- `output/mode_b/prediction_{月份}.html` — HTML 可视化报告（颜色编码）
- `output/mode_b/prediction_llm_report_{月份}.md` / `.html` — LLM 分析报告
- `output/mode_b/backtest_{月份}.json` — 回测结果（双层准确率 + 偏差分析）
- `output/mode_b/backtest_{月份}.html` — HTML 回测报告

### 数据缓存

`mode_b/data/` 下的缓存文件按 cutoff 日期命名，存在则直接加载跳过 API 采集：

- `turnover_test_{YYYYMMDD}.json` — 换手率测试结果
- `month_end_mc_{YYYYMMDD}.json` — 月末市值映射
- `universe_market_cap_{YYYYMMDD}.json` — 选股范畴市值数据

## Web API

`api_main.py` 提供以下端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat/` | 同步对话，返回完整结果 |
| POST | `/api/chat/stream/` | SSE 流式对话，实时推送进度 |
| GET | `/health` | 健康检查 |
| GET | `/` | 前端首页 |
| Mount | `/static` | 静态资源 |
| Mount | `/output` | 报告文件服务 |

SSE 流式端点推送三种事件：`progress`（处理进度）、`done`（最终结果）、`error`（错误）。前端固定使用流式端点以获得实时进度反馈。

## 已知限制

- **历史子指数分类缺失**：倒推重建的历史成分股不保留子指数归属，缓冲区矩阵对未知分类的股票统一使用非成分股门槛（77%/93%），可能导致边界股票的子指数分级偏保守
- **数据源稳定性**：东方财富 push2 API 在部分网络环境不可用，已改用新浪和百度作为替代
- **单用户上下文**：Web 模式使用全局单例 ChatHandler，多浏览器窗口共享上下文，适合单用户本地使用
- **幽灵 Add 问题**：少数股票有 HSCI Add 记录但不在当前成分股列表且无 Delete 记录，导致重建数量偏多，系统通过 `count_after` 校验并记录警告
