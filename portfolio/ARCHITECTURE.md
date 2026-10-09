# AI 市场风险管理平台 · 技术架构白皮书

> 本项目为在华泰证券 AI 市场风险管理岗实习期间，独立负责三个 AI 应用系统的 0→1 架构设计与全栈开发，并构建统一的工作台门户。本文档面向技术评审，描述系统真实架构、AI 流水线、关键技术决策与验证指标。

---

## 0. 总览

平台由 **三个业务系统** 与 **一个统一门户** 组成，覆盖市场风险管理的三类典型场景：

| 系统 | 业务场景 | 核心 AI 能力 | 交互形态 |
|---|---|---|---|
| A股指数成份股调整预测 | 预判指数定期调样，辅助提前布局 | 量化规则 + LLM 增强双模式预测 | CLI / FastAPI / 原生 Web |
| 港股通标的池监控 | 监控港股通标的可能调出风险 | 两层模型（HSCI 覆盖 + 港股通映射） | FastAPI / 原生 Web |
| 播客舆情分析 | 将财经播客语音转为对 A 股的可执行洞察 | LangGraph 多智能体 + ASR + RAG + 跨市场传导 | Streamlit |

三者共享同一套工程范式：**配置 / 数据驱动的智能体编排 + LLM 增强 + 回测验证闭环**，并通过统一门户提供一致的产品外壳与状态感知。

整体架构见图 [`architecture_diagram.svg`](./architecture_diagram.svg)。

---

## 1. 统一门户（Portal）

**定位**：面向终端用户（风险经理 / 研究员）的工作台外壳，将三个异构系统聚合为"一个产品"。

**设计要点**
- **工作台布局**：左侧常驻导航（概览 + 三个模块）+ 顶部标题栏 + 主工作区，模块以 iframe 内嵌，进入系统不丢失导航上下文。
- **状态感知**：加载时对每个系统地址做轻量 `no-cors` 探活，在侧栏、概览卡片、聚合状态条三处同步显示 ●运行中 / ○未启动，并提供手动「刷新状态」。
- **设计系统**：统一的品牌色板、Inter 字体、内联 SVG 线性图标、克制 hover；开发向内容（启动命令）折叠，对非技术用户只暴露"模块状态 + 打开"。
- **零逻辑侵入**：门户为纯静态 HTML，不改动任何业务系统代码，仅通过可配置的地址（默认 `localhost:8000/8001/8501`）嵌入。

**价值**：把"三个各做各的网页"收敛为"一个平台"，降低跨系统使用成本，也统一了视觉与交互语言。

---

## 2. 系统一：A股指数成份股调整预测系统

### 2.1 业务价值
指数定期调样会影响数百亿被动资金的流向。系统提前预测可能被 **调入 / 调出** 的成份股，为风险管理与合规监测提供前瞻信号。

### 2.2 架构：配置驱动的 Multi-Agent 编排
核心设计是 **"调度引擎与业务解耦"**——新增一种预测能力，只需实现一个 Agent 并在 `skills/skill.md` 的 YAML 工作流里加一步，编排引擎零改动。

- `skills/skill.md`：YAML frontmatter 定义 `predict_adjustment` / `backtest` 两条工作流；
- `core/skill_loader.py`：解析工作流定义；
- `core/orchestrator.py`：通用执行引擎，按工作流串联 Agent；
- `core/agent_registry.py`：逻辑名 → 工厂映射，注册 **6 个 Agent**：Data / Rule / Predict / Backtest / Report / LLMReport；
- `rules/csi300.md`：以 YAML frontmatter 承载量化参数（样本 300、缓冲区 rank 240/360、cutoff 日期），规则与代码分离。

### 2.3 双预测模式（可解释性 + 定性增强）
由 `use_llm` 切换：
- **纯量化模式**：基于缓冲区的排名规则，结果完全可解释、可追溯；
- **量化 + LLM 增强模式**：在规则输出基础上，用 LLM 补充流动性、政策、事件等定性因素，给出更稳健的判断。

### 2.4 多入口与意图识别
- 三入口：CLI 交互对话 / FastAPI（`/api/predict`、`/api/backtest`、`/api/chat`）/ 原生 HTML 单页（`static/index.html`）；
- `interface/chat_handler.py`：正则快速通道 + LLM 兜底的两级意图识别，支持 4 类意图与多轮上下文继承。

### 2.5 数据与模型
- 数据：Tushare 主源 + AKShare 备用 + SQLite 缓存；
- LLM：阿里云百炼 qwen（OpenAI 兼容接口）。

### 2.6 报告与验证
- 报告 3 格式（md / json / html）+ LLM 叙述版；HTML 含置信度着色的股票卡片；
- **回测验证**（reports/）：
  - 2025-06 综合 **F1 = 71.43%**（LLM 与纯量化一致）；
  - 2026-06 综合 **F1 = 70.73%**（调入 48.78% / 调出 92.68%；调入命中 10/19、调出 19/19）；
  - 预测名单（2026-06）：调入 22 / 备选 5 / 调出 22，置信度 0.886。

---

## 3. 系统二：港股通标的池监控系统

### 3.1 业务价值
港股通标的被调出会带来流动性与合规风险。系统持续监控临近调出门槛的股票，并支持对历史调样做预测与回测。

### 3.2 双模式
- **Mode A 持续扫描**（`main_a.py`）：识别接近调出门槛的风险标的；
- **Mode B 预测 / 回测**（`main_b.py`，子命令 `predict` / `backtest`）。

### 3.3 两层模型（`mode_b/core/orchestrator_b.py`）
- **Layer1 HSCI 覆盖率**：判断标的是否仍满足恒生综合大型/中型股指数资格；
- **Layer2 港股通映射**：在 Layer1 基础上套用港股通纳入 / 剔除映射规则。

### 3.4 工程要点
- FastAPI（`api_main.py`，`/api/chat/`、`/api/chat/stream/`），支持 SSE 流式输出；
- 两级意图识别（`interface/chat_handler.py` 正则 + LLM 兜底）；
- **前端为原生 HTML 单页**（非 Streamlit），与门户设计语言统一；
- 数据源：akshare + 上交所 XLS + 恒生 CSV/PDF；**缓存采用 JSON 断点续采**（非 SQLite），支持长任务中断恢复；
- LLM：阿里百炼 qwen3.7-plus。

### 3.5 验证指标
- 回测（`output/mode_b/backtest_2026-03.json`，单期）：
  - Layer1 **P=77.0% / R=81.7% / F1=79.3%**；
  - Layer2 **P=68.3% / R=73.7% / F1=70.9%**；
- HSCI 成分 534 只；扫描（2026-08-06）评估 803 只 / 池 617 只（高 / 中 / 低风险 2 / 5 / 17）。

---

## 4. 系统三：播客舆情分析系统

### 4.1 业务价值
财经播客蕴含大量非结构化市场观点。系统将播客语音转为文本，提取对 A 股板块的情绪与事件，并传导到具体概念板块，为研究员提供"语音 → 可执行洞察"的分析链路。

### 4.2 编排：LangGraph StateGraph
- `orchestration/graph.py` 使用 `langgraph.graph.StateGraph` 定义编排图，含条件路由；
- 串联 **7 个 Agent**：collector（采集）/ transcriber（转写）/ retriever（检索）/ sentiment（情感）/ analyst（分析）/ crossmarket（跨市场传导）/ reporter（报告）；
- 业务 Agent 的 LLM 调用通过 OpenAI 兼容 SDK 自实现，编排与执行解耦。

### 4.3 关键流水线
- **ASR**：阿里百炼 `qwen-audio-3.0-asr-flash-filetrans`（DashScope filetrans 协议：上传 → 提交任务 → 轮询），转录文本存于内存 state（不落 SQLite，保证大音频不撑爆库）；
- **RAG**（`ENABLE_RAG` 默认 `false` 以省配额）：chunk → embedding → ChromaDB → 语义检索 + BM25（RRF 融合）→ qwen3-rerank 精排；
- **Map-Reduce**：`graph.py` 将长文本按 4000/500 重叠分块，并发跑 Sentiment / Analyst 后聚合，解决长转录上下文溢出；
- **跨市场传导（crossmarket）**：retrieve-then-generate —— LLM 提取中文主题词 → 对东财 **504 个概念板块**（`data/concept_boards.json`）做子串检索取 top-5 → LLM 选定标准板块名，完成"观点 → 板块"的映射。

### 4.4 前端与存储
- Streamlit 前端（`web/app.py`，自然语言 / 采集 / 上传三页签）；
- SQLite 4 表（podcasts / episodes / transcripts / analysis_results）；实测 29 podcasts / 30 episodes；
- 报告为固定 6 小节 Markdown + ECharts HTML；CLI 用 Typer（无 FastAPI）。

### 4.5 情感分析口径（重要校正）
当前代码为 **LLM 主导、FinBERT 仅作兜底**，并非早期课题报告中的"0.7 LLM + 0.3 FinBERT 融合"。白皮书以代码现状为准。

---

## 5. 技术亮点与我的贡献

1. **三个系统的 0→1 架构设计**：覆盖 Multi-Agent 编排、LangGraph 状态图、RAG、ASR 等主流 AI 工程范式；
2. **配置 / 数据驱动的编排**：`skill.md` 工作流 + Agent 注册表，使能力扩展无需改动调度引擎；
3. **可解释性与 LLM 增强的平衡**：量化规则保证可追溯，LLM 补充定性因素；
4. **回测验证闭环**：用 F1 / P / R 量化系统有效性，而非仅靠 demo 演示；
5. **工程鲁棒性**：流式输出（SSE）、意图识别（正则 + LLM 兜底）、长任务断点续采、多环境隔离（conda）；
6. **统一产品化**：将三个异构系统收敛为一致的工作台门户，补齐状态感知与设计系统。

---

## 6. 技术栈总表

| 层 | CSI300 | 港股通 | 播客 |
|---|---|---|---|
| 编排 | 自研 orchestrator + skill.md | 自研 orchestrator_b | LangGraph StateGraph |
| 接口 | CLI / FastAPI / 原生 HTML | FastAPI / 原生 HTML | Streamlit / Typer |
| LLM | 阿里云百炼 qwen | 阿里云百炼 qwen3.7-plus | 阿里云百炼 qwen |
| ASR | — | — | DashScope qwen-audio-3.0-asr-flash |
| 检索 | — | — | ChromaDB + BM25 + qwen3-rerank |
| 数据 | Tushare / AKShare / SQLite | akshare / 上交所 / 恒生 / JSON | SQLite / 东财概念板块 |
| 报告 | md / json / html | HTML | Markdown + ECharts HTML |

---

## 7. 可改进方向（未来工作）

- **部署形态**：当前为本地单机工具，可迁移为自托管 Web 应用（反向代理同源 + 简单鉴权），实现"给网址即用"；
- **RAG 常态化**：在配额可控前提下默认开启检索增强，提升跨市场传导的事实依据；
- **统一前端框架**：三个前端目前分别为原生 HTML / 原生 HTML / Streamlit，可统一技术栈以降低维护成本；
- **多租户与配额**：若对外开放，需补充账号体系与 API 配额隔离。

---

## 附录：回测指标汇总

| 系统 | 期次 | 指标 | 数值 |
|---|---|---|---|
| CSI300 | 2025-06 | 综合 F1 | 71.43% |
| CSI300 | 2026-06 | 综合 F1 | 70.73% |
| CSI300 | 2026-06 | 调出召回 | 92.68%（19/19） |
| 港股通 | 2026-03 | Layer1 F1 | 79.3% |
| 港股通 | 2026-03 | Layer2 F1 | 70.9% |
