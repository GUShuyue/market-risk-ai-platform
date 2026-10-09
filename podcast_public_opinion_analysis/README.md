# 播客舆情分析系统（Podcast Public Opinion Analysis）

> 将财经播客语音转为文本，提取对 A 股板块的情绪与事件，并传导到具体概念板块，为研究员提供「语音 → 可执行洞察」的分析链路。

---

## 1. 项目简介

财经播客蕴含大量非结构化的市场观点与情绪。本系统把播客音频自动转写为文本，经多智能体流水线完成**情感分析、要点提炼、跨市场传导**，最终生成可直接用于研究的舆情分析报告。

- **输入**：播客 RSS / 指定节目最新若干集，或本地音频文件（mp3 等）
- **输出**：结构化 Markdown 舆情报告 + ECharts HTML 可视化
- **价值**：把"听一期播客"压缩为"一张可读、可检索、可映射到板块"的洞察卡片

---

## 2. 核心能力

- **语音转写（ASR）**：长音频自动切片、异步转写，文本不落库以规避大文件撑爆数据库
- **情感与要点分析**：LLM 主导的情感判断 + 要点抽取（FinBERT 仅作兜底）
- **跨市场传导（crossmarket）**：把播客观点映射到 A 股具体概念板块
- **检索增强（RAG）**：可选开启，对转写文本做语义 + BM25 混合检索后重排
- **长文本处理**：Map-Reduce 分块并发分析后聚合，解决长转录上下文溢出

---

## 3. 系统架构：LangGraph StateGraph

`orchestration/graph.py` 使用 `langgraph.graph.StateGraph` 定义编排图，含条件路由，串联 **7 个 Agent**：

| Agent | 职责 |
|---|---|
| collector | 采集播客元数据与最新节目 |
| transcriber | 音频转写（DashScope ASR） |
| retriever | 检索增强（RAG，可选） |
| sentiment | 情感与情绪分析 |
| analyst | 要点与投资逻辑提炼 |
| crossmarket | 观点 → A股概念板块传导 |
| reporter | 报告编排与输出 |

业务 Agent 的 LLM 调用通过 OpenAI 兼容 SDK 自实现，编排与执行解耦。

---

## 4. AI 流水线

- **ASR**：阿里百炼 `qwen-audio-3.0-asr-flash-filetrans`（DashScope filetrans 协议：上传 → 提交任务 → 轮询）；转录文本存于内存 state（不落 SQLite）。
- **RAG**（`ENABLE_RAG` 默认 `false` 以省配额）：chunk → embedding → ChromaDB → 语义检索 + BM25（RRF 融合）→ `qwen3-rerank` 精排。
- **Map-Reduce**：`graph.py` 将长文本按 4000 / 500 重叠分块，并发跑 Sentiment / Analyst 后聚合。
- **跨市场传导（crossmarket）**：retrieve-then-generate —— LLM 提取中文主题词 → 对东方财富 **504 个概念板块**（`data/concept_boards.json`）做子串检索取 top-5 → LLM 选定标准板块名。

---

## 5. 技术栈

| 层 | 选型 |
|---|---|
| 编排 | LangGraph `StateGraph` |
| 接口 | Streamlit（前端）/ Typer（CLI） |
| LLM | 阿里云百炼 qwen（OpenAI 兼容接口） |
| ASR | DashScope `qwen-audio-3.0-asr-flash` |
| 检索 | ChromaDB + BM25 + `qwen3-rerank` |
| 存储 | SQLite（4 表：podcasts / episodes / transcripts / analysis_results） |

---

## 6. 快速开始

### 环境
- Python 3.11+
- conda 环境：`podcast_agent`
- 需配置大模型与 ASR 的 API 密钥（阿里云百炼 / DashScope），具体见代码配置

### 安装与启动

```bash
conda activate podcast_agent
cd podcast_public_opinion_analysis/podcast_agent
pip install -e ".[all]"          # 含 RAG / embedding / 金融数据依赖

# 启动 Web 界面（默认 http://localhost:8501）
python -m streamlit run src/podcast_agent/web/app.py --server.port 8501
```

> 注意：启动命令必须在 `podcast_agent/` 目录下执行（`src/podcast_agent/web/app.py` 为包内相对路径）。

---

## 7. 目录结构（要点）

```
podcast_public_opinion_analysis/
└── podcast_agent/
    ├── src/podcast_agent/
    │   ├── agents/          # 7 个 Agent 实现
    │   ├── orchestration/   # graph.py：LangGraph 编排
    │   ├── web/app.py       # Streamlit 前端（自然语言 / 采集 / 上传）
    │   ├── data/            # concept_boards.json（504 概念板块）
    │   └── ...
    ├── output/              # 生成的分析报告
    └── pyproject.toml
```

---

## 8. 说明

- **情感分析口径**：当前代码为 **LLM 主导、FinBERT 仅作兜底**，并非早期课题报告中的「0.7 LLM + 0.3 FinBERT 融合」。
- **RAG 默认关闭**（`ENABLE_RAG=false`）以节省 API 配额，开启后分析多一步事实检索与重排。
- **报告格式**：固定 6 小节 Markdown + ECharts HTML。
