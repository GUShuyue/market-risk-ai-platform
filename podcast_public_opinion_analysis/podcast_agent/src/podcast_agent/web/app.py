"""
Streamlit 前端 — 播客舆情分析系统

两种交互方式:
  1. 自然语言: 用 LLM 解析指令 → 采集播客分析 / 上传文件分析
  2. 表单兜底: 手动填播客名+集数 / 上传本地 mp3

启动:
  streamlit run src/podcast_agent/web/app.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

# 确保 src 目录在 sys.path，脚本独立运行也能导入 podcast_agent 包
_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from podcast_agent.config import get_config  # noqa: E402
from podcast_agent.orchestration.graph import (  # noqa: E402
    run_local_files_pipeline,
    run_pipeline,
)
from podcast_agent.web.parser import parse_intent  # noqa: E402

st.set_page_config(page_title="播客舆情分析系统", page_icon="🎙️", layout="wide")


# ================================================================
# 全局样式 — 与门户 / 其它子系统统一的设计语言
# ================================================================

_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* 隐藏 Streamlit 默认外壳，营造产品感 */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="stHeader"] { display: none; }

    html, body, [class], [data-testid] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important; }

    /* 全局背景 */
    .stApp { background-color: #f5f7fb; }

    /* 主内容容器留白 */
    .main .block-container { padding-top: 1.4rem; padding-left: 2.2rem; padding-right: 2.2rem; max-width: 1180px; }

    /* 状态圆点 */
    .pa-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }

    /* 顶部细品牌条 */
    .pa-topbar { height: 3px; background: linear-gradient(90deg, #1f5fbf, #1a4f9e 60%, #6f9bd8); border-radius: 999px 999px 0 0; }

    /* 侧栏 — 浅色 + 细边框 */
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e7ebf2; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.1rem; padding-left: 1.1rem; padding-right: 1.1rem; }
    [data-testid="stSidebar"] h4 { color: #1a4f9e; font-size: 13px !important; font-weight: 700 !important; letter-spacing: .02em; }
    [data-testid="stSidebar"] .stMarkdown p { color: #586478; font-size: 13px; }

    /* 品牌头部 */
    .pa-header {
        display: flex; align-items: center; gap: 13px;
        padding: 4px 2px 16px; margin-bottom: 6px;
        border-bottom: 1px solid #e7ebf2;
    }
    .pa-logo {
        width: 42px; height: 42px; border-radius: 12px; flex: 0 0 42px;
        background: linear-gradient(135deg, #1f5fbf, #1a4f9e);
        display: grid; place-items: center;
        box-shadow: 0 8px 18px -6px rgba(31,95,191,.55);
    }
    .pa-logo svg { width: 22px; height: 22px; }
    .pa-title { font-size: 19px; font-weight: 700; color: #18212f; line-height: 1.2; }
    .pa-sub { font-size: 12.5px; color: #8a94a6; margin-top: 2px; }

    /* 主按钮（华泰蓝） */
    .stButton > button[data-baseweb="button"] {
        background-color: #1f5fbf; border-color: #1f5fbf; color: #ffffff;
        border-radius: 10px; font-weight: 600; padding: 0.5rem 1.1rem;
        transition: background-color .15s ease, box-shadow .15s ease;
        box-shadow: 0 6px 16px -6px rgba(31,95,191,.5);
    }
    .stButton > button[data-baseweb="button"]:hover { background-color: #1a4f9e; border-color: #1a4f9e; }
    .stButton > button[kind="secondary"] { color: #1a4f9e; border-color: #d6dde6; background: #fff; box-shadow: none; }
    .stButton > button[kind="secondary"]:hover { border-color: #1f5fbf; color: #1a4f9e; }

    /* 输入框焦点描边 */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 11px; border-color: #d9e0ea;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #1f5fbf; box-shadow: 0 0 0 3px rgba(31,95,191,.12);
    }

    /* Tab 样式（品牌色选中态） */
    [data-baseweb="tab-list"] { gap: 4px; }
    [data-baseweb="tab"] { font-weight: 600; border-radius: 9px 9px 0 0; padding: 0.5rem 1.1rem; }
    [data-baseweb="tab"][aria-selected="true"] { color: #1a4f9e; background: #eaf1fb; }
    [data-baseweb="tab-highlight"] { background-color: #1f5fbf !important; }

    /* 状态容器圆角 */
    [data-testid="stStatus"] { border-radius: 12px; }
    [data-testid="stStatus"] > div { font-weight: 600; }

    /* 分隔线更柔和 */
    hr { border-color: #e7ebf2 !important; }

    /* 报告预览容器留白 */
    [data-testid="stHorizontalBlock"] iframe { border-radius: 12px; border: 1px solid #e7ebf2; }

    /* 示例指令 chips */
    .pa-examples { display: flex; gap: 9px; flex-wrap: wrap; margin: 10px 0 4px; }
    div[data-testid="stButton"] > button { box-shadow: none; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ================================================================
# 工具函数
# ================================================================

def _status_line(label: str, text: str, ok: bool) -> str:
    """带彩色状态圆点的行（纯 CSS 圆点，非 emoji）"""
    color = "#15a06a" if ok else "#e8920c"
    return (
        f'<span class="pa-dot" style="background:{color}"></span>'
        f"{label}: <strong>{text}</strong>"
    )


def _show_result(state: dict, status) -> None:
    """展示 pipeline 运行结果（报告下载 + 内嵌预览）"""
    errors = state.get("errors", [])
    if errors:
        st.warning("部分阶段出错: " + "；".join(str(e) for e in errors))

    reporter = state.get("reporter_output")
    report_path = getattr(reporter, "report_html_path", None) if reporter else None

    if report_path and Path(report_path).exists():
        status.update(label="分析完成", state="complete")
        st.success(f"报告已生成: {report_path}")

        html_content = Path(report_path).read_text(encoding="utf-8")
        st.download_button(
            "下载报告 (HTML)",
            data=html_content,
            file_name=Path(report_path).name,
            mime="text/html",
        )
        st.markdown("---")
        st.markdown("**报告预览**")
        try:
            st.components.v1.html(html_content, height=820, scrolling=True)
        except Exception as e:  # noqa: BLE001
            st.info(f"内嵌预览失败（可下载后用浏览器打开）: {e}")
    else:
        status.update(label="完成（未生成报告）", state="error")
        st.warning("未生成报告，请查看上方错误信息")


def _run_collect(names: list, episodes: int) -> None:
    """采集指定播客最新集并分析"""
    max_results = max(len(names) * episodes, 1)
    with st.status(f"正在采集 {len(names)} 个播客并分析…", expanded=True) as status:
        st.write(f"播客: {', '.join(names)}")
        st.write(f"每播客 {episodes} 集")
        try:
            state = run_pipeline(
                podcast_names=names,
                episodes_per_podcast=episodes,
                max_results=max_results,
            )
            _show_result(state, status)
        except Exception as e:  # noqa: BLE001
            status.update(label="运行失败", state="error")
            st.error(f"运行失败: {e}")


def _run_upload(files) -> None:
    """上传本地 mp3 并分析"""
    tmp_dir = tempfile.mkdtemp(prefix="pa_upload_")
    paths = []
    for f in files:
        p = os.path.join(tmp_dir, f.name)
        with open(p, "wb") as fh:
            fh.write(f.getbuffer())
        paths.append(p)

    with st.status(f"正在转录并分析 {len(paths)} 个文件…", expanded=True) as status:
        st.write("文件: " + "、".join(os.path.basename(p) for p in paths))
        try:
            state = run_local_files_pipeline(paths)
            _show_result(state, status)
        except Exception as e:  # noqa: BLE001
            status.update(label="运行失败", state="error")
            st.error(f"运行失败: {e}")


# ================================================================
# 页面
# ================================================================

config = get_config()

# 品牌头部（替代默认 st.title）
st.markdown(
    """
    <div class="pa-topbar"></div>
    <div class="pa-header">
        <div class="pa-logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"
                 stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="2" width="6" height="12" rx="3"/>
                <path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="17" x2="12" y2="22"/>
            </svg>
        </div>
        <div>
            <div class="pa-title">播客舆情分析系统</div>
            <div class="pa-sub">采集播客节目或上传音频，生成 A 股舆情分析报告</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 系统状态
with st.sidebar:
    st.markdown("#### 系统状态")
    st.markdown(
        _status_line("Listen Notes API", "已配置" if config.listennotes_available else "未配置", config.listennotes_available),
        unsafe_allow_html=True,
    )
    st.markdown(
        _status_line("LLM API", config.llm_model, config.llm_available),
        unsafe_allow_html=True,
    )
    st.markdown(
        _status_line("RAG", "启用" if config.enable_rag else "关闭", config.enable_rag),
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**默认目标播客**")
    for p in config.target_podcasts:
        st.markdown(f"- {p}")


def process_nl(text: str) -> None:
    """解析自然语言指令并执行（自然语言页与示例指令共用）"""
    intent = parse_intent(text)
    if intent["action"] == "collect":
        names = intent["podcast_names"]
        eps = intent.get("episodes_per_podcast", 1)
        if not names:
            st.warning("没有识别到播客名，请在下方「采集播客」页手动填写")
        else:
            st.success(f"已识别：采集 **{len(names)}** 个播客，每播客 **{eps}** 集")
            _run_collect(names, eps)
    elif intent["action"] == "upload":
        st.info("请直接选择本地 mp3 文件，无需切换页签：")
        up = st.file_uploader(
            "选择 mp3 文件", type=["mp3"], accept_multiple_files=True, key="nl_upload"
        )
        if st.button("开始转录并分析", type="primary", key="nl_upload_run", disabled=not up):
            _run_upload(up)
    else:
        st.warning("无法识别指令，请换种说法，或使用右侧「采集播客 / 上传文件」页")


tab_nl, tab_collect, tab_upload = st.tabs(["自然语言", "采集播客", "上传文件"])

# --- Tab 1: 自然语言（主路径） ---
with tab_nl:
    st.markdown("用一句话描述需求，例如：**采集 Odd Lots 最新 3 集分析**")
    examples = ["采集 Odd Lots 最新 3 集分析", "采集 FT News Briefing 最新 2 集", "上传本地 mp3 进行舆情分析"]
    ex_cols = st.columns(3)
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            process_nl(ex)
    nl_input = st.text_input(
        "指令",
        placeholder="例如：采集 FT News Briefing 和 Odd Lots 最新 3 集分析",
    )
    if st.button("解析并执行", type="primary", disabled=not nl_input.strip()):
        process_nl(nl_input)

# --- Tab 2: 采集播客 ---
with tab_collect:
    st.markdown("手动指定播客名和每播客集数")
    default_names_text = "\n".join(config.target_podcasts)
    names_text = st.text_area(
        "播客名（每行一个，或用逗号分隔）",
        value=default_names_text,
        height=160,
    )
    episodes = st.number_input("每播客采集集数", min_value=1, max_value=20, value=1)
    if st.button("开始采集并分析", type="primary", key="collect_run"):
        names = [n.strip() for n in re.split(r"[\n,，]+", names_text) if n.strip()]
        if not names:
            st.warning("请至少填写一个播客名")
        else:
            _run_collect(names, int(episodes))

# --- Tab 3: 上传文件 ---
with tab_upload:
    st.markdown("上传本地 mp3 文件（可多选），绕过 Listen Notes API 直接转录分析")
    uploaded = st.file_uploader(
        "选择 mp3 文件",
        type=["mp3"],
        accept_multiple_files=True,
    )
    if st.button("开始转录并分析", type="primary", key="upload_run", disabled=not uploaded):
        _run_upload(uploaded)
