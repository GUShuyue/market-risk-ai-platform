"""
播客舆情 Multi-Agent 系统 — 主入口

用法:
  # 运行完整管道 (Demo模式)
  python -m podcast_agent.main

  # 使用真实API
  python -m podcast_agent.main --use-api

  # 查看数据库状态
  python -m podcast_agent.main --status
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from .config import get_config

app = typer.Typer(
    name="podcast-agent",
    help="播客舆情 Multi-Agent 分析系统",
    add_completion=False,
)

console = Console()


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@app.command()
def run(
    query: str = typer.Option("", "--query", "-q", help="指定播客名称（空则采集 TARGET_PODCASTS 目标列表）"),
    max_results: int = typer.Option(10, "--max", "-m", help="采集总集数上限"),
    skip_transcription: bool = typer.Option(
        False, "--skip-transcription/--no-skip-transcription",
        help="跳过转录阶段"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """
    运行完整的多Agent管道
    """
    setup_logging(verbose)

    from .orchestration.graph import run_pipeline

    # 显示启动信息
    config = get_config()

    rprint(Panel.fit(
        "[bold blue]🎯 Podcast Agent Orchestrator[/bold blue]\n\n"
        f"  GPU: [{'red' if not config.gpu_available else 'green'}]"
        f"{'不可用 (CPU模式)' if not config.gpu_available else '可用'}[/]\n"
        f"  Listen Notes API: [{'yellow' if not config.listennotes_available else 'green'}]"
        f"{'未配置 (Demo模式)' if not config.listennotes_available else '已配置'}[/]\n"
        f"  LLM API: [{'yellow' if not config.llm_available else 'green'}]"
        f"{'未配置' if not config.llm_available else '已配置'}[/] ({config.llm_base_url})\n"
        f"  LLM 模型: {config.llm_model}\n"
        f"  Embedding: {config.embedding_provider} ({config.embedding_model})\n"
        f"  跳过转录: {'是' if skip_transcription else '否'}\n",
        title="系统状态",
        border_style="blue",
    ))

    # 运行管道
    rprint("\n[bold]开始执行管道...[/bold]\n")

    final_state = run_pipeline(
        search_query=query,
        max_results=max_results,
        skip_transcription=skip_transcription,
    )

    # 显示结果
    _display_results(final_state)


@app.command()
def run_local(
    file_path: str = typer.Option(
        "", "--file", "-f", help="本地 MP3 文件路径（默认: data/input_podcast/ 下第一个文件）"
    ),
    all_files: bool = typer.Option(
        False, "--all", "-a", help="分析 data/input_podcast/ 下全部 MP3，汇总为一份综合报告"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """
    从本地 MP3 文件直接运行分析管道（绕过 Listen Notes API）

    工作流: 本地文件 → 转录 → 分析 → 报告

    --all: 目录下全部文件独立分析，合并为一份综合报告
    不指定 --file / --all 时，自动使用 data/input_podcast/ 下第一个 mp3 文件
    """
    setup_logging(verbose)

    from .orchestration.graph import run_local_file_pipeline, run_local_files_pipeline

    config = get_config()

    if all_files:
        input_dir = config.data_dir / "input_podcast"
        input_dir.mkdir(parents=True, exist_ok=True)
        mp3_files = sorted(input_dir.glob("*.mp3"))
        if not mp3_files:
            rprint("[red]错误: data/input_podcast/ 下没有 MP3 文件[/red]")
            raise typer.Exit(1)

        rprint(Panel.fit(
            "[bold blue]🎯 Podcast Agent — 本地多文件模式[/bold blue]\n\n"
            f"  文件数: {len(mp3_files)}\n"
            + "".join(f"  • {f.name}\n" for f in mp3_files)
            + f"  LLM API: [{'yellow' if not config.llm_available else 'green'}]"
            f"{'未配置' if not config.llm_available else '已配置'}[/] ({config.llm_base_url})\n"
            f"  LLM 模型: {config.llm_model}\n"
            f"  模式: 逐文件转录 → 独立分析 → 综合报告\n",
            title="系统状态",
            border_style="blue",
        ))

        rprint("\n[bold]开始执行管道...[/bold]\n")

        final_state = run_local_files_pipeline([str(f) for f in mp3_files])
        _display_results(final_state)
        return

    # 自动检测文件
    if not file_path:
        input_dir = config.data_dir / "input_podcast"
        input_dir.mkdir(parents=True, exist_ok=True)
        mp3_files = sorted(input_dir.glob("*.mp3"))
        if not mp3_files:
            rprint("[red]错误: data/input_podcast/ 下没有 MP3 文件[/red]")
            rprint("[yellow]请将 MP3 文件放入该目录，或使用 --file 指定路径[/yellow]")
            raise typer.Exit(1)
        file_path = str(mp3_files[0])
        if len(mp3_files) > 1:
            rprint(f"[yellow]找到 {len(mp3_files)} 个 MP3 文件，使用第一个: {mp3_files[0].name}[/yellow]")

    if not Path(file_path).exists():
        rprint(f"[red]错误: 文件不存在 — {file_path}[/red]")
        raise typer.Exit(1)

    if not file_path.lower().endswith(".mp3"):
        rprint("[yellow]警告: 文件扩展名不是 .mp3，可能无法正常转录[/yellow]")

    rprint(Panel.fit(
        "[bold blue]🎯 Podcast Agent — 本地文件模式[/bold blue]\n\n"
        f"  文件: {Path(file_path).name}\n"
        f"  LLM API: [{'yellow' if not config.llm_available else 'green'}]"
        f"{'未配置' if not config.llm_available else '已配置'}[/] ({config.llm_base_url})\n"
        f"  LLM 模型: {config.llm_model}\n"
        f"  模式: 单文件 → 转录 → 分析 → 报告\n",
        title="系统状态",
        border_style="blue",
    ))

    rprint("\n[bold]开始执行管道...[/bold]\n")

    final_state = run_local_file_pipeline(file_path)

    _display_results(final_state)


@app.command()
def status():
    """查看数据库和系统状态"""
    config = get_config()

    # 系统信息
    rprint(Panel.fit(
        f"  GPU 可用: [{'red' if not config.gpu_available else 'green'}]{config.gpu_available}[/]\n"
        f"  设备: {config.device}\n"
        f"  ASR 转录: {config.asr_model}\n"
        f"  Embedding: {config.embedding_provider} ({config.embedding_model})\n"
        f"  Rerank: {config.rerank_model}\n"
        f"  Listen Notes API: [{'yellow' if not config.listennotes_available else 'green'}]"
        f"{'未配置' if not config.listennotes_available else '已配置'}[/]\n"
        f"  LLM API: [{'yellow' if not config.llm_available else 'green'}]"
        f"{'未配置' if not config.llm_available else '已配置'}[/] ({config.llm_base_url})\n"
        f"  LLM 模型: {config.llm_model}\n"
        f"  数据库: {config.sqlite_path}\n",
        title="系统状态",
        border_style="blue",
    ))

    # 数据库状态
    try:
        from .utils.db import get_session
        from .models.database import Podcast, Episode, Transcript, AnalysisResult

        with get_session() as session:
            podcast_count = session.query(Podcast).count()
            episode_count = session.query(Episode).count()
            transcript_count = session.query(Transcript).count()
            analysis_count = session.query(AnalysisResult).count()

            table = Table(title="数据库统计")
            table.add_column("表", style="cyan")
            table.add_column("记录数", style="green", justify="right")

            table.add_row("Podcasts (播客)", str(podcast_count))
            table.add_row("Episodes (节目)", str(episode_count))
            table.add_row("Transcripts (转录)", str(transcript_count))
            table.add_row("Analysis Results (分析)", str(analysis_count))

            console.print(table)

            # 显示最近的节目
            if episode_count > 0:
                episodes = session.query(Episode).order_by(
                    Episode.created_at.desc()
                ).limit(5).all()

                ep_table = Table(title="最近采集的节目")
                ep_table.add_column("标题", style="white")
                ep_table.add_column("播客", style="cyan")
                ep_table.add_column("已转录", justify="center")
                ep_table.add_column("已分析", justify="center")

                for ep in episodes:
                    ep_table.add_row(
                        ep.title[:60] + ("..." if len(ep.title) > 60 else ""),
                        ep.podcast.title if ep.podcast else "-",
                        "✅" if ep.is_transcribed else "❌",
                        "✅" if ep.is_analyzed else "❌",
                    )

                console.print(ep_table)

    except Exception as e:
        rprint(f"[red]数据库查询失败: {e}[/red]")


@app.command()
def search(
    query: str = typer.Option("", "--query", "-q", help="指定播客名称（空则采集 TARGET_PODCASTS 目标列表）"),
    max_results: int = typer.Option(10, "--max", "-m", help="采集总集数上限"),
):
    """仅运行 Collector Agent 采集指定播客的最新节目"""
    setup_logging(True)

    from .agents.collector import CollectorAgent

    agent = CollectorAgent()
    result = agent.execute({
        "search_query": query,
        "max_results": max_results,
    })

    table = Table(title=f"搜索结果 ({result.filtered_count}/{result.total_found})")
    table.add_column("播客", style="cyan")
    table.add_column("节目标题", style="white")
    table.add_column("时长", justify="right")

    for ep in result.episodes:
        mins = ep.audio_length_sec // 60
        table.add_row(
            ep.podcast_title[:30],
            ep.episode_title[:60],
            f"{mins}min",
        )

    console.print(table)


def _display_results(state: dict) -> None:
    """显示管道执行结果"""
    collector = state.get("collector_output")

    rprint("\n" + "=" * 60)

    if collector:
        table = Table(title="管道执行结果")
        table.add_column("阶段", style="cyan")
        table.add_column("状态", style="green")
        table.add_column("详情", style="white")

        # Collector
        status_icon = "✅" if collector.agent_status.value == "success" else "❌"
        table.add_row(
            "Collector",
            status_icon,
            f"找到 {collector.total_found} 集, 过滤后 {collector.filtered_count} 集"
        )

        # Transcriber
        transcriber = state.get("transcriber_output")
        if transcriber:
            s = "✅" if transcriber.agent_status.value == "success" else "❌"
            et_count = len(transcriber.episode_transcriptions)
            detail = f"转录 {et_count} 集" if et_count else (transcriber.error_message or "-")
            table.add_row("Transcriber", s, detail[:80])

        # Per-Episode Analysis
        episode_analyses = state.get("episode_analyses", [])
        if episode_analyses:
            s = "✅"
            sentiments = []
            for ea in episode_analyses:
                if ea.sentiment_output and ea.sentiment_output.agent_status.value == "success":
                    sentiments.append(ea.sentiment_output.overall_sentiment.value)
            dist = ", ".join(f"{k}={v}" for k, v in __import__("collections").Counter(sentiments).most_common())
            detail = f"分析 {len(episode_analyses)} 集 | 情感: {dist}"
            table.add_row("Analyze (多集)", s, detail[:80])

        # 后续阶段（兼容旧版单集）
        stages = [
            ("Retriever", "retriever_output"),
            ("Sentiment", "sentiment_output"),
            ("Analyst", "analyst_output"),
            ("CrossMarket", "crossmarket_output"),
        ]
        for stage_name, key in stages:
            output = state.get(key)
            if output:
                s = "✅" if output.agent_status.value == "success" else (
                    "⏭️" if output.agent_status.value == "skipped" else "❌"
                )
                detail = output.error_message or "-"
                table.add_row(stage_name, s, detail[:80])

        # Reporter
        reporter = state.get("reporter_output")
        if reporter:
            s = "✅" if reporter.agent_status.value == "success" else "❌"
            detail = reporter.error_message or "-"
            if hasattr(reporter, 'report_html_path') and reporter.report_html_path:
                detail = f"报告: {reporter.report_html_path}"
            table.add_row("Reporter", s, detail[:80])

        console.print(table)

    # 显示采集到的节目
    if collector and collector.episodes:
        rprint("\n[bold]采集到的节目:[/bold]")
        for i, ep in enumerate(collector.episodes[:5], 1):
            rprint("  {}. [cyan]{}[/cyan] - {}".format(i, ep.podcast_title, ep.episode_title[:80]))

    # 显示多集分析摘要
    episode_analyses = state.get("episode_analyses", [])
    if episode_analyses:
        rprint("\n[bold]各集分析摘要:[/bold]")
        for i, ea in enumerate(episode_analyses, 1):
            sent = ea.sentiment_output
            cm = ea.crossmarket_output
            if sent and sent.agent_status.value == "success":
                score_str = f"{sent.sentiment_score:+.2f}"
                color = "green" if sent.sentiment_score > 0.2 else (
                    "red" if sent.sentiment_score < -0.2 else "yellow"
                )
                sector_count = len(cm.a_share_impacts) if (cm and cm.agent_status.value == "success") else 0
                rprint("  {}. [cyan]{}[/cyan] → 情感: [{}]{}[/{}], 板块: {} 个".format(
                    i, ea.episode_title[:60], color, score_str, color, sector_count
                ))


if __name__ == "__main__":
    app()