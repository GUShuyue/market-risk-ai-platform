"""Render _build_html with dummy data (single & multi episode) and syntax-check the inline JS via node --check."""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from podcast_agent.agents.reporter import ReporterAgent
from podcast_agent.models.messages import (
    AShareImpact, AnalystOutput, CrossMarketOutput, EpisodeAnalysis,
    ReporterInput, SentimentOutput,
)


def extract_inline_js(html: str) -> str:
    start = html.index("<script>", html.index("</style>")) + len("<script>")
    end = html.rindex("</script>")
    return html[start:end]


def check_js(name: str, js: str) -> bool:
    p = Path(tempfile.gettempdir()) / f"pa_{name}.js"
    p.write_text(js, encoding="utf-8")
    r = subprocess.run(["node", "--check", str(p)], capture_output=True, text=True)
    ok = r.returncode == 0
    print(f"[{name}] {'OK' if ok else 'FAIL'}")
    if not ok:
        print(r.stderr[:800])
    return ok


agent = ReporterAgent()

chart_data_multi = {
    "sentiment_score": -0.1, "sentiment_label": "bearish",
    "finbert_scores": {"positive": 0.1, "negative": 0.26, "neutral": 0.64},
    "sector_impacts": [
        {"name": "人工智能", "direction": "bullish", "strength": 0.8, "channel": "政策联动",
         "reasoning": "r", "mention_count": 1},
    ],
    "csi300_trend": [{"date": "2026-08-13", "csi300": 3926.96, "change_pct": -0.5}],
    "market_breadth": {"up": 667, "down": 924, "flat": 48},
    "southbound_net": 2.87, "northbound_note": "note",
    "key_factors": {"positive": ["a"], "negative": ["b"]},
    "entities": [{"name": "X", "type": "company", "sentiment": "neutral", "score": 0.0}],
    "risk_warnings": ["w1"],
    "episode_summaries": [
        {"podcast": "P1", "episode": "E1", "score": 0.3},
        {"podcast": "P2", "episode": "E2", "score": -0.4},
    ],
    "consensus": {"consensus_level": "高"},
    "episode_count": 2, "csi300": 3926.96, "csi300_change": -0.5,
}

chart_data_single = dict(chart_data_multi, episode_summaries=[], episode_count=1,
                         consensus={"consensus_level": "N/A"})

ok = True
for name, multi in (("multi", True), ("single", False)):
    inp = ReporterInput(
        podcast_title="PT", episode_title="ET", episode_id="EID",
        episode_analyses=[EpisodeAnalysis(podcast_title="P1", episode_title="E1")] if multi else [],
    )
    html = agent._build_html(inp, "# 测试\n\n正文", chart_data_multi if multi else chart_data_single, "test")
    inlined = "echarts.min.js" not in html and "Apache Software Foundation" in html
    print(f"[{name}] echarts inlined: {inlined}")
    ok = check_js(name, extract_inline_js(html)) and inlined and ok

sys.exit(0 if ok else 1)
