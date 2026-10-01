"""Figures and tables for the dashboard, built from a saved run (runs/*.json).

Kept free of Streamlit so they can be tested and reused in the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import plotly.graph_objects as go

from ..core.trace import TraceNode, TraceStore
from ..probabilistic.hmm_engine import KILL_CHAIN

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
STAGE_COLORS = {
    "act": "#d62728", "plan": "#9467bd", "failure": "#e377c2", "defense": "#8c564b", "proof": "#1f77b4",
    "kb": "#7f7f7f", "knowledge": "#bcbd22", "grounding": "#17becf", "frame": "#aec7e8", "belief": "#ff7f0e",
    "ml": "#2ca02c", "rule": "#98df8a",
}
LAYER = {"internet": 0, "dmz": 1, "server": 2, "workstation": 2, "domain_controller": 3, "database": 4}


# --------------------------------------------------------------------------- runs


def list_runs(directory: Path = RUNS_DIR) -> List[Path]:
    return sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if directory.exists() else []


def load_run(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def trace_store(run: Dict[str, Any]) -> TraceStore:
    return TraceStore.from_dict(run["trace"])


def graph_of(run: Dict[str, Any]) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in run["topology"]["nodes"]:
        g.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    g.add_edges_from(tuple(e) for e in run["topology"]["edges"])
    return g


def layout(g: nx.DiGraph) -> Dict[str, tuple]:
    """Layered left-to-right layout: internet, DMZ, servers and workstations, DC, databases."""
    columns: Dict[int, List[str]] = {}
    for n, d in g.nodes(data=True):
        columns.setdefault(LAYER.get(d.get("type", "workstation"), 2), []).append(n)
    pos = {}
    for x, nodes in columns.items():
        nodes.sort()
        for i, n in enumerate(nodes):
            pos[n] = (float(x), float(i - (len(nodes) - 1) / 2))
    return pos


# --------------------------------------------------------------------------- figures


def network_figure(run: Dict[str, Any], tick: int, show_truth: bool = True) -> go.Figure:
    """Hosts coloured by the agent's P(compromised) at `tick`; isolated hosts as squares; blocked links dashed."""
    g = graph_of(run)
    pos = layout(g)
    row = run["timeline"][max(0, min(tick, len(run["timeline"])) - 1)]
    probs = row.get("host_compromise_prob", {})
    blocked = {tuple(e) for e in row["blocked"]}
    fig = go.Figure()
    for u, v in g.edges():
        is_blocked = (u, v) in blocked
        fig.add_trace(go.Scatter(
            x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines", hoverinfo="skip", showlegend=False,
            line={"color": "#d62728" if is_blocked else "#b0b0b0", "width": 2.5 if is_blocked else 1.2,
                  "dash": "dash" if is_blocked else "solid"}))
    nodes = list(g.nodes)
    targets = set(run["topology"]["targets"])
    compromised = set(row["compromised"])
    isolated = set(row["isolated"])
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
        text=[n + (" ★" if n in targets else "") for n in nodes], textposition="top center",
        marker={
            "size": [26 if n in targets else 20 for n in nodes],
            "color": [probs.get(n, 0.0) for n in nodes], "colorscale": "YlOrRd", "cmin": 0.0, "cmax": 1.0,
            "colorbar": {"title": "P(compromised)"},
            "symbol": ["square" if n in isolated else "circle" for n in nodes],
            "line": {"color": ["#000000" if (show_truth and n in compromised) else "#666666" for n in nodes],
                     "width": [4 if (show_truth and n in compromised) else 1 for n in nodes]},
        },
        hovertext=[f"{n}<br>P(compromised)={probs.get(n, 0.0):.2f}<br>isolated={n in isolated}"
                   + (f"<br>actually compromised={n in compromised}" if show_truth else "") for n in nodes],
        hoverinfo="text", showlegend=False))
    fig.update_layout(height=420, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      title=f"Network at tick {row['tick']} (squares: isolated, dashed red: blocked"
                            + (", thick outline: actually compromised)" if show_truth else ")"),
                      xaxis={"visible": False}, yaxis={"visible": False}, plot_bgcolor="rgba(0,0,0,0)")
    return fig


def phase_figure(run: Dict[str, Any]) -> go.Figure:
    """HMM phase posteriors over time, with the true phase as a step line."""
    ticks = [r["tick"] for r in run["timeline"]]
    fig = go.Figure()
    for phase in KILL_CHAIN:
        fig.add_trace(go.Scatter(x=ticks, y=[r.get("phase_posterior", {}).get(phase, 0.0) for r in run["timeline"]],
                                 mode="lines", stackgroup="phases", name=phase))
    fig.add_trace(go.Scatter(x=ticks, y=[KILL_CHAIN.index(r["phase"]) / (len(KILL_CHAIN) - 1)
                                         for r in run["timeline"]],
                             mode="lines", line={"color": "black", "dash": "dot", "shape": "hv"},
                             name="true phase (scaled)"))
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis={"range": [0, 1]},
                      title="Kill-chain phase posterior (HMM filter)", xaxis_title="tick")
    return fig


def actions_table(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for r in run["timeline"]:
        for a in r["actions"]:
            rows.append({"tick": r["tick"], "action": a["action"], "success": a["success"], "command": a["command"],
                         "trace_id": a["trace_id"]})
    return rows


# --------------------------------------------------------------------------- explanation


def lineage(run: Dict[str, Any], trace_id: str) -> List[TraceNode]:
    """The action and every trace entry it came from, nearest first: act -> plan -> ... -> ml."""
    return trace_store(run).lineage(trace_id)


def lineage_dot(run: Dict[str, Any], trace_id: str, max_label: int = 60) -> str:
    """Graphviz DOT of an action's provenance graph (drawn bottom-up: detection at the bottom)."""
    nodes = lineage(run, trace_id)
    ids = {n.trace_id for n in nodes}
    lines = ["digraph lineage {", '  rankdir="BT";', '  node [shape=box, style="rounded,filled", fontsize=10];']
    for n in nodes:
        label = n.summary if len(n.summary) <= max_label else n.summary[:max_label - 1] + "…"
        label = label.replace('"', "'")
        lines.append(f'  "{n.trace_id}" [label="{n.stage}\\n{label}", fillcolor="{STAGE_COLORS.get(n.stage, "#ffffff")}40"];')
    for n in nodes:
        for p in n.parents:
            if p in ids:
                lines.append(f'  "{p}" -> "{n.trace_id}";')
    lines.append("}")
    return "\n".join(lines)


def explanation(run: Dict[str, Any], trace_id: str) -> List[str]:
    """The lineage as plain sentences, action first."""
    return [f"{n.stage}: {n.summary}" for n in lineage(run, trace_id)]


def run_label(path: Path, run: Optional[Dict[str, Any]] = None) -> str:
    run = run or load_run(path)
    m = run["metrics"]
    status = "breached" if m["breached"] else "no breach"
    return f"{run['scenario']} · {run['agent']} · seed {run['seed']} ({status})"
