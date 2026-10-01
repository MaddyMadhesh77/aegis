"""A.E.G.I.S. dashboard: streamlit run aegis_system/dashboard/app.py

    sidebar   run a scenario with a chosen agent, or replay a saved run from runs/
    network   hosts coloured by the agent's P(compromised) at the selected tick
    timeline  HMM kill-chain phase posteriors
    actions   every action taken; select one to see its lineage back to the detection
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_system.agents import AGENTS  # noqa: E402
from aegis_system.dashboard.visualizer import (  # noqa: E402
    RUNS_DIR, actions_table, lineage, lineage_dot, list_runs, load_run, network_figure, phase_figure, run_label,
)
from aegis_system.loop import run_episode  # noqa: E402
from aegis_system.simulation.scenarios import available_scenarios, load_scenario  # noqa: E402

st.set_page_config(page_title="A.E.G.I.S. dashboard", layout="wide")
st.title("A.E.G.I.S. — autonomous cyber defence")

# --------------------------------------------------------------------------- sidebar: choose a run

with st.sidebar:
    st.header("Scenario")
    scenario = st.selectbox("Scenario", available_scenarios(), index=0)
    agent = st.selectbox("Agent", sorted(AGENTS), index=sorted(AGENTS).index("goal_based"))
    seed = st.number_input("Seed", min_value=0, max_value=999, value=0, step=1)
    ticks = st.slider("Ticks", 10, 60, 30)
    if st.button("Run scenario", type="primary"):
        with st.spinner("Running the closed loop..."):
            result = run_episode(agent, scenario, ticks, int(seed))
            st.session_state["run_path"] = str(result.save(RUNS_DIR))
    st.header("Replay")
    runs = list_runs()
    if runs:
        labels = {run_label(p): p for p in runs}
        current = st.session_state.get("run_path")
        names = list(labels)
        default = next((i for i, p in enumerate(labels.values()) if str(p) == current), 0)
        choice = st.selectbox("Saved runs", names, index=default)
        st.session_state["run_path"] = str(labels[choice])

if "run_path" not in st.session_state:
    st.info("Run a scenario from the sidebar, or create one from the command line: "
            "`python -m aegis_system.main --scenario solarwinds`.")
    st.stop()

run = load_run(Path(st.session_state["run_path"]))
spec = load_scenario(run["scenario"], run["seed"])
st.caption(f"{spec.title} · agent **{run['agent']}** · seed {run['seed']}")

# --------------------------------------------------------------------------- PEAS measures

m = run["metrics"]
cols = st.columns(6)
cols[0].metric("Breached", "yes" if m["breached"] else "no")
cols[1].metric("Time to containment", "never" if m["time_to_containment"] is None else f"{m['time_to_containment']} ticks")
cols[2].metric("Uptime", f"{100 * m['uptime']:.1f}%")
cols[3].metric("False isolations", m["false_isolations"])
cols[4].metric("Blocked links", m["blocked_edges"])
cols[5].metric("Mean proof depth", f"{m['proof_depth']:.1f}")

# --------------------------------------------------------------------------- network and timeline

left, right = st.columns([3, 2])
with left:
    tick = st.slider("Tick", 1, len(run["timeline"]), len(run["timeline"]))
    show_truth = st.checkbox("Show ground truth (simulation only)", value=True)
    st.plotly_chart(network_figure(run, tick, show_truth), width="stretch")
with right:
    st.plotly_chart(phase_figure(run), width="stretch")

# --------------------------------------------------------------------------- actions and their lineage

st.subheader("Actions")
rows = actions_table(run)
if not rows:
    st.write("The agent took no action in this run.")
    st.stop()

table = pd.DataFrame(rows)
event = st.dataframe(table.drop(columns=["trace_id"]), width="stretch", hide_index=True,
                     on_select="rerun", selection_mode="single-row", key="actions")
selected = event.selection.rows[0] if event is not None and event.selection.rows else 0
action = rows[selected]
st.subheader(f"Why: {action['action']} at tick {action['tick']}")
if not action["trace_id"]:
    st.write("This action has no trace entry.")
else:
    st.caption("Click a row above to explain another action. Arrows point from evidence to conclusion.")
    graph_col, text_col = st.columns([3, 2])
    with graph_col:
        st.graphviz_chart(lineage_dot(run, action["trace_id"]))
    with text_col:
        for node in lineage(run, action["trace_id"]):
            with st.expander(f"{node.stage}: {node.summary[:80]}"):
                st.json(node.payload)
