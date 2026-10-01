"""Phase 7: agents, the closed loop, PEAS metrics, the CLI, and the dashboard."""

import json
from pathlib import Path

import numpy as np
import pytest

from aegis_system.agents import (
    AGENTS, BeliefTracker, GoalBasedAgent, Percept, Perception, UtilityAgent,
)
from aegis_system.core.environment import ActionCommand, NetworkEnvironment
from aegis_system.core.trace import TraceStore
from aegis_system.core.types import Alert, Proof
from aegis_system.dashboard.visualizer import (
    actions_table, explanation, lineage_dot, network_figure, phase_figure,
)
from aegis_system.loop import compare_agents, contained, run_episode
from aegis_system.main import main
from aegis_system.simulation.scenarios import available_scenarios, load_scenario, make_environment


@pytest.fixture(scope="module")
def env():
    return make_environment("solarwinds", seed=0)


@pytest.fixture(scope="module")
def perception(env):
    return Perception(env, seed=0)


@pytest.fixture(scope="module")
def goal_run(perception):
    return run_episode("goal_based", "solarwinds", ticks=20, seed=0, perception=perception)


# --------------------------------------------------------------------------- scenarios and environment


def test_scenarios_load():
    assert {"solarwinds", "random"} <= set(available_scenarios())
    spec = load_scenario("solarwinds")
    assert spec.targets == ["DC", "ExchangeServer"] and spec.entry == "Internet"
    assert spec.graph.has_edge("AdminWS", "DC")
    with pytest.raises(FileNotFoundError):
        load_scenario("nope")


def test_lookalike_flows_are_benign_and_off_by_default():
    plain = NetworkEnvironment(seed=0)
    assert plain.lookalike_rate == 0.0
    noisy = make_environment("solarwinds", seed=0, lookalike_rate=1.0)
    for _ in range(3):
        noisy.step()
    flows = noisy.observe().flows
    benign = [f for f in flows if not f.malicious]
    assert benign and all(f.features["bytes_out"] > 0 for f in benign)


def test_sensing_after_an_action_is_scoped_to_the_hosts_it_touched():
    e = make_environment("solarwinds", seed=0, p_fail=0.0)
    e.state.compromised |= {"OrionServer", "WS1"}
    report = e.apply(ActionCommand("isolate", "OrionServer")).observed_state
    assert report["compromised"] == ["OrionServer"] and report["scope"] == ["OrionServer"]
    assert e.sensed_state()["compromised"] == ["OrionServer", "WS1"]  # unscoped: full view


# --------------------------------------------------------------------------- shared components


def test_perception_calibrates_its_false_alarm_rate(perception, env):
    assert 0.0 < perception.false_alarm_rate < 0.2
    e = make_environment("solarwinds", seed=3)
    for _ in range(8):
        e.step()
    alerts = perception.alerts(e.observe().flows, 8, None)
    assert all(a.host != e.entry for a in alerts)


def test_belief_tracker_ignores_silence_from_isolated_hosts():
    tracker = BeliefTracker(["A", "B"], false_alarm=0.01)
    alert = [Alert("A", "r2l", 0.99)]
    tracker.update(alert, 1, None)
    p = tracker.update(alert, 2, None).host_compromise_prob["A"]
    for t in range(3, 10):
        b = tracker.update([], t, None, unobserved={"A"})
    assert b.host_compromise_prob["A"] == pytest.approx(p, abs=0.02)
    assert b.host_compromise_prob["B"] < 0.05
    assert np.isclose(sum(b.phase_posterior.values()), 1.0)


# --------------------------------------------------------------------------- agents


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_every_agent_acts_and_explains_its_actions(name, perception):
    result = run_episode(name, "solarwinds", ticks=15, seed=0, perception=perception)
    store = TraceStore.from_dict(result.trace)
    actions = [a for row in result.timeline for a in row["actions"]]
    assert actions, f"{name} never acted"
    for a in actions:
        stages = {n.stage for n in store.lineage(a["trace_id"])}
        assert stages & {"ml", "rule"}, f"{a['action']} does not trace back to a detection"


def test_goal_agent_treats_a_compromised_target_as_a_violation(env, perception):
    agent = GoalBasedAgent(env, seed=0, perception=perception)
    agent.proven = {"ExchangeServer": Proof(None, [], True)}
    percept = Percept(1, [], isolated=set(), blocked=set())
    assert ("ExchangeServer", "ExchangeServer") in agent.goal_violations(percept)
    percept.isolated.add("ExchangeServer")
    assert agent.goal_violations(percept) == []


def test_goal_agent_cuts_paths_from_critical_hosts_instead_of_isolating(env, perception):
    agent = GoalBasedAgent(env, seed=0, perception=perception)
    g = agent.graph.copy()
    cut = agent._choke(g, ["DC"], ["ExchangeServer"])
    assert cut == {("DC", "ExchangeServer")}


def test_utility_agent_weighs_risk_against_downtime(env, perception):
    agent = UtilityAgent(env, seed=0, perception=perception)
    g = agent.graph.copy()
    assert agent.value_at_risk(g, "OrionServer", agent.targets) == pytest.approx(19.0)
    assert agent.value_at_risk(g, "WS2", agent.targets) == pytest.approx(19.0)
    g.remove_edge("FileServer", "DC")
    assert agent.value_at_risk(g, "WS2", agent.targets) == 0.0


# --------------------------------------------------------------------------- loop and metrics


def test_episode_metrics_and_timeline(goal_run):
    m = goal_run.metrics
    for key in ("time_to_containment", "breached", "uptime", "false_isolations", "proof_depth"):
        assert key in m
    assert 0.0 <= m["uptime"] <= 1.0
    assert m["proof_depth"] > 0
    assert len(goal_run.timeline) == 20
    row = goal_run.timeline[-1]
    assert np.isclose(sum(row["phase_posterior"].values()), 1.0)
    assert set(row["host_compromise_prob"]) == {n["id"] for n in goal_run.topology["nodes"]} - {"Internet"}


def test_full_pipeline_lineage_for_goal_agent(goal_run):
    store = TraceStore.from_dict(goal_run.trace)
    action = next(a for row in goal_run.timeline for a in row["actions"])
    stages = [n.stage for n in store.lineage(action["trace_id"])]
    assert stages[0] == "act"
    assert {"plan", "defense", "proof", "grounding", "belief", "ml"} <= set(stages)


def test_containment_definition():
    e = make_environment("solarwinds", seed=0, p_fail=0.0)
    assert contained(e)
    e.state.compromised.add("OrionServer")
    assert not contained(e)
    e.state.isolated.add("OrionServer")
    assert contained(e)
    e.state.compromised.add("DC")  # a target held live is never contained
    assert not contained(e)


def test_runs_are_reproducible(perception):
    a = run_episode("utility", "solarwinds", ticks=12, seed=4, perception=perception)
    b = run_episode("utility", "solarwinds", ticks=12, seed=4, perception=perception)
    assert a.metrics == b.metrics and a.timeline == b.timeline


def test_compare_agents_table():
    rows = compare_agents("solarwinds", seeds=[0], ticks=12, agents=["reflex", "goal_based"])
    assert [r["agent"] for r in rows] == ["reflex", "goal_based"]
    assert all(r["episodes"] == 1 and 0 <= r["mean_uptime"] <= 1 for r in rows)


# --------------------------------------------------------------------------- CLI


def test_cli_runs_solarwinds_end_to_end(tmp_path, capsys):
    assert main(["--scenario", "solarwinds", "--ticks", "15", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "PEAS performance measures" in out and "[t=" in out
    saved = json.loads((tmp_path / "solarwinds_goal_based_seed0.json").read_text())
    assert saved["timeline"] and saved["trace"]


# --------------------------------------------------------------------------- dashboard


def test_visualizer_builds_figures_and_explanations(goal_run):
    run = goal_run.to_json()
    assert len(network_figure(run, 10).data) > 1
    assert len(phase_figure(run).data) == 7
    rows = actions_table(run)
    dot = lineage_dot(run, rows[0]["trace_id"])
    assert dot.startswith("digraph") and "->" in dot
    assert explanation(run, rows[0]["trace_id"])[0].startswith("act:")


def test_dashboard_app_renders_lineage(goal_run, tmp_path):
    from streamlit.testing.v1 import AppTest

    path = goal_run.save(tmp_path)
    app = Path(__file__).resolve().parents[1] / "aegis_system" / "dashboard" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=120)
    at.session_state["run_path"] = str(path)
    at.run()
    assert not at.exception
    assert any(s.value.startswith("Why:") for s in at.subheader)
    assert any(e.label.startswith("ml:") for e in at.expander)
    assert [m.label for m in at.metric][:3] == ["Breached", "Time to containment", "Uptime"]
