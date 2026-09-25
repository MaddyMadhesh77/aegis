"""Phase 4: STRIPS search, partial-order planning, execution with replanning, conditional plans."""

import itertools
import random

import networkx as nx
import pytest

from aegis_system.core.environment import NetworkEnvironment
from aegis_system.core.logic import Const, Literal, parse_atom
from aegis_system.core.trace import TraceStore
from aegis_system.planning.domain import SCHEMAS, defense_problem, merge_observation, observed_atoms, to_command
from aegis_system.planning.executor import (
    Branch, Executor, SensingAction, branch_outcomes, conditional_plan, contained_goal, format_conditional,
    unknown_state,
)
from aegis_system.planning.partial_order import PartialOrderPlanner, _order_graph, format_layers, layers, to_plan
from aegis_system.planning.strips_planner import (
    ActionSchema, Problem, STRIPSPlanner, compare_planners, ground, h_add, h_max, literals, negated, plan_cost,
    progression, regression, urgency_from_ttc, validate,
)
from aegis_system.search.graph_builder import build_graph, example_topology

G = example_topology()
WS1, WS2 = "Workstation1", "Workstation2"
COMPROMISED = [WS1, WS2]
POP_GOAL = f"MemoryCaptured({WS1}) & Isolated({WS1}) & Blocked(MailServer, {WS2})"
CLEAN_GOAL = f"Clean({WS1}) & Blocked(MailServer, {WS2})"
BIG_GOAL = f"Restored({WS1}) & Patched(FileServer) & Isolated({WS2})"
SMALL_GOAL = f"Clean({WS1})"
# Uninformed search wanders through combinations of cheap BlockEdge actions, so the
# heuristic comparisons run on a three-host network.
TINY = build_graph(
    {WS1: {"criticality": 0.1, "cves": [{"id": "CVE-SIM-1", "cvss": 8.0}]},
     WS2: {"criticality": 0.1}, "Server": {"criticality": 0.8, "value": 9.0}},
    [("Internet", WS1), (WS1, WS2), (WS2, "Server")],
)


def a(text):
    return parse_atom(text)


def names(plan):
    return [str(x) for x in plan]


# --------------------------------------------------------------------------- representation


def test_literals_split_positive_and_negative():
    pos, neg = literals("Compromised(h) & ~Isolated(h)")
    assert [str(p) for p in pos] == ["Compromised(h)"] and [str(n) for n in neg] == ["Isolated(h)"]
    with pytest.raises(ValueError):
        literals("A | B")


def test_schema_parse():
    s = ActionSchema.parse("Isolate(h)", pre="Compromised(h) & ~Isolated(h)", add="Isolated(h)")
    assert s.name == "Isolate" and len(s.params) == 1 and s.pre_neg == (a("Isolated(h)"),)
    with pytest.raises(ValueError):
        ActionSchema.parse("Isolate(Host1)")


def test_grounding_uses_static_facts_and_compiles_negations():
    p = defense_problem(G, COMPROMISED, POP_GOAL)
    blocks = [x for x in p.actions if x.name == "BlockEdge"]
    assert len(blocks) == G.number_of_edges()  # only real links, not every pair of hosts
    iso = next(x for x in p.actions if str(x) == f"Isolate({WS1})")
    assert iso.pre == {a(f"Compromised({WS1})"), negated(a(f"Isolated({WS1})"))}
    assert negated(a(f"Isolated({WS1})")) in iso.delete
    assert negated(a(f"Isolated({WS1})")) in p.init
    assert not any(x.pred in ("Host", "Link") for x in p.init)  # static facts are compiled away


def test_negative_goal():
    schemas = [ActionSchema.parse("Clean(h)", pre="Infected(h)", delete="Infected(h)")]
    p = ground(schemas, ["H"], [a("Infected(H)")], "~Infected(H)")
    r = progression(p)
    assert names(r.plan) == ["Clean(H)"]


# --------------------------------------------------------------------------- state-space planners


PLANNER_CASES = [
    (search, heuristic, goal)
    for goal in (POP_GOAL, CLEAN_GOAL, BIG_GOAL)
    for search in (progression, regression)
    for heuristic in ("hmax", "hadd")
    # forward search with the weak h_max needs ~14k expansions on BIG_GOAL: too slow for a unit test
    if not (search is progression and heuristic == "hmax" and goal == BIG_GOAL)
]


@pytest.mark.parametrize("search, heuristic, goal", PLANNER_CASES)
def test_planners_find_valid_plans(search, heuristic, goal):
    p = defense_problem(G, COMPROMISED, goal)
    r = search(p, heuristic)
    assert r.solved and validate(p, r.plan)
    assert r.cost == pytest.approx(plan_cost(p, r.plan))


def test_admissible_searches_agree_on_optimal_cost():
    p = defense_problem(TINY, [WS1], SMALL_GOAL)
    costs = {(row["search"], row["heuristic"]): row["cost"] for row in compare_planners(p)
             if row["heuristic"] in ("none", "hmax")}
    assert len({round(c, 9) for c in costs.values()}) == 1


def test_heuristics_reduce_nodes_expanded():
    p = defense_problem(TINY, [WS1], SMALL_GOAL)
    none, hmax, hadd = (progression(p, h).nodes_expanded for h in ("none", "hmax", "hadd"))
    assert hadd < hmax < none


def test_hmax_admissible_hadd_not():
    p = defense_problem(G, [WS1], f"MemoryCaptured({WS1}) & Clean({WS1})")
    optimal = progression(p, "hmax").cost
    assert h_max(p, p.init) <= optimal + 1e-9
    assert h_add(p, p.init) > optimal  # MemoryCaptured is counted twice


@pytest.mark.parametrize("seed", range(5))
def test_planners_survive_shuffled_action_lists(seed):
    p = defense_problem(G, COMPROMISED, BIG_GOAL)
    reference = regression(p, "hmax").cost  # optimal: h_max is admissible
    actions = list(p.actions)
    random.Random(seed).shuffle(actions)
    shuffled = Problem(p.init, p.goal, tuple(actions), p.negatable, p.urgency)
    r = regression(shuffled, "hmax")
    assert r.solved and validate(shuffled, r.plan) and r.cost == pytest.approx(reference)
    r = progression(shuffled, "hadd")
    assert r.solved and validate(shuffled, r.plan)


def test_unsolvable_problem():
    p = defense_problem(G, [], f"Clean({WS1})")  # nothing is compromised, so nothing can be cleaned
    for search in (progression, regression):
        assert not search(p, "hadd").solved


def test_urgency_prefers_fast_actions():
    assert urgency_from_ttc(None) == 0.0
    assert urgency_from_ttc(0.5) > 0.9 > urgency_from_ttc(240)
    slow = progression(defense_problem(G, COMPROMISED, CLEAN_GOAL, ttc_hours=None)).plan
    fast = progression(defense_problem(G, COMPROMISED, CLEAN_GOAL, ttc_hours=0.5)).plan
    assert f"Reimage({WS1})" in names(slow) and f"KillProcess({WS1})" not in names(slow)
    assert f"KillProcess({WS1})" in names(fast)


def test_strips_planner_default_is_optimal():
    p = defense_problem(G, COMPROMISED, BIG_GOAL)
    plan = STRIPSPlanner().plan(p)  # regression + h_max
    assert validate(p, plan)
    assert plan_cost(p, plan) == pytest.approx(regression(p, "hmax").cost)
    assert plan_cost(p, plan) <= plan_cost(p, progression(p, "hadd").plan) + 1e-9


# --------------------------------------------------------------------------- partial-order planning


def test_pop_produces_parallel_layers():
    p = defense_problem(G, COMPROMISED, POP_GOAL)
    r = PartialOrderPlanner(p).plan()
    assert r.solved
    stages = layers(r.plan)
    assert format_layers(stages) == \
        f"[{{BlockEdge(MailServer, {WS2}), DumpMemory({WS1})}}, {{Isolate({WS1})}}]"
    assert max(len(layer) for layer in stages) >= 2


def test_pop_orders_threatened_steps():
    # Isolate deletes NotIsolated, which DumpMemory needs, so DumpMemory must come first.
    p = defense_problem(G, COMPROMISED, POP_GOAL)
    plan = PartialOrderPlanner(p).plan().plan
    ids = {str(act): sid for sid, act in plan.steps.items() if act is not None}
    order = _order_graph(plan)
    assert nx.has_path(order, ids[f"DumpMemory({WS1})"], ids[f"Isolate({WS1})"])


@pytest.mark.parametrize("goal", [POP_GOAL, CLEAN_GOAL, BIG_GOAL])
def test_every_linearization_of_pop_plan_is_valid(goal):
    p = defense_problem(G, COMPROMISED, goal)
    plan = PartialOrderPlanner(p).plan().plan
    order = _order_graph(plan)
    for linear in itertools.islice(nx.all_topological_sorts(order), 500):
        assert validate(p, [plan.steps[s] for s in linear if plan.steps[s] is not None])


def test_pop_causal_links_are_supported():
    p = defense_problem(G, COMPROMISED, BIG_GOAL)
    plan = PartialOrderPlanner(p).plan().plan
    for producer, atom, consumer in plan.links:
        adds = p.init if producer == 0 else plan.steps[producer].add
        assert atom in adds
        assert consumer == 1 or atom in plan.steps[consumer].pre
    record = to_plan(plan)
    assert any(len(layer) >= 2 for layer in record.steps)
    assert ("Start", "Finish") in record.orderings


def test_pop_survives_shuffled_actions():
    p = defense_problem(G, COMPROMISED, BIG_GOAL)
    actions = list(p.actions)
    random.Random(3).shuffle(actions)
    shuffled = Problem(p.init, p.goal, tuple(actions), p.negatable, p.urgency)
    plan = PartialOrderPlanner(shuffled).plan().plan
    assert validate(shuffled, [x for layer in layers(plan) for x in layer])


# --------------------------------------------------------------------------- execution


def compromised_env(p_fail=0.0, persistent=(), seed=0, env_cls=NetworkEnvironment):
    env = env_cls(G, seed=seed, p_fail=p_fail)
    env.state.compromised |= set(COMPROMISED)
    env.state.persistent |= set(persistent)
    return env


class FailFirst(NetworkEnvironment):
    """Fails the first attempt of one action, then behaves normally (injected failure)."""

    fail_action = "isolate"

    def apply(self, action):
        if action.name == self.fail_action and not getattr(self, "_failed", False):
            self._failed = True
            saved, self.p_fail = self.p_fail, 1.0
            try:
                return super().apply(action)
            finally:
                self.p_fail = saved
        return super().apply(action)


def test_command_mapping_and_observation():
    p = defense_problem(G, COMPROMISED, POP_GOAL)
    iso = next(x for x in p.actions if str(x) == f"Isolate({WS1})")
    assert str(to_command(iso)) == f"isolate({WS1})"
    block = next(x for x in p.actions if str(x) == f"BlockEdge(MailServer, {WS2})")
    cmd = to_command(block)
    assert (cmd.name, cmd.target, cmd.destination) == ("block_edge", "MailServer", WS2)
    sensed = {"compromised": [WS1], "isolated": [], "blocked": [("MailServer", WS2)]}
    assert observed_atoms(sensed) == {a(f"Compromised({WS1})"), a(f"Blocked(MailServer, {WS2})")}
    belief = merge_observation(p, p.init | {a(f"Clean({WS2})")}, sensed)
    assert a(f"Clean({WS2})") in belief and a(f"Compromised({WS2})") not in belief


def test_executor_runs_plan_to_goal_without_failures():
    env = compromised_env()
    p = defense_problem(G, COMPROMISED, BIG_GOAL)
    report = Executor(env).execute(p)
    assert report.success and report.replans == 0
    assert all(r.success for r in report.results)
    assert WS2 in env.state.isolated and WS1 not in env.state.compromised
    assert all(r.command for r in report.results)


def test_injected_failure_triggers_replan_that_reaches_goal():
    env = compromised_env(env_cls=FailFirst)
    p = defense_problem(G, COMPROMISED, POP_GOAL)
    report = Executor(env).execute(p)
    assert report.success
    assert report.replans == 1
    assert "Isolate" in report.failures[0]
    assert names(r.step for r in report.results).count(f"Isolate({WS1})") == 2


def test_persistence_is_learned_and_plan_switches_to_reimage():
    env = compromised_env(persistent=[WS2])
    p = defense_problem(G, COMPROMISED, f"Clean({WS1}) & Clean({WS2})", ttc_hours=0.5)
    report = Executor(env).execute(p)
    assert report.success and report.replans == 1
    assert f"KillProcess({WS2})" in report.executed and f"Reimage({WS2})" in report.executed
    assert a(f"Persistent({WS2})") in report.final_state
    assert not env.state.compromised


def test_replans_are_capped():
    env = compromised_env(p_fail=1.0)
    p = defense_problem(G, COMPROMISED, POP_GOAL)
    report = Executor(env, max_replans=2).execute(p)
    assert not report.success and report.replans == 2 and len(report.failures) == 3


def test_invariant_violation_counts_as_failure():
    env = compromised_env()
    p = defense_problem(G, COMPROMISED, f"Isolated({WS1})")
    invariant = Literal(a(f"Isolated({WS1})"), positive=False)
    report = Executor(env, max_replans=0, invariants=[invariant]).execute(p)
    assert not report.success and "invariant" in report.failures[0]


@pytest.mark.parametrize("planner", ["regression", "progression"])
def test_state_space_executors_also_work(planner):
    env = compromised_env()
    p = defense_problem(G, COMPROMISED, CLEAN_GOAL)
    report = Executor(env, planner=planner).execute(p)
    assert report.success and all(len(layer) == 1 for layer in report.plans[0])


def test_execution_trace_links_actions_back_through_replans():
    trace = TraceStore()
    objective = trace.record("defense", "best defence for the incident", {})
    env = compromised_env(env_cls=FailFirst)
    p = defense_problem(G, COMPROMISED, POP_GOAL)
    report = Executor(env, trace=trace).execute(p, parent_trace=objective)
    stages = [n.stage for n in trace.lineage(report.results[-1].trace_id)]
    assert stages[0] == "act"
    assert {"plan", "failure", "defense"} <= set(stages)
    assert len(trace.by_stage("plan")) == 2


# --------------------------------------------------------------------------- conditional planning


def test_conditional_plan_branches_on_sensing():
    p = defense_problem(G, [WS1], f"Isolated({WS1})")
    state = unknown_state(p, [WS1], [WS2])
    goal = contained_goal([WS1, WS2])
    plan = conditional_plan(p, state, goal, [SensingAction(WS2)], max_depth=4,
                            relevant=lambda x: x.name == "Isolate")
    assert plan is not None
    assert any(isinstance(step, Branch) for step in plan)
    outcomes = branch_outcomes(plan, state)
    assert {labels[0] for labels, _ in outcomes} == {"compromised", "clean"}
    assert all(goal(final) for _, final in outcomes)
    compromised_branch = next(final for labels, final in outcomes if labels == ["compromised"])
    assert a(f"Isolated({WS2})") in compromised_branch
    clean_branch = next(final for labels, final in outcomes if labels == ["clean"])
    assert a(f"Isolated({WS2})") not in clean_branch  # no needless isolation
    assert "CheckHostStatus" in format_conditional(plan)


def test_conditional_plan_fails_when_depth_too_small():
    p = defense_problem(G, [WS1], f"Isolated({WS1})")
    state = unknown_state(p, [WS1], [WS2])
    assert conditional_plan(p, state, contained_goal([WS1, WS2]), [SensingAction(WS2)], max_depth=1,
                            relevant=lambda x: x.name == "Isolate") is None


def test_schemas_cover_every_environment_command():
    p = defense_problem(G, COMPROMISED, POP_GOAL)
    kinds = {to_command(x).name for x in p.actions}
    assert kinds == {"isolate", "block_edge", "reimage", "kill_process", "patch", "restore", "collect_memory"}
    assert len(SCHEMAS) == 7
    assert Const(WS1) in {arg for x in p.actions for arg in x.args}
