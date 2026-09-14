import pytest

from aegis_system.knowledge.frame_system import HostFrame, ThreatActorFrame
from aegis_system.knowledge.semantic_net import SemanticNet
from aegis_system.perception.classifier import AttackClassifier
from aegis_system.perception.regressor import TimeToCompromiseRegressor
from aegis_system.planning.executor import ExecutionAgent
from aegis_system.planning.partial_order import PartialOrderPlanner, Step
from aegis_system.probabilistic.bayes_net import BayesNet
from aegis_system.probabilistic.hmm_engine import HMM
from aegis_system.reasoning.fol_engine import FOLKnowledgeBase
from aegis_system.reasoning.resolution import resolve_clauses
from aegis_system.reasoning.unifier import unify
from aegis_system.search.adversarial_search import minimax_decision
from aegis_system.search.graph_builder import build_graph
from aegis_system.search.path_search import bfs_search, astar_search
from aegis_system.planning.strips_planner import STRIPSPlanner, Action
from aegis_system.simulation.simulator import generate_attack_simulation


def test_unify_success_and_failure():
    left = ("Compromised", "x")
    right = ("Compromised", "Host1")
    result = unify(left, right)
    assert result == {"x": "Host1"}

    left = ("Trusts", "x", "y")
    right = ("Trusts", "Host1", "Host1")
    assert unify(left, right) == {"x": "Host1", "y": "Host1"}

    assert unify(("Compromised", "x"), ("Trusts", "Host1")) is None


def test_hmm_forward_and_viterbi():
    states = ["Recon", "Exploit", "Exfiltration"]
    observations = [0, 1, 1]
    hmm = HMM(
        states=states,
        observations=[0, 1],
        start_probs={"Recon": 0.6, "Exploit": 0.2, "Exfiltration": 0.2},
        trans_probs={
            "Recon": {"Recon": 0.7, "Exploit": 0.2, "Exfiltration": 0.1},
            "Exploit": {"Recon": 0.1, "Exploit": 0.7, "Exfiltration": 0.2},
            "Exfiltration": {"Recon": 0.05, "Exploit": 0.15, "Exfiltration": 0.8},
        },
        emit_probs={
            "Recon": {0: 0.8, 1: 0.2},
            "Exploit": {0: 0.3, 1: 0.7},
            "Exfiltration": {0: 0.2, 1: 0.8},
        },
    )

    forward = hmm.forward(observations)
    assert sum(forward[-1].values()) > 0.0

    viterbi_path = hmm.viterbi(observations)
    assert len(viterbi_path) == len(observations)
    assert viterbi_path[0] in states


def test_graph_search_routes_to_target():
    graph = {
        "A": {"B": 1, "C": 2},
        "B": {"D": 1},
        "C": {"D": 1},
        "D": {},
    }

    bfs_result = bfs_search(graph, "A", "D")
    astar_result = astar_search(graph, "A", "D")

    assert bfs_result[-1] == "D"
    assert astar_result[-1] == "D"


def test_strips_planner_generates_action_sequence():
    action1 = Action(
        "scan",
        preconditions={"connected"},
        add_effects={"vulnerable"},
        delete_effects=set(),
    )
    action2 = Action(
        "isolate",
        preconditions={"vulnerable"},
        add_effects={"contained"},
        delete_effects={"vulnerable"},
    )

    planner = STRIPSPlanner([action1, action2])
    plan = planner.plan({"connected"}, {"contained"})

    assert plan is not None
    assert "scan" in [step.name for step in plan]
    assert "isolate" in [step.name for step in plan]


def test_bayes_net_exact_inference():
    net = BayesNet(
        variables=["A", "B", "C"],
        cpts={
            "A": {(): {True: 0.6, False: 0.4}},
            "B": {"A": {True: {True: 0.9, False: 0.1}, False: {True: 0.2, False: 0.8}}},
            "C": {"A": {True: {True: 0.7, False: 0.3}, False: {True: 0.1, False: 0.9}}},
        },
    )

    result = net.infer("B", {"A": True})
    assert result[True] > result[False]


def test_resolution_reduces_complementary_literals():
    clauses = [
        {"Compromised(x)"},
        {"~Compromised(x)", "Suspicious(x)"},
    ]
    result = resolve_clauses(clauses[0], clauses[1])
    assert "Suspicious(x)" in result or result == {"Suspicious(x)"}


def test_semantic_net_tracks_relations():
    net = SemanticNet()
    net.add_relation("Mimikatz", "instance_of", "Credential_Dumping")
    net.add_relation("Credential_Dumping", "subtechnique_of", "T1003")

    assert net.has_relation("Mimikatz", "instance_of", "Credential_Dumping")
    assert net.path_exists("Mimikatz", "T1003")


def test_partial_order_planner_orders_independent_steps():
    steps = [
        Step("block_port", {"network_access"}, {"blocked"}, set()),
        Step("dump_memory", {"host_access"}, {"memory_acquired"}, set()),
    ]

    planner = PartialOrderPlanner(steps)
    plan = planner.plan({"network_access", "host_access"}, {"blocked", "memory_acquired"})

    assert plan is not None
    assert len(plan) == 2


def test_attack_classifier_and_ttc_regressor_work():
    X = [[0.1, 0.2], [0.8, 0.9], [0.7, 0.8], [0.2, 0.1]]
    y = ["benign", "attack", "attack", "benign"]
    classifier = AttackClassifier()
    classifier.fit(X, y)

    pred = classifier.predict([[0.75, 0.85]])
    assert pred[0] == "attack"

    reg = TimeToCompromiseRegressor()
    reg.fit([[0.1], [0.9], [0.7], [0.2]], [3.0, 1.0, 1.5, 4.5])
    assert reg.predict([[0.8]])[0] >= 0


def test_graph_builder_and_minimax_engine():
    graph = build_graph(
        {
            "HostA": {"HostB": 1},
            "HostB": {"HostC": 1},
            "HostC": {},
        }
    )
    assert "HostA" in graph and "HostC" in graph

    action_values = {"attack": 10, "defend": 1}
    result = minimax_decision(action_values, maximizing=True)
    assert result in action_values


def test_execution_agent_uses_failover_plan():
    agent = ExecutionAgent()
    action = agent.execute_action("isolate", {"host": "Host1"})
    assert action["status"] in {"success", "replan"}


def test_knowledge_frames_and_fol_engine_support_reasoning():
    host = HostFrame("Host1", {"os": "Windows", "privilege_level": "admin"})
    attacker = ThreatActorFrame("APT29", {"tactic": "Credential Dumping"})

    kb = FOLKnowledgeBase()
    kb.assert_fact("Compromised(Host1)")
    kb.assert_fact("Trusts(Host1, DomainController)")

    assert host.name == "Host1"
    assert attacker.name == "APT29"
    assert "Compromised(Host1)" in kb.facts


def test_attack_simulation_generates_actionable_report():
    report = generate_attack_simulation()
    assert report["network"]["start"] == "HostA"
    assert "path" in report and len(report["path"]) > 0
    assert "actions" in report and len(report["actions"]) > 0
