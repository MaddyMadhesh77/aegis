import pytest

from aegis_system.core.logic import parse_atom

from aegis_system.knowledge.frame_system import HostFrame, ThreatActorFrame
from aegis_system.knowledge.semantic_net import SemanticNet
from aegis_system.perception.classifier import AttackClassifier
from aegis_system.perception.regressor import TimeToCompromiseRegressor
from aegis_system.reasoning.fol_engine import FOLKnowledgeBase
from aegis_system.search.adversarial_search import minimax_decision
from aegis_system.search.graph_builder import from_adjacency
from aegis_system.search.path_search import bfs_search, astar_search
from aegis_system.simulation.simulator import generate_attack_simulation


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


def test_semantic_net_tracks_relations():
    net = SemanticNet()
    net.add_relation("Mimikatz", "instance_of", "Credential_Dumping")
    net.add_relation("Credential_Dumping", "subtechnique_of", "T1003")

    assert net.has_relation("Mimikatz", "instance_of", "Credential_Dumping")
    assert net.path_exists("Mimikatz", "T1003")


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
    graph = from_adjacency(
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


def test_knowledge_frames_and_fol_engine_support_reasoning():
    host = HostFrame("Host1", {"os": "Windows", "privilege_level": "admin"})
    attacker = ThreatActorFrame("APT29", {"tactic": "Credential Dumping"})

    kb = FOLKnowledgeBase()
    kb.tell("Compromised(Host1)")
    kb.tell("Trusts(Host1, DomainController)")

    assert host.name == "Host1"
    assert attacker.name == "APT29"
    assert parse_atom("Compromised(Host1)") in set(kb.facts())


def test_attack_simulation_generates_actionable_report():
    report = generate_attack_simulation()
    assert report["network"]["start"] == "HostA"
    assert "path" in report and len(report["path"]) > 0
    assert "actions" in report and len(report["actions"]) > 0
