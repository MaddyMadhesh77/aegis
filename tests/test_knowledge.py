"""Phase 5: semantic net, frames, production system, OWL ontology, and grounding."""

import shutil
import tomllib
from pathlib import Path

import pytest

from aegis_system.core.logic import Atom, Const, parse_atom
from aegis_system.core.trace import TraceStore
from aegis_system.core.types import Alert, BeliefState
from aegis_system.knowledge.frame_system import Frame, FrameSystem
from aegis_system.knowledge.grounding import Grounder, ground, incident_demo
from aegis_system.knowledge.ontology_manager import OntologyManager
from aegis_system.knowledge.production_system import ProductionSystem, Rule, compare, flow_facts, threshold_rule
from aegis_system.knowledge.semantic_net import SemanticNet
from aegis_system.search.graph_builder import example_topology


@pytest.fixture(scope="module")
def net():
    return SemanticNet.from_attack_json()


# --------------------------------------------------------------------------- semantic net


def test_attack_subset_loads(net):
    techniques = net.sources("is_a", "Technique") | net.sources("subtechnique_of", "T1003") | \
        net.sources("subtechnique_of", "T1021") | net.sources("subtechnique_of", "T1550") | \
        net.sources("subtechnique_of", "T1195")
    assert 15 <= len(techniques) <= 25
    assert net.name("T1003") == "OS Credential Dumping"
    assert net.tactic_of("T1003") == "credential-access"


def test_inherits_nearest_value_wins(net):
    assert net.inherits("T1003.001", "requires_privilege") == "SYSTEM"  # from its parent T1003
    assert net.inherits("T1110", "requires_privilege") == "user"        # from the generic Technique node
    net2 = SemanticNet()
    net2.add_relation("Mimikatz", "instance_of", "CredentialTool")
    net2.add_relation("CredentialTool", "is_a", "Tool")
    net2.set_attr("Tool", "risk", "medium")
    net2.set_attr("CredentialTool", "risk", "high")
    assert net2.inherits("Mimikatz", "risk") == "high"
    assert net2.inherits("Mimikatz", "missing", "none") == "none"


def test_subtechnique_inherits_tactic_and_mitigations(net):
    assert net.tactic_of("T1003.001") == "credential-access"
    mitigations = net.mitigations_for("T1003.001")
    assert "M1025" in mitigations                         # its own
    assert set(net.mitigations_for("T1003")) <= set(mitigations)  # its parent's


def test_transitive_closure():
    n = SemanticNet()
    for a, b in [("A", "B"), ("B", "C"), ("C", "D")]:
        n.add_relation(a, "lateral", b)
    assert n.transitive_closure("lateral") == {("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"),
                                              ("C", "D")}


def test_alert_class_mapping_and_tools(net):
    assert net.technique_for_alert("credential_dumping") == "T1003"
    assert net.technique_for_alert("probe") == "T1046"
    assert net.technique_for_alert("unknown") is None
    assert net.has_relation("Mimikatz", "implements", "T1003.001")
    assert net.path_exists("Mimikatz", "credential-access")
    assert "T1003" in net.techniques_by_tactic("credential-access")


def test_kill_chain_phase_matches_hmm_states(net):
    from aegis_system.probabilistic.hmm_engine import KILL_CHAIN
    for tactic in net.sources("is_a", "Tactic"):
        assert net.inherits(tactic, "kill_chain_phase") in KILL_CHAIN


# --------------------------------------------------------------------------- frames


def test_frame_get_order_value_if_needed_default_parent():
    fs = FrameSystem()
    host = fs.generic("host")
    host.define("os", default="Windows")
    host.define("label", if_needed=lambda f, s: f"host {f.name}")
    server = fs.generic("server", "host")
    server.define("os", default="Linux")
    web = fs.instance("Web1", "server")
    ws = fs.instance("WS1", "host", {"os": "macOS"})

    assert ws.get("os") == "macOS"            # own value
    assert web.get("os") == "Linux"           # nearest default
    assert web.get("label") == "host Web1"    # inherited if_needed, computed for the instance
    assert web.get("missing", 42) == 42
    assert web.is_a("host") and not ws.is_a("server")
    assert fs.instances_of("host") == [web, ws]


def test_if_added_and_if_removed_demons_are_inherited():
    events = []
    fs = FrameSystem()
    host = fs.generic("host")
    host.define("status", if_added=lambda f, s, v: events.append(("added", f.name, v)),
                if_removed=lambda f, s, v: events.append(("removed", f.name, v)))
    h = fs.instance("H1", "host")
    h.set("status", "compromised")
    h.remove("status")
    h.remove("status")  # nothing to remove: no demon
    assert events == [("added", "H1", "compromised"), ("removed", "H1", "compromised")]
    assert h.get("status") is None


def test_initial_slot_values_do_not_fire_demons_and_names_are_unique():
    fired = []
    fs = FrameSystem()
    fs.generic("host").define("x", if_added=lambda *a: fired.append(a))
    fs.instance("H", "host", {"x": 1})
    assert fired == []
    with pytest.raises(ValueError):
        fs.instance("H", "host")


def test_plain_frame_still_works():
    f = Frame("Host1", {"os": "Windows"})
    assert f.get("os") == "Windows" and f.to_dict() == {"os": "Windows"}


# --------------------------------------------------------------------------- production system


def test_conflict_resolution_salience_then_specificity_then_recency():
    ps = ProductionSystem([
        Rule.make("general", ["Alert(h)"], asserts=["Seen(h)"]),
        Rule.make("specific", ["Alert(h)", "Critical(h)"], asserts=["Escalate(h)"]),
        Rule.make("urgent", ["Alert(h)"], asserts=["Page(h)"], salience=10),
    ])
    ps.assert_fact("Alert(A)")
    ps.assert_fact("Critical(A)")
    ps.assert_fact("Alert(B)")
    order = [(f.rule, f.facts[0].args[0].name) for f in ps.run()]
    # salience first (newest fact first within it), then the more specific rule, then recency
    assert order == [("urgent", "B"), ("urgent", "A"), ("specific", "A"), ("general", "B"), ("general", "A")]


def test_refraction_rule_never_fires_twice_on_same_facts():
    ps = ProductionSystem([Rule.make("count", ["Alert(h)"], asserts=["Seen(h)"])])
    ps.assert_fact("Alert(A)")
    assert len(ps.run()) == 1
    assert ps.run() == []
    ps.assert_fact("Alert(B)")
    assert [f.rule for f in ps.run()] == ["count"]


def test_negation_retraction_tests_and_actions():
    called = []
    ps = ProductionSystem([
        Rule.make("isolate", ["Compromised(h)"], negated=["Isolated(h)"], asserts=["Isolated(h)"],
                  action=lambda sys, theta: called.append(str(theta[next(iter(theta))]))),
        Rule.make("clear", ["Isolated(h)", "Clean(h)"], retracts=["Compromised(h)"]),
        Rule.make("big", ["Bytes(f, b)"], tests=[compare("b", ">", 5000)], asserts=["Large(f)"]),
    ])
    for fact in ["Compromised(H1)", "Clean(H1)", "Bytes(F1, 9000)", "Bytes(F2, 10)"]:
        ps.assert_fact(fact)
    ps.run()
    assert parse_atom("Isolated(H1)") in ps.memory
    assert parse_atom("Compromised(H1)") not in ps.memory
    assert parse_atom("Large(F1)") in ps.memory and parse_atom("Large(F2)") not in ps.memory
    assert called == ["H1"]
    with pytest.raises(ValueError):
        ps.assert_fact("Alert(x)")  # working memory is ground


def test_threshold_rules_classify_flows_once():
    rules = [
        threshold_rule("exfil", [("bytes_out", ">", 50000.0), ("duration", ">", 10.0)], "exfiltration",
                       confidence=0.97),
        threshold_rule("scan", [("syn_ratio", ">", 0.8)], "probe"),
        threshold_rule("big", [("bytes_out", ">", 50000.0)], "suspicious"),
    ]
    ps = ProductionSystem(rules)
    for fact in flow_facts("f1", {"bytes_out": 90000, "duration": 20, "syn_ratio": 0.0}) + \
            flow_facts("f2", {"bytes_out": 60, "duration": 0.1, "syn_ratio": 0.95}):
        ps.assert_fact(fact)
    ps.run()
    classified = {(f.args[0].name, f.args[1].name) for f in ps.facts("Classified")}
    # the more specific exfil rule wins for f1, and a flow is classified only once
    assert classified == {("f1", "exfiltration"), ("f2", "probe")}
    assert Atom("Confidence", (Const("f1"), Const("exfiltration"), Const("0.970"))) in ps.memory


# --------------------------------------------------------------------------- ontology


@pytest.fixture(scope="module", params=["fallback", "hermit"])
def onto(request):
    if request.param == "hermit" and shutil.which("java") is None:
        pytest.skip("HermiT needs Java")
    o = OntologyManager()
    o.load_topology(example_topology(), {"CVE-SIM-0101": ["T1190"]})
    assert o.reason(use_hermit=request.param == "hermit") == request.param
    return o


def test_defined_classes_are_inferred(onto):
    assert onto.critical_vulnerabilities() == ["CVE_SIM_0101", "CVE_SIM_0401"]  # cvss >= 9.0
    assert onto.exposed_hosts() == ["DomainController", "WebServer"]
    assert onto.is_subclass("DomainController", "Host")
    assert onto.is_subclass("CredentialAccessTechnique", "Technique")


def test_lateral_movement_is_transitive(onto):
    assert onto.reachable_from("MailServer") == ["Database", "DomainController", "FileServer", "Workstation1",
                                                 "Workstation2"]


def test_sparql_over_attack_data(onto):
    assert onto.mitigations_for("T1003.001") == SemanticNet.from_attack_json().mitigations_for("T1003.001")
    assert "T1003" in onto.techniques_of_tactic("credential-access")
    rows = onto.sparql("SELECT ?t WHERE { ?t a:exploits ?v . ?v a:cvssScore ?s . FILTER(?s >= 9.0) }")
    assert onto.names(rows) == ["T1190"]


def test_without_reasoning_nothing_is_inferred():
    o = OntologyManager()
    o.load_topology(example_topology())
    assert o.exposed_hosts() == []
    assert o.reachable_from("MailServer") == ["Workstation1", "Workstation2"]


# --------------------------------------------------------------------------- grounding


def test_credential_dumping_alert_proves_compromised_dc():
    demo = incident_demo()
    assert demo["proved_before"] is False
    grounded = demo["grounded"]
    assert {str(a) for a in grounded.atoms} == {"Compromised(Host3)", "CredentialsDumped(Host3)",
                                                "Uses(Attacker, T1003)"}
    proof = demo["proof"]
    assert proof.proved
    kb = demo["grounder"].kb
    assert kb.ask("Compromised(DC)", method="backward").proved
    assert kb.ask("Compromised(DC)", method="forward").proved


def test_proof_trace_shows_every_step_back_to_detection():
    demo = incident_demo()
    trace = demo["trace"]
    lineage = trace.lineage(demo["proof"].trace_id)
    stages = [n.stage for n in lineage]
    assert stages[0] == "proof"
    assert {"kb", "grounding", "ml", "hmm", "knowledge"} <= set(stages)
    told = [n.summary for n in lineage if n.stage == "kb"]
    assert "tell Compromised(Host3)" in told and "tell CredentialsDumped(Host3)" in told
    assert "tell HasAdminSession(Host3, DC)" in told
    grounding = next(n for n in lineage if n.stage == "grounding")
    assert "T1003" in grounding.summary


def test_privilege_demon_tells_privesc_and_records_trace():
    demo = incident_demo()
    grounder, trace = demo["grounder"], demo["trace"]
    assert grounder.frames["Host3"].get("privilege_level") == "SYSTEM"
    frame_nodes = trace.by_stage("frame")
    assert len(frame_nodes) == 1 and "if_added demon" in frame_nodes[0].summary
    assert trace.get(frame_nodes[0].parents[0]).stage == "grounding"
    proof = grounder.kb.ask("PrivEsc(Host3)")
    assert proof.proved
    assert "frame" in [n.stage for n in trace.lineage(proof.trace_id)]


def test_incident_frame_fills_from_semantic_net():
    frames = incident_demo()["grounded"].frames
    assert frames["incident"]["tactic"] == "credential-access"
    assert frames["incident"]["kill_chain_phase"] == "Exploitation"
    assert "M1043" in frames["incident"]["mitigations"]
    assert frames["host"]["status"] == "compromised" and frames["host"]["admin_sessions"] == ["DC"]


def _grounder():
    return Grounder(example_topology(), trace=TraceStore())


def test_below_threshold_alerts_are_not_grounded():
    g = _grounder()
    result = g.ground(Alert("Workstation1", "credential_dumping", 0.95),
                      BeliefState({}, {"Workstation1": 0.2}))
    assert result.atoms == set()
    assert "below threshold" in g.trace.get(result.trace_id).summary
    assert not g.kb.ask("Compromised(Workstation1)").proved


def test_reconnaissance_marks_host_targeted_not_compromised():
    g = _grounder()
    atoms = ground(Alert("WebServer", "probe", 0.9), None, g)
    assert {str(a) for a in atoms} == {"Targeted(WebServer)", "Uses(Attacker, T1046)"}
    assert g.frames["WebServer"].get("status") == "targeted"
    assert g.frames["WebServer"].get("privilege_level") == "user"  # no demon


def test_unknown_alert_class_and_host():
    g = _grounder()
    assert g.ground(Alert("WebServer", "teleport", 0.99)).atoms == set()
    assert g.ground(Alert("Nowhere", "credential_dumping", 0.99)).atoms == set()


def test_topology_facts_on_example_network():
    g = _grounder()
    facts = {str(f) for f in g.topology_facts()}
    assert "HasAdminSession(Workstation2, DomainController)" in facts
    assert "Trusts(DomainController, FileServer)" in facts
    g.ground(Alert("Workstation2", "lsass_access", 0.9))  # sub-technique of T1003
    assert g.kb.ask("Compromised(DomainController)").proved


def test_ontology_data_ships_with_the_package():
    pyproject = tomllib.loads(Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text())
    assert "data/*.json" in pyproject["tool"]["setuptools"]["package-data"]["aegis_system.knowledge"]
