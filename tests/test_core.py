import pytest
from hypothesis import given, strategies as st

from aegis_system.core.environment import (
    FEATURES, KILL_CHAIN, ActionCommand, NetworkEnvironment, iptables_command,
)
from aegis_system.core.logic import (
    And, Atom, Const, Exists, Fn, ForAll, Iff, Implies, Literal, Not, Or, Var,
    clause_str, free_variables, is_ground, parse, parse_atom, parse_term, rename_variables, term_variables,
)
from aegis_system.core.trace import TraceStore
from aegis_system.search.graph_builder import (
    MIN_WEIGHT, NO_CVE_WEIGHT, build_graph, critical_assets, example_topology, from_adjacency, generate_topology,
)

x, y, z = Var("x"), Var("y"), Var("z")


# ----------------------------------------------------------------------------- logic


def test_parse_trust_rule():
    f = parse("forall x forall y forall z (Compromised(x) & Trusts(y, x) & TransfersAuth(x, y, z) -> Compromised(y))")
    expected = ForAll(x, ForAll(y, ForAll(z, Implies(
        And(And(Atom("Compromised", (x,)), Atom("Trusts", (y, x))), Atom("TransfersAuth", (x, y, z))),
        Atom("Compromised", (y,)),
    ))))
    assert f == expected


def test_parse_precedence_and_associativity():
    assert parse("A | B & C") == Or(Atom("A"), And(Atom("B"), Atom("C")))
    assert parse("~A & B") == And(Not(Atom("A")), Atom("B"))
    assert parse("A -> B -> C") == Implies(Atom("A"), Implies(Atom("B"), Atom("C")))
    assert parse("A <-> B") == Iff(Atom("A"), Atom("B"))
    assert parse("A => B") == parse("A -> B")


def test_parse_quantifier_variants():
    assert parse("forall x, y P(x, y)") == parse("forall x y P(x, y)") == ForAll(x, ForAll(y, Atom("P", (x, y))))
    assert parse("exists h . Vulnerable(h)") == Exists(Var("h"), Atom("Vulnerable", (Var("h"),)))
    assert parse("forall x exists y Owns(y, x)") == ForAll(x, Exists(y, Atom("Owns", (y, x))))


def test_parse_terms():
    assert parse_term("Host1") == Const("Host1")
    assert parse_term("x") == Var("x")
    assert parse_term("f(x, Host1)") == Fn("f", (x, Const("Host1")))
    assert parse_atom("Port(Host1, 443)") == Atom("Port", (Const("Host1"), Const("443")))


@pytest.mark.parametrize("bad", ["P(x", "forall P(x)", "p(x)", "A &", "A B"])
def test_parse_rejects_bad_input(bad):
    with pytest.raises(SyntaxError):
        parse(bad)


@pytest.mark.parametrize("text", [
    "forall x forall y (Compromised(x) & Trusts(y, x) -> Compromised(y))",
    "exists h (Vulnerable(h) & ~Patched(h))",
    "~~(A | B) <-> (C -> D)",
    "Owns(f(x), Host1)",
    "Raining",
])
def test_formula_round_trips_through_str(text):
    f = parse(text)
    assert parse(str(f)) == f


def test_free_variables_and_ground():
    f = parse("forall x (P(x, y) & Q(z))")
    assert free_variables(f) == [y, z]
    assert not is_ground(Atom("P", (x,)))
    assert is_ground(parse_atom("Compromised(Host1)"))


def test_rename_variables_is_consistent_and_fresh():
    clause = frozenset({Literal(Atom("P", (x, y))), Literal(Atom("Q", (x,)), False)})
    renamed = rename_variables(clause)
    new_vars = set(term_variables(renamed))
    assert len(new_vars) == 2 and not new_vars & {x, y}
    # the same variable maps to the same fresh variable everywhere
    p = next(l for l in renamed if l.atom.pred == "P")
    q = next(l for l in renamed if l.atom.pred == "Q")
    assert p.atom.args[0] == q.atom.args[0]


def test_literal_and_clause_display():
    lit = Literal(parse_atom("Compromised(Host1)"))
    assert str(lit.negate()) == "~Compromised(Host1)"
    assert lit.negate().negate() == lit
    assert clause_str(frozenset()) == "[]"


_names = st.sampled_from(["Host1", "Host2", "DC"])
_vars = st.sampled_from(["x", "y", "z"])
_atoms = st.builds(
    lambda p, args: f"{p}({', '.join(args)})",
    st.sampled_from(["P", "Q", "Trusts"]),
    st.lists(st.one_of(_names, _vars), min_size=1, max_size=3),
)
_formulas = st.recursive(
    _atoms,
    lambda inner: st.one_of(
        st.builds(lambda a: f"~{a}", inner),
        st.builds(lambda a, b, op: f"({a} {op} {b})", inner, inner, st.sampled_from(["&", "|", "->", "<->"])),
        st.builds(lambda v, a, q: f"({q} {v} {a})", _vars, inner, st.sampled_from(["forall", "exists"])),
    ),
    max_leaves=8,
)


@given(_formulas)
def test_round_trip_property(text):
    f = parse(text)
    assert parse(str(f)) == f


# ----------------------------------------------------------------------------- trace


def test_trace_lineage_walks_back_to_detection():
    store = TraceStore()
    ml = store.record("ml", "attack on Host3", {"confidence": 0.92})
    hmm = store.record("hmm", "phase=Exploitation", parents=[ml])
    bn = store.record("bayes", "P(compromised)=0.81", parents=[ml])
    proof = store.record("proof", "Compromised(DC)", parents=[hmm, bn])
    act = store.record("action", "isolate(Host3)", parents=[proof])

    stages = [n.stage for n in store.lineage(act)]
    assert stages[0] == "action" and stages[1] == "proof"
    assert set(stages) == {"action", "proof", "hmm", "bayes", "ml"}
    assert stages.count("ml") == 1  # shared ancestor appears once
    assert store.get(ml).payload["confidence"] == 0.92
    assert [n.trace_id for n in store.by_stage("ml")] == [ml]


def test_trace_rejects_unknown_parent():
    with pytest.raises(KeyError):
        TraceStore().record("plan", "x", parents=["missing-1"])


# ----------------------------------------------------------------------------- graph builder


def test_edge_weight_is_exploit_difficulty():
    g = build_graph(
        {"A": {}, "B": {"cves": [{"id": "C1", "cvss": 9.8}, {"id": "C2", "cvss": 5.0}]},
         "C": {"cves": [{"id": "C3", "cvss": 10.0}]}, "D": {}},
        [("A", "B"), ("A", "C"), ("A", "D"), ("D", "E")],
    )
    g2 = build_graph({"E": {"cves": [{"id": "C4", "cvss": 6.5}]}}, [("D", "E")])
    assert g2["D"]["E"]["weight"] == pytest.approx(3.5)
    assert g["A"]["B"]["weight"] == MIN_WEIGHT  # 10 - 9.8 is floored at MIN_WEIGHT
    assert g["A"]["C"]["weight"] == MIN_WEIGHT
    assert g["A"]["D"]["weight"] == NO_CVE_WEIGHT


def test_from_adjacency_adds_neighbor_only_nodes():
    g = from_adjacency({"A": {"B": 1, "C": 2}})
    assert set(g.nodes) == {"A", "B", "C"}
    assert g["A"]["C"]["weight"] == 2.0


def test_example_and_generated_topologies():
    g = example_topology()
    assert set(critical_assets(g)) == {"DomainController", "Database"}

    gen = generate_topology(seed=7)
    assert gen.number_of_nodes() == 1 + 2 + 12 + 4 + 1 + 2
    for node in gen.nodes:
        assert node == "Internet" or __import__("networkx").has_path(gen, "Internet", node)
    assert list(generate_topology(seed=7).edges) == list(gen.edges)  # seeded


# ----------------------------------------------------------------------------- environment


def test_environment_runs_100_ticks_deterministically():
    def run():
        env = NetworkEnvironment(seed=3)
        obs = env.run(100)
        return env.ground_truth(), [len(o.flows) for o in obs], env.events

    first, second = run(), run()
    assert first == second
    truth = first[0]
    assert truth["tick"] == 100
    assert truth["phase"] in KILL_CHAIN


def test_attacker_progresses_without_defence():
    env = NetworkEnvironment(seed=1, attacker_success=1.0, phase_advance=1.0)
    env.run(30)
    assert env.breached
    assert env.state.phase == "Exfiltration"
    # cheapest route: Internet -> WebServer -> AppServer -> Database
    assert env.state.compromised == {"WebServer", "AppServer", "Database"}
    compromises = [e for e in env.events if e["event"] == "compromise"]
    assert [e["dst"] for e in compromises] == ["WebServer", "AppServer", "Database"]


def test_observation_features_and_hidden_label():
    env = NetworkEnvironment(seed=2, benign_flows=5)
    env.step()
    obs = env.observe()
    assert len(obs.flows) == 6
    assert sum(f.malicious for f in obs.flows) == 1
    matrix = obs.feature_matrix()
    assert matrix.shape == (6, len(FEATURES))
    assert (matrix >= 0).all()


def test_isolation_stops_lateral_movement():
    env = NetworkEnvironment(seed=0, p_fail=0.0, attacker_success=1.0, phase_advance=1.0)
    for host in ("WebServer", "MailServer"):
        assert env.apply(ActionCommand("isolate", host)).success
    env.run(30)
    assert env.state.compromised == set()
    assert not env.breached
    assert env.is_contained()


def test_blocking_edges_and_containment():
    env = NetworkEnvironment(seed=0, p_fail=0.0)
    env.state.compromised.add("Workstation2")
    assert not env.is_contained()
    env.apply(ActionCommand("block_edge", "Workstation2", "DomainController"))
    env.apply(ActionCommand("block_edge", "Workstation2", "FileServer"))
    assert env.is_contained()


def test_actions_can_fail_and_persistence_survives_kill():
    env = NetworkEnvironment(seed=0, p_fail=1.0)
    assert not env.apply(ActionCommand("isolate", "WebServer")).success
    assert "WebServer" not in env.state.isolated

    env = NetworkEnvironment(seed=0, p_fail=0.0)
    env.state.compromised |= {"Workstation1", "Workstation2"}
    env.state.persistent.add("Workstation2")
    assert env.apply(ActionCommand("kill_process", "Workstation1")).success
    assert not env.apply(ActionCommand("kill_process", "Workstation2")).success
    assert env.state.compromised == {"Workstation2"}


def test_apply_validates_and_returns_dry_run_command():
    env = NetworkEnvironment(seed=0, p_fail=0.0)
    result = env.apply(ActionCommand("block_edge", "Internet", "WebServer"))
    assert result.command == "iptables -I FORWARD -s Internet -d WebServer -j DROP"
    assert "iptables" in iptables_command(ActionCommand("isolate", "Host1"))
    with pytest.raises(ValueError):
        env.apply(ActionCommand("format_disk", "WebServer"))
    with pytest.raises(ValueError):
        env.apply(ActionCommand("isolate", "NoSuchHost"))
    with pytest.raises(ValueError):
        env.apply(ActionCommand("block_edge", "Internet", "Database"))
