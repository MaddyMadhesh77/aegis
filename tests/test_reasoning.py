"""Phase 1: unification, clause form, resolution, chaining and propositional inference."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aegis_system.core.logic import (
    And, Atom, Const, Exists, Fn, ForAll, Literal, Not, Or, Var, parse, parse_atom,
)
from aegis_system.core.trace import TraceStore
from aegis_system.reasoning.chaining import Rule, as_definite, backward_chain, forward_chain
from aegis_system.reasoning.cnf import (
    distribute_or_over_and, drop_universals, eliminate_implications, is_tautology, move_not_inwards,
    skolemize, standardize_variables, to_cnf,
)
from aegis_system.reasoning.fol_engine import FOLKnowledgeBase, demo_kb
from aegis_system.reasoning.propositional import dpll, dpll_entails, dpll_satisfiable, tt_entails
from aegis_system.reasoning.resolution import factors, format_proof, refute, resolve, subsumes
from aegis_system.reasoning.unifier import compose, substitute, unify

x, y, z = Var("x"), Var("y"), Var("z")
A, B = Const("A"), Const("B")


def lit(text: str) -> Literal:
    text = text.strip()
    if text.startswith("~"):
        return Literal(parse_atom(text[1:]), False)
    return Literal(parse_atom(text), True)


def clause(*texts: str):
    return frozenset(lit(t) for t in texts)


# --------------------------------------------------------------------------- unifier


def test_unify_binds_variables_to_constants():
    assert unify(parse_atom("Compromised(x)"), parse_atom("Compromised(Host1)")) == {x: Const("Host1")}
    theta = unify(parse_atom("Trusts(x, y)"), parse_atom("Trusts(Host1, Host1)"))
    assert theta == {x: Const("Host1"), y: Const("Host1")}


def test_unify_fails_on_different_predicates_and_constants():
    assert unify(parse_atom("Compromised(x)"), parse_atom("Trusts(Host1)")) is None
    assert unify(parse_atom("Trusts(A, x)"), parse_atom("Trusts(B, x)")) is None
    assert unify(parse_atom("Trusts(x)"), parse_atom("Trusts(x, y)")) is None


def test_unify_occurs_check():
    assert unify(x, Fn("f", (x,))) is None
    assert unify(parse_atom("P(x, f(x))"), parse_atom("P(f(y), y)")) is None


def test_unify_follows_variable_chains():
    theta = unify(parse_atom("P(x, y, x)"), parse_atom("P(y, z, A)"))
    assert theta is not None
    assert substitute(x, theta) == substitute(y, theta) == substitute(z, theta) == A


def test_unify_does_not_modify_callers_substitution():
    theta = {x: A}
    result = unify(y, B, theta)
    assert theta == {x: A}
    assert result == {x: A, y: B}


def test_compose_applies_first_then_second():
    t1, t2 = {x: Fn("f", (y,))}, {y: A}
    for term in (x, y, Fn("g", (x, y))):
        assert substitute(term, compose(t1, t2)) == substitute(substitute(term, t1), t2)


_vars = st.sampled_from([Var("u"), Var("v"), Var("w")])
_consts = st.sampled_from([Const("A"), Const("B")])
_terms = st.recursive(
    st.one_of(_vars, _consts),
    lambda children: st.builds(lambda name, args: Fn(name, tuple(args)),
                               st.sampled_from(["f", "g"]), st.lists(children, min_size=1, max_size=2)),
    max_leaves=6,
)
_atoms = st.builds(lambda args: Atom("P", tuple(args)), st.lists(_terms, min_size=2, max_size=2))


@settings(max_examples=300, deadline=None)
@given(_atoms, _atoms)
def test_unifier_makes_both_sides_equal(a, b):
    theta = unify(a, b)
    if theta is not None:
        assert substitute(a, theta) == substitute(b, theta)


@settings(max_examples=200, deadline=None)
@given(_terms, _terms)
def test_unify_is_symmetric_in_success(a, b):
    assert (unify(a, b) is None) == (unify(b, a) is None)


# --------------------------------------------------------------------------- CNF passes


def test_eliminate_implications():
    f = eliminate_implications(parse("P -> Q"))
    assert f == Or(Not(Atom("P")), Atom("Q"))
    f = eliminate_implications(parse("P <-> Q"))
    assert f == And(Or(Not(Atom("P")), Atom("Q")), Or(Atom("P"), Not(Atom("Q"))))


def test_move_not_inwards_de_morgan_and_quantifiers():
    assert move_not_inwards(parse("~(P & Q)")) == Or(Not(Atom("P")), Not(Atom("Q")))
    assert move_not_inwards(parse("~(P | Q)")) == And(Not(Atom("P")), Not(Atom("Q")))
    assert move_not_inwards(parse("~~P")) == Atom("P")
    assert move_not_inwards(parse("~forall x P(x)")) == Exists(x, Not(parse_atom("P(x)")))
    assert move_not_inwards(parse("~exists x P(x)")) == ForAll(x, Not(parse_atom("P(x)")))


def test_standardize_variables_gives_each_quantifier_its_own_variable():
    f = standardize_variables(parse("(forall x P(x)) & (forall x Q(x))"))
    assert isinstance(f, And)
    assert f.left.var != f.right.var
    assert f.left.body.args == (f.left.var,)
    assert f.right.body.args == (f.right.var,)


def test_skolemize_uses_constant_outside_universals_and_function_inside():
    f = skolemize(parse("exists x P(x)"))
    assert isinstance(f, Atom) and isinstance(f.args[0], Const) and f.args[0].name.startswith("Sk")

    f = skolemize(parse("forall y exists x Owns(y, x)"))
    body = f.body
    skolem = body.args[1]
    assert isinstance(skolem, Fn) and skolem.name.startswith("Sk") and skolem.args == (y,)


def test_drop_universals():
    assert drop_universals(parse("forall x forall y R(x, y)")) == parse_atom("R(x, y)")
    with pytest.raises(TypeError):
        drop_universals(parse("exists x P(x)"))


def test_distribute_or_over_and():
    f = distribute_or_over_and(parse("(P & Q) | R"))
    assert f == And(Or(Atom("P"), Atom("R")), Or(Atom("Q"), Atom("R")))


def test_to_cnf_full_pipeline():
    clauses = to_cnf(parse("forall x (Compromised(x) -> exists c (Stolen(c, x)))"))
    assert len(clauses) == 1
    (c,) = clauses
    neg = [l for l in c if not l.positive]
    pos = [l for l in c if l.positive]
    assert neg[0].atom.pred == "Compromised" and pos[0].atom.pred == "Stolen"
    skolem = pos[0].atom.args[0]
    assert isinstance(skolem, Fn) and skolem.args == (neg[0].atom.args[0],)


def test_to_cnf_drops_tautologies():
    assert to_cnf(parse("P | ~P")) == []
    assert is_tautology(clause("P(x)", "~P(x)"))


# --------------------------------------------------------------------------- resolution


def test_resolve_complementary_literals():
    result = resolve(clause("Compromised(Host1)"), clause("~Compromised(x)", "Suspicious(x)"))
    assert result == [clause("Suspicious(Host1)")]
    assert resolve(clause("P(A)"), clause("~P(B)")) == []
    assert resolve(clause("P(A)"), clause("~P(A)")) == [frozenset()]


def test_resolve_renames_clauses_apart():
    # Without renaming, x would have to be both A and B.
    result = resolve(clause("P(x, A)"), clause("~P(B, x)"))
    assert result == [frozenset()]


def test_factoring_merges_unifiable_literals():
    result = [c for c, _ in factors(clause("P(x)", "P(A)", "Q(x)"))]
    assert clause("P(A)", "Q(A)") in result


def test_subsumption():
    assert subsumes(clause("P(x)"), clause("P(A)", "Q(B)"))
    assert not subsumes(clause("P(A)"), clause("P(x)"))
    assert not subsumes(clause("P(x)", "Q(x)"), clause("P(A)", "Q(B)"))


def test_refutation_proves_compromised_dc():
    kb = demo_kb()
    proof = kb.ask("Compromised(DC)", method="resolution")
    assert proof.proved
    assert proof.steps[-1].clause == frozenset()
    derived = [s for s in proof.steps if s.parents]
    assert derived and all(s.rule in ("resolution", "factoring") for s in derived)
    ids = {s.clause_id for s in proof.steps}
    assert all(p in ids for s in proof.steps for p in s.parents)


def test_refutation_proof_uses_skolem_constant():
    proof = demo_kb().ask("Compromised(DC)")
    skolem_inputs = [s for s in proof.steps if s.rule == "kb" and "Sk" in str(sorted(map(str, s.clause)))]
    assert skolem_inputs, format_proof(proof)


def test_refutation_answers_goal_with_variables():
    proof = demo_kb().ask("Compromised(w)")
    assert proof.proved
    assert {str(v): str(t) for v, t in proof.bindings.items()} in ({"w": "Host1"}, {"w": "DC"})


def test_refutation_fails_on_non_entailed_goal():
    kb = demo_kb()
    assert not kb.ask("Compromised(Kerberos)").proved
    assert not kb.ask("~Compromised(DC)").proved


def test_refutation_handles_disjunctive_knowledge():
    # Not provable by chaining: needs reasoning by cases.
    kb = FOLKnowledgeBase()
    kb.tell("Phished(Host1) | Exploited(Host1)")
    kb.tell("forall h (Phished(h) -> Compromised(h))")
    kb.tell("forall h (Exploited(h) -> Compromised(h))")
    assert kb.ask("Compromised(Host1)").proved
    assert not kb.ask("Compromised(Host1)", method="backward").proved


def test_refute_takes_raw_clauses():
    kb = [clause("~P(x)", "Q(x)"), clause("P(A)")]
    assert refute(kb, parse("Q(A)")).proved


# --------------------------------------------------------------------------- chaining


def _rules(*texts):
    kb = FOLKnowledgeBase()
    for t in texts:
        kb.tell(t)
    return kb.rules


def test_as_definite():
    rule = as_definite(clause("~P(x)", "~Q(x)", "R(x)"))
    assert rule.head == parse_atom("R(x)") and set(rule.body) == {parse_atom("P(x)"), parse_atom("Q(x)")}
    assert as_definite(clause("P(x)", "Q(x)")) is None
    assert as_definite(clause("~P(x)")) is None


def test_forward_chain_reaches_fixpoint():
    rules = _rules(
        "forall x forall y (Edge(x, y) -> Reach(x, y))",
        "forall x forall y forall z (Reach(x, y) & Edge(y, z) -> Reach(x, z))",
        "Edge(A, B)", "Edge(B, C)", "Edge(C, D)",
    )
    result = forward_chain(rules)
    reach = {(str(f.args[0]), str(f.args[1])) for f in result.facts if f.pred == "Reach"}
    assert reach == {("A", "B"), ("B", "C"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D")}
    # running again on the result adds nothing
    again = forward_chain(rules, result.facts)
    assert again.facts == result.facts


def test_forward_chain_derives_compromised_dc_with_support():
    kb = demo_kb()
    result = forward_chain(kb.rules)
    dc = parse_atom("Compromised(DC)")
    assert dc in result.facts
    support = result.support(dc)
    assert any(r.head.pred == "Malware" for r in support)


def test_backward_chain_proves_compromised_dc():
    kb = demo_kb()
    answers = list(backward_chain(kb.rules, parse_atom("Compromised(DC)")))
    assert answers and answers[0] == {}


def test_backward_chain_enumerates_answers():
    kb = demo_kb()
    hosts = {str(theta[Var("w")]) for theta in backward_chain(kb.rules, parse_atom("Compromised(w)"))}
    assert hosts == {"Host1", "DC"}


def test_backward_chain_depth_limit_stops_left_recursion():
    rules = _rules(
        "forall x forall z (Reach(x, z) & Edge(z, A) -> Reach(x, A))",  # left recursive
        "Edge(B, A)",
    )
    assert list(backward_chain(rules, parse_atom("Reach(B, A)"), depth_limit=10)) == []


def test_backward_chain_uses_fresh_variables_per_rule_use():
    rules = [
        Rule(parse_atom("Parent(A, B)")),
        Rule(parse_atom("Parent(B, C)")),
        Rule(parse_atom("Grand(x, z)"), (parse_atom("Parent(x, y)"), parse_atom("Parent(y, z)"))),
    ]
    answers = list(backward_chain(rules, parse_atom("Grand(A, w)")))
    assert answers == [{Var("w"): Const("C")}]


# --------------------------------------------------------------------------- propositional


ALERT_KB = [
    parse("PortScan & FailedLogins -> BruteForce"),
    parse("BruteForce & NewAdminAccount -> PrivEsc"),
    parse("PrivEsc | LateralMove -> Incident"),
    parse("PortScan"),
    parse("FailedLogins"),
    parse("NewAdminAccount"),
]


def test_tt_entails_alert_correlation():
    assert tt_entails(ALERT_KB, parse("Incident"))
    assert tt_entails(ALERT_KB, parse("PrivEsc"))
    assert not tt_entails(ALERT_KB, parse("LateralMove"))


def test_dpll_agrees_with_truth_table():
    for goal in ("Incident", "PrivEsc", "LateralMove", "~LateralMove", "BruteForce & ~Incident"):
        g = parse(goal)
        assert dpll_entails(ALERT_KB, g) == tt_entails(ALERT_KB, g), goal


def test_dpll_unit_propagation_and_unsat():
    assert dpll([clause("P"), clause("~P")]) is None
    model = dpll([clause("P"), clause("~P", "Q"), clause("~Q", "R")])
    assert model[Atom("P")] and model[Atom("Q")] and model[Atom("R")]


def test_dpll_returns_satisfying_model():
    f = parse("(A | B) & (~A | C) & (~B | ~C) & (A | ~C)")
    model = dpll_satisfiable(f)
    assert model is not None
    full = {s: model.get(s, False) for s in (Atom("A"), Atom("B"), Atom("C"))}
    from aegis_system.reasoning.propositional import pl_true
    assert pl_true(f, full)


def test_dpll_works_on_ground_atoms_with_arguments():
    kb = [parse("FailedLogin(Host3) & Mimikatz(Host3) -> CredDump(Host3)"),
          parse("FailedLogin(Host3)"), parse("Mimikatz(Host3)")]
    assert dpll_entails(kb, parse("CredDump(Host3)"))


_symbols = st.sampled_from([Atom("P"), Atom("Q"), Atom("R"), Atom("S")])
_props = st.recursive(
    _symbols,
    lambda c: st.one_of(st.builds(Not, c), st.builds(And, c, c), st.builds(Or, c, c)),
    max_leaves=8,
)


@settings(max_examples=150, deadline=None)
@given(st.lists(_props, min_size=1, max_size=3), _props)
def test_dpll_entailment_matches_truth_table(kb, alpha):
    assert dpll_entails(kb, alpha) == tt_entails(kb, alpha)


# --------------------------------------------------------------------------- knowledge base


def test_all_three_methods_prove_compromised_dc():
    kb = demo_kb()
    for method in ("resolution", "forward", "backward"):
        proof = kb.ask("Compromised(DC)", method=method)
        assert proof.proved, method
        assert proof.method == method


def test_ask_all_by_backward_chaining():
    answers = demo_kb().ask_all("Compromised(w)")
    assert {str(a[Var("w")]) for a in answers} == {"Host1", "DC"}


def test_ask_rejects_unknown_method_and_non_atomic_chaining_goals():
    kb = demo_kb()
    with pytest.raises(ValueError):
        kb.ask("Compromised(DC)", method="magic")
    with pytest.raises(ValueError):
        kb.ask("Compromised(DC) & Compromised(Host1)", method="backward")


def test_proof_trace_links_back_to_told_formulas():
    trace = TraceStore()
    alert = trace.record("ml", "credential dumping on Host1", {})
    kb = FOLKnowledgeBase(trace)
    kb.tell("forall x forall y (Compromised(x) & Trusts(y, x) -> Compromised(y))")
    kb.tell("Compromised(Host1)", trace_id=alert)
    kb.tell("Trusts(DC, Host1)")
    kb.tell("Trusts(Printer, Host2)")  # irrelevant to the proof

    proof = kb.ask("Compromised(DC)")
    assert proof.trace_id is not None
    lineage = trace.lineage(proof.trace_id)
    stages = [n.stage for n in lineage]
    assert stages[0] == "proof" and "ml" in stages
    summaries = " ".join(n.summary for n in lineage)
    assert "Printer" not in summaries
    assert "[]" in trace.get(proof.trace_id).payload["proof"]
