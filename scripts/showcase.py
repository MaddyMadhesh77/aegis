"""Walk through every A.E.G.I.S. feature, phase by phase, printing what each one produces.

    python scripts/showcase.py              # all phases (about 1-2 minutes)
    python scripts/showcase.py --phase 5    # one phase
    python scripts/showcase.py --phase 1 4  # several phases

Everything is seeded, so the output is the same on every run.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def sub(title: str) -> None:
    print(f"\n--- {title}")


def table(rows, cols, widths=None) -> None:
    widths = widths or [max(len(str(c)), *(len(_fmt(r.get(c))) for r in rows)) for c in cols]
    print("  ".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(f"{_fmt(r.get(c)):<{w}}" for c, w in zip(cols, widths)))


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "inf" if v == float("inf") else f"{v:.3f}"
    return str(v)


# --------------------------------------------------------------------------- phases


def phase0() -> None:
    header("Phase 0: shared logic, provenance and the simulated network")
    from aegis_system.core.environment import ActionCommand, NetworkEnvironment
    from aegis_system.core.logic import parse

    text = "forall x forall y (Compromised(x) & Trusts(y, x) -> Compromised(y))"
    f = parse(text)
    print(f"parsed:     {text}\nprinted:    {f}\nround trip: {parse(str(f)) == f}")
    env = NetworkEnvironment(seed=0)
    env.run(100)
    print(f"\n100 simulated ticks with seed 0 -> {env.ground_truth()}")
    result = env.apply(ActionCommand("isolate", "WebServer"))
    print(f"isolate(WebServer) -> success={result.success}; nothing is executed, the command is returned:\n"
          f"  {result.command}")


def phase1() -> None:
    header("Phase 1: reasoning (unification, CNF, resolution, chaining, DPLL)")
    from aegis_system.core.logic import Fn, Var, parse
    from aegis_system.reasoning import demo_kb, dpll_entails, format_proof, tt_entails, unify
    from aegis_system.reasoning.cnf import to_cnf

    sub("Unification")
    print("unify(Compromised(x), Compromised(Host1)) =",
          {str(k): str(v) for k, v in unify(parse("Compromised(x)"), parse("Compromised(Host1)")).items()})
    print("unify(x, f(x))  (occurs check)          =", unify(Var("x"), Fn("f", (Var("x"),))))
    sub("Clause form with Skolemization")
    print("exists m (Malware(m) & RunsOn(m, Host1))  ->",
          [" | ".join(map(str, c)) for c in to_cnf(parse("exists m (Malware(m) & RunsOn(m, Host1))"))])
    sub("Resolution refutation of Compromised(DC)")
    kb = demo_kb()
    print(format_proof(kb.ask("Compromised(DC)")))
    for method in ("forward", "backward"):
        print(f"{method:>8} chaining proves Compromised(DC): {kb.ask('Compromised(DC)', method).proved}")
    print("answer to Compromised(w):", [str(a[Var('w')]) for a in kb.ask_all("Compromised(w)")])
    sub("Propositional alert correlation")
    alerts = [parse(s) for s in ("PortScan & FailedLogins -> BruteForce", "BruteForce & NewAdminAccount -> PrivEsc",
                                 "PortScan", "FailedLogins", "NewAdminAccount")]
    print(f"KB |= PrivEsc     truth table: {tt_entails(alerts, parse('PrivEsc'))}   DPLL: "
          f"{dpll_entails(alerts, parse('PrivEsc'))}")
    print(f"KB |= LateralMove truth table: {tt_entails(alerts, parse('LateralMove'))}  DPLL: "
          f"{dpll_entails(alerts, parse('LateralMove'))}")


def phase2() -> None:
    header("Phase 2: search (paths, heuristics, firewall local search, alpha-beta)")
    from aegis_system.search.adversarial_search import NetworkGame, best_defense, node_count_experiment
    from aegis_system.search.graph_builder import critical_assets, example_topology, generate_topology
    from aegis_system.search.local_search import FirewallProblem, hill_climb, simulated_annealing
    from aegis_system.search.path_search import (
        choke_edges, compare_algorithms, cvss_hop_heuristic, hop_heuristic, verify_consistency,
    )

    g = generate_topology(seed=0)
    targets = critical_assets(g)
    sub(f"Path search from the Internet to {targets} ({g.number_of_nodes()} hosts, {g.number_of_edges()} links)")
    table(compare_algorithms(g, "Internet", targets), ["algorithm", "cost", "hops", "nodes_expanded"])
    print(f"\nA* heuristic w_min * hops is consistent: {verify_consistency(g, hop_heuristic(g, targets), targets)}")
    try:
        verify_consistency(example_topology(), cvss_hop_heuristic(example_topology(), critical_assets(example_topology())),
                           critical_assets(example_topology()))
    except AssertionError as e:
        print(f"original CVSS x hops heuristic is NOT: {e}")
    ex = example_topology()
    print(f"choke edges on the example network: {sorted(choke_edges(ex, 'Internet', critical_assets(ex)))}")
    sub("Alpha-beta pruning (nodes visited per search depth)")
    game = NetworkGame(ex)
    table(node_count_experiment(game, game.initial_state(), range(2, 8)),
          ["depth", "minimax", "alphabeta", "alphabeta_ordered", "value", "values_agree"])
    d = best_defense(game, ["MailServer"], depth=4)
    print(f"best defence with MailServer compromised: {d.choke_point} (goal {sorted(map(str, d.goal_atoms))})")
    sub("Firewall configuration by local search")
    flows = [("Internet", "WebServer"), ("Internet", "MailServer"), ("WebServer", "AppServer"),
             ("AppServer", "Database"), ("Workstation1", "FileServer")]
    p = FirewallProblem(ex, required_flows=flows)
    hc, sa = hill_climb(p, restarts=5, seed=0), simulated_annealing(p, seed=0)
    print(f"cost with every link open: {p.cost(p.all_open()):.2f}")
    print(f"hill climbing:       {hc.cost:.2f}  closes {sorted(set(p.edges) - hc.state)}")
    print(f"simulated annealing: {sa.cost:.2f}  closes {sorted(set(p.edges) - sa.state)}")


def phase3() -> None:
    header("Phase 3: probabilistic inference (Bayes net, DBN, HMM)")
    import numpy as np

    from aegis_system.probabilistic.bayes_net import attack_chain_network
    from aegis_system.probabilistic.dbn import slow_and_low_demo
    from aegis_system.probabilistic.hmm_engine import kill_chain_hmm

    net = attack_chain_network()
    e = {"PortScan": True, "PrivEsc": True}
    sub("Bayesian network: P(Exfiltration | PortScan, PrivEsc)")
    print(f"enumeration           {net.enumerate_ask('Exfiltration', e)[True]:.6f}")
    print(f"variable elimination  {net.variable_elimination('Exfiltration', e)[True]:.6f}")
    print(f"likelihood weighting  {net.likelihood_weighting('Exfiltration', e, 5000, seed=0)[True]:.6f}  (5000 samples)")
    print(f"explaining away: P(Phishing | VulnExploit) = {net.variable_elimination('PhishingEmail', {'VulnExploit': True})[True]:.3f}"
          f" -> {net.variable_elimination('PhishingEmail', {'VulnExploit': True, 'PortScan': True})[True]:.3f} once a port scan is seen")
    sub("Kill-chain HMM on 10,000 observations")
    hmm = kill_chain_hmm()
    states, obs = hmm.sample(10_000, seed=1)
    t = time.time()
    gamma = hmm.smooth(obs)
    print(f"log-likelihood {hmm.log_likelihood(obs):.1f} (no underflow), smoothing rows sum to 1: "
          f"{np.allclose(gamma.sum(axis=1), 1)}, {time.time() - t:.2f}s")
    # Exfiltration never ends in this model, so a long run is mostly easy exfiltration steps;
    # accuracy is measured on 200 short attacks instead.
    hits_s = hits_v = total = 0
    for seed in range(200):
        st, ob = hmm.sample(40, seed=seed)
        truth = np.array([hmm.states.index(x) for x in st])
        hits_s += int(np.sum(hmm.smooth(ob).argmax(axis=1) == truth))
        hits_v += int(np.sum(np.array([hmm.states.index(x) for x in hmm.viterbi(ob)]) == truth))
        total += len(st)
    print(f"phase recovered per step over 200 attacks of 40 steps: smoothing {hits_s / total:.1%}, "
          f"Viterbi {hits_v / total:.1%}")
    early_states, early_obs = hmm.sample(40, seed=3)
    smooth = [hmm.states[i] for i in hmm.smooth(early_obs).argmax(axis=1)]
    print("first 20 steps of a 40-step attack (true phase / smoothed estimate):")
    for t in range(0, 20, 4):
        print("  " + "   ".join(f"t={i:<2} {early_states[i]:<13}/ {smooth[i]:<13}" for i in range(t, t + 4)))
    sub("DBN against a slow-and-low attacker (one action every 10 ticks from tick 20)")
    demo = slow_and_low_demo()
    print("tick:            " + "  ".join(f"{t:>5}" for t in range(0, 300, 30)))
    print("P(Foothold):     " + "  ".join(f"{demo['belief'][t]:>5.2f}" for t in range(0, 300, 30)))
    print(f"threshold detector (3 alerts / 10 ticks) ever fired: {any(demo['detector'])}")


def phase4() -> None:
    header("Phase 4: planning and execution")
    from aegis_system.core.environment import NetworkEnvironment
    from aegis_system.planning.domain import defense_problem
    from aegis_system.planning.executor import (
        Executor, SensingAction, conditional_plan, contained_goal, format_conditional, unknown_state,
    )
    from aegis_system.planning.partial_order import PartialOrderPlanner, format_layers, layers
    from aegis_system.planning.strips_planner import STRIPSPlanner, compare_planners
    from aegis_system.search.graph_builder import example_topology

    g = example_topology()
    ws = ["Workstation1", "Workstation2"]
    goal = "Clean(Workstation1) & Blocked(MailServer, Workstation2)"
    p = defense_problem(g, ws, goal)
    sub(f"Goal: {goal}")
    table(compare_planners(p, max_nodes=50_000), ["search", "heuristic", "cost", "length", "nodes_expanded"])
    print("\nplan (regression + h_max):", [str(a) for a in STRIPSPlanner().plan(p)])
    print("same goal, 30 minutes to compromise:",
          [str(a) for a in STRIPSPlanner().plan(defense_problem(g, ws, goal, ttc_hours=0.5))])
    sub("Partial-order plan with parallel steps")
    pop = PartialOrderPlanner(defense_problem(
        g, ws, "MemoryCaptured(Workstation1) & Isolated(Workstation1) & Blocked(MailServer, Workstation2)")).plan()
    print(format_layers(layers(pop.plan)))
    sub("Execution with an injected failure (first Isolate fails, attacker keeps persistence on Workstation2)")

    class FailFirst(NetworkEnvironment):
        def apply(self, action):
            if action.name == "isolate" and not getattr(self, "_failed", False):
                self._failed, saved, self.p_fail = True, self.p_fail, 1.0
                try:
                    return super().apply(action)
                finally:
                    self.p_fail = saved
            return super().apply(action)

    env = FailFirst(g, seed=0, p_fail=0.0)
    env.state.compromised |= set(ws)
    env.state.persistent.add("Workstation2")
    report = Executor(env).execute(defense_problem(g, ws, "Isolated(Workstation1) & Clean(Workstation2)",
                                                   ttc_hours=0.5))
    print(f"success={report.success}  replans={report.replans}")
    for r in report.results:
        print(f"  {'ok  ' if r.success else 'FAIL'} {str(r.step):<28} {r.command}")
    for f in report.failures:
        print(f"  failure: {f}")
    sub("Conditional plan with a sensing action")
    cp = defense_problem(g, ["Workstation1"], "Isolated(Workstation1)")
    state = unknown_state(cp, ["Workstation1"], ["Workstation2"])
    plan = conditional_plan(cp, state, contained_goal(["Workstation1", "Workstation2"]),
                            [SensingAction("Workstation2")], relevant=lambda a: a.name == "Isolate")
    print(format_conditional(plan))


def phase5() -> None:
    header("Phase 5: knowledge representation and grounding")
    from aegis_system.knowledge.grounding import incident_demo
    from aegis_system.knowledge.ontology_manager import OntologyManager
    from aegis_system.knowledge.production_system import ProductionSystem, Rule
    from aegis_system.knowledge.semantic_net import SemanticNet
    from aegis_system.reasoning.resolution import format_proof
    from aegis_system.search.graph_builder import example_topology

    net = SemanticNet.from_attack_json()
    sub("Semantic net (ATT&CK subset)")
    print(f"T1003.001 {net.name('T1003.001')}: tactic {net.tactic_of('T1003.001')}, "
          f"requires {net.inherits('T1003.001', 'requires_privilege')} (inherited from T1003)")
    print(f"mitigations: {[f'{m} {net.name(m)}' for m in net.mitigations_for('T1003.001')]}")
    sub("Production system (salience, then specificity, then recency)")
    ps = ProductionSystem([Rule.make("general", ["Alert(h)"], asserts=["Seen(h)"]),
                           Rule.make("specific", ["Alert(h)", "Critical(h)"], asserts=["Escalate(h)"]),
                           Rule.make("urgent", ["Alert(h)"], asserts=["Page(h)"], salience=10)])
    for fact in ("Alert(A)", "Critical(A)", "Alert(B)"):
        ps.assert_fact(fact)
    for f in ps.run():
        print(f"  cycle {f.cycle}: {f.rule:<8} on {', '.join(map(str, f.facts)):<24} -> {', '.join(map(str, f.asserted))}")
    sub("OWL ontology")
    onto = OntologyManager()
    onto.load_topology(example_topology(), {"CVE-SIM-0101": ["T1190"]})
    print(f"reasoner: {onto.reason()}")
    print(f"ExposedHost (has a CVSS >= 9 vulnerability): {onto.exposed_hosts()}")
    print(f"lateralMoveTo* from MailServer (transitive):  {onto.reachable_from('MailServer')}")
    sub("Grounding: credential dumping on Host3 proves Compromised(DC)")
    demo = incident_demo()
    print(f"provable before the alert: {demo['proved_before']}")
    print(f"atoms from the alert: {sorted(map(str, demo['grounded'].atoms))}")
    print(format_proof(demo["proof"]))
    print("\ntrace of the proof (nearest first):")
    for n in demo["trace"].lineage(demo["proof"].trace_id):
        print(f"  {n.stage:<10} {n.summary[:90]}")


def phase6() -> None:
    header("Phase 6: perception and adversarial ML")
    from aegis_system.perception.adversarial_test import adversarial_training, robustness_curves
    from aegis_system.perception.classifier import (
        compare_models, model_zoo, one_standard_error_choice, tree_paths,
    )
    from aegis_system.perception.datasets import NSL_CATEGORICAL, NSL_MUTABLE, load_nsl_kdd, nsl_kdd_available
    from aegis_system.perception.regressor import compare_regressors, generate_ttc_dataset

    sub("Time-to-compromise regression (synthetic target, documented formula)")
    rows, _ = compare_regressors(*generate_ttc_dataset(2000, seed=0), seed=0)
    table(rows, ["model", "rmse", "mae", "r2"])
    print(f"Lasso sets to zero: {rows[1]['zeroed']}")
    if not nsl_kdd_available():
        print("\nNSL-KDD not downloaded: run `python scripts/fetch_nsl_kdd.py` for the classification part.")
        return
    sub("Model comparison on NSL-KDD (quick: 8,000 training rows, 4,000 test rows)")
    X, y = load_nsl_kdd(sample=8000)
    Xt, yt = load_nsl_kdd("test", sample=4000)
    rows, fitted = compare_models(X, y, Xt, yt, model_zoo(fast=True), categorical=NSL_CATEGORICAL, n_jobs=-1)
    table(rows, ["model", "cv_f1_macro", "test_f1_macro", "test_roc_auc_ovr", "complexity"])
    print(f"one-standard-error choice: {one_standard_error_choice(rows)}")
    print("\ntop rules extracted from the decision tree:")
    for r in sorted(tree_paths(fitted["decision_tree"], min_samples=50), key=lambda r: -r.samples)[:3]:
        print(f"  {r}")
    sub("FGSM against the MLP, before and after adversarial training")
    robust = adversarial_training(fitted["mlp"], X, y, NSL_MUTABLE, eps=1.0)
    curves = robustness_curves(fitted["mlp"], robust, Xt, yt, NSL_MUTABLE, [0.0, 0.5, 1.0, 2.0])
    print("epsilon:               " + "  ".join(f"{r['epsilon']:>5}" for r in curves["standard"]))
    for k, label in (("standard", "standard MLP:"), ("adversarially_trained", "adversarially trained:")):
        print(f"{label:<23}" + "  ".join(f"{r['accuracy']:>5.3f}" for r in curves[k]))
    print("(full-size version with plots: python scripts/perception_report.py)")


def phase7() -> None:
    header("Phase 7: agents, closed loop and dashboard")
    from aegis_system.core.trace import TraceStore
    from aegis_system.loop import compare_agents, run_episode

    sub("Goal-based agent (full pipeline) on the SolarWinds network, seed 0")
    run = run_episode("goal_based", "solarwinds", ticks=30, seed=0)
    for k, v in run.metrics.items():
        print(f"  {k:<20} {_fmt(v)}")
    store = TraceStore.from_dict(run.trace)
    action = next(a for row in run.timeline for a in row["actions"])
    print(f"\nwhy {action['action']}? (action first, detection last)")
    for n in store.lineage(action["trace_id"]):
        print(f"  {n.stage:<10} {n.summary[:90]}")
    sub("All four agents on the same incidents (seeds 0-2)")
    table(compare_agents("solarwinds", seeds=range(3), ticks=30),
          ["agent", "contained", "breaches", "mean_time_to_containment", "mean_uptime", "mean_false_isolations",
           "mean_proof_depth"])
    print("\nlive trace:  python -m aegis_system.main --scenario solarwinds")
    print("dashboard:   streamlit run aegis_system/dashboard/app.py")


PHASES = {0: phase0, 1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5, 6: phase6, 7: phase7}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", type=int, nargs="*", choices=sorted(PHASES), help="phases to show (default: all)")
    args = parser.parse_args()
    for n in args.phase or sorted(PHASES):
        start = time.time()
        PHASES[n]()
        print(f"\n[phase {n} done in {time.time() - start:.1f}s]")


if __name__ == "__main__":
    main()
