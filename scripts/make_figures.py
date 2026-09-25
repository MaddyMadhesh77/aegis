"""Regenerate the report figures and tables for Phases 2-4 (search, probabilistic inference, planning).

    python scripts/make_figures.py            # writes into figures/
    python scripts/make_figures.py --out DIR

Every random choice uses a fixed seed, so the outputs are reproducible.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis_system.planning.domain import defense_problem  # noqa: E402
from aegis_system.planning.partial_order import PartialOrderPlanner, format_layers, layers  # noqa: E402
from aegis_system.planning.strips_planner import compare_planners  # noqa: E402
from aegis_system.probabilistic.bayes_net import attack_chain_network  # noqa: E402
from aegis_system.probabilistic.dbn import slow_and_low_demo  # noqa: E402
from aegis_system.probabilistic.hmm_engine import KILL_CHAIN, kill_chain_hmm  # noqa: E402
from aegis_system.search.adversarial_search import NetworkGame, node_count_experiment  # noqa: E402
from aegis_system.search.graph_builder import critical_assets, example_topology, generate_topology  # noqa: E402
from aegis_system.search.local_search import FirewallProblem, hill_climb, simulated_annealing  # noqa: E402
from aegis_system.search.path_search import compare_algorithms  # noqa: E402

SEED = 0
REQUIRED_FLOWS = [("Internet", "WebServer"), ("Internet", "MailServer"), ("WebServer", "AppServer"),
                  ("AppServer", "Database"), ("Workstation1", "FileServer")]


def write_csv(path: Path, rows, fields) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def search_figures(out: Path) -> None:
    g = generate_topology(seed=SEED)
    rows = compare_algorithms(g, "Internet", critical_assets(g))
    write_csv(out / "path_search_comparison.csv", rows, ["algorithm", "cost", "hops", "nodes_expanded"])

    game = NetworkGame(example_topology())
    counts = node_count_experiment(game, game.initial_state(), range(1, 8))
    write_csv(out / "alphabeta_node_counts.csv", counts,
              ["depth", "minimax", "alphabeta", "alphabeta_ordered", "value", "values_agree"])
    fig, ax = plt.subplots(figsize=(6, 4))
    depths = [r["depth"] for r in counts]
    for key, label in (("minimax", "Minimax"), ("alphabeta", "Alpha-beta"),
                       ("alphabeta_ordered", "Alpha-beta, ordered moves")):
        ax.plot(depths, [r[key] for r in counts], marker="o", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Search depth (plies)")
    ax.set_ylabel("Nodes visited (log scale)")
    ax.set_title("Pruning in the attacker-defender game")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "alphabeta_node_counts.png", dpi=150)
    plt.close(fig)

    problem = FirewallProblem(example_topology(), required_flows=REQUIRED_FLOWS)
    hc = hill_climb(problem, restarts=10, seed=SEED)
    sa = simulated_annealing(problem, seed=SEED)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(hc.history, label="current")
    a1.plot(hc.best_history, label="best so far")
    a1.set_title(f"Hill climbing, 10 restarts (best {hc.cost:.2f})")
    a2.plot(sa.history, label="current", alpha=0.7)
    a2.plot(sa.best_history, label="best so far")
    a2.set_title(f"Simulated annealing (best {sa.cost:.2f})")
    for a in (a1, a2):
        a.set_xlabel("Iteration")
        a.set_ylabel("Cost (risk + flow penalty)")
        a.legend()
    fig.tight_layout()
    fig.savefig(out / "local_search_objective.png", dpi=150)
    plt.close(fig)


def probabilistic_figures(out: Path) -> None:
    net = attack_chain_network()
    e = {"PortScan": True, "PrivEsc": True}
    exact = net.variable_elimination("Exfiltration", e)[True]
    ns = [100, 300, 1000, 3000, 10000]
    lw = [net.likelihood_weighting("Exfiltration", e, n, seed=SEED)[True] for n in ns]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(exact, color="k", linestyle="--", label=f"Variable elimination ({exact:.3f})")
    ax.plot(ns, lw, marker="o", label="Likelihood weighting")
    ax.set_xscale("log")
    ax.set_xlabel("Samples")
    ax.set_ylabel("P(Exfiltration | PortScan, PrivEsc)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bayes_net_convergence.png", dpi=150)
    plt.close(fig)

    demo = slow_and_low_demo(seed=SEED)
    t = np.arange(len(demo["belief"]))
    alerts = [i for i, ev in enumerate(demo["evidence"]) if ev["Alert"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, demo["belief"], label="DBN P(Foothold | alerts)")
    ax.plot(t, np.array(demo["detector"], dtype=float), label="Sliding-window detector (3 alerts / 10 ticks)")
    ax.vlines(alerts, 0, 0.08, color="grey", label="Alert")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Probability / detector output")
    ax.set_title("Slow-and-low APT: one action every 10 ticks from tick 20")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(out / "dbn_slow_and_low.png", dpi=150)
    plt.close(fig)

    hmm = kill_chain_hmm()
    states, obs = hmm.sample(40, seed=SEED)
    gamma = hmm.smooth(obs)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stackplot(np.arange(len(obs)), gamma.T, labels=KILL_CHAIN)
    ax.plot([KILL_CHAIN.index(s) / (len(KILL_CHAIN) - 1) for s in states], "k--", lw=1, label="true phase (scaled)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("P(phase | all alerts)")
    ax.set_title("Kill-chain phase posterior (forward-backward)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "hmm_phase_posterior.png", dpi=150)
    plt.close(fig)


PLANNING_GOALS = {
    "contain": "MemoryCaptured(Workstation1) & Isolated(Workstation1) & Blocked(MailServer, Workstation2)",
    "clean": "Clean(Workstation1) & Blocked(MailServer, Workstation2)",
    "recover": "Restored(Workstation1) & Patched(FileServer) & Isolated(Workstation2)",
}


def planning_tables(out: Path) -> None:
    g = example_topology()
    rows = []
    for name, goal in PLANNING_GOALS.items():
        problem = defense_problem(g, ["Workstation1", "Workstation2"], goal)
        for row in compare_planners(problem, max_nodes=50_000):
            rows.append({"goal": name, **row})
        pop = PartialOrderPlanner(problem).plan()
        rows.append({"goal": name, "search": "pop", "heuristic": "", "cost": "",
                     "length": sum(len(layer) for layer in layers(pop.plan)),
                     "nodes_expanded": pop.stats.nodes, "layers": format_layers(layers(pop.plan))})
    write_csv(out / "planner_comparison.csv", rows,
              ["goal", "search", "heuristic", "cost", "length", "nodes_expanded", "layers"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    search_figures(args.out)
    probabilistic_figures(args.out)
    planning_tables(args.out)
    for f in sorted(args.out.iterdir()):
        print(f)


if __name__ == "__main__":
    main()
