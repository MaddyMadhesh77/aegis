"""A.E.G.I.S. closed-loop demo: python -m aegis_system.main --scenario solarwinds

Runs the sense-think-plan-act loop against the simulated network. Every trace
entry is printed as it is recorded (detections, beliefs, grounding, proofs,
defence objectives, plans, actions), and the run is saved to runs/ for the
dashboard (streamlit run aegis_system/dashboard/app.py).

    --agent goal_based|utility|model_based|reflex   (default goal_based, the full pipeline)
    --compare                                         run every agent on several seeds, print the PEAS table
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from .agents import AGENTS
from .loop import compare_agents, run_episode
from .simulation.scenarios import available_scenarios, load_scenario

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
STAGE_WIDTH = 9


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aegis_system.main", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="solarwinds", choices=available_scenarios())
    parser.add_argument("--agent", default="goal_based", choices=sorted(AGENTS))
    parser.add_argument("--ticks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RUNS_DIR, help="where to save the run for the dashboard")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    parser.add_argument("--compare", action="store_true", help="compare all agents on seeds 0..4")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning)

    spec = load_scenario(args.scenario, args.seed)
    print(f"A.E.G.I.S. | {spec.title} | agent={args.agent} | seed={args.seed} | {args.ticks} ticks")
    print(f"targets: {', '.join(spec.targets)}\n")

    if args.compare:
        rows = compare_agents(args.scenario, range(5), args.ticks)
        cols = ["agent", "contained", "breaches", "mean_time_to_containment", "mean_uptime",
                "mean_false_isolations", "mean_proof_depth", "mean_actions"]
        print("  ".join(f"{c:>24}" if i else f"{c:<12}" for i, c in enumerate(cols)))
        for r in rows:
            cells = [f"{r['agent']:<12}"] + [f"{_fmt(r[c]):>24}" for c in cols[1:]]
            print("  ".join(cells))
        return 0

    def show(tick, stage, summary):
        if not args.quiet:
            print(f"[t={tick:02d}] {stage:<{STAGE_WIDTH}} {summary}")

    result = run_episode(args.agent, args.scenario, args.ticks, args.seed, on_trace=show)
    path = result.save(args.out)
    m = result.metrics
    print("\nPEAS performance measures")
    for key in ("breached", "contained_at_end", "time_to_containment", "uptime", "false_isolations",
                "isolations", "blocked_edges", "proof_depth", "compromised_hosts"):
        print(f"  {key:<20} {_fmt(m[key])}")
    print(f"\nrun saved to {path}")
    print("dashboard: streamlit run aegis_system/dashboard/app.py")
    return 0


def _fmt(v) -> str:
    if v is None:
        return "never"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


if __name__ == "__main__":
    sys.exit(main())
