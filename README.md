# A.E.G.I.S.

A.E.G.I.S. (Autonomous Expert for Grounded Inference and Security) is a neuro-symbolic cyber-defence agent. It detects attacks in network flows, tracks the attacker's kill-chain phase and which hosts are compromised, turns detections into logic, proves what is compromised, plans a containment response, and carries it out on a simulated network. Every action can be traced back to the detection behind it.

Everything runs against a **simulated** network: no real traffic is read and no firewall is changed. Actions print the `iptables` command a real deployment would run.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_nsl_kdd.py   # NSL-KDD (~25 MB) into data/, needed for the perception report
pytest -q                         # ~250 tests, about 1-2 minutes
```

The OWL reasoner (HermiT) needs Java. Without Java the ontology falls back to an equivalent Python reasoner automatically.

## Run the demo

```bash
python -m aegis_system.main --scenario solarwinds              # full pipeline, one line per trace entry
python -m aegis_system.main --scenario solarwinds --compare    # all four agents, PEAS table over 5 seeds
streamlit run aegis_system/dashboard/app.py                    # network, phase timeline, click-to-explain
```

Options: `--agent goal_based|utility|model_based|reflex`, `--scenario solarwinds|random`, `--ticks`, `--seed`.
Runs are saved to `runs/` and can be replayed in the dashboard.

The SolarWinds scenario supplies the network (Orion server, admin workstation, domain controller, Exchange) and the critical targets. The attack is played by the simulator's standard kill-chain attacker; the scenario file's `steps` describe the real campaign for reference and are not executed.

## Report figures

```bash
python scripts/make_figures.py        # search, probabilistic inference, planning   -> figures/
python scripts/perception_report.py   # model comparison, curves, FGSM, TTC        -> figures/perception/
```

Both use fixed seeds, so the figures are reproducible.

## Layout

| Package | Contents |
|---|---|
| `core/` | Shared logic terms and formulas, records passed between stages, the trace store, the simulated network |
| `reasoning/` | Unification, CNF, resolution refutation, forward/backward chaining, DPLL, the FOL knowledge base |
| `search/` | Attack graph, BFS/DFS/IDDFS/UCS/A* with a consistent heuristic, firewall local search, minimax and alpha-beta |
| `probabilistic/` | Bayesian network with variable elimination, two-slice DBN with particle filter, kill-chain HMM |
| `planning/` | STRIPS progression/regression with h_max/h_add, partial-order planner, monitored executor with replanning |
| `knowledge/` | ATT&CK semantic net, frames with demons, production system, OWL ontology, grounding of alerts into atoms |
| `perception/` | NSL-KDD loader, model zoo with 1-SE selection, rule extraction, time-to-compromise regression, FGSM |
| `agents.py`, `loop.py`, `main.py` | Reflex, model-based, goal-based and utility agents; the closed loop and PEAS metrics; the CLI |
| `dashboard/` | Streamlit app and the figures it draws |

## Things to state in the report

- "Provably sound" applies to the logic layer only: resolution is sound, the ML and probabilistic layers are not.
- The adversarial game is treated as perfect-information (the defender sees the attacker's position).
- Time-to-compromise targets come from a documented formula plus noise; no public TTC dataset exists.
- The simulated benign traffic includes 5% look-alike flows (scans, admin sessions, backups); without them the detector is near-perfect and the agent comparison is uninformative.
