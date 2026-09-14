# A.E.G.I.S.

A.E.G.I.S. (Autonomous Expert for Grounded Inference and Security) is a neuro-symbolic cyber defense architecture that combines statistical detection, probabilistic reasoning, symbolic proof, adversarial search, and planning.

## Architecture

- Perception: classification, regression, and adversarial tests
- Probabilistic reasoning: Bayesian networks and HMM
- Knowledge grounding: ontologies, semantic networks, frames
- Symbolic reasoning: FOL, unification, resolution
- Search: graph traversal, A*, minimax
- Planning: STRIPS and partial-order planning
- Interface: dashboard and policy loops

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Module roadmap

- `aegis_system/perception/`
- `aegis_system/probabilistic/`
- `aegis_system/knowledge/`
- `aegis_system/reasoning/`
- `aegis_system/search/`
- `aegis_system/planning/`
- `aegis_system/dashboard/`
- `aegis_system/main.py`

## Deliverable milestones

1. Core algorithms and tests
2. Network topology and graph search
3. Probabilistic inference engines
4. Symbolic FOL and resolution integration
5. Planner and execution loop
6. Dashboard and system orchestration
