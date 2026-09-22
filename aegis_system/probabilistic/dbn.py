"""Two-slice dynamic Bayesian network over boolean state variables.

    prior       a BayesNet over the state variables at t = 0
    transition  P(X_t | parents in slice t-1), one CPT per state variable
    sensor      P(E_t | parents in slice t), one CPT per evidence variable

Exact filtering keeps the belief as a joint distribution over all state
variables (2^n entries, fine for the handful of variables used here); the
particle filter is the scalable alternative. Evidence at a step may be partial:
evidence variables left out of the dict are treated as unobserved.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .bayes_net import BayesNet

Belief = np.ndarray  # shape (2,) * n_state, indexed by the state variables in order


@dataclass
class _CPT:
    parents: Tuple[str, ...]
    p_true: np.ndarray  # P(var = True | parents), one axis per parent


class DBN:
    def __init__(self, prior: BayesNet):
        self.prior = prior
        self.state_vars: List[str] = list(prior.order)
        self.transition: Dict[str, _CPT] = {}
        self.sensor: Dict[str, _CPT] = {}
        self._states = list(itertools.product((0, 1), repeat=len(self.state_vars)))
        self._T: Optional[np.ndarray] = None

    # ----------------------------------------------------------------- building

    def add_transition(self, var: str, prev_parents: Sequence[str], p_true) -> None:
        """P(var_t = True | prev_parents at t-1), indexed by parent values (False=0, True=1)."""
        self._add(self.transition, var, self.state_vars, prev_parents, p_true)
        self._T = None

    def add_sensor(self, evidence_var: str, parents: Sequence[str], p_true) -> None:
        """P(evidence_var_t = True | parents at t)."""
        if evidence_var in self.state_vars:
            raise ValueError(f"{evidence_var!r} is a state variable")
        self._add(self.sensor, evidence_var, None, parents, p_true)

    def _add(self, table, var, allowed, parents, p_true) -> None:
        if allowed is not None and var not in allowed:
            raise ValueError(f"{var!r} is not a state variable")
        for p in parents:
            if p not in self.state_vars:
                raise ValueError(f"Parent {p!r} is not a state variable")
        p = np.asarray(p_true, dtype=float)
        if p.shape != (2,) * len(parents) or np.any((p < 0) | (p > 1)):
            raise ValueError(f"CPT for {var!r} must have shape {(2,) * len(parents)} with entries in [0, 1]")
        table[var] = _CPT(tuple(parents), p)

    def _check_complete(self) -> None:
        missing = [v for v in self.state_vars if v not in self.transition]
        if missing:
            raise ValueError(f"No transition model for {missing}")

    def _p(self, cpt: _CPT, value: int, state: Tuple[int, ...]) -> float:
        p = float(cpt.p_true[tuple(state[self.state_vars.index(q)] for q in cpt.parents)])
        return p if value else 1.0 - p

    # ----------------------------------------------------------------- exact filtering

    def initial_belief(self) -> Belief:
        b = np.array([self.prior.joint_probability(dict(zip(self.state_vars, map(bool, s)))) for s in self._states])
        return b.reshape((2,) * len(self.state_vars))

    def transition_matrix(self) -> np.ndarray:
        """T[i, j] = P(state j at t | state i at t-1), over the joint states in product order."""
        self._check_complete()
        if self._T is None:
            S = len(self._states)
            T = np.ones((S, S))
            for i, prev in enumerate(self._states):
                for j, cur in enumerate(self._states):
                    for k, var in enumerate(self.state_vars):
                        T[i, j] *= self._p(self.transition[var], cur[k], prev)
            self._T = T
        return self._T

    def sensor_likelihood(self, evidence: Mapping[str, bool]) -> np.ndarray:
        """P(evidence | state) for every joint state."""
        like = np.ones(len(self._states))
        for var, value in evidence.items():
            if var not in self.sensor:
                raise KeyError(f"No sensor model for {var!r}")
            for j, s in enumerate(self._states):
                like[j] *= self._p(self.sensor[var], int(bool(value)), s)
        return like

    def filter_step(self, belief: Belief, evidence: Mapping[str, bool], predict: bool = True) -> Belief:
        """One step of exact forward inference: predict through the transition model, then weight by evidence.

        Use predict=False for the first slice, whose belief is the prior itself.
        """
        b = belief.reshape(-1)
        if predict:
            b = b @ self.transition_matrix()
        b = b * self.sensor_likelihood(evidence)
        total = b.sum()
        if total <= 0:
            raise ZeroDivisionError("evidence has probability 0 under the model")
        return (b / total).reshape(belief.shape)

    def filter(self, evidence_seq: Sequence[Mapping[str, bool]]) -> List[Belief]:
        """Beliefs after each step; the first evidence applies to slice 0."""
        beliefs: List[Belief] = []
        b = self.initial_belief()
        for t, e in enumerate(evidence_seq):
            b = self.filter_step(b, e, predict=t > 0)
            beliefs.append(b)
        return beliefs

    def marginal(self, belief: Belief, var: str) -> float:
        """P(var = True) under a joint belief."""
        k = self.state_vars.index(var)
        axes = tuple(i for i in range(len(self.state_vars)) if i != k)
        return float(belief.sum(axis=axes)[1])

    # ----------------------------------------------------------------- particle filter

    def particle_filter(
        self, evidence_seq: Sequence[Mapping[str, bool]], n: int = 1000, seed: Optional[int] = 0
    ) -> List[Dict[str, float]]:
        """Sequential importance resampling. Returns P(var = True) estimates per step."""
        self._check_complete()
        rng = np.random.default_rng(seed)
        nv = len(self.state_vars)
        particles = np.zeros((n, nv), dtype=int)
        for i in range(n):  # sample slice 0 from the prior network
            for k, var in enumerate(self.state_vars):
                node = self.prior.nodes[var]
                row = node.cpt[tuple(particles[i, self.state_vars.index(q)] for q in node.parents)]
                particles[i, k] = rng.random() < row[1]
        estimates: List[Dict[str, float]] = []
        for t, e in enumerate(evidence_seq):
            if t > 0:
                new = np.zeros_like(particles)
                for k, var in enumerate(self.state_vars):
                    cpt = self.transition[var]
                    idx = tuple(particles[:, self.state_vars.index(q)] for q in cpt.parents)
                    new[:, k] = rng.random(n) < cpt.p_true[idx]
                particles = new
            w = np.ones(n)
            for var, value in e.items():
                cpt = self.sensor[var]
                p = cpt.p_true[tuple(particles[:, self.state_vars.index(q)] for q in cpt.parents)]
                w *= p if value else 1.0 - p
            if w.sum() <= 0:
                w = np.ones(n)  # every particle contradicts the evidence: keep them rather than fail
            w = w / w.sum()
            estimates.append({var: float(w @ particles[:, k]) for k, var in enumerate(self.state_vars)})
            particles = particles[rng.choice(n, size=n, p=w)]
        return estimates


# --------------------------------------------------------------------------- demonstration


def apt_dbn(alert_rate: float = 0.1, false_alarm: float = 0.02) -> DBN:
    """Foothold (persistent APT presence) and Exfiltrating, observed through Alert and Spike sensors.

    Foothold is sticky: once present it almost never disappears on its own.
    alert_rate is how often the attacker's activity raises an alert per tick, so a
    slow-and-low attacker acting every k ticks has alert_rate about 1/k.
    """
    prior = BayesNet()
    prior.add_boolean("Foothold", [], 0.01)
    prior.add_boolean("Exfiltrating", ["Foothold"], [0.0, 0.05])
    dbn = DBN(prior)
    dbn.add_transition("Foothold", ["Foothold"], [0.002, 0.999])
    dbn.add_transition("Exfiltrating", ["Foothold", "Exfiltrating"], [[0.0, 0.1], [0.02, 0.7]])
    dbn.add_sensor("Alert", ["Foothold"], [false_alarm, false_alarm + (1 - false_alarm) * alert_rate])
    dbn.add_sensor("Spike", ["Exfiltrating"], [0.05, 0.8])
    return dbn


def slow_and_low_evidence(ticks: int = 300, every: int = 10, start: int = 20, false_alarm: float = 0.02,
                          seed: Optional[int] = 0) -> List[Dict[str, bool]]:
    """Alerts from an attacker who arrives at `start` and then acts only once every `every` ticks."""
    rng = np.random.default_rng(seed)
    seq = []
    for t in range(ticks):
        attacker_acts = t >= start and (t - start) % every == 0
        seq.append({"Alert": bool(attacker_acts or rng.random() < false_alarm)})
    return seq


def sliding_window_detector(evidence_seq: Sequence[Mapping[str, bool]], window: int = 10, threshold: int = 3
                            ) -> List[bool]:
    """Fires at t when at least `threshold` alerts fall in the last `window` ticks."""
    alerts = [bool(e.get("Alert", False)) for e in evidence_seq]
    return [sum(alerts[max(0, t - window + 1):t + 1]) >= threshold for t in range(len(alerts))]


def slow_and_low_demo(ticks: int = 300, every: int = 10, seed: Optional[int] = 0) -> Dict[str, List]:
    """DBN belief in Foothold versus a threshold detector on the same slow-and-low alert stream."""
    evidence = slow_and_low_evidence(ticks, every, seed=seed)
    dbn = apt_dbn(alert_rate=1.0 / every)
    belief = [dbn.marginal(b, "Foothold") for b in dbn.filter(evidence)]
    return {"evidence": evidence, "belief": belief, "detector": sliding_window_detector(evidence)}
