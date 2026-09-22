"""Hidden Markov model of the attacker's kill-chain phase.

Parameters are numpy arrays: pi[i] = P(X_0 = i), A[i, j] = P(X_t+1 = j | X_t = i),
B[i, k] = P(E_t = k | X_t = i). Observations may be given as symbols or indices.

The forward and backward passes are scaled: each alpha_t is normalized and the
normalizer c_t is kept, so a sequence of any length stays in floating-point range
and log P(e_1:T) = sum_t log c_t. Viterbi works in log space for the same reason.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

KILL_CHAIN = ("Recon", "Weaponization", "Delivery", "Exploitation", "C2", "Exfiltration")
# Alert classes produced by the perception layer (the NSL-KDD attack categories).
ALERT_CLASSES = ("normal", "probe", "dos", "r2l", "u2r")

Symbol = Hashable


class HMM:
    def __init__(
        self,
        states: Sequence[str],
        observations: Sequence[Symbol],
        start_probs: Union[Mapping[str, float], Sequence[float], np.ndarray],
        trans_probs: Union[Mapping[str, Mapping[str, float]], np.ndarray],
        emit_probs: Union[Mapping[str, Mapping[Symbol, float]], np.ndarray],
    ):
        self.states = list(states)
        self.observations = list(observations)
        n, m = len(self.states), len(self.observations)
        self.pi = self._vector(start_probs, self.states)
        self.A = self._matrix(trans_probs, self.states, self.states)
        self.B = self._matrix(emit_probs, self.states, self.observations)
        if self.pi.shape != (n,) or self.A.shape != (n, n) or self.B.shape != (n, m):
            raise ValueError("parameter shapes do not match the states and observations")
        for name, arr in (("start", self.pi), ("transition", self.A), ("emission", self.B)):
            if np.any(arr < 0) or not np.allclose(arr.sum(axis=-1), 1.0):
                raise ValueError(f"{name} probabilities must be non-negative and sum to 1 per row")
        self._obs_index = {o: k for k, o in enumerate(self.observations)}

    @staticmethod
    def _vector(p, keys) -> np.ndarray:
        if isinstance(p, Mapping):
            return np.array([p.get(k, 0.0) for k in keys], dtype=float)
        return np.asarray(p, dtype=float)

    @staticmethod
    def _matrix(p, rows, cols) -> np.ndarray:
        if isinstance(p, Mapping):
            return np.array([[p.get(r, {}).get(c, 0.0) for c in cols] for r in rows], dtype=float)
        return np.asarray(p, dtype=float)

    def encode(self, obs: Sequence[Symbol]) -> np.ndarray:
        """Observation symbols to indices. Integers already in range pass through when they are not symbols."""
        out = np.empty(len(obs), dtype=int)
        for t, o in enumerate(obs):
            if o in self._obs_index:
                out[t] = self._obs_index[o]
            elif isinstance(o, (int, np.integer)) and 0 <= o < len(self.observations):
                out[t] = int(o)
            else:
                raise ValueError(f"Unknown observation {o!r}")
        return out

    # ----------------------------------------------------------------- inference

    def forward(self, obs: Sequence[Symbol]) -> Tuple[np.ndarray, np.ndarray]:
        """Scaled forward pass. Returns (alpha_hat, c): alpha_hat[t] = P(X_t | e_1:t), c[t] = P(e_t | e_1:t-1)."""
        o = self.encode(obs)
        T, n = len(o), len(self.states)
        alpha = np.zeros((T, n))
        c = np.zeros(T)
        for t in range(T):
            a = (self.pi if t == 0 else alpha[t - 1] @ self.A) * self.B[:, o[t]]
            c[t] = a.sum()
            if c[t] <= 0:
                raise ZeroDivisionError(f"observation {obs[t]!r} at t={t} has probability 0 under the model")
            alpha[t] = a / c[t]
        return alpha, c

    def backward(self, obs: Sequence[Symbol], c: Optional[np.ndarray] = None) -> np.ndarray:
        """Scaled backward pass, using the forward scale factors c."""
        o = self.encode(obs)
        if c is None:
            _, c = self.forward(obs)
        T, n = len(o), len(self.states)
        beta = np.zeros((T, n))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, o[t + 1]] * beta[t + 1])) / c[t + 1]
        return beta

    def step(self, belief: Optional[np.ndarray], observation: Symbol) -> np.ndarray:
        """One online filtering step: P(X_t | e_1:t) from P(X_t-1 | e_1:t-1); belief None means t = 0."""
        k = int(self.encode([observation])[0])
        b = (self.pi if belief is None else np.asarray(belief) @ self.A) * self.B[:, k]
        total = b.sum()
        if total <= 0:
            raise ZeroDivisionError(f"observation {observation!r} has probability 0 under the model")
        return b / total

    def filter(self, obs: Sequence[Symbol]) -> np.ndarray:
        """P(X_t | e_1:t) for every t."""
        return self.forward(obs)[0]

    def smooth(self, obs: Sequence[Symbol]) -> np.ndarray:
        """Forward-backward: gamma[t] = P(X_t | e_1:T). Each row sums to 1."""
        alpha, c = self.forward(obs)
        gamma = alpha * self.backward(obs, c)
        return gamma / gamma.sum(axis=1, keepdims=True)

    def log_likelihood(self, obs: Sequence[Symbol]) -> float:
        return float(np.log(self.forward(obs)[1]).sum())

    def viterbi(self, obs: Sequence[Symbol]) -> List[str]:
        """Most likely state sequence, computed in log space."""
        o = self.encode(obs)
        with np.errstate(divide="ignore"):
            log_pi, log_A, log_B = np.log(self.pi), np.log(self.A), np.log(self.B)
        T, n = len(o), len(self.states)
        delta = log_pi + log_B[:, o[0]]
        back = np.zeros((T, n), dtype=int)
        for t in range(1, T):
            scores = delta[:, None] + log_A  # scores[i, j]: best path ending i, then i -> j
            back[t] = scores.argmax(axis=0)
            delta = scores.max(axis=0) + log_B[:, o[t]]
        path = [int(delta.argmax())]
        for t in range(T - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return [self.states[i] for i in reversed(path)]

    def posterior_dicts(self, matrix: np.ndarray) -> List[Dict[str, float]]:
        return [{s: float(p) for s, p in zip(self.states, row)} for row in matrix]

    # ----------------------------------------------------------------- simulation and learning

    def sample(self, T: int, seed: Optional[int] = 0) -> Tuple[List[str], List[Symbol]]:
        rng = np.random.default_rng(seed)
        states, obs = [], []
        x = rng.choice(len(self.states), p=self.pi)
        for t in range(T):
            if t > 0:
                x = rng.choice(len(self.states), p=self.A[x])
            states.append(self.states[x])
            obs.append(self.observations[rng.choice(len(self.observations), p=self.B[x])])
        return states, obs

    def baum_welch(
        self, sequences: Sequence[Sequence[Symbol]], n_iter: int = 50, tol: float = 1e-6, pseudocount: float = 1e-3
    ) -> List[float]:
        """Fit pi, A and B to observation sequences by expectation-maximization (in place).

        Returns the total log-likelihood before each iteration; it never decreases
        (up to the small pseudocount, which keeps unseen events from becoming impossible).
        Zero entries in A stay zero, so a left-to-right structure is preserved.
        """
        mask = self.A > 0
        history: List[float] = []
        for _ in range(n_iter):
            pi_num = np.zeros_like(self.pi)
            A_num = np.zeros_like(self.A)
            B_num = np.zeros_like(self.B)
            total_ll = 0.0
            for seq in sequences:
                o = self.encode(seq)
                alpha, c = self.forward(seq)
                beta = self.backward(seq, c)
                total_ll += float(np.log(c).sum())
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)
                pi_num += gamma[0]
                for t in range(len(o) - 1):
                    xi = alpha[t][:, None] * self.A * (self.B[:, o[t + 1]] * beta[t + 1])[None, :] / c[t + 1]
                    A_num += xi
                for k in range(len(self.observations)):
                    B_num[:, k] += gamma[o == k].sum(axis=0)
            history.append(total_ll)
            self.pi = (pi_num + pseudocount) / (pi_num + pseudocount).sum()
            A_num = np.where(mask, A_num + pseudocount, 0.0)
            self.A = A_num / A_num.sum(axis=1, keepdims=True)
            self.B = (B_num + pseudocount) / (B_num + pseudocount).sum(axis=1, keepdims=True)
            if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
                break
        return history


def kill_chain_hmm() -> HMM:
    """Six-phase kill-chain model observed through the perception layer's alert classes.

    Left-to-right: each phase persists or advances one step. Emissions are
    hand-set (to be re-estimated with baum_welch on simulated alert streams):
    reconnaissance looks like probing, delivery and exploitation like remote-to-local
    and privilege-escalation attacks, C2 is mostly quiet, exfiltration is bulk traffic.
    """
    n = len(KILL_CHAIN)
    A = np.zeros((n, n))
    stay = [0.80, 0.70, 0.75, 0.75, 0.85, 0.90]
    for i in range(n):
        A[i, i] = stay[i]
        if i + 1 < n:
            A[i, i + 1] = 1 - stay[i]
        else:
            A[i, i] = 1.0
    #             normal probe  dos   r2l   u2r
    B = np.array([[0.30, 0.60, 0.05, 0.04, 0.01],   # Recon
                  [0.80, 0.10, 0.02, 0.06, 0.02],   # Weaponization (mostly off-network)
                  [0.35, 0.05, 0.05, 0.50, 0.05],   # Delivery
                  [0.20, 0.05, 0.05, 0.30, 0.40],   # Exploitation
                  [0.60, 0.05, 0.05, 0.25, 0.05],   # C2 (low and slow)
                  [0.25, 0.02, 0.38, 0.30, 0.05]])  # Exfiltration (bulk transfer looks like dos)
    pi = np.array([0.90, 0.05, 0.05, 0.0, 0.0, 0.0])
    return HMM(KILL_CHAIN, ALERT_CLASSES, pi, A, B)
