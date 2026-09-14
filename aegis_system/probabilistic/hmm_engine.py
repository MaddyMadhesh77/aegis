from __future__ import annotations

from typing import Dict, List, Sequence


class HMM:
    def __init__(
        self,
        states: Sequence[str],
        observations: Sequence[str],
        start_probs: Dict[str, float],
        trans_probs: Dict[str, Dict[str, float]],
        emit_probs: Dict[str, Dict[str, float]],
    ):
        self.states = list(states)
        self.observations = list(observations)
        self.start_probs = start_probs
        self.trans_probs = trans_probs
        self.emit_probs = emit_probs

    def forward(self, observation_sequence: Sequence[str]) -> List[Dict[str, float]]:
        alpha = []
        for t, obs in enumerate(observation_sequence):
            current = {}
            for state in self.states:
                if t == 0:
                    prob = self.start_probs.get(state, 0.0) * self.emit_probs[state].get(obs, 0.0)
                else:
                    prob = sum(
                        alpha[-1].get(prev_state, 0.0) * self.trans_probs.get(prev_state, {}).get(state, 0.0)
                        for prev_state in self.states
                    ) * self.emit_probs[state].get(obs, 0.0)
                current[state] = prob
            alpha.append(current)
        return alpha

    def viterbi(self, observation_sequence: Sequence[str]) -> List[str]:
        dp: List[Dict[str, float]] = []
        backpointer: List[Dict[str, str]] = []

        for t, obs in enumerate(observation_sequence):
            current = {}
            back = {}
            for state in self.states:
                if t == 0:
                    score = self.start_probs.get(state, 0.0) * self.emit_probs[state].get(obs, 0.0)
                    current[state] = score
                    back[state] = state
                else:
                    best_prev_state = max(
                        self.states,
                        key=lambda prev: dp[-1].get(prev, 0.0) * self.trans_probs.get(prev, {}).get(state, 0.0),
                    )
                    score = dp[-1].get(best_prev_state, 0.0) * self.trans_probs.get(best_prev_state, {}).get(state, 0.0)
                    score *= self.emit_probs[state].get(obs, 0.0)
                    current[state] = score
                    back[state] = best_prev_state
            dp.append(current)
            backpointer.append(back)

        last_state = max(self.states, key=lambda st: dp[-1].get(st, 0.0))
        path = [last_state]
        for t in range(len(observation_sequence) - 1, 0, -1):
            last_state = backpointer[t][last_state]
            path.append(last_state)
        path.reverse()
        return path
