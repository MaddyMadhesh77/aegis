from __future__ import annotations

from aegis_system.perception.classifier import AttackClassifier
from aegis_system.perception.regressor import TimeToCompromiseRegressor
from aegis_system.planning.executor import ExecutionAgent
from aegis_system.planning.strips_planner import STRIPSPlanner, Action
from aegis_system.probabilistic.hmm_engine import HMM
from aegis_system.search.graph_builder import build_graph
from aegis_system.search.path_search import bfs_search


class AEGISSystem:
    def __init__(self):
        self.classifier = AttackClassifier()
        self.regressor = TimeToCompromiseRegressor()
        self.planner = STRIPSPlanner([
            Action("scan", {"connected"}, {"vulnerable"}, set()),
            Action("isolate", {"vulnerable"}, {"contained"}, {"vulnerable"}),
        ])
        self.executor = ExecutionAgent()
        self.hmm = HMM(
            states=["Recon", "Exploit", "Exfiltration"],
            observations=[0, 1],
            start_probs={"Recon": 0.6, "Exploit": 0.2, "Exfiltration": 0.2},
            trans_probs={
                "Recon": {"Recon": 0.7, "Exploit": 0.2, "Exfiltration": 0.1},
                "Exploit": {"Recon": 0.1, "Exploit": 0.7, "Exfiltration": 0.2},
                "Exfiltration": {"Recon": 0.05, "Exploit": 0.15, "Exfiltration": 0.8},
            },
            emit_probs={
                "Recon": {0: 0.8, 1: 0.2},
                "Exploit": {0: 0.3, 1: 0.7},
                "Exfiltration": {0: 0.2, 1: 0.8},
            },
        )

    def process_telemetry(self, samples, labels):
        self.classifier.fit(samples, labels)
        return self.classifier.predict(samples)

    def estimate_ttc(self, X, y):
        self.regressor.fit(X, y)
        return self.regressor.predict(X)

    def find_path(self, graph, start, target):
        return bfs_search(build_graph(graph), start, target)

    def plan_response(self, initial_state, goal_state):
        return self.planner.plan(initial_state, goal_state)

    def execute_response(self, action_name, context):
        return self.executor.execute_action(action_name, context)


def main() -> None:
    system = AEGISSystem()
    print("A.E.G.I.S. system initialized.")
    print(system.hmm.viterbi([0, 1, 1]))


if __name__ == "__main__":
    main()
