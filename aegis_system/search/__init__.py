"""Search: attack paths, firewall configuration, and the attacker-defender game."""

from .adversarial_search import NetworkGame, alphabeta, best_defense, minimax, minimax_decision
from .graph_builder import build_graph, example_topology, generate_topology
from .local_search import FirewallProblem, hill_climb, simulated_annealing
from .path_search import (
    SearchResult, astar, astar_search, bfs, bfs_search, choke_edges, dfs, dfs_search, hop_heuristic, iddfs, ucs,
    verify_consistency,
)

__all__ = [
    "FirewallProblem", "NetworkGame", "SearchResult", "alphabeta", "astar", "astar_search", "best_defense", "bfs",
    "bfs_search", "build_graph", "choke_edges", "dfs", "dfs_search", "example_topology", "generate_topology",
    "hill_climb", "hop_heuristic", "iddfs", "minimax", "minimax_decision", "simulated_annealing", "ucs",
    "verify_consistency",
]
