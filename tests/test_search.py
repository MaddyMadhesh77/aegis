"""Phase 2: path search, local search over firewall rules, and the attacker-defender game."""

import math

import networkx as nx
import pytest

from aegis_system.search.adversarial_search import (
    GameState, Move, NetworkGame, SearchStats, alphabeta, best_defense, minimax, node_count_experiment,
)
from aegis_system.search.graph_builder import critical_assets, example_topology, from_adjacency, generate_topology
from aegis_system.search.local_search import FirewallProblem, hill_climb, simulated_annealing
from aegis_system.search.path_search import (
    astar, bfs, choke_edges, compare_algorithms, cvss_hop_heuristic, dfs, hop_distances, hop_heuristic, iddfs,
    reconstruct, ucs, verify_consistency,
)

SMALL = from_adjacency({
    "A": {"B": 1, "C": 4},
    "B": {"C": 1, "D": 5},
    "C": {"D": 1},
    "D": {},
    "E": {"A": 1},  # cannot be reached from A
})


# --------------------------------------------------------------------------- path search


def test_reconstruct():
    assert reconstruct({"A": None, "B": "A", "C": "B"}, "C") == ["A", "B", "C"]


def test_every_search_returns_path_cost_and_expansions():
    for fn in (bfs, dfs, iddfs, ucs, astar):
        path, cost, expanded = fn(SMALL, "A", "D")
        assert path[0] == "A" and path[-1] == "D", fn.__name__
        assert cost == sum(SMALL[u][v]["weight"] for u, v in zip(path, path[1:]))
        assert expanded >= 1


def test_bfs_finds_fewest_hops_and_ucs_lowest_cost():
    assert len(bfs(SMALL, "A", "D").path) == 3
    assert ucs(SMALL, "A", "D") == (["A", "B", "C", "D"], 3.0, ucs(SMALL, "A", "D").nodes_expanded)


def test_iddfs_finds_shallowest_goal():
    assert len(iddfs(SMALL, "A", "D").path) == 3


def test_unreachable_goal():
    for fn in (bfs, dfs, iddfs, ucs, astar):
        r = fn(SMALL, "A", "E")
        assert r.path == [] and r.cost == math.inf, fn.__name__


def test_start_is_goal():
    for fn in (bfs, dfs, iddfs, ucs, astar):
        assert fn(SMALL, "A", "A").path == ["A"]


def test_multiple_goals_reach_the_nearest():
    assert ucs(SMALL, "A", {"C", "D"}).path == ["A", "B", "C"]


def test_dict_graphs_are_accepted():
    assert bfs({"A": {"B": 1}, "B": {}}, "A", "B").path == ["A", "B"]


def test_hop_distances_reverse_bfs():
    assert hop_distances(SMALL, "D") == {"D": 0, "C": 1, "B": 1, "A": 2, "E": 3}


@pytest.mark.parametrize("seed", range(5))
def test_astar_ucs_and_dijkstra_agree_on_generated_topologies(seed):
    g = generate_topology(seed=seed)
    targets = critical_assets(g)
    rows = {r["algorithm"]: r for r in compare_algorithms(g, "Internet", targets)}
    assert rows["astar"]["cost"] == pytest.approx(rows["networkx_dijkstra"]["cost"])
    assert rows["ucs"]["cost"] == pytest.approx(rows["networkx_dijkstra"]["cost"])
    assert rows["astar"]["nodes_expanded"] <= rows["ucs"]["nodes_expanded"]
    assert verify_consistency(g, hop_heuristic(g, targets), targets)


def test_astar_expands_fewer_nodes_than_ucs():
    g = generate_topology(seed=0)
    targets = critical_assets(g)
    assert astar(g, "Internet", targets).nodes_expanded < ucs(g, "Internet", targets).nodes_expanded


def test_original_cvss_heuristic_is_not_consistent():
    g = example_topology()
    targets = critical_assets(g)
    with pytest.raises(AssertionError):
        verify_consistency(g, cvss_hop_heuristic(g, targets), targets)


def test_choke_edges_disconnect_entry_from_targets():
    g = example_topology()
    targets = critical_assets(g)
    cut = choke_edges(g, "Internet", targets)
    assert cut
    h = g.copy()
    h.remove_edges_from(cut)
    assert not any(nx.has_path(h, "Internet", t) for t in targets)


# --------------------------------------------------------------------------- local search


FLOWS = [("Internet", "WebServer"), ("Internet", "MailServer"), ("WebServer", "AppServer"),
         ("AppServer", "Database"), ("Workstation1", "FileServer")]


def test_firewall_objective():
    p = FirewallProblem(example_topology(), required_flows=FLOWS)
    assert p.broken_flows(p.all_open()) == []
    closed = frozenset()
    assert p.cost(closed) == pytest.approx(p.flow_penalty * len(FLOWS))  # no risk, every flow broken
    assert all(0 <= v <= 1 for v in p.reach_probabilities(p.all_open()).values())


def test_hill_climb_improves_on_open_network_and_is_seeded():
    p = FirewallProblem(example_topology(), required_flows=FLOWS)
    r = hill_climb(p, restarts=3, seed=1)
    assert r.cost < p.cost(p.all_open())
    assert r.cost == pytest.approx(p.cost(r.state))
    assert r.best_history == sorted(r.best_history, reverse=True)
    assert hill_climb(p, restarts=3, seed=1).history == r.history


def test_simulated_annealing_improves_on_open_network():
    p = FirewallProblem(example_topology(), required_flows=FLOWS)
    r = simulated_annealing(p, steps=1500, seed=0)
    assert r.cost < p.cost(p.all_open())
    assert len(r.history) == len(r.best_history) > 0
    assert min(r.history) == pytest.approx(r.cost)


# --------------------------------------------------------------------------- adversarial search


@pytest.fixture
def game():
    return NetworkGame(example_topology())


def test_moves_and_results(game):
    s = game.initial_state()
    attacks = game.moves(s)
    assert {m.target for m in attacks} == {"WebServer", "MailServer"}
    s2 = game.result(s, attacks[0])
    assert attacks[0].target in s2.attacker and not s2.attacker_to_move
    defences = game.moves(s2)
    assert Move("pass") in defences
    assert any(m.kind == "block" for m in defences) and any(m.kind == "isolate" for m in defences)
    blocked = game.result(s2, next(m for m in defences if m.kind == "block"))
    assert blocked.defender_cost > 0 and blocked.attacker_to_move


def test_blocking_every_frontier_edge_leaves_attacker_only_pass(game):
    s = GameState(frozenset({"Internet"}), blocked=frozenset({("Internet", "WebServer"), ("Internet", "MailServer")}))
    assert game.moves(s) == [Move("pass")]


@pytest.mark.parametrize("depth", range(1, 6))
def test_alphabeta_matches_minimax(game, depth):
    s = game.initial_state()
    mm_value, _ = minimax(game, s, depth)
    for ordered in (False, True):
        ab_value, _ = alphabeta(game, s, depth, order_moves=ordered)
        assert ab_value == pytest.approx(mm_value)


def test_alphabeta_matches_minimax_mid_game():
    g = generate_topology(n_workstations=6, n_servers=2, seed=3)
    game = NetworkGame(g)
    s = game.initial_state(["DMZ1"])
    assert alphabeta(game, s, 4)[0] == pytest.approx(minimax(game, s, 4)[0])


def test_pruning_and_ordering_reduce_nodes(game):
    rows = node_count_experiment(game, game.initial_state(), [4, 5, 6])
    for r in rows:
        assert r["values_agree"]
        assert r["alphabeta"] <= r["minimax"]
        assert r["alphabeta_ordered"] <= r["alphabeta"]
    assert rows[-1]["alphabeta_ordered"] < rows[-1]["minimax"] / 2


def test_best_defense_returns_goal_atoms(game):
    objective = best_defense(game, ["MailServer"], depth=4)
    assert objective.choke_point is not None
    if objective.choke_point.kind in ("block", "isolate"):
        assert len(objective.goal_atoms) == 1
    assert objective.minimax_value == pytest.approx(
        alphabeta(game, GameState(frozenset({"Internet", "MailServer"}), attacker_to_move=False), 4)[0])


def test_search_stats_counts_nodes(game):
    stats = SearchStats()
    minimax(game, game.initial_state(), 2, stats)
    assert stats.nodes == 1 + 2 + sum(len(game.moves(game.result(game.initial_state(), m)))
                                      for m in game.moves(game.initial_state()))
