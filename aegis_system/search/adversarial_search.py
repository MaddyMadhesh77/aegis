"""Attacker-defender game on the network, solved with minimax and alpha-beta.

The game is treated as perfect-information: the defender sees which hosts the
attacker holds (the design-fix choice; the HMM supplies that estimate in the
running system). Players alternate, attacker (Max) first.

    Attacker  compromises one host reachable over an open edge from a host it holds,
              or passes if it has no move.
    Defender  blocks one frontier edge, isolates one host, or does nothing.
              Each action has an operational cost.

Evaluation, from the attacker's point of view (Max):

    value of compromised hosts
    + THREAT_WEIGHT * value of hosts one open edge away from the attacker
    + defender's accumulated operational cost

The plan describes the score as "exposed value minus defender cost". That is the
defender's utility; in a zero-sum game with the attacker as Max, the defender's
spending must *raise* Max's score, otherwise blocking everything would be free.

Search depth counts plies (one move by one player).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from ..core.logic import Atom, Const
from ..core.types import DefenseObjective
from .graph_builder import critical_assets, max_cvss

Edge = Tuple[str, str]
THREAT_WEIGHT = 0.5
BLOCK_COST = 0.3            # cost of one firewall rule
ISOLATION_COST = 3.0        # multiplied by the host's criticality (downtime)


@dataclass(frozen=True)
class Move:
    kind: str                  # attack | block | isolate | pass
    target: str = ""
    source: str = ""

    def __str__(self) -> str:
        if self.kind in ("attack", "block"):
            return f"{self.kind}({self.source}, {self.target})"
        if self.kind == "isolate":
            return f"isolate({self.target})"
        return "pass"


@dataclass(frozen=True)
class GameState:
    attacker: FrozenSet[str]
    blocked: FrozenSet[Edge] = frozenset()
    isolated: FrozenSet[str] = frozenset()
    defender_cost: float = 0.0
    attacker_to_move: bool = True


@dataclass
class SearchStats:
    nodes: int = 0


class NetworkGame:
    def __init__(self, graph: nx.DiGraph, entry: str = "Internet", protected: Optional[Sequence[str]] = None):
        self.graph = graph
        self.entry = entry
        self.protected = set(protected) if protected is not None else set(critical_assets(graph))

    def initial_state(self, compromised: Iterable[str] = ()) -> GameState:
        return GameState(frozenset({self.entry, *compromised}))

    # ------------------------------------------------------------------ rules

    def frontier(self, s: GameState) -> List[Edge]:
        """Open edges from a live attacker host to a host the attacker does not hold."""
        edges = []
        for u in s.attacker:
            if u in s.isolated:
                continue
            for v in self.graph.successors(u):
                if v not in s.attacker and v not in s.isolated and (u, v) not in s.blocked:
                    edges.append((u, v))
        return sorted(edges)

    def moves(self, s: GameState) -> List[Move]:
        if s.attacker_to_move:
            targets: Dict[str, str] = {}
            for u, v in self.frontier(s):
                targets.setdefault(v, u)  # one move per target host; the source does not matter
            return [Move("attack", v, u) for v, u in sorted(targets.items())] or [Move("pass")]
        moves = [Move("pass")]
        frontier = self.frontier(s)
        moves += [Move("block", v, u) for u, v in frontier]
        threatened = sorted({v for _, v in frontier})
        holders = sorted(h for h in s.attacker if h != self.entry and h not in s.isolated)
        moves += [Move("isolate", h) for h in threatened + holders]
        return moves

    def result(self, s: GameState, m: Move) -> GameState:
        if m.kind == "attack":
            return GameState(s.attacker | {m.target}, s.blocked, s.isolated, s.defender_cost, False)
        if m.kind == "block":
            return GameState(s.attacker, s.blocked | {(m.source, m.target)}, s.isolated,
                             s.defender_cost + BLOCK_COST, True)
        if m.kind == "isolate":
            cost = ISOLATION_COST * self.graph.nodes[m.target].get("criticality", 0.0)
            return GameState(s.attacker, s.blocked, s.isolated | {m.target}, s.defender_cost + cost, True)
        return GameState(s.attacker, s.blocked, s.isolated, s.defender_cost, not s.attacker_to_move)

    def evaluate(self, s: GameState) -> float:
        nodes = self.graph.nodes
        held = sum(nodes[h].get("value", 0.0) for h in s.attacker if h != self.entry)
        threat = sum(nodes[v].get("value", 0.0) for v in {v for _, v in self.frontier(s)})
        return held + THREAT_WEIGHT * threat + s.defender_cost

    def is_terminal(self, s: GameState) -> bool:
        return bool(self.protected) and self.protected <= s.attacker

    # ------------------------------------------------------------------ ordering

    def ordered_moves(self, s: GameState) -> List[Move]:
        """Most promising moves first: high-CVSS targets and admin-credential hosts.

        For the attacker these are the best attacks; for the defender, blocking or
        isolating them is the best defence. Pass goes last for both.
        """
        nodes = self.graph.nodes

        def key(m: Move) -> Tuple:
            if m.kind == "pass":
                return (1, 0.0, 0.0, str(m))
            attrs = nodes[m.target]
            return (0, -float(attrs.get("admin_credentials", False)) - attrs.get("value", 0.0) / 10,
                    -max_cvss(attrs), str(m))

        return sorted(self.moves(s), key=key)


# --------------------------------------------------------------------------- search


def minimax(game: NetworkGame, state: GameState, depth: int, stats: Optional[SearchStats] = None
            ) -> Tuple[float, Optional[Move]]:
    """Plain minimax to `depth` plies. Returns (value, best move)."""
    stats = stats if stats is not None else SearchStats()
    stats.nodes += 1
    if depth == 0 or game.is_terminal(state):
        return game.evaluate(state), None
    maximizing = state.attacker_to_move
    best_value, best_move = (-math.inf if maximizing else math.inf), None
    for m in game.moves(state):
        value, _ = minimax(game, game.result(state, m), depth - 1, stats)
        if (maximizing and value > best_value) or (not maximizing and value < best_value):
            best_value, best_move = value, m
    return best_value, best_move


def alphabeta(
    game: NetworkGame,
    state: GameState,
    depth: int,
    alpha: float = -math.inf,
    beta: float = math.inf,
    order_moves: bool = True,
    stats: Optional[SearchStats] = None,
) -> Tuple[float, Optional[Move]]:
    """Minimax with alpha-beta pruning. Same value as minimax; fewer nodes with good ordering."""
    stats = stats if stats is not None else SearchStats()
    stats.nodes += 1
    if depth == 0 or game.is_terminal(state):
        return game.evaluate(state), None
    moves = game.ordered_moves(state) if order_moves else game.moves(state)
    best_move = None
    if state.attacker_to_move:
        value = -math.inf
        for m in moves:
            child, _ = alphabeta(game, game.result(state, m), depth - 1, alpha, beta, order_moves, stats)
            if child > value:
                value, best_move = child, m
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # beta cut-off: Min will never allow this branch
    else:
        value = math.inf
        for m in moves:
            child, _ = alphabeta(game, game.result(state, m), depth - 1, alpha, beta, order_moves, stats)
            if child < value:
                value, best_move = child, m
            beta = min(beta, value)
            if alpha >= beta:
                break  # alpha cut-off
    return value, best_move


def node_count_experiment(game: NetworkGame, state: GameState, depths: Iterable[int]) -> List[Dict[str, float]]:
    """Nodes visited by minimax, alpha-beta, and alpha-beta with move ordering, per depth."""
    rows = []
    for d in depths:
        counts = {}
        values = {}
        for name, run in (
            ("minimax", lambda st: minimax(game, state, d, st)),
            ("alphabeta", lambda st: alphabeta(game, state, d, order_moves=False, stats=st)),
            ("alphabeta_ordered", lambda st: alphabeta(game, state, d, order_moves=True, stats=st)),
        ):
            stats = SearchStats()
            values[name], _ = run(stats)
            counts[name] = stats.nodes
        rows.append({"depth": d, **counts, "value": values["minimax"],
                     "values_agree": len({round(v, 9) for v in values.values()}) == 1})
    return rows


def best_defense(game: NetworkGame, compromised: Iterable[str], depth: int = 4) -> DefenseObjective:
    """The defender's best immediate move when the attacker holds `compromised`.

    The search starts with the defender to move. The result names the move as the
    choke point and states its effect as goal atoms for the planner.
    """
    state = game.initial_state(compromised)
    state = GameState(state.attacker, attacker_to_move=False)
    value, move = alphabeta(game, state, depth)
    atoms = set()
    if move is not None and move.kind == "block":
        atoms.add(Atom("Blocked", (Const(move.source), Const(move.target))))
    elif move is not None and move.kind == "isolate":
        atoms.add(Atom("Isolated", (Const(move.target),)))
    return DefenseObjective(goal_atoms=atoms, choke_point=move, minimax_value=value)


def minimax_decision(values: Dict[str, float], maximizing: bool = True) -> str:
    """One-ply decision over precomputed action values (kept for older callers)."""
    if not values:
        raise ValueError("No values provided for minimax decision")
    pick = max if maximizing else min
    return pick(values, key=lambda key: values[key])
