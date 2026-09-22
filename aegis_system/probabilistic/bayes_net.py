"""Discrete Bayesian networks on numpy factors.

A CPT is an array with one axis per parent (in the order given) and a last axis
for the node itself, so cpt[p1, p2, ..., x] = P(X = x | parents). Every row must
sum to 1. Boolean variables use the domain (False, True), so index 1 means True.

Inference:
    enumerate_ask          exact, sums the full joint (exponential; reference for tests)
    variable_elimination   exact, eliminates hidden variables one at a time
    likelihood_weighting   approximate, samples non-evidence variables
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

Value = Hashable
BOOL = (False, True)


@dataclass
class Factor:
    vars: Tuple[str, ...]
    table: np.ndarray  # one axis per variable, in the order of `vars`

    def __post_init__(self) -> None:
        if self.table.ndim != len(self.vars):
            raise ValueError(f"table has {self.table.ndim} axes for {len(self.vars)} variables")

    def multiply(self, other: "Factor") -> "Factor":
        out_vars = self.vars + tuple(v for v in other.vars if v not in self.vars)
        letters = {v: string.ascii_letters[i] for i, v in enumerate(out_vars)}
        spec = (f"{''.join(letters[v] for v in self.vars)},{''.join(letters[v] for v in other.vars)}"
                f"->{''.join(letters[v] for v in out_vars)}")
        return Factor(out_vars, np.einsum(spec, self.table, other.table))

    def sum_out(self, var: str) -> "Factor":
        i = self.vars.index(var)
        return Factor(self.vars[:i] + self.vars[i + 1:], self.table.sum(axis=i))

    def reduce(self, var: str, index: int) -> "Factor":
        """Fix var to the value at `index` and drop its axis."""
        if var not in self.vars:
            return self
        i = self.vars.index(var)
        return Factor(self.vars[:i] + self.vars[i + 1:], np.take(self.table, index, axis=i))

    def normalize(self) -> "Factor":
        total = self.table.sum()
        if total <= 0:
            raise ZeroDivisionError("factor sums to zero (evidence has probability 0)")
        return Factor(self.vars, self.table / total)

    def __mul__(self, other: "Factor") -> "Factor":
        return self.multiply(other)


@dataclass
class Node:
    name: str
    parents: Tuple[str, ...]
    cpt: np.ndarray
    domain: Tuple[Value, ...]


class BayesNet:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.order: List[str] = []  # topological: parents are always added first

    # ----------------------------------------------------------------- building

    def add_node(self, name: str, parents: Sequence[str], cpt, domain: Sequence[Value] = BOOL) -> None:
        """Add a node. Parents must already be in the network, which rules out cycles."""
        if name in self.nodes:
            raise ValueError(f"{name!r} is already in the network (adding it again could create a cycle)")
        for p in parents:
            if p not in self.nodes:
                raise ValueError(f"Parent {p!r} of {name!r} must be added first; cycles are not allowed")
        cpt = np.asarray(cpt, dtype=float)
        expected = tuple(len(self.nodes[p].domain) for p in parents) + (len(domain),)
        if cpt.shape != expected:
            raise ValueError(f"CPT for {name!r} has shape {cpt.shape}, expected {expected}")
        if np.any(cpt < 0) or not np.allclose(cpt.sum(axis=-1), 1.0):
            raise ValueError(f"Every row of the CPT for {name!r} must be a distribution summing to 1")
        self.nodes[name] = Node(name, tuple(parents), cpt, tuple(domain))
        self.order.append(name)

    def add_boolean(self, name: str, parents: Sequence[str], p_true) -> None:
        """Add a boolean node from P(name = True | parents), indexed by parent values (False=0, True=1)."""
        p = np.asarray(p_true, dtype=float)
        self.add_node(name, parents, np.stack([1.0 - p, p], axis=-1))

    def factor(self, name: str) -> Factor:
        n = self.nodes[name]
        return Factor(n.parents + (name,), n.cpt)

    def index(self, var: str, value: Value) -> int:
        try:
            return self.nodes[var].domain.index(value)
        except ValueError:
            raise ValueError(f"{value!r} is not in the domain of {var!r}: {self.nodes[var].domain}") from None

    def _distribution(self, var: str, probs: np.ndarray) -> Dict[Value, float]:
        return {v: float(p) for v, p in zip(self.nodes[var].domain, probs)}

    def _check(self, X: str, e: Mapping[str, Value]) -> None:
        for v in [X, *e]:
            if v not in self.nodes:
                raise KeyError(f"Unknown variable {v!r}")
        if X in e:
            raise ValueError(f"Query variable {X!r} is also evidence")

    # ----------------------------------------------------------------- exact: enumeration

    def joint_probability(self, assignment: Mapping[str, Value]) -> float:
        p = 1.0
        for name in self.order:
            n = self.nodes[name]
            idx = tuple(self.index(q, assignment[q]) for q in n.parents) + (self.index(name, assignment[name]),)
            p *= n.cpt[idx]
        return p

    def enumerate_ask(self, X: str, e: Optional[Mapping[str, Value]] = None) -> Dict[Value, float]:
        """P(X | e) by summing the full joint over every hidden variable."""
        e = dict(e or {})
        self._check(X, e)
        probs = np.array([self._enumerate_all(0, {**e, X: x}) for x in self.nodes[X].domain])
        if probs.sum() <= 0:
            raise ZeroDivisionError("evidence has probability 0")
        return self._distribution(X, probs / probs.sum())

    def _enumerate_all(self, i: int, assignment: Dict[str, Value]) -> float:
        if i == len(self.order):
            return 1.0
        name = self.order[i]
        n = self.nodes[name]
        parent_idx = tuple(self.index(q, assignment[q]) for q in n.parents)
        if name in assignment:
            return n.cpt[parent_idx + (self.index(name, assignment[name]),)] * self._enumerate_all(i + 1, assignment)
        return sum(n.cpt[parent_idx + (k,)] * self._enumerate_all(i + 1, {**assignment, name: v})
                   for k, v in enumerate(n.domain))

    # ----------------------------------------------------------------- exact: variable elimination

    def ancestors(self, variables: Sequence[str]) -> Set[str]:
        result: Set[str] = set()
        stack = list(variables)
        while stack:
            v = stack.pop()
            if v not in result:
                result.add(v)
                stack.extend(self.nodes[v].parents)
        return result

    def variable_elimination(
        self,
        X: str,
        e: Optional[Mapping[str, Value]] = None,
        order: Union[str, Sequence[str]] = "min-degree",
    ) -> Dict[Value, float]:
        """P(X | e) by variable elimination.

        Variables that are not ancestors of X or the evidence are dropped first:
        they sum to 1 and cannot affect the answer. `order` is "min-degree"
        (greedy: eliminate the variable whose new factor has the fewest variables),
        "topological", or an explicit list of the hidden variables.
        """
        e = dict(e or {})
        self._check(X, e)
        relevant = self.ancestors([X, *e])
        factors = []
        for name in self.order:
            if name in relevant:
                f = self.factor(name)
                for var, val in e.items():
                    f = f.reduce(var, self.index(var, val))
                factors.append(f)
        hidden = [v for v in self.order if v in relevant and v != X and v not in e]

        if order == "topological":
            elimination = hidden
        elif order == "min-degree":
            elimination = self._min_degree_order(hidden, factors)
        else:
            elimination = list(order)
            if set(elimination) != set(hidden):
                raise ValueError(f"Elimination order must list exactly the hidden variables {sorted(hidden)}")

        for var in elimination:
            related = [f for f in factors if var in f.vars]
            factors = [f for f in factors if var not in f.vars]
            product = related[0]
            for f in related[1:]:
                product = product * f
            factors.append(product.sum_out(var))

        result = factors[0]
        for f in factors[1:]:
            result = result * f
        return self._distribution(X, result.normalize().table)

    @staticmethod
    def _min_degree_order(hidden: Sequence[str], factors: Sequence[Factor]) -> List[str]:
        scopes = [set(f.vars) for f in factors]
        remaining = list(hidden)
        order: List[str] = []
        while remaining:
            def degree(v: str) -> Tuple[int, int]:
                merged = set().union(*(s for s in scopes if v in s))
                return (len(merged) - 1, remaining.index(v))
            var = min(remaining, key=degree)
            merged = set().union(*(s for s in scopes if var in s)) - {var}
            scopes = [s for s in scopes if var not in s] + [merged]
            remaining.remove(var)
            order.append(var)
        return order

    # ----------------------------------------------------------------- approximate

    def likelihood_weighting(
        self, X: str, e: Optional[Mapping[str, Value]] = None, n: int = 10_000, seed: Optional[int] = 0
    ) -> Dict[Value, float]:
        """P(X | e) estimated from n samples weighted by the likelihood of the evidence."""
        e = dict(e or {})
        self._check(X, e)
        rng = np.random.default_rng(seed)
        weights = np.zeros(len(self.nodes[X].domain))
        for _ in range(n):
            w, sample = 1.0, {}
            for name in self.order:
                node = self.nodes[name]
                row = node.cpt[tuple(sample[q] for q in node.parents)]
                if name in e:
                    k = self.index(name, e[name])
                    w *= row[k]
                else:
                    k = int(rng.choice(len(row), p=row))
                sample[name] = k
            weights[sample[X]] += w
        if weights.sum() <= 0:
            raise ZeroDivisionError("no sample was consistent with the evidence")
        return self._distribution(X, weights / weights.sum())


# --------------------------------------------------------------------------- demonstration


MAX_EXPLOITABILITY = 3.9  # largest CVSS v3 exploitability sub-score (8.22 * 0.85 * 0.77 * 0.85 * 0.85 = 3.887)


def exploit_probability(exploitability_subscore: float) -> float:
    """Modelling assumption stated in the report: P(exploit) = exploitability / 3.9, capped to [0.01, 0.99]."""
    return float(np.clip(exploitability_subscore / MAX_EXPLOITABILITY, 0.01, 0.99))


def attack_chain_network(exploitability: float = 2.8) -> BayesNet:
    """PortScan -> VulnExploit <- PhishingEmail;  VulnExploit -> PrivEsc -> LateralMove -> Exfiltration.

    VulnExploit is a noisy-OR of its two causes. The scan cause succeeds with the
    CVSS-derived exploit probability (2.8 is the sub-score of a typical network
    RCE with low attack complexity and low privileges required).
    """
    p_exploit = exploit_probability(exploitability)
    p_phish, leak = 0.6, 0.01
    net = BayesNet()
    net.add_boolean("PortScan", [], 0.3)
    net.add_boolean("PhishingEmail", [], 0.2)
    ve = np.zeros((2, 2))
    for scan in (0, 1):
        for phish in (0, 1):
            ve[scan, phish] = 1 - (1 - leak) * (1 - p_exploit) ** scan * (1 - p_phish) ** phish
    net.add_boolean("VulnExploit", ["PortScan", "PhishingEmail"], ve)
    net.add_boolean("PrivEsc", ["VulnExploit"], [0.02, 0.7])
    net.add_boolean("LateralMove", ["PrivEsc"], [0.05, 0.8])
    net.add_boolean("Exfiltration", ["LateralMove"], [0.01, 0.75])
    return net
