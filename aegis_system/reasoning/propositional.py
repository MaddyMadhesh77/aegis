"""Propositional inference for alert correlation.

Ground atoms such as PortScan or FailedLogin(Host3) act as proposition symbols, so
the same formulas used by the FOL engine work here unchanged.

tt_entails   model checking by truth-table enumeration (exponential, kept as the reference)
dpll         satisfiability with unit propagation and pure-literal elimination
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

from ..core.logic import And, Atom, Clause, Exists, ForAll, Formula, Iff, Implies, Literal, Not, Or, conjoin, is_ground
from .cnf import to_cnf

Model = Dict[Atom, bool]


def prop_symbols(f: Union[Formula, Iterable[Formula]]) -> List[Atom]:
    """The atoms of a ground formula, in first-occurrence order."""
    found: List[Atom] = []

    def walk(node: object) -> None:
        if isinstance(node, Atom):
            if not is_ground(node):
                raise ValueError(f"Propositional inference needs ground atoms, got {node}")
            if node not in found:
                found.append(node)
        elif isinstance(node, Not):
            walk(node.arg)
        elif isinstance(node, (And, Or, Implies, Iff)):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, (ForAll, Exists)):
            raise ValueError("Propositional inference does not handle quantifiers")
        elif isinstance(node, (list, tuple, set, frozenset)):
            for item in node:
                walk(item)

    walk(f)
    return found


def pl_true(f: Formula, model: Model) -> Optional[bool]:
    """Truth value of f in a (possibly partial) model; None if it is not yet determined."""
    if isinstance(f, Atom):
        return model.get(f)
    if isinstance(f, Not):
        v = pl_true(f.arg, model)
        return None if v is None else not v
    if isinstance(f, And):
        a, b = pl_true(f.left, model), pl_true(f.right, model)
        if a is False or b is False:
            return False
        return None if a is None or b is None else True
    if isinstance(f, Or):
        a, b = pl_true(f.left, model), pl_true(f.right, model)
        if a is True or b is True:
            return True
        return None if a is None or b is None else False
    if isinstance(f, Implies):
        return pl_true(Or(Not(f.left), f.right), model)
    if isinstance(f, Iff):
        a, b = pl_true(f.left, model), pl_true(f.right, model)
        return None if a is None or b is None else a == b
    raise ValueError(f"Not a propositional formula: {f!r}")


def _as_formula(kb: Union[Formula, Sequence[Formula]]) -> Formula:
    if isinstance(kb, (list, tuple)):
        return conjoin(*kb)
    return kb  # type: ignore[return-value]


def tt_entails(kb: Union[Formula, Sequence[Formula]], alpha: Formula) -> bool:
    """KB |= alpha, checked by enumerating every model of the symbols involved."""
    kb_f = _as_formula(kb)
    symbols = prop_symbols([kb_f, alpha])

    def check_all(i: int, model: Model) -> bool:
        if i == len(symbols):
            return not pl_true(kb_f, model) or bool(pl_true(alpha, model))
        s = symbols[i]
        return check_all(i + 1, {**model, s: True}) and check_all(i + 1, {**model, s: False})

    return check_all(0, {})


# --------------------------------------------------------------------------- DPLL


def dpll(clauses: Iterable[Clause], model: Optional[Model] = None) -> Optional[Model]:
    """A satisfying model of the clauses, or None if they are unsatisfiable.

    Symbols that do not matter may be missing from the returned model.
    """
    clauses = [frozenset(c) for c in clauses]
    prop_symbols([lit.atom for c in clauses for lit in c])  # rejects non-ground clauses
    return _dpll(clauses, dict(model or {}))


def _dpll(clauses: List[Clause], model: Model) -> Optional[Model]:
    while True:
        open_clauses: List[List[Literal]] = []
        for clause in clauses:
            unassigned: List[Literal] = []
            satisfied = False
            for lit in clause:
                value = model.get(lit.atom)
                if value is None:
                    unassigned.append(lit)
                elif value == lit.positive:
                    satisfied = True
                    break
            if satisfied:
                continue
            if not unassigned:
                return None  # clause false under the model
            open_clauses.append(unassigned)
        if not open_clauses:
            return model

        unit = next((c[0] for c in open_clauses if len(c) == 1), None)
        if unit is not None:
            model = {**model, unit.atom: unit.positive}
            continue

        polarity: Dict[Atom, Set[bool]] = {}
        for c in open_clauses:
            for lit in c:
                polarity.setdefault(lit.atom, set()).add(lit.positive)
        pure = next(((a, p) for a, p in polarity.items() if len(p) == 1), None)
        if pure is not None:
            model = {**model, pure[0]: next(iter(pure[1]))}
            continue
        break

    # branch on the symbol that occurs most often among the open clauses
    counts: Dict[Atom, int] = {}
    for c in open_clauses:
        for lit in c:
            counts[lit.atom] = counts.get(lit.atom, 0) + 1
    symbol = max(counts, key=lambda a: (counts[a], str(a)))
    for value in (True, False):
        result = _dpll(clauses, {**model, symbol: value})
        if result is not None:
            return result
    return None


def dpll_satisfiable(f: Union[Formula, Sequence[Formula]]) -> Optional[Model]:
    return dpll(to_cnf(_as_formula(f)))


def dpll_entails(kb: Union[Formula, Sequence[Formula]], alpha: Formula) -> bool:
    """KB |= alpha iff KB & ~alpha is unsatisfiable."""
    return dpll_satisfiable(And(_as_formula(kb), Not(alpha))) is None
