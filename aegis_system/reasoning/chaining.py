"""Forward and backward chaining over definite clauses.

A definite clause has exactly one positive literal. It is stored as a Rule
`head <- body`; a fact is a rule with an empty body.

forward_chain   agenda-based, runs to a fixpoint (data-driven, for the alert stream)
backward_chain  SLD resolution as a generator of substitutions (goal-driven, for queries)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from ..core.logic import Atom, Clause, free_variables, is_ground, rename_variables
from .unifier import Substitution, resolve_bindings, substitute, unify


@dataclass(frozen=True)
class Rule:
    head: Atom
    body: Tuple[Atom, ...] = ()
    label: object = field(default=None, compare=False)  # where the rule came from, e.g. a KB clause id

    @property
    def is_fact(self) -> bool:
        return not self.body

    def __str__(self) -> str:
        if not self.body:
            return str(self.head)
        return f"{self.head} <- {' & '.join(str(a) for a in self.body)}"


def as_definite(clause: Clause, label: object = None) -> Optional[Rule]:
    """The clause as a Rule, or None if it does not have exactly one positive literal."""
    positive = [lit.atom for lit in clause if lit.positive]
    if len(positive) != 1:
        return None
    body = tuple(sorted((lit.atom for lit in clause if not lit.positive), key=str))
    return Rule(positive[0], body, label)


# --------------------------------------------------------------------------- forward


@dataclass
class ForwardChainResult:
    facts: Set[Atom]
    # derived fact -> (rule used, the facts that matched its body)
    derivations: Dict[Atom, Tuple[Rule, Tuple[Atom, ...]]]
    rule_firings: int = 0

    def support(self, fact: Atom) -> List[Rule]:
        """Every rule (facts included) that the derivation of fact depends on."""
        used: List[Rule] = []
        stack = [fact]
        seen: Set[Atom] = set()
        while stack:
            f = stack.pop()
            if f in seen or f not in self.derivations:
                continue
            seen.add(f)
            rule, premises = self.derivations[f]
            used.append(rule)
            stack.extend(premises)
        return used


def forward_chain(rules: Iterable[Rule], facts: Iterable[Atom] = (), max_facts: int = 100_000) -> ForwardChainResult:
    """Derive every fact entailed by the rules and facts.

    Facts may be given as rules with empty bodies or through `facts`. Facts must be
    ground. Each new fact is pushed on an agenda; when it is popped, it is matched
    against each body atom of each rule and the rest of the body is joined with the
    known facts. Terminates for function-free (Datalog) rules; max_facts guards
    rules with function symbols.
    """
    rules = list(rules)
    proper = [r for r in rules if r.body]
    known: Set[Atom] = set()
    by_pred: Dict[str, List[Atom]] = {}
    derivations: Dict[Atom, Tuple[Rule, Tuple[Atom, ...]]] = {}
    agenda: Deque[Atom] = deque()
    firings = 0

    def add(fact: Atom, rule: Rule, premises: Tuple[Atom, ...]) -> None:
        if not is_ground(fact):
            raise ValueError(f"Forward chaining needs ground facts; {rule} derived {fact}")
        if fact in known:
            return
        known.add(fact)
        by_pred.setdefault(fact.pred, []).append(fact)
        derivations[fact] = (rule, premises)
        agenda.append(fact)

    for r in rules:
        if r.is_fact:
            add(r.head, r, ())
    for f in facts:
        add(f, Rule(f, (), "given"), ())

    while agenda:
        fact = agenda.popleft()
        for rule in proper:
            for i, atom in enumerate(rule.body):
                if atom.pred != fact.pred:
                    continue
                theta = unify(atom, fact)
                if theta is None:
                    continue
                rest = rule.body[:i] + rule.body[i + 1:]
                for theta2, matched in _join(rest, by_pred, theta):
                    premises = matched[:i] + (fact,) + matched[i:]
                    firings += 1
                    add(substitute(rule.head, theta2), rule, premises)
                    if len(known) >= max_facts:
                        return ForwardChainResult(known, derivations, firings)
    return ForwardChainResult(known, derivations, firings)


def _join(atoms: Sequence[Atom], by_pred: Dict[str, List[Atom]], theta: Substitution
          ) -> Iterator[Tuple[Substitution, Tuple[Atom, ...]]]:
    if not atoms:
        yield theta, ()
        return
    first, rest = atoms[0], atoms[1:]
    for fact in list(by_pred.get(first.pred, ())):
        theta2 = unify(first, fact, theta)
        if theta2 is not None:
            for theta3, matched in _join(rest, by_pred, theta2):
                yield theta3, (fact,) + matched


def forward_ask(rules: Iterable[Rule], goal: Atom) -> Iterator[Substitution]:
    """Bindings of the goal's variables for every derived fact that matches it."""
    result = forward_chain(rules)
    goal_vars = free_variables(goal)
    for fact in sorted(result.facts, key=str):
        theta = unify(goal, fact)
        if theta is not None:
            yield {v: substitute(v, theta) for v in goal_vars}


# --------------------------------------------------------------------------- backward


def backward_chain(rules: Sequence[Rule], goal: Atom, depth_limit: int = 50) -> Iterator[Substitution]:
    """SLD resolution: yield one substitution for the goal's variables per proof found.

    Rules are tried in order and renamed to fresh variables at every use. A proof
    branch deeper than depth_limit is abandoned, so left-recursive rules cannot loop
    forever (the search is then incomplete beyond that depth).
    """
    for theta, _ in prove_backward(rules, goal, depth_limit):
        yield theta


def prove_backward(rules: Sequence[Rule], goal: Atom, depth_limit: int = 50
                   ) -> Iterator[Tuple[Substitution, Tuple[Rule, ...]]]:
    """Like backward_chain, but also yields the rules each proof used."""
    goal_vars = free_variables(goal)
    for theta, used in _bc_and([goal], {}, depth_limit, list(rules)):
        theta = resolve_bindings(theta)
        yield {v: substitute(v, theta) for v in goal_vars}, used


def _bc_or(goal: Atom, theta: Substitution, depth: int, rules: List[Rule]
           ) -> Iterator[Tuple[Substitution, Tuple[Rule, ...]]]:
    if depth <= 0:
        return
    goal = substitute(goal, theta)
    for rule in rules:
        if rule.head.pred != goal.pred:
            continue
        fresh: Rule = Rule(*rename_variables((rule.head, rule.body)), rule.label)  # type: ignore[misc]
        theta2 = unify(fresh.head, goal, theta)
        if theta2 is None:
            continue
        for theta3, used in _bc_and(list(fresh.body), theta2, depth - 1, rules):
            yield theta3, (rule,) + used


def _bc_and(goals: List[Atom], theta: Substitution, depth: int, rules: List[Rule]
            ) -> Iterator[Tuple[Substitution, Tuple[Rule, ...]]]:
    if not goals:
        yield theta, ()
        return
    first, rest = goals[0], goals[1:]
    for theta2, used1 in _bc_or(first, theta, depth, rules):
        for theta3, used2 in _bc_and(rest, theta2, depth, rules):
            yield theta3, used1 + used2
