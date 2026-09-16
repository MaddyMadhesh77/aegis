"""Binary resolution with factoring, and refutation with the set-of-support strategy.

`refute(kb, goal)` adds the clauses of ~goal to the knowledge base and searches for
the empty clause. Only clauses descended from ~goal are resolved against each
other or against the KB (set of support), so the KB is never saturated on its own.

Goals with free variables are answered with Green's answer literal: ~goal | Answer(x..)
is converted instead of ~goal, and a clause made only of Answer literals counts as
the empty clause. Its arguments are the bindings for the goal's variables.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from ..core.logic import Atom, Clause, Const, Formula, Not, Or, Var, clause_str, free_variables, rename_variables
from ..core.types import Proof, ResolutionStep
from .cnf import is_tautology, to_cnf
from .unifier import Substitution, resolve_bindings, substitute, unify

ANSWER = "Answer"


def standardize_apart(clause: Clause) -> Clause:
    return rename_variables(clause)  # type: ignore[return-value]


def resolvents(c1: Clause, c2: Clause) -> Iterator[Tuple[Clause, Substitution]]:
    """Every binary resolvent of c1 and c2 with the unifier that produced it.

    c2 is renamed apart from c1 first, so the clauses never share variables.
    """
    renaming: Dict[Var, Var] = {}
    c2 = rename_variables(c2, renaming)  # type: ignore[assignment]
    for l1 in c1:
        for l2 in c2:
            if l1.positive == l2.positive or l1.atom.pred != l2.atom.pred:
                continue
            theta = unify(l1.atom, l2.atom)
            if theta is None:
                continue
            theta = resolve_bindings(theta)
            rest = (c1 - {l1}) | (c2 - {l2})
            yield frozenset(substitute(lit, theta) for lit in rest), _unrename(theta, renaming)


def _unrename(theta: Substitution, renaming: Dict[Var, Var]) -> Substitution:
    """Express theta over c2's original variable names, so proofs can be read against the input."""
    back = {fresh: orig for orig, fresh in renaming.items()}
    return {back.get(v, v): t for v, t in theta.items()}


def resolve(c1: Clause, c2: Clause) -> List[Clause]:
    """All binary resolvents of two clauses."""
    return [clause for clause, _ in resolvents(c1, c2)]


def factors(clause: Clause) -> Iterator[Tuple[Clause, Substitution]]:
    """Clauses obtained by unifying two literals of the same sign (needed for completeness)."""
    lits = list(clause)
    for a, b in itertools.combinations(lits, 2):
        if a.positive != b.positive or a.atom.pred != b.atom.pred:
            continue
        theta = unify(a.atom, b.atom)
        if theta:
            theta = resolve_bindings(theta)
            yield frozenset(substitute(lit, theta) for lit in clause), theta


def subsumes(c: Clause, d: Clause) -> bool:
    """True if c theta-subsumes d: some theta maps every literal of c into d."""
    if len(c) > len(d):
        return False
    d_keys = {(lit.positive, lit.atom.pred) for lit in d}
    if any((lit.positive, lit.atom.pred) not in d_keys for lit in c):
        return False
    frozen = _freeze(d)  # d's variables become constants, so only c's variables get bound
    lits = sorted(c, key=lambda lit: -len(lit.atom.args))

    def match(i: int, theta: Substitution) -> bool:
        if i == len(lits):
            return True
        for target in frozen:
            if target.positive == lits[i].positive and target.atom.pred == lits[i].atom.pred:
                result = unify(lits[i].atom, target.atom, theta)
                if result is not None and match(i + 1, result):
                    return True
        return False

    return match(0, {})


def _freeze(clause: Clause) -> Clause:
    mapping = {v: Const(f"?{v.name}") for v in free_variables(clause)}
    return substitute(clause, mapping) if mapping else clause  # type: ignore[return-value]


def is_answer_clause(clause: Clause) -> bool:
    return bool(clause) and all(lit.positive and lit.atom.pred == ANSWER for lit in clause)


def refute(
    kb: Iterable[Clause],
    goal: Formula,
    max_iterations: int = 500,
    max_clauses: int = 5000,
    max_clause_length: int = 8,
) -> Proof:
    """Prove goal from kb by deriving the empty clause from kb + CNF(~goal).

    Returns a Proof whose steps are the derivation of the empty clause, in order,
    each with its parent clause ids and unifier. On failure the steps are empty.

    First-order entailment is only semi-decidable, so the search is capped by the
    number of given clauses, the number of clauses kept, and the length of a kept
    resolvent. A failed proof therefore means "not found within the limits".
    """
    goal_vars = free_variables(goal)
    negated: Formula = Not(goal)
    if goal_vars:
        negated = Or(negated, Atom(ANSWER, tuple(goal_vars)))

    steps: Dict[int, ResolutionStep] = {}
    counter = itertools.count(1)

    def add(clause: Clause, parents: Tuple[int, ...], theta: Substitution, rule: str) -> ResolutionStep:
        step = ResolutionStep(next(counter), clause, parents, dict(theta), rule)
        steps[step.clause_id] = step
        return step

    usable: List[ResolutionStep] = []
    for clause in kb:
        if not is_tautology(clause) and not any(subsumes(u.clause, clause) for u in usable):
            usable.append(add(clause, (), {}, "kb"))
    support: List[ResolutionStep] = [add(c, (), {}, "negated_goal") for c in to_cnf(negated)]

    def finish(step: ResolutionStep) -> Proof:
        bindings = {}
        if step.clause:  # answer clause: read the bindings off its (first) Answer literal
            answer = min(step.clause, key=str).atom
            bindings = dict(zip(goal_vars, answer.args))
        return Proof(goal, _derivation(step, steps), True, bindings, len(steps))

    for step in support:
        if not step.clause or is_answer_clause(step.clause):
            return finish(step)

    for _ in range(max_iterations):
        if not support:
            break
        support.sort(key=lambda s: (len(s.clause), s.clause_id))
        given = support.pop(0)
        usable.append(given)
        new: List[Tuple[Clause, Tuple[int, ...], Substitution, str]] = []
        for other in usable:
            for clause, theta in resolvents(given.clause, other.clause):
                new.append((clause, (given.clause_id, other.clause_id), theta, "resolution"))
        for clause, theta in factors(given.clause):
            new.append((clause, (given.clause_id,), theta, "factoring"))

        for clause, parents, theta, rule in new:
            if is_tautology(clause) or len(clause) > max_clause_length:
                continue
            if any(subsumes(s.clause, clause) for s in itertools.chain(usable, support)):
                continue
            step = add(clause, parents, theta, rule)
            if not clause or is_answer_clause(clause):
                return finish(step)
            support = [s for s in support if not subsumes(clause, s.clause)]  # backward subsumption
            support.append(step)
            if len(steps) >= max_clauses:
                return Proof(goal, [], False, {}, len(steps))

    return Proof(goal, [], False, {}, len(steps))


def _derivation(final: ResolutionStep, steps: Dict[int, ResolutionStep]) -> List[ResolutionStep]:
    """The steps the final clause depends on, parents before children."""
    needed: Dict[int, ResolutionStep] = {}
    stack = [final.clause_id]
    while stack:
        cid = stack.pop()
        if cid not in needed:
            needed[cid] = steps[cid]
            stack.extend(steps[cid].parents)
    return [needed[cid] for cid in sorted(needed)]


def format_proof(proof: Proof) -> str:
    """Human-readable proof, one numbered clause per line."""
    if not proof.proved:
        return f"{proof.goal}: not proved ({proof.clauses_generated} clauses generated)"
    lines = []
    for s in proof.steps:
        if s.parents:
            sub = ", ".join(f"{v}/{t}" for v, t in sorted(s.substitution.items(), key=lambda kv: kv[0].name))
            origin = f"{s.rule} of {', '.join(map(str, s.parents))}" + (f"  {{{sub}}}" if sub else "")
        else:
            origin = s.rule
        lines.append(f"{s.clause_id:>4}. {clause_str(s.clause):<50} [{origin}]")
    return "\n".join(lines)


def proof_inputs(proof: Proof) -> Sequence[ResolutionStep]:
    """The KB clauses a proof actually used."""
    return [s for s in proof.steps if s.rule == "kb"]
