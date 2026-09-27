"""Forward-chaining production system with conflict resolution.

Working memory holds ground atoms, each stamped with the cycle it was added in.
A rule has positive conditions (atom patterns with variables), negated
conditions (no matching fact may exist), tests on the bindings, and an action:
atoms to assert and/or retract, and/or a Python callable.

Each recognize-act cycle (matching is incremental, see ProductionSystem):
  1. build the conflict set: every instantiation (rule, bindings, matched facts)
     not fired before (refraction: a rule never fires twice on the same facts);
  2. order it by salience, then specificity (more conditions and tests first),
     then recency (the newest matched facts first);
  3. fire the first instantiation only.

Rules extracted from the Phase 6 decision tree arrive through `threshold_rule`.
"""

from __future__ import annotations

import itertools
import operator
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

from ..core.logic import Atom, Const, Term, Var, is_ground, parse_atom
from ..reasoning.unifier import Substitution, substitute, unify

Test = Callable[[Substitution], bool]
Action = Callable[["ProductionSystem", Substitution], None]

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq,
       "!=": operator.ne}


def _atoms(items: Iterable[Union[str, Atom]]) -> Tuple[Atom, ...]:
    return tuple(parse_atom(i) if isinstance(i, str) else i for i in items)


def number(term: Term) -> float:
    if not isinstance(term, Const):
        raise TypeError(f"Expected a numeric constant, got {term}")
    return float(term.name)


def compare(var: str, op: str, value: float) -> Test:
    """A test that the number bound to `var` satisfies `op value`, e.g. compare("b", ">", 5000)."""
    v, fn = Var(var), OPS[op]

    def test(theta: Substitution) -> bool:
        return fn(number(substitute(v, theta)), value)

    test.__name__ = f"{var} {op} {value}"
    return test


@dataclass
class Rule:
    name: str
    conditions: Tuple[Atom, ...]
    asserts: Tuple[Atom, ...] = ()
    retracts: Tuple[Atom, ...] = ()
    negated: Tuple[Atom, ...] = ()
    tests: Tuple[Test, ...] = ()
    action: Optional[Action] = None
    salience: int = 0

    @classmethod
    def make(cls, name: str, conditions: Sequence[Union[str, Atom]], asserts: Sequence[Union[str, Atom]] = (),
             retracts: Sequence[Union[str, Atom]] = (), negated: Sequence[Union[str, Atom]] = (),
             tests: Sequence[Test] = (), action: Optional[Action] = None, salience: int = 0) -> "Rule":
        return cls(name, _atoms(conditions), _atoms(asserts), _atoms(retracts), _atoms(negated), tuple(tests),
                   action, salience)

    @property
    def specificity(self) -> int:
        return len(self.conditions) + len(self.negated) + len(self.tests)

    def __str__(self) -> str:
        lhs = [str(c) for c in self.conditions] + [f"~{n}" for n in self.negated] + \
              [getattr(t, "__name__", "test") for t in self.tests]
        rhs = [str(a) for a in self.asserts] + [f"retract {r}" for r in self.retracts]
        return f"{self.name}: {' & '.join(lhs)} => {', '.join(rhs) or 'action'}"


@dataclass(frozen=True)
class Instantiation:
    rule: Rule = field(compare=False)
    rule_name: str
    facts: Tuple[Atom, ...]
    bindings: Tuple[Tuple[Var, Term], ...]

    @property
    def theta(self) -> Substitution:
        return dict(self.bindings)


@dataclass
class Firing:
    cycle: int
    rule: str
    facts: Tuple[Atom, ...]
    asserted: Tuple[Atom, ...]
    retracted: Tuple[Atom, ...]


class ProductionSystem:
    """Working memory is indexed by predicate and by each constant argument, and the
    agenda (matched instantiations) is maintained incrementally: asserting a fact
    only looks for matches that use that fact, and retracting one drops the
    instantiations built on it. Negated conditions are re-checked when the
    conflict set is built, since a later fact can block an earlier match.
    """

    def __init__(self, rules: Iterable[Rule] = ()):
        self.rules: List[Rule] = []
        self.memory: Dict[Atom, int] = {}       # fact -> timestamp
        self.fired: Set[Tuple[str, Tuple[Atom, ...]]] = set()
        self.log: List[Firing] = []
        self.cycle = 0
        self._clock = itertools.count(1)
        self._by_pred: Dict[str, Set[Atom]] = {}
        self._by_arg: Dict[Tuple[str, int, Term], Set[Atom]] = {}
        self._by_pair: Dict[Tuple[str, int, Term, int, Term], Set[Atom]] = {}
        self._agenda: Dict[Tuple[str, Tuple[Atom, ...]], Instantiation] = {}
        # (pred,) or (pred, position, constant) -> conditions that a new fact could match
        self._cond_index: Dict[Tuple, List[Tuple[Rule, int]]] = {}
        for rule in rules:
            self.add_rule(rule)

    # ----------------------------------------------------------------- working memory

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)
        for i, cond in enumerate(rule.conditions):
            const = next(((j, a) for j, a in enumerate(cond.args) if isinstance(a, Const)), None)
            key = (cond.pred,) if const is None else (cond.pred, const[0], const[1])
            self._cond_index.setdefault(key, []).append((rule, i))
        for theta, facts in self._matches(rule.conditions, {}):
            self._offer(rule, theta, facts)

    def assert_fact(self, fact: Union[str, Atom]) -> bool:
        f = parse_atom(fact) if isinstance(fact, str) else fact
        if not is_ground(f):
            raise ValueError(f"Working memory holds ground facts only: {f}")
        if f in self.memory:
            return False
        self.memory[f] = next(self._clock)
        self._by_pred.setdefault(f.pred, set()).add(f)
        for i, a in enumerate(f.args):
            self._by_arg.setdefault((f.pred, i, a), set()).add(f)
        for (i, a), (j, b) in itertools.combinations(enumerate(f.args), 2):
            self._by_pair.setdefault((f.pred, i, a, j, b), set()).add(f)
        candidates = list(self._cond_index.get((f.pred,), ()))
        for j, a in enumerate(f.args):
            candidates += self._cond_index.get((f.pred, j, a), ())
        for rule, i in candidates:
            theta = unify(rule.conditions[i], f)
            if theta is None:
                continue
            rest = rule.conditions[:i] + rule.conditions[i + 1:]
            for theta2, matched in self._matches(rest, theta):
                self._offer(rule, theta2, matched[:i] + (f,) + matched[i:])
        return True

    def retract(self, fact: Union[str, Atom]) -> bool:
        f = parse_atom(fact) if isinstance(fact, str) else fact
        if self.memory.pop(f, None) is None:
            return False
        self._by_pred[f.pred].discard(f)
        for i, a in enumerate(f.args):
            self._by_arg[(f.pred, i, a)].discard(f)
        for (i, a), (j, b) in itertools.combinations(enumerate(f.args), 2):
            self._by_pair[(f.pred, i, a, j, b)].discard(f)
        self._agenda = {k: v for k, v in self._agenda.items() if f not in v.facts}
        return True

    def facts(self, pred: Optional[str] = None) -> List[Atom]:
        return sorted((f for f in self.memory if pred is None or f.pred == pred), key=self.memory.get)

    # ----------------------------------------------------------------- matching

    def _candidates(self, pattern: Atom) -> Iterable[Atom]:
        consts = [(i, a) for i, a in enumerate(pattern.args) if isinstance(a, Const)]
        if len(consts) >= 2:
            (i, a), (j, b) = consts[:2]
            return list(self._by_pair.get((pattern.pred, i, a, j, b), ()))
        best: Optional[Set[Atom]] = self._by_pred.get(pattern.pred, set())
        for i, a in enumerate(pattern.args):
            if isinstance(a, Const):
                bucket = self._by_arg.get((pattern.pred, i, a), set())
                if len(bucket) < len(best):
                    best = bucket
        return list(best)

    def _matches(self, conditions: Sequence[Atom], theta: Substitution) -> Iterator[Tuple[Substitution, Tuple[Atom, ...]]]:
        if not conditions:
            yield theta, ()
            return
        first = substitute(conditions[0], theta)
        for fact in self._candidates(first):
            theta2 = unify(first, fact, theta)
            if theta2 is not None:
                for theta3, matched in self._matches(conditions[1:], theta2):
                    yield theta3, (fact,) + matched

    def _offer(self, rule: Rule, theta: Substitution, facts: Tuple[Atom, ...]) -> None:
        key = (rule.name, facts)
        if key in self.fired or key in self._agenda:
            return
        try:
            if not all(t(theta) for t in rule.tests):
                return
        except (TypeError, ValueError):
            return  # a test on a non-numeric binding simply does not match
        bindings = tuple(sorted(theta.items(), key=lambda kv: kv[0].name))
        self._agenda[key] = Instantiation(rule, rule.name, facts, bindings)

    def _blocked(self, inst: Instantiation) -> bool:
        theta = inst.theta
        for n in inst.rule.negated:
            pattern = substitute(n, theta)
            if any(unify(pattern, f) is not None for f in self._candidates(pattern)):
                return True
        return False

    def conflict_set(self) -> List[Instantiation]:
        found = [inst for inst in self._agenda.values() if not self._blocked(inst)]
        found.sort(key=self._priority)
        return found

    def _priority(self, inst: Instantiation):
        stamps = sorted((self.memory[f] for f in inst.facts), reverse=True)
        # Python sorts ascending, so negate everything that should come first when larger.
        return (-inst.rule.salience, -inst.rule.specificity, [-s for s in stamps], inst.rule_name)

    # ----------------------------------------------------------------- execution

    def step(self) -> Optional[Firing]:
        """One recognize-act cycle. Returns the firing, or None if the conflict set is empty."""
        conflict = self.conflict_set()
        if not conflict:
            return None
        inst = conflict[0]
        self.cycle += 1
        key = (inst.rule_name, inst.facts)
        self.fired.add(key)
        self._agenda.pop(key, None)
        theta = inst.theta
        asserted = tuple(a for a in (substitute(x, theta) for x in inst.rule.asserts) if self.assert_fact(a))
        retracted = tuple(r for r in (substitute(x, theta) for x in inst.rule.retracts) if self.retract(r))
        if inst.rule.action is not None:
            inst.rule.action(self, theta)
        firing = Firing(self.cycle, inst.rule_name, inst.facts, asserted, retracted)
        self.log.append(firing)
        return firing

    def run(self, max_cycles: int = 1000) -> List[Firing]:
        fired = []
        for _ in range(max_cycles):
            f = self.step()
            if f is None:
                break
            fired.append(f)
        return fired


def threshold_rule(name: str, tests: Sequence[Tuple[str, str, float]], conclusion: str, salience: int = 0,
                   confidence: Optional[float] = None) -> Rule:
    """A rule over flow features, as produced by walking a decision tree.

    tests are (feature, op, threshold). Flows are facts Feature(flow, name, value);
    the rule asserts Classified(flow, conclusion) when every test holds.
    """
    flow = Var("flow")
    conditions = []
    checks = []
    for i, (feature, op, threshold) in enumerate(tests):
        v = Var(f"v{i}")
        conditions.append(Atom("Feature", (flow, Const(feature), v)))
        checks.append(compare(v.name, op, threshold))
    asserts = [Atom("Classified", (flow, Const(conclusion)))]
    if confidence is not None:
        asserts.append(Atom("Confidence", (flow, Const(conclusion), Const(f"{confidence:.3f}"))))
    return Rule(name, tuple(conditions), tuple(asserts), (), (Atom("Classified", (flow, Var("_any"))),),
                tuple(checks), None, salience)


def flow_facts(flow_id: str, features: Dict[str, float]) -> List[Atom]:
    return [Atom("Feature", (Const(flow_id), Const(k), Const(repr(float(v))))) for k, v in features.items()]
