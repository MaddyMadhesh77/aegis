"""First-order knowledge base with three inference methods.

`tell` converts each formula to clause form. Every clause is kept for resolution;
definite clauses are also kept as rules for forward and backward chaining, which
therefore ignore any non-definite knowledge (disjunctive facts, negative facts).

    kb = FOLKnowledgeBase()
    kb.tell("forall x forall y (Compromised(x) & Trusts(y, x) -> Compromised(y))")
    kb.tell("Compromised(Host1)")
    kb.tell("Trusts(DC, Host1)")
    kb.ask("Compromised(DC)")                      # resolution refutation
    kb.ask("Compromised(DC)", method="backward")   # SLD resolution

When the knowledge base has a TraceStore, each told formula and each answer is
recorded, and a proof's trace node lists the told formulas it used as parents.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

from ..core.logic import Atom, Clause, Formula, clause_str, free_variables, parse
from ..core.trace import TraceStore
from ..core.types import Proof
from .chaining import Rule, as_definite, forward_chain, prove_backward
from .cnf import to_cnf
from .resolution import format_proof, refute
from .unifier import Substitution, substitute, unify

METHODS = ("resolution", "forward", "backward")


class FOLKnowledgeBase:
    def __init__(self, trace: Optional[TraceStore] = None):
        self.trace = trace
        self.formulas: List[Formula] = []
        self.clauses: List[Clause] = []
        self.rules: List[Rule] = []
        self._clause_source: Dict[Clause, Optional[str]] = {}  # clause -> trace id of the told formula

    # ----------------------------------------------------------------- tell

    def tell(self, formula: Union[str, Formula], trace_id: Optional[str] = None) -> List[Clause]:
        """Add a formula; returns the clauses it became.

        trace_id links the formula to the pipeline stage that produced it (e.g. grounding).
        """
        f = parse(formula) if isinstance(formula, str) else formula
        if self.trace is not None:
            trace_id = self.trace.record("kb", f"tell {f}", {"formula": str(f)}, [trace_id])
        self.formulas.append(f)
        added: List[Clause] = []
        for clause in to_cnf(f):
            if clause in self._clause_source:
                continue
            self.clauses.append(clause)
            self._clause_source[clause] = trace_id
            rule = as_definite(clause, label=clause)
            if rule is not None:
                self.rules.append(rule)
            added.append(clause)
        return added

    def tell_all(self, formulas: Sequence[Union[str, Formula]], trace_id: Optional[str] = None) -> None:
        for f in formulas:
            self.tell(f, trace_id)

    # ----------------------------------------------------------------- ask

    def ask(self, goal: Union[str, Formula], method: str = "resolution", record: bool = True, **limits) -> Proof:
        """Try to prove goal. Free variables in goal are existential; `bindings` holds one answer.

        limits: max_iterations / max_clauses for resolution, depth_limit for backward.
        record=False skips the trace entry (for cheap probes whose failures are not worth logging).
        """
        g = parse(goal) if isinstance(goal, str) else goal
        if method == "resolution":
            proof = refute(self.clauses, g, **limits)
            used = [s.clause for s in proof.steps if s.rule == "kb"]
        elif method in ("forward", "backward"):
            if not isinstance(g, Atom):
                raise ValueError(f"{method} chaining answers single atoms, got {g}")
            proof, used = (self._ask_forward if method == "forward" else self._ask_backward)(g, **limits)
        else:
            raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
        proof.method = method
        if record:
            self._record(proof, used)
        return proof

    def ask_all(self, goal: Union[str, Atom], depth_limit: int = 50, limit: int = 100) -> List[Substitution]:
        """Every distinct answer to an atomic goal, by backward chaining."""
        g = parse(goal) if isinstance(goal, str) else goal
        answers: List[Substitution] = []
        for theta, _ in prove_backward(self.rules, g, depth_limit):  # type: ignore[arg-type]
            if theta not in answers:
                answers.append(theta)
                if len(answers) >= limit:
                    break
        return answers

    def entails(self, goal: Union[str, Formula], method: str = "resolution") -> bool:
        return self.ask(goal, method).proved

    def _ask_forward(self, goal: Atom) -> Tuple[Proof, List[Clause]]:
        result = forward_chain(self.rules)
        goal_vars = free_variables(goal)
        for fact in sorted(result.facts, key=str):
            theta = unify(goal, fact)
            if theta is not None:
                bindings = {v: substitute(v, theta) for v in goal_vars}
                used = [r.label for r in result.support(fact)]
                return Proof(goal, [], True, bindings, len(result.facts)), used
        return Proof(goal, [], False, {}, len(result.facts)), []

    def _ask_backward(self, goal: Atom, depth_limit: int = 50) -> Tuple[Proof, List[Clause]]:
        for theta, rules in prove_backward(self.rules, goal, depth_limit):
            return Proof(goal, [], True, theta, len(rules)), [r.label for r in rules]
        return Proof(goal, [], False, {}, 0), []

    def _record(self, proof: Proof, used: List[Clause]) -> None:
        if self.trace is None:
            return
        parents: List[str] = []
        for clause in used:
            tid = self._clause_source.get(clause)
            if tid is not None and tid not in parents:
                parents.append(tid)
        status = "proved" if proof.proved else "not proved"
        proof.trace_id = self.trace.record(
            "proof",
            f"{proof.goal} {status} by {proof.method}",
            {
                "goal": str(proof.goal),
                "method": proof.method,
                "proved": proof.proved,
                "bindings": {str(v): str(t) for v, t in proof.bindings.items()},
                "clauses_used": [clause_str(c) for c in used],
                "proof": format_proof(proof) if proof.method == "resolution" else "",
            },
            parents,
        )

    # ----------------------------------------------------------------- inspection

    def facts(self) -> Iterator[Atom]:
        """Ground facts told directly (not derived)."""
        for rule in self.rules:
            if rule.is_fact:
                yield rule.head

    def __len__(self) -> int:
        return len(self.clauses)


def demo_kb(trace: Optional[TraceStore] = None) -> FOLKnowledgeBase:
    """The lateral-movement knowledge base used in the report.

    Host1's compromise is not stated directly: it follows from an existential fact
    ("some malware runs on Host1"), so the proof of Compromised(DC) goes through a
    Skolem constant.
    """
    kb = FOLKnowledgeBase(trace)
    kb.tell_all([
        "forall x forall y forall z (Compromised(x) & Trusts(y, x) & TransfersAuth(x, y, z) -> Compromised(y))",
        "forall m forall h (Malware(m) & RunsOn(m, h) -> Compromised(h))",
        "exists m (Malware(m) & RunsOn(m, Host1))",
        "Trusts(DC, Host1)",
        "TransfersAuth(Host1, DC, Kerberos)",
    ])
    return kb
