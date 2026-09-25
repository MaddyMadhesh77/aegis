"""Plan execution against the simulated network, with monitoring and replanning.

Execution goes layer by layer through a partial-order plan. After every action
the executor senses the environment and checks that

  * the action's observable effects hold (Isolated(h) really appears, Compromised(h) really went away), and
  * the invariants hold (literals that must stay true, e.g. ~Isolated(DomainController)).

On a failure (a patch did not apply, the attacker kept persistence) the belief
state is updated with what was observed, the failure is recorded in the trace,
and the executor replans from the observed state. Replans are capped.

Belief update: observable predicates come from the sensors; other atoms (Clean,
Restored) are carried over from the plan's predictions, and only for actions
that were confirmed to work.

conditional_plan() is a depth-limited AND-OR search for plans with one kind of
sensing action, CheckHostStatus(h), which branches on whether h is compromised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from ..core.environment import NetworkEnvironment
from ..core.logic import Atom, Const, Literal
from ..core.trace import TraceStore
from ..core.types import ExecutionResult
from .domain import FAILURE_FACTS, OBSERVABLE, merge_observation, to_command
from .partial_order import PartialOrderPlanner, format_layers, layers
from .strips_planner import Action, Problem, negated, progression, regression

State = frozenset


@dataclass
class ExecutionReport:
    success: bool
    results: List[ExecutionResult] = field(default_factory=list)
    replans: int = 0
    failures: List[str] = field(default_factory=list)
    plans: List[List[List[Action]]] = field(default_factory=list)
    final_state: State = frozenset()
    trace_id: Optional[str] = None

    @property
    def executed(self) -> List[str]:
        return [str(r.step) for r in self.results]


class Executor:
    def __init__(
        self,
        env: NetworkEnvironment,
        max_replans: int = 5,
        invariants: Sequence[Literal] = (),
        trace: Optional[TraceStore] = None,
        planner: str = "pop",  # "pop", "regression" or "progression"
    ):
        if planner not in ("pop", "regression", "progression"):
            raise ValueError("planner must be 'pop', 'regression' or 'progression'")
        self.env = env
        self.max_replans = max_replans
        self.invariants = list(invariants)
        self.trace = trace
        self.planner = planner

    # ----------------------------------------------------------------- planning

    def make_plan(self, problem: Problem) -> Optional[List[List[Action]]]:
        """Layered plan: POP when possible (it gives parallel steps), otherwise an optimal
        state-space plan with one action per layer.

        The fallback is regression with h_max: h_max is admissible, so the plan is
        cheapest, and regression expanded the fewest nodes on the defence goals.
        """
        if self.planner == "pop":
            result = PartialOrderPlanner(problem).plan()
            if result.solved:
                return layers(result.plan)
        search = progression if self.planner == "progression" else regression
        linear = search(problem, "hmax").plan
        return None if linear is None else [[a] for a in linear]

    # ----------------------------------------------------------------- execution

    def execute(self, problem: Problem, plan: Optional[List[List[Action]]] = None,
                parent_trace: Optional[str] = None) -> ExecutionReport:
        report = ExecutionReport(False)
        belief = problem.init
        if plan is None:
            plan = self.make_plan(problem)
        plan_tid = self._record("plan", "initial plan", plan, [parent_trace])

        while True:
            if plan is None:
                report.failures.append("no plan reaches the goal from the observed state")
                break
            report.plans.append(plan)
            failure, belief, last_tid = self._run_layers(problem, plan, belief, report, plan_tid)
            if failure is None and problem.is_goal(belief):
                report.success = True
                break
            reason = failure or "goal not reached after the plan finished"
            report.failures.append(reason)
            fail_tid = self._record("failure", reason, None, [last_tid or plan_tid])
            if report.replans >= self.max_replans:
                break
            report.replans += 1
            replanned = problem.with_init(problem.positive_view(belief))
            plan = self.make_plan(replanned)
            plan_tid = self._record("plan", f"replan {report.replans}", plan, [fail_tid])

        report.final_state = belief
        report.trace_id = plan_tid
        return report

    def _run_layers(self, problem: Problem, plan: List[List[Action]], belief: State, report: ExecutionReport,
                    plan_tid: Optional[str]) -> Tuple[Optional[str], State, Optional[str]]:
        last_tid = None
        for layer in plan:
            for action in layer:
                if not action.applicable(belief):
                    missing = sorted(str(p) for p in action.pre - belief)
                    return f"{action} no longer applicable (missing {', '.join(missing)})", belief, last_tid
                result = self.env.apply(to_command(action))
                predicted = action.apply(belief)
                observed = merge_observation(problem, predicted, result.observed_state)
                problems = self._check(action, observed, result.success)
                result.step = action
                last_tid = self._record("act", f"{action} -> {result.command}",
                                        {"success": not problems, "command": result.command}, [plan_tid])
                result.trace_id = last_tid
                report.results.append(result)
                if problems:
                    result.success = False
                    # credit only what the sensors confirm, plus what the failure teaches
                    belief = merge_observation(problem, belief, result.observed_state)
                    if action.name in FAILURE_FACTS:
                        belief = problem.complete(problem.positive_view(belief)
                                                  | {Atom(FAILURE_FACTS[action.name], action.args)})
                    return f"{action} failed: {'; '.join(problems)}", belief, last_tid
                belief = observed
            if problem.is_goal(belief):
                break
        return None, belief, last_tid

    def _check(self, action: Action, state: State, env_success: bool) -> List[str]:
        problems = []
        if not env_success:
            problems.append("environment reported failure")
        for p in action.add:
            if p.pred in OBSERVABLE and p not in state:
                problems.append(f"expected {p}")
        for p in action.delete:
            if p.pred in OBSERVABLE and p in state:
                problems.append(f"expected ~{p}")
        for lit in self.invariants:
            if (lit.atom in state) != lit.positive:
                problems.append(f"invariant {lit} violated")
        return problems

    def _record(self, stage: str, summary: str, payload, parents) -> Optional[str]:
        if self.trace is None:
            return None
        if isinstance(payload, list):
            payload = {"layers": format_layers(payload)}
        elif payload is None:
            payload = {}
        return self.trace.record(stage, summary, payload, parents)


# --------------------------------------------------------------------------- conditional planning


@dataclass(frozen=True)
class SensingAction:
    """CheckHostStatus(h): requires Unknown(h); one outcome per truth value of Compromised(h)."""

    host: str

    @property
    def atom(self) -> Atom:
        return Atom("Compromised", (Const(self.host),))

    @property
    def unknown(self) -> Atom:
        return Atom("Unknown", (Const(self.host),))

    def applicable(self, state: State) -> bool:
        return self.unknown in state

    def outcomes(self, state: State) -> Dict[str, State]:
        base = state - {self.unknown}
        return {
            "compromised": (base - {negated(self.atom)}) | {self.atom},
            "clean": (base - {self.atom}) | {negated(self.atom)},
        }

    def __str__(self) -> str:
        return f"CheckHostStatus({self.host})"


@dataclass
class Branch:
    sensing: SensingAction
    cases: Dict[str, "ConditionalPlan"]


ConditionalPlan = List[Union[Action, Branch]]


def unknown_state(problem: Problem, known_compromised: Iterable[str], suspected: Iterable[str]) -> State:
    """Belief where the suspected hosts' status is unknown (neither Compromised nor NotCompromised)."""
    state = set(problem.init)
    for h in suspected:
        c = Atom("Compromised", (Const(h),))
        state -= {c, negated(c)}
        state.add(Atom("Unknown", (Const(h),)))
    for h in known_compromised:
        c = Atom("Compromised", (Const(h),))
        state = (state - {negated(c)}) | {c}
    return frozenset(state)


def contained_goal(hosts: Iterable[str]) -> Callable[[State], bool]:
    """Every host's status is known, and every compromised host is isolated."""
    hosts = list(hosts)

    def test(state: State) -> bool:
        for h in hosts:
            c = (Const(h),)
            if Atom("Unknown", c) in state:
                return False
            if Atom("Compromised", c) in state and Atom("Isolated", c) not in state:
                return False
        return True

    return test


def conditional_plan(
    problem: Problem,
    state: State,
    goal_test: Callable[[State], bool],
    sensing: Sequence[SensingAction] = (),
    max_depth: int = 4,
    relevant: Optional[Callable[[Action], bool]] = None,
) -> Optional[ConditionalPlan]:
    """AND-OR search (iterative deepening up to max_depth) for a plan that reaches the goal in every outcome.

    OR nodes choose an action; AND nodes must solve every outcome of a sensing action.
    `relevant` filters the deterministic actions considered, to keep branching low.
    """
    actions = [a for a in problem.actions if relevant is None or relevant(a)]

    def or_search(s: State, path: Tuple[State, ...], depth: int) -> Optional[ConditionalPlan]:
        if goal_test(s):
            return []
        if depth == 0 or s in path:
            return None
        for sense in sensing:
            if sense.applicable(s):
                cases = {}
                for label, outcome in sense.outcomes(s).items():
                    sub = or_search(outcome, path + (s,), depth - 1)
                    if sub is None:
                        break
                    cases[label] = sub
                else:
                    return [Branch(sense, cases)]
        for a in actions:
            if a.applicable(s):
                sub = or_search(a.apply(s), path + (s,), depth - 1)
                if sub is not None:
                    return [a] + sub
        return None

    for depth in range(max_depth + 1):
        plan = or_search(state, (), depth)
        if plan is not None:
            return plan
    return None


def branch_outcomes(plan: ConditionalPlan, state: State) -> List[Tuple[List[str], State]]:
    """Every (outcome labels, final state) reached by following the plan."""
    for i, step in enumerate(plan):
        if isinstance(step, Branch):
            out = []
            for label, outcome in step.sensing.outcomes(state).items():
                for labels, final in branch_outcomes(step.cases[label], outcome):
                    out.append(([label] + labels, final))
            return out
        if not step.applicable(state):
            raise ValueError(f"{step} is not applicable at step {i}")
        state = step.apply(state)
    return [([], state)]


def format_conditional(plan: ConditionalPlan, indent: int = 0) -> str:
    pad = "  " * indent
    lines = []
    for step in plan:
        if isinstance(step, Branch):
            lines.append(f"{pad}{step.sensing}")
            for label, sub in step.cases.items():
                lines.append(f"{pad}if {label}:")
                lines.append(format_conditional(sub, indent + 1) if sub else f"{pad}  (nothing)")
        else:
            lines.append(f"{pad}{step}")
    return "\n".join(lines)


class ExecutionAgent:
    """Kept for older callers: wraps an Executor around a fresh simulated environment."""

    def __init__(self, env: Optional[NetworkEnvironment] = None):
        self.executor = Executor(env or NetworkEnvironment())

    def execute(self, problem: Problem) -> ExecutionReport:
        return self.executor.execute(problem)


def hosts_in(action: Action) -> Set[str]:
    return {a.name for a in action.args if isinstance(a, Const)}
