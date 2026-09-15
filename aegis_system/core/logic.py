"""First-order logic terms and formulas shared by reasoning, knowledge and planning.

Terms:     Var, Const, Fn
Atoms:     Atom(pred, args)            -- also used as STRIPS literals by the planner
Formulas:  Atom, Not, And, Or, Implies, Iff, ForAll, Exists
Clauses:   frozenset[Literal]

`parse` reads a small text syntax:

    forall x forall y (Compromised(x) & Trusts(y, x) -> Compromised(y))
    exists h (Vulnerable(h) & ~Patched(h))

In the text syntax an identifier starting with a lowercase letter is a variable
(or a function symbol when followed by "("); everything else is a constant or a
predicate. Operator precedence, tightest first: ~, &, |, ->, <->. Quantifier
bodies extend as far to the right as possible.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple, Union


# --------------------------------------------------------------------------- terms


@dataclass(frozen=True)
class Var:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Fn:
    name: str
    args: Tuple["Term", ...]

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(a) for a in self.args)})"


Term = Union[Var, Const, Fn]


# --------------------------------------------------------------------------- formulas


@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple[Term, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.pred
        return f"{self.pred}({', '.join(str(a) for a in self.args)})"


@dataclass(frozen=True)
class Not:
    arg: "Formula"

    def __str__(self) -> str:
        return f"~{_wrap(self.arg)}"


@dataclass(frozen=True)
class And:
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} & {self.right})"


@dataclass(frozen=True)
class Or:
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} | {self.right})"


@dataclass(frozen=True)
class Implies:
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} -> {self.right})"


@dataclass(frozen=True)
class Iff:
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} <-> {self.right})"


@dataclass(frozen=True)
class ForAll:
    var: Var
    body: "Formula"

    def __str__(self) -> str:
        return f"(forall {self.var} {self.body})"


@dataclass(frozen=True)
class Exists:
    var: Var
    body: "Formula"

    def __str__(self) -> str:
        return f"(exists {self.var} {self.body})"


Formula = Union[Atom, Not, And, Or, Implies, Iff, ForAll, Exists]


def _wrap(f: Formula) -> str:
    return str(f) if isinstance(f, (Atom, Not)) or str(f).startswith("(") else f"({f})"


# --------------------------------------------------------------------------- clauses


@dataclass(frozen=True)
class Literal:
    atom: Atom
    positive: bool = True

    def negate(self) -> "Literal":
        return Literal(self.atom, not self.positive)

    def __str__(self) -> str:
        return str(self.atom) if self.positive else f"~{self.atom}"


Clause = FrozenSet[Literal]


def clause_str(clause: Clause) -> str:
    if not clause:
        return "[]"
    return " | ".join(sorted(str(lit) for lit in clause))


# --------------------------------------------------------------------------- helpers


def conjoin(*formulas: Formula) -> Formula:
    if not formulas:
        raise ValueError("conjoin needs at least one formula")
    result = formulas[0]
    for f in formulas[1:]:
        result = And(result, f)
    return result


def disjoin(*formulas: Formula) -> Formula:
    if not formulas:
        raise ValueError("disjoin needs at least one formula")
    result = formulas[0]
    for f in formulas[1:]:
        result = Or(result, f)
    return result


def term_variables(t: object) -> Iterator[Var]:
    """Yield every variable occurrence in a term, atom, literal, clause or formula."""
    if isinstance(t, Var):
        yield t
    elif isinstance(t, (Fn, Atom)):
        for a in t.args:
            yield from term_variables(a)
    elif isinstance(t, Literal):
        yield from term_variables(t.atom)
    elif isinstance(t, Not):
        yield from term_variables(t.arg)
    elif isinstance(t, (And, Or, Implies, Iff)):
        yield from term_variables(t.left)
        yield from term_variables(t.right)
    elif isinstance(t, (ForAll, Exists)):
        yield t.var
        yield from term_variables(t.body)
    elif isinstance(t, (frozenset, set, list, tuple)):
        for item in t:
            yield from term_variables(item)


def free_variables(f: object, bound: FrozenSet[Var] = frozenset()) -> List[Var]:
    """Free variables of a formula, in first-occurrence order."""
    seen: List[Var] = []

    def walk(node: object, bound: FrozenSet[Var]) -> None:
        if isinstance(node, Var):
            if node not in bound and node not in seen:
                seen.append(node)
        elif isinstance(node, (Fn, Atom)):
            for a in node.args:
                walk(a, bound)
        elif isinstance(node, Literal):
            walk(node.atom, bound)
        elif isinstance(node, Not):
            walk(node.arg, bound)
        elif isinstance(node, (And, Or, Implies, Iff)):
            walk(node.left, bound)
            walk(node.right, bound)
        elif isinstance(node, (ForAll, Exists)):
            walk(node.body, bound | {node.var})
        elif isinstance(node, (frozenset, set, list, tuple)):
            for item in node:
                walk(item, bound)

    walk(f, bound)
    return seen


def is_ground(t: object) -> bool:
    return next(term_variables(t), None) is None


_fresh_counter = itertools.count(1)


def fresh_var(base: Var) -> Var:
    """A variable no other code has used, keeping the original name for readability."""
    return Var(f"{base.name.split('#')[0]}#{next(_fresh_counter)}")


def rename_variables(t: object, mapping: Optional[Dict[Var, Var]] = None) -> object:
    """Rename every variable to a fresh one (standardizing apart)."""
    mapping = {} if mapping is None else mapping

    def walk(node: object) -> object:
        if isinstance(node, Var):
            if node not in mapping:
                mapping[node] = fresh_var(node)
            return mapping[node]
        if isinstance(node, Fn):
            return Fn(node.name, tuple(walk(a) for a in node.args))
        if isinstance(node, Atom):
            return Atom(node.pred, tuple(walk(a) for a in node.args))
        if isinstance(node, Literal):
            return Literal(walk(node.atom), node.positive)
        if isinstance(node, Not):
            return Not(walk(node.arg))
        if isinstance(node, (And, Or, Implies, Iff)):
            return type(node)(walk(node.left), walk(node.right))
        if isinstance(node, (ForAll, Exists)):
            return type(node)(walk(node.var), walk(node.body))
        if isinstance(node, frozenset):
            return frozenset(walk(item) for item in node)
        if isinstance(node, tuple):
            return tuple(walk(item) for item in node)
        return node

    return walk(t)


# --------------------------------------------------------------------------- parser


_TOKEN_RE = re.compile(
    r"\s*(?:(?P<op><->|<=>|->|=>|[~!&|(),.])|(?P<name>[A-Za-z_][A-Za-z0-9_']*|\d+))"
)
_KEYWORDS = {"forall", "exists", "not", "and", "or"}


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if not match or match.end() == pos:
            raise SyntaxError(f"Unexpected character at {pos}: {text[pos:pos + 10]!r}")
        tokens.append(match.group("op") or match.group("name"))
        pos = match.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return tokens


class _Parser:
    def __init__(self, text: str):
        self.tokens = _tokenize(text)
        self.pos = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of input")
        if expected is not None and tok != expected:
            raise SyntaxError(f"Expected {expected!r}, got {tok!r}")
        self.pos += 1
        return tok

    def parse(self) -> Formula:
        f = self.iff()
        if self.peek() is not None:
            raise SyntaxError(f"Unexpected token {self.peek()!r}")
        return f

    def iff(self) -> Formula:
        left = self.implies()
        while self.peek() in ("<->", "<=>"):
            self.take()
            left = Iff(left, self.implies())
        return left

    def implies(self) -> Formula:
        left = self.disjunction()
        if self.peek() in ("->", "=>"):
            self.take()
            return Implies(left, self.implies())  # right associative
        return left

    def disjunction(self) -> Formula:
        left = self.conjunction()
        while self.peek() in ("|", "or"):
            self.take()
            left = Or(left, self.conjunction())
        return left

    def conjunction(self) -> Formula:
        left = self.unary()
        while self.peek() in ("&", "and"):
            self.take()
            left = And(left, self.unary())
        return left

    def unary(self) -> Formula:
        tok = self.peek()
        if tok in ("~", "!", "not"):
            self.take()
            return Not(self.unary())
        if tok in ("forall", "exists"):
            self.take()
            variables = [self.variable()]
            while self.peek() == ",":
                self.take()
                variables.append(self.variable())
            while self.peek() is not None and self.peek()[0].islower() and self.peek() not in _KEYWORDS \
                    and self.tokens[self.pos + 1:self.pos + 2] != ["("]:
                variables.append(self.variable())
            if self.peek() == ".":
                self.take()
            body = self.iff()
            quant = ForAll if tok == "forall" else Exists
            for v in reversed(variables):
                body = quant(v, body)
            return body
        if tok == "(":
            self.take()
            f = self.iff()
            self.take(")")
            return f
        return self.atom()

    def variable(self) -> Var:
        name = self.take()
        if not name[0].islower() or name in _KEYWORDS:
            raise SyntaxError(f"Expected a variable, got {name!r}")
        return Var(name)

    def atom(self) -> Atom:
        name = self.take()
        if not (name[0].isupper() or name[0] == "_" or name[0].isdigit()):
            raise SyntaxError(f"Predicate names start with an uppercase letter: {name!r}")
        if self.peek() != "(":
            return Atom(name, ())
        return Atom(name, self.arguments())

    def arguments(self) -> Tuple[Term, ...]:
        self.take("(")
        args: List[Term] = []
        if self.peek() != ")":
            args.append(self.term())
            while self.peek() == ",":
                self.take()
                args.append(self.term())
        self.take(")")
        return tuple(args)

    def term(self) -> Term:
        name = self.take()
        if self.peek() == "(":
            return Fn(name, self.arguments())
        if name[0].islower():
            return Var(name)
        return Const(name)


def parse(text: str) -> Formula:
    """Parse a formula from text. See the module docstring for the syntax."""
    return _Parser(text).parse()


def parse_atom(text: str) -> Atom:
    f = parse(text)
    if not isinstance(f, Atom):
        raise SyntaxError(f"Expected a single atom, got {f}")
    return f


def parse_term(text: str) -> Term:
    p = _Parser(text)
    t = p.term()
    if p.peek() is not None:
        raise SyntaxError(f"Unexpected token {p.peek()!r}")
    return t
