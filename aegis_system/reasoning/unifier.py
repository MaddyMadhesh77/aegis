from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


Term = Any


def occurs_in(var: str, term: Any) -> bool:
    if term == var:
        return True
    if isinstance(term, tuple):
        return any(occurs_in(var, part) for part in term)
    if isinstance(term, list):
        return any(occurs_in(var, part) for part in term)
    return False


def unify(left: Any, right: Any, subst: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if subst is None:
        subst = {}

    if left == right:
        return subst

    if isinstance(left, str) and left[0].islower() and left not in {"True", "False"}:
        if left in subst:
            return unify(subst[left], right, subst)
        if isinstance(right, str) and right[0].islower() and right not in {"True", "False"}:
            subst[left] = right
            return subst
        subst[left] = right
        return subst

    if isinstance(right, str) and right[0].islower() and right not in {"True", "False"}:
        if right in subst:
            return unify(left, subst[right], subst)
        subst[right] = left
        return subst

    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            return None
        for l_item, r_item in zip(left, right):
            result = unify(l_item, r_item, subst)
            if result is None:
                return None
            subst = result
        return subst

    return None
