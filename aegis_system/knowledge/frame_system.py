"""Frames with facets and procedural attachments (demons).

A slot has the facets value, default, if_added, if_needed and if_removed.
Frames inherit slot definitions through is_a (the `parent` frame).

    get(slot)    tries, on the frame and then on each ancestor in turn:
                 the value, then the if_needed demon (called with the frame that
                 was asked, so a generic frame can compute values for its
                 instances), then the default.
    set(slot, v) stores v, then runs the nearest if_added demon.
    remove(slot) deletes the value, then runs the nearest if_removed demon.

Demons receive (frame, slot, value) and are inherited like any other facet,
so a demon on the generic Host frame fires for every host instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

Demon = Callable[["Frame", str, Any], Any]
NeededDemon = Callable[["Frame", str], Any]
_MISSING = object()


@dataclass
class Slot:
    value: Any = _MISSING
    default: Any = _MISSING
    if_added: Optional[Demon] = None
    if_needed: Optional[NeededDemon] = None
    if_removed: Optional[Demon] = None


class Frame:
    def __init__(self, name: str, slots: Optional[Dict[str, Any]] = None, parent: Optional["Frame"] = None,
                 system: Optional["FrameSystem"] = None, generic: bool = False):
        self.name = name
        self.generic = generic  # a class-like frame (Host) rather than an individual (Host3)
        self.parent = parent
        self.system = system
        self.slots: Dict[str, Slot] = {}
        for k, v in (slots or {}).items():
            self.slots[k] = Slot(value=v)  # initial values do not fire demons

    # ----------------------------------------------------------------- structure

    def lineage(self) -> Iterator["Frame"]:
        """This frame, then its parent, and so on up the is_a chain."""
        f: Optional[Frame] = self
        while f is not None:
            yield f
            f = f.parent

    def is_a(self, other: "Frame | str") -> bool:
        name = other if isinstance(other, str) else other.name
        return any(f.name == name for f in self.lineage())

    def define(self, slot: str, default: Any = _MISSING, if_added: Optional[Demon] = None,
               if_needed: Optional[NeededDemon] = None, if_removed: Optional[Demon] = None) -> "Frame":
        """Declare facets for a slot (typically on a generic frame)."""
        s = self.slots.setdefault(slot, Slot())
        if default is not _MISSING:
            s.default = default
        s.if_added = if_added or s.if_added
        s.if_needed = if_needed or s.if_needed
        s.if_removed = if_removed or s.if_removed
        return self

    # ----------------------------------------------------------------- access

    def get(self, slot: str, default: Any = None) -> Any:
        for f in self.lineage():
            s = f.slots.get(slot)
            if s is None:
                continue
            if s.value is not _MISSING:
                return s.value
            if s.if_needed is not None:
                value = s.if_needed(self, slot)
                if value is not None:
                    return value
            if s.default is not _MISSING:
                return s.default
        return default

    def own(self, slot: str) -> Any:
        s = self.slots.get(slot)
        return None if s is None or s.value is _MISSING else s.value

    def set(self, slot: str, value: Any) -> None:
        self.slots.setdefault(slot, Slot()).value = value
        demon = self._facet(slot, "if_added")
        if demon is not None:
            demon(self, slot, value)

    def remove(self, slot: str) -> None:
        s = self.slots.get(slot)
        if s is None or s.value is _MISSING:
            return
        old, s.value = s.value, _MISSING
        demon = self._facet(slot, "if_removed")
        if demon is not None:
            demon(self, slot, old)

    def _facet(self, slot: str, facet: str) -> Any:
        for f in self.lineage():
            s = f.slots.get(slot)
            if s is not None and getattr(s, facet) is not None:
                return getattr(s, facet)
        return None

    def slot_names(self) -> List[str]:
        names: List[str] = []
        for f in self.lineage():
            for k in f.slots:
                if k not in names:
                    names.append(k)
        return names

    def to_dict(self) -> Dict[str, Any]:
        return {k: self.get(k) for k in self.slot_names()}

    def __repr__(self) -> str:
        parent = f" is_a {self.parent.name}" if self.parent else ""
        return f"<Frame {self.name}{parent}>"


class FrameSystem:
    """A registry of frames, so frames can be looked up and created by name."""

    def __init__(self) -> None:
        self.frames: Dict[str, Frame] = {}

    def generic(self, name: str, parent: Optional[str] = None) -> Frame:
        return self._add(Frame(name, parent=self.frames[parent] if parent else None, system=self, generic=True))

    def instance(self, name: str, of: str, slots: Optional[Dict[str, Any]] = None) -> Frame:
        return self._add(Frame(name, slots, parent=self.frames[of], system=self))

    def _add(self, frame: Frame) -> Frame:
        if frame.name in self.frames:
            raise ValueError(f"Frame {frame.name!r} already exists")
        self.frames[frame.name] = frame
        return frame

    def __getitem__(self, name: str) -> Frame:
        return self.frames[name]

    def __contains__(self, name: str) -> bool:
        return name in self.frames

    def instances_of(self, generic: str) -> List[Frame]:
        """Individual (non-generic) frames that are, directly or indirectly, a `generic`."""
        return [f for f in self.frames.values() if not f.generic and f.is_a(generic)]


# Simple named frames kept for older callers.
class HostFrame(Frame):
    pass


class ThreatActorFrame(Frame):
    pass

