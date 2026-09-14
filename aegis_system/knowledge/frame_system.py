from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Frame:
    name: str
    slots: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HostFrame(Frame):
    pass


@dataclass
class ThreatActorFrame(Frame):
    pass
