"""Knowledge representation and ontology layer."""

from .frame_system import Frame, HostFrame, ThreatActorFrame
from .semantic_net import SemanticNet

__all__ = ["Frame", "HostFrame", "ThreatActorFrame", "SemanticNet"]
