from __future__ import annotations

from typing import Any, Dict, List


def generate_attack_simulation() -> Dict[str, Any]:
    network = {"start": "HostA", "target": "DomainController", "edges": {"HostA": ["HostB"], "HostB": ["DomainController"]}}
    path = ["HostA", "HostB", "DomainController"]
    actions = [
        {"name": "block_port", "target": "HostB"},
        {"name": "isolate", "target": "HostA"},
    ]
    return {"network": network, "path": path, "actions": actions}
