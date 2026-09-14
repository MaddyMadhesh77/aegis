from __future__ import annotations

from typing import Any, Dict, Optional


class ExecutionAgent:
    """Minimal execution layer for defensive actions."""

    def execute_action(self, action_name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        if action_name in {"isolate", "block_port", "kill_process"}:
            return {"action": action_name, "status": "success", "context": context}
        return {"action": action_name, "status": "replan", "context": context}
