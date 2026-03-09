from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ProjectStateStore:
    path: Path

    @classmethod
    def from_path(cls, path: str | Path) -> "ProjectStateStore":
        return cls(path=Path(path).resolve())

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def save(self, state: dict[str, Any]) -> None:
        state["last_update"] = self._now_iso()
        self.path.write_text(json.dumps(state, indent=2) + "\n")

    def append_event(self, state: dict[str, Any], event: str) -> None:
        state.setdefault("event_log", [])
        state["event_log"].append({"time": self._now_iso(), "event": event})

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().astimezone().replace(microsecond=0).isoformat()
