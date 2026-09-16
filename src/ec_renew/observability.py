"""Structured event log (JSONL).

Doubles as checkpoint source and as the data feed for the explainability
module. Records are written with ``sort_keys=True`` so a replayed run produces
byte-identical output.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlEventLog:
    def __init__(self, path: Path | str, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class NullEventLog:
    """No-op sink for tests that do not care about the log."""

    def emit(self, event: str, **fields: Any) -> None:
        return None

    def close(self) -> None:
        return None