from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class EventLogger:
    """
    Structured event logger (JSON Lines).

    Firmware-like logging discipline:
    - fixed fields (ts, stage, event)
    - meta dict reserved for structured diagnostics
    - one event per line (append-only)
    """
    log_path: Path

    def log(self, stage: str, event: str, meta: Optional[Dict[str, Any]] = None) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "stage": stage,
            "event": event,
            "meta": meta or {},
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
