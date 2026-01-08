from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict

from .config import RTAConfig
from .schemas import RunStatus, StageStatus


def _slugify(text: str, max_len: int = 32) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\-_ ]+", "", text)
    text = re.sub(r"\s+", "-", text)
    return text[:max_len] if len(text) > max_len else text


def new_run_dir(cfg: RTAConfig, query: str) -> Path:
    """
    Create a new run directory with a stable naming convention.

    Convention:
    YYYY-MM-DD_HHMMSS_<slug>
    """
    ts = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    slug = _slugify(query)
    run_id = f"{ts}_{slug}"
    run_dir = Path(cfg.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def init_status(run_dir: Path) -> RunStatus:
    status = RunStatus(run_id=run_dir.name, stages=StageStatus())
    write_json(run_dir / "status.json", status.model_dump())
    return status


def update_status(run_dir: Path, status: RunStatus) -> None:
    write_json(run_dir / "status.json", status.model_dump())


def write_report_md(run_dir: Path, report_md: str) -> None:
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")
