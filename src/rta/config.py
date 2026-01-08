from __future__ import annotations

from pydantic import BaseModel, Field
import os


class RTAConfig(BaseModel):
    """
    Global configuration for pipeline execution.

    Notes:
    - Keep config serializable (JSON) for run reproducibility.
    - Add fields incrementally as phases progress.
    """
    runs_dir: str = Field(default_factory=lambda: os.getenv("RTA_RUNS_DIR", "runs"))
    max_papers: int = 50
    debug_store_llm_raw: bool = False  # Phase 1: store raw prompt/response cautiously


DEFAULT_CONFIG = RTAConfig()
