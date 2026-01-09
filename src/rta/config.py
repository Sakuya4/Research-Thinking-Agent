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

    # retrieval
    retrieval_mode: str = Field(default_factory=lambda: os.getenv("RTA_RETRIEVAL_MODE", "live"))
    max_papers: int = Field(default_factory=lambda: int(os.getenv("RTA_MAX_PAPERS", "80")))
    min_year: int = Field(default_factory=lambda: int(os.getenv("RTA_MIN_YEAR", "2020")))
    max_year: int = Field(default_factory=lambda: int(os.getenv("RTA_MAX_YEAR", "2026")))
    cache_ttl_hours: int = Field(default_factory=lambda: int(os.getenv("RTA_CACHE_TTL_HOURS", "24")))

    debug_store_llm_raw: bool = False  # store raw prompt/response cautiously


DEFAULT_CONFIG = RTAConfig()
