"""
Global Configuration for RTA Pipeline.
File: src/rta/config.py
"""
from __future__ import annotations
import os
from pydantic_settings import BaseSettings
from pydantic import Field

class RTAConfig(BaseSettings):
    """
    Global configuration for pipeline execution.
    Automatically reads environment variables starting with RTA_ (e.g., RTA_MAX_PAPERS).
    """
    
    # ----------------------------------------------------------------
    # 1. Output Settings
    # ----------------------------------------------------------------
    runs_dir: str = Field(default="runs", description="Directory to store run outputs")

    # ----------------------------------------------------------------
    # 2. Retrieval Settings
    # ----------------------------------------------------------------
    retrieval_mode: str = Field(default="live", description="Mode: 'live' (Real API) or 'mock' (Fake Data)")
    
    # [UPDATED] Default set to 10 to prevent Rate Limit & Speed up Stage 3/4
    max_papers: int = Field(default=2, description="Max papers to retrieve and analyze")
    
    min_year: int = Field(default=2020, description="Filter papers published after this year")
    max_year: int = Field(default=2026, description="Filter papers published before this year")
    
    cache_ttl_hours: int = Field(default=24, description="How long to keep search results in cache")

    # ----------------------------------------------------------------
    # 3. Debugging
    # ----------------------------------------------------------------
    debug_store_llm_raw: bool = Field(default=False, description="Store raw LLM prompts/responses for debugging")

    class Config:
        # This allows you to set config via env vars like RTA_MAX_PAPERS=20
        env_prefix = "RTA_"
        # Allow extra fields in env vars without crashing
        extra = "ignore"

# Global default instance
DEFAULT_CONFIG = RTAConfig()