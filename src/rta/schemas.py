from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class InputPayload(BaseModel):
    """
    User input payload (stable interface).
    """
    query: str = Field(..., description="User's initial research idea / keywords")
    context: Optional[str] = Field(default=None, description="Optional context/constraints")


class QueryPlan(BaseModel):
    """
    Stage 1 output schema: query expansion / search planning.
    """
    expanded_queries: List[str]
    must_include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)
    target_subtasks: List[str] = Field(default_factory=list)
    notes: str = ""


class PaperItem(BaseModel):
    """
    Minimal paper metadata representation.
    """
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    url: str = ""
    source: str = ""  # arXiv / SemanticScholar / IEEE / etc.


class RetrievalResult(BaseModel):
    """
    Stage 2 output schema: literature retrieval.
    """
    queries_used: List[str]
    papers: List[PaperItem]
    dedup_before: int
    dedup_after: int
    warnings: List[str] = Field(default_factory=list)


class Cluster(BaseModel):
    """
    Cluster representation for topic structuring.
    """
    cluster_id: int
    name: str
    description: str
    paper_indices: List[int] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    typical_methods: List[str] = Field(default_factory=list)


class TopicStructuringResult(BaseModel):
    """
    Stage 3 output schema: topic structuring / task decomposition.
    """
    clusters: List[Cluster]
    recommended_pipeline: List[str] = Field(default_factory=list)
    main_directions: List[str] = Field(default_factory=list)


class FinalOutput(BaseModel):
    """
    Final packaged output (what you show to users / UI / AI Studio).
    """
    query: str
    main_directions: List[str]
    recommended_pipeline: List[str]
    clusters: List[Cluster]
    top_papers: List[PaperItem] = Field(default_factory=list)


class StageStatus(BaseModel):
    stage1: str = "pending"  # ok/fail/pending
    stage2: str = "pending"
    stage3: str = "pending"


class RunStatus(BaseModel):
    run_id: str
    stages: StageStatus
    error: Optional[Dict[str, Any]] = None
