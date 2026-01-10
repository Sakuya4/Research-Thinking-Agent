from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ---------- Stage 0: Input ----------
class InputPayload(BaseModel):
    query: str
    context: Optional[str] = None


# ---------- Stage 1: Query plan ----------
class QueryPlan(BaseModel):
    expanded_queries: List[str] = Field(default_factory=list)
    must_include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)
    target_subtasks: List[str] = Field(default_factory=list)
    notes: str = ""


# ---------- Stage 2: Retrieval ----------
class PaperItem(BaseModel):
    paper_id: str

    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    url: str = ""
    venue: str = ""
    citation_count: Optional[int] = None
    source: str = ""


class RetrievalResult(BaseModel):
    queries_used: List[str] = Field(default_factory=list)
    papers: List[PaperItem] = Field(default_factory=list)
    dedup_before: int = 0
    dedup_after: int = 0
    warnings: List[str] = Field(default_factory=list)


# ---------- Stage 3: Topic structuring ----------
class TopicCluster(BaseModel):
    cluster_id: str
    name: str
    description: str
    paper_indices: List[int] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    typical_methods: List[str] = Field(default_factory=list)


class TopicStructuringResult(BaseModel):
    clusters: List[TopicCluster] = Field(default_factory=list)
    main_directions: List[str] = Field(default_factory=list)
    recommended_pipeline: List[str] = Field(default_factory=list)


# ---------- Final output ----------
class FinalOutput(BaseModel):
    query: str
    main_directions: List[str]
    recommended_pipeline: List[str]
    clusters: List["TopicCluster"]
    top_papers: List["PaperItem"]
    reasoning: Optional["ReasoningResult"] = None


# ---------- Run status ----------
class StageStatus(BaseModel):
    stage1: str = "pending"
    stage2: str = "pending"
    stage3: str = "pending"
    stage4: str = "pending"


class RunStatus(BaseModel):
    run_id: str
    stages: StageStatus = Field(default_factory=StageStatus)
    error: Optional[Dict[str, Any]] = None
