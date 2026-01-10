from __future__ import annotations

from typing import Dict, List, Literal
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    paper_id: str
    evidence: str


class ClusteredPaper(BaseModel):
    paper_id: str
    title: str
    why_included: str


class Cluster(BaseModel):
    cluster_id: str
    cluster_name: str
    description: str
    papers: List[ClusteredPaper]
    key_methods: List[str] = Field(default_factory=list)
    time_span: Dict[str, int] = Field(default_factory=dict)


class ReasoningClaim(BaseModel):
    claim_id: str
    claim_type: Literal["trend", "comparison", "limitation", "consensus"]
    statement: str
    supporting_papers: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchGap(BaseModel):
    gap_id: str
    description: str
    related_clusters: List[str] = Field(default_factory=list)
    supporting_papers: List[str] = Field(default_factory=list)
    significance: str = ""


class ReasoningResult(BaseModel):
    clusters: List[Cluster] = Field(default_factory=list)
    claims: List[ReasoningClaim] = Field(default_factory=list)
    research_gaps: List[ResearchGap] = Field(default_factory=list)
    meta: Dict[str, str] = Field(default_factory=dict)
