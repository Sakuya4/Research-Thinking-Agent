from __future__ import annotations

# core models
from .core import (
    InputPayload,
    QueryPlan,
    PaperItem,
    RetrievalResult,
    TopicCluster,
    TopicStructuringResult,
    FinalOutput,
    StageStatus,
    RunStatus,
)

# reasoning models
from .reasoning import (
    Evidence,
    ClusteredPaper,
    Cluster,
    ReasoningClaim,
    ResearchGap,
    ReasoningResult,
)


FinalOutput.model_rebuild()
TopicStructuringResult.model_rebuild()

__all__ = [
    "InputPayload",
    "QueryPlan",
    "PaperItem",
    "RetrievalResult",
    "TopicCluster",
    "TopicStructuringResult",
    "FinalOutput",
    "StageStatus",
    "RunStatus",
    "Evidence",
    "ClusteredPaper",
    "Cluster",
    "ReasoningClaim",
    "ResearchGap",
    "ReasoningResult",
]
