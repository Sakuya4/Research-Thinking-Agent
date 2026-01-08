from __future__ import annotations

from ..logger import EventLogger
from ..schemas import RetrievalResult, TopicStructuringResult, Cluster


def topic_structuring_mock(rr: RetrievalResult, logger: EventLogger) -> TopicStructuringResult:
    """
    Mock implementation of Stage 3: topic structuring / task decomposition.

    Engineering notes:
    - Phase 3A: embedding + clustering replaces the cluster assignment.
    - Phase 3B: Gemini summarizes clusters into directions/pipeline.
    """
    n = len(rr.papers)
    if n == 0:
        raise ValueError("No papers to structure (retrieval empty).")

    clusters = [
        Cluster(
            cluster_id=0,
            name="Problem Formulation & Task Setup",
            description="Papers that define problem scope, objectives, and task formulations.",
            paper_indices=list(range(0, min(4, n))),
            keywords=["problem", "formulation", "setup"],
            typical_methods=["problem decomposition", "taxonomy building"],
        ),
        Cluster(
            cluster_id=1,
            name="Methods & Training Strategies",
            description="Papers focusing on model families, training recipes, and optimization.",
            paper_indices=list(range(min(4, n), min(8, n))),
            keywords=["method", "training", "optimization"],
            typical_methods=["supervised learning", "reinforcement learning", "self-training"],
        ),
        Cluster(
            cluster_id=2,
            name="Evaluation & Benchmarks",
            description="Papers about datasets, benchmarks, and evaluation protocols.",
            paper_indices=list(range(min(8, n), n)),
            keywords=["benchmark", "metrics", "evaluation"],
            typical_methods=["benchmarking", "ablation studies"],
        ),
    ]

    tsr = TopicStructuringResult(
        clusters=clusters,
        main_directions=[
            "Direction A: Define the research problem and its sub-tasks",
            "Direction B: Identify method families and training strategies",
            "Direction C: Establish evaluation protocols and benchmarks",
        ],
        recommended_pipeline=[
            "Generate expanded search queries and constraints",
            "Retrieve candidate papers from multiple sources (arXiv/Scholar/IEEE)",
            "Cluster papers into subtopics and label them",
            "Derive task decomposition and method candidates per subtopic",
            "Output a structured research path for the user",
        ],
    )
    logger.log("stage3", "mock_structured", {"clusters": len(tsr.clusters)})
    return tsr
