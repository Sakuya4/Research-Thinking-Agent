from __future__ import annotations

from typing import List
from ..config import RTAConfig
from ..logger import EventLogger
from ..schemas import QueryPlan, RetrievalResult, PaperItem


def retrieval_mock(cfg: RTAConfig, qp: QueryPlan, logger: EventLogger) -> RetrievalResult:
    """
    Mock implementation of Stage 2: literature retrieval.

    Engineering notes:
    - Phase 2 will replace with arXiv + Semantic Scholar retrieval.
    - Must provide dedup stats and warnings.
    """
    mock_papers: List[PaperItem] = []
    for i in range(min(cfg.max_papers, 12)):
        mock_papers.append(
            PaperItem(
                title=f"Mock Paper {i+1}: {qp.expanded_queries[0]}",
                authors=["A. Author", "B. Author"],
                year=2024,
                abstract="This is a mock abstract used for Phase 0 pipeline validation.",
                url="https://example.com/mock",
                source="mock",
            )
        )

    dedup_before = len(mock_papers)
    # Mock dedup: do nothing
    dedup_after = len(mock_papers)

    rr = RetrievalResult(
        queries_used=qp.expanded_queries,
        papers=mock_papers,
        dedup_before=dedup_before,
        dedup_after=dedup_after,
        warnings=[],
    )
    logger.log("stage2", "mock_retrieved", {"papers": len(rr.papers)})
    return rr
