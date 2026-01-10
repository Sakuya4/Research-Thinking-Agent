from __future__ import annotations

import hashlib
from typing import List

from ..config import RTAConfig
from ..logger import EventLogger
from ..schemas import QueryPlan, RetrievalResult, PaperItem


def _mock_paper_id(source: str, title: str, url: str) -> str:
    """
    Deterministic paper_id for mocks (stable across runs).
    """
    base = f"{source}|{url}|{title}".encode("utf-8")
    return hashlib.sha1(base).hexdigest()[:12]


def retrieval_mock(cfg: RTAConfig, qp: QueryPlan, logger: EventLogger) -> RetrievalResult:
    """
    Mock implementation of literature retrieval.

    Notes:
    - Used for pipeline validation without network calls.
    - Must output stable paper_id for downstream reasoning/evidence linking.
    """
    mock_papers: List[PaperItem] = []

    topic = (qp.expanded_queries[0] if qp.expanded_queries else "unknown topic")

    for i in range(min(cfg.max_papers, 12)):
        title = f"Mock Paper {i+1}: {topic}"
        url = f"https://example.com/mock/{i+1}"

        mock_papers.append(
            PaperItem(
                paper_id=_mock_paper_id("mock", title, url),
                title=title,
                authors=["A. Author", "B. Author"],
                year=2024,
                abstract="This is a mock abstract used for pipeline validation.",
                url=url,
                source="mock",
            )
        )

    dedup_before = len(mock_papers)
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
