from __future__ import annotations

from typing import Set
from ..schemas.reasoning import ReasoningResult


def validate_reasoning(result: ReasoningResult, valid_paper_ids: Set[str]) -> None:
    """
    Make "thinking" evaluable:
    - every supporting_papers/evidence paper_id must exist in retrieved set
    - research_gaps supporting_papers must exist
    - related_clusters must refer to existing cluster_ids
    """
    cluster_ids = {c.cluster_id for c in result.clusters}

    # claims
    for cl in result.claims:
        for pid in cl.supporting_papers:
            if pid not in valid_paper_ids:
                raise ValueError(f"ReasoningClaim {cl.claim_id} references unknown paper_id: {pid}")
        for ev in cl.evidence:
            if ev.paper_id not in valid_paper_ids:
                raise ValueError(f"Evidence references unknown paper_id: {ev.paper_id}")

    # gaps
    for g in result.research_gaps:
        for pid in g.supporting_papers:
            if pid not in valid_paper_ids:
                raise ValueError(f"ResearchGap {g.gap_id} references unknown paper_id: {pid}")
        for cid in g.related_clusters:
            if cid not in cluster_ids:
                raise ValueError(f"ResearchGap {g.gap_id} references unknown cluster_id: {cid}")
