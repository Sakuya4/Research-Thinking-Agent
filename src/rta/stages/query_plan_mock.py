from __future__ import annotations

from ..schemas import InputPayload, QueryPlan
from ..logger import EventLogger


def query_plan_mock(user_input: InputPayload, logger: EventLogger) -> QueryPlan:
    """
    Mock implementation of Stage 1: query expansion / search planning.

    Engineering notes:
    - In Phase 1, replace this function with Gemini-based planner.
    - Keep output schema stable to avoid ripple changes downstream.
    """
    base = user_input.query.strip()
    expanded = [
        base,
        f"{base} survey",
        f"{base} dataset",
        f"{base} benchmark",
        f"{base} method taxonomy",
        f"{base} training strategy",
        f"{base} evaluation metrics",
        f"{base} agent planning",
        f"{base} retrieval augmented generation",
        f"{base} workflow",
    ]

    qp = QueryPlan(
        expanded_queries=expanded[:12],
        must_include=["method", "approach", "framework"],
        exclude=["tutorial", "blog", "advertisement"],
        target_subtasks=[
            "problem definition",
            "task decomposition",
            "data/benchmark identification",
            "method families and training",
            "evaluation protocols",
        ],
        notes="Mock query plan (Phase 0).",
    )
    logger.log("stage1", "mock_generated", {"n": len(qp.expanded_queries)})
    return qp
