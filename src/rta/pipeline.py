from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .config import RTAConfig
from .logger import EventLogger
from .run_manager import new_run_dir, write_json, init_status, update_status, write_report_md

from .schemas import (
    InputPayload,
    QueryPlan,
    RetrievalResult,
    TopicStructuringResult,
    FinalOutput,
)

from .stages.query_plan_gemini import stage_query_plan_gemini
from .stages.retrieval_mock import retrieval_mock
from .stages.retrieval_live import retrieval_live
from .stages.topic_structuring_mock import topic_structuring_mock
from .stages.reasoning_agent import run_reasoning_agent

from .validators.reasoning_validator import validate_reasoning


def run_pipeline(cfg: RTAConfig, user_input: InputPayload) -> Tuple[str, Path]:
    """
    Execute pipeline with run artifacts persisted.

    Stages:
      1) query plan (LLM)
      2) retrieval (mock/live)
      3) topic structuring (mock for now)
      4) reasoning agent (LLM + validator)
    """
    run_dir = new_run_dir(cfg, user_input.query)
    logger = EventLogger(log_path=run_dir / "logs.jsonl")

    status = init_status(run_dir)
    write_json(run_dir / "input.json", user_input.model_dump())
    write_json(run_dir / "config.json", cfg.model_dump())

    # ---- Stage 1 ----
    try:
        logger.log("stage1", "start", {"query": user_input.query})
        qp: QueryPlan = stage_query_plan_gemini(user_input, logger)
        write_json(run_dir / "query_plan.json", qp.model_dump())
        status.stages.stage1 = "ok"
        logger.log("stage1", "done", {"expanded_queries": len(qp.expanded_queries)})
    except Exception as e:
        status.stages.stage1 = "fail"
        status.error = {"stage": "stage1", "message": str(e)}
        logger.log("stage1", "fail", {"error": str(e)})
        update_status(run_dir, status)
        raise

    update_status(run_dir, status)

    # ---- Stage 2 ----
    try:
        logger.log("stage2", "start", {"queries": len(qp.expanded_queries), "mode": cfg.retrieval_mode})

        if cfg.retrieval_mode.lower() == "mock":
            rr: RetrievalResult = retrieval_mock(cfg, qp, logger)
        else:
            rr: RetrievalResult = retrieval_live(cfg, qp, logger)

        write_json(run_dir / "retrieval.json", rr.model_dump())
        status.stages.stage2 = "ok"
        logger.log("stage2", "done", {
            "papers": len(rr.papers),
            "dedup_after": rr.dedup_after,
            "warnings": len(rr.warnings),
        })
    except Exception as e:
        status.stages.stage2 = "fail"
        status.error = {"stage": "stage2", "message": str(e)}
        logger.log("stage2", "fail", {"error": str(e)})
        update_status(run_dir, status)
        raise

    update_status(run_dir, status)

    # ---- Stage 3 ----
    try:
        logger.log("stage3", "start", {"papers": len(rr.papers)})
        tsr: TopicStructuringResult = topic_structuring_mock(rr, logger)
        write_json(run_dir / "topic_structuring.json", tsr.model_dump())
        status.stages.stage3 = "ok"
        logger.log("stage3", "done", {"clusters": len(tsr.clusters)})
    except Exception as e:
        status.stages.stage3 = "fail"
        status.error = {"stage": "stage3", "message": str(e)}
        logger.log("stage3", "fail", {"error": str(e)})
        update_status(run_dir, status)
        raise

    update_status(run_dir, status)

    # ---- Stage 4 ----
    try:
        logger.log("stage4", "start", {"papers": len(rr.papers)})

        reasoning = run_reasoning_agent(
            query=user_input.query,
            papers=rr.papers,
            logger=logger,
            prompt_dir=Path("src/rta/prompts"),
        )

        # validator expects that claims/gaps cite valid paper_id
        paper_ids = {cp.paper_id for c in reasoning.clusters for cp in c.papers}
        validate_reasoning(reasoning, paper_ids)

        write_json(run_dir / "reasoning.json", reasoning.model_dump())
        status.stages.stage4 = "ok"

        logger.log("stage4", "done", {
            "clusters": len(reasoning.clusters),
            "claims": len(reasoning.claims),
            "gaps": len(reasoning.research_gaps),
        })

    except Exception as e:
        status.stages.stage4 = "fail"
        status.error = {"stage": "stage4", "message": str(e)}
        logger.log("stage4", "fail", {"error": str(e)})
        update_status(run_dir, status)
        raise

    update_status(run_dir, status)

    # ---- Final packaging ----
    final = FinalOutput(
        query=user_input.query,
        main_directions=tsr.main_directions,
        recommended_pipeline=tsr.recommended_pipeline,
        clusters=tsr.clusters,
        top_papers=rr.papers[: min(10, len(rr.papers))],
        reasoning=reasoning,
    )
    write_json(run_dir / "final_output.json", final.model_dump())

    report = _render_report_md(final)
    write_report_md(run_dir, report)
    logger.log("pipeline", "done", {"run_id": run_dir.name})

    return run_dir.name, run_dir


def _render_report_md(final: FinalOutput) -> str:
    lines = []
    lines.append("# Research Thinking Report\n")
    lines.append(f"**Query**: {final.query}\n")

    lines.append("## Main Directions\n")
    for d in final.main_directions:
        lines.append(f"- {d}")

    lines.append("\n## Recommended Pipeline\n")
    for i, step in enumerate(final.recommended_pipeline, 1):
        lines.append(f"{i}. {step}")

    lines.append("\n## Topic Clusters\n")
    for c in final.clusters:
        lines.append(f"### [{c.cluster_id}] {c.name}")
        lines.append(c.description)
        if c.keywords:
            lines.append(f"- Keywords: {', '.join(c.keywords)}")
        if c.typical_methods:
            lines.append(f"- Methods: {', '.join(c.typical_methods)}")
        lines.append("")

    lines.append("## Top Papers (Preview)\n")
    for p in final.top_papers:
        lines.append(f"- **{p.title}** ({p.year or 'n/a'}) — {p.source}")

    # Reasoning summary (if any field exists)
    if getattr(final, "reasoning", None) is not None:
        lines.append("\n## Reasoning Agent Output\n")
        lines.append(f"- Clusters: {len(final.reasoning.clusters)}")
        lines.append(f"- Claims: {len(final.reasoning.claims)}")
        lines.append(f"- Research gaps: {len(final.reasoning.research_gaps)}")

    lines.append("")
    return "\n".join(lines)
