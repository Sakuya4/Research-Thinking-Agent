from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from .llm.gemini_client import GeminiClient
from .logger import EventLogger
from .schemas import InputPayload, QueryPlan


class AgentReply(BaseModel):
    topic_summary: str = Field(..., description="1 short paragraph in English.")
    key_terms: list[dict[str, str]] = Field(
        ..., description="3-5 items of {term, definition} in English."
    )
    suggested_directions: list[str] = Field(
        ..., description="3-5 bullet points in English."
    )
    suggested_search_queries: list[str] = Field(
        ..., description="Up to 5 queries in English."
    )


_SYSTEM = """You are a research thinking agent.
Given a topic and (optionally) a query plan, produce a helpful user-facing response.

Hard requirements:
- Output MUST be valid JSON only (no markdown fences, no extra text).
- ALL strings MUST be in English.
- Keep it compact. Do NOT output long paragraphs.
"""

_SCHEMA_HINT = """{
  "topic_summary": "string",
  "key_terms": [{"term":"string","definition":"string"}],
  "suggested_directions": ["string"],
  "suggested_search_queries": ["string"]
}"""


def _extract_json_block(text: str) -> str:
    """
    Extract the first {...} block if the model accidentally adds extra text.
    """
    t = text.strip()

    # Remove common markdown fences if any
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)

    # Find first JSON object block
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    return t


def _repair_json_via_llm(
    client: GeminiClient,
    logger: EventLogger,
    broken_json_text: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Ask Gemini to repair broken JSON into valid JSON.
    """
    system = """You fix broken JSON.
Return ONLY valid JSON. No markdown. No explanations.
All strings must be in English."""

    user_payload = {
        "task": "repair_json",
        "broken_json_text": broken_json_text,
        "schema_hint": _SCHEMA_HINT,
    }

    repaired_text, meta = client.generate_json(
        logger=logger,
        stage="reply_repair",
        system=system,
        user=json.dumps(user_payload, ensure_ascii=False),
        schema_hint=_SCHEMA_HINT,
        temperature=0.0,
        max_output_tokens=2000,
    )
    return repaired_text, meta


def build_agent_reply(
    user_input: InputPayload,
    logger: EventLogger,
    query_plan: Optional[QueryPlan],
    run_dir: Path,
) -> AgentReply:
    client = GeminiClient.from_env()

    # IMPORTANT: keep reply compact to avoid truncation
    payload: Dict[str, Any] = {
        "topic": user_input.query,
        "context": user_input.context or "",
        "query_plan": query_plan.model_dump() if query_plan else None,
        "output_rules": {
            "language": "English only",
            "topic_summary": "max 60 words",
            "key_terms": "3-5 items, each definition max 25 words",
            "suggested_directions": "3-5 items, each max 18 words",
            "suggested_search_queries": "max 5 queries, short",
        },
    }

    raw_text, meta = client.generate_json(
        logger=logger,
        stage="reply",
        system=_SYSTEM,
        user=json.dumps(payload, ensure_ascii=False),
        schema_hint=_SCHEMA_HINT,
        temperature=0.2,
        max_output_tokens=2000,  # key fix: avoid truncation
    )

    logger.log(
        "reply",
        "raw_preview",
        {
            "preview": raw_text[:240],
            "model": meta.get("model"),
            "latency_ms": meta.get("latency_ms"),
        },
    )

    # ---- Attempt 1: parse directly (with extraction) ----
    candidate = _extract_json_block(raw_text)
    try:
        obj = json.loads(candidate)
    except Exception as e:
        logger.log("reply", "json_parse_failed", {"attempt": 1, "reason": str(e)})

        # ---- Attempt 2-3: repair up to 2 times ----
        last_err: Optional[Exception] = None
        repaired_candidate = candidate

        for attempt in (2, 3):
            repaired_text, rmeta = _repair_json_via_llm(client, logger, repaired_candidate)
            repaired_candidate = _extract_json_block(repaired_text)

            logger.log(
                "reply",
                "json_repair_preview",
                {"attempt": attempt, "preview": repaired_candidate[:200], "model": rmeta.get("model")},
            )

            try:
                obj = json.loads(repaired_candidate)
                last_err = None
                break
            except Exception as e2:
                last_err = e2
                logger.log("reply", "json_parse_failed", {"attempt": attempt, "reason": str(e2)})

        if last_err is not None:
            logger.log("reply", "schema_validated", {"ok": False, "reason": f"json_parse_error_after_repairs: {last_err}"})
            raise RuntimeError(f"Reply JSON parse failed after repairs: {last_err}")

    # ---- Pydantic validation ----
    try:
        reply = AgentReply.model_validate(obj)
        logger.log("reply", "schema_validated", {"ok": True})
        return reply
    except ValidationError as e:
        logger.log("reply", "schema_validated", {"ok": False, "reason": "pydantic_validation_error"})
        raise RuntimeError(f"Reply schema validation failed: {e}")


def print_agent_reply(reply: AgentReply, run_dir: Path) -> None:
    print("\nRTA: Topic summary")
    print(f"{reply.topic_summary}\n")

    print("RTA: Key terms (glossary)")
    for item in reply.key_terms:
        term = (item.get("term") or "").strip()
        definition = (item.get("definition") or "").strip()
        if term and definition:
            print(f"  - {term}: {definition}")
    print("")

    print("RTA: Suggested research directions")
    for i, d in enumerate(reply.suggested_directions, 1):
        print(f"  {i}. {d}")
    print("")

    print("RTA: Suggested search queries")
    for q in reply.suggested_search_queries:
        print(f"  - {q}")
    print("")

    print("RTA: Saved outputs")
    print(f"  - {run_dir}")
    print(f"  - {run_dir / 'query_plan.json'}")
    print("")
