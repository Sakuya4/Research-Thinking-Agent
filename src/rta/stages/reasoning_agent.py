from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..llm.gemini_client import GeminiClient
from ..logger import EventLogger
from ..schemas.reasoning import ReasoningResult


DEFAULT_SYSTEM_PROMPT = """You are a Research Reasoning Agent.

Your task is to synthesize retrieved academic papers into structured, verifiable research reasoning.

Hard rules:
- Output MUST be valid JSON only. No markdown. No extra text.
- Every claim MUST be supported by one or more paper_ids.
- Do NOT invent papers, methods, or results.
- If evidence is insufficient, explicitly state a research gap.
- Output MUST follow the provided JSON schema exactly.
- All strings MUST be in English.

Be concise and structured.
"""

DEFAULT_USER_PROMPT = """Topic: {{query}}

You are given a list of academic papers in JSON: {{papers_json}}

Produce:
1) clusters
2) claims
3) research_gaps
4) meta

Return ONLY JSON.
"""


# -------- JSON helpers --------

def _extract_first_complete_json_object(text: str) -> str:
    """
    Extract the first complete JSON object via brace-depth matching.
    Works even if the model adds extra text before/after.
    """
    t = text.strip()

    # Remove code fences if any
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)

    start = t.find("{")
    if start == -1:
        return t

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(t)):
        ch = t[i]

        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]

    # not closed -> return from first "{"
    return t[start:]


def _repair_json_via_llm(
    client: GeminiClient,
    logger: EventLogger,
    broken_json: str,
    schema_hint: str,
) -> str:
    system = (
        "You fix broken JSON.\n"
        "Return ONLY valid JSON. No markdown fences. No explanations.\n"
        "All strings must be in English.\n"
        "Must conform to the provided schema."
    )
    payload = {
        "task": "repair_json",
        "broken_json": broken_json,
    }
    raw, meta = client.generate_json(
        logger=logger,
        stage="stage4_repair",
        system=system,
        user=json.dumps(payload, ensure_ascii=False),
        schema_hint=schema_hint,
        temperature=0.0,
        max_output_tokens=4096,
    )
    logger.log("stage4", "repair_meta", {"model": meta.get("model"), "latency_ms": meta.get("latency_ms")})
    return raw


def ensure_reasoning_prompts(prompt_dir: Path) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)

    system_p = prompt_dir / "reasoning_system.txt"
    user_p = prompt_dir / "reasoning_user.txt"

    created = False
    if not system_p.exists():
        system_p.write_text(DEFAULT_SYSTEM_PROMPT, encoding="utf-8")
        created = True
    if not user_p.exists():
        user_p.write_text(DEFAULT_USER_PROMPT, encoding="utf-8")
        created = True
    if created:
        pass


def _schema_hint_reasoning() -> str:
    return """{
  "clusters": [
    {
      "cluster_id": "string",
      "cluster_name": "string",
      "description": "string",
      "papers": [
        {"paper_id":"string","title":"string","why_included":"string"}
      ],
      "key_methods": ["string"],
      "time_span": {"start": 2020, "end": 2026}
    }
  ],
  "claims": [
    {
      "claim_id":"string",
      "claim_type":"trend|comparison|limitation|consensus",
      "statement":"string",
      "supporting_papers":["string"],
      "evidence":[{"paper_id":"string","evidence":"string"}],
      "confidence": 0.0
    }
  ],
  "research_gaps": [
    {
      "gap_id":"string",
      "description":"string",
      "related_clusters":["string"],
      "supporting_papers":["string"],
      "significance":"string"
    }
  ],
  "meta": {"model":"string","notes":"string"}
}"""


def _build_user_prompt(query: str, papers: List[Dict[str, Any]], tmpl: str) -> str:
    papers_json = json.dumps(papers, ensure_ascii=False, indent=2)
    out = tmpl.replace("{{query}}", query)
    out = out.replace("{{papers_json}}", papers_json)
    return out


def run_reasoning_agent(
    *,
    query: str,
    papers: List[Any],
    logger: EventLogger,
    prompt_dir: Path,
) -> ReasoningResult:
    """
    Robust reasoning stage:
    - ensure prompts exist
    - parse JSON strictly
    - if truncated/invalid -> repair up to 2 times
    - if still invalid -> retry with fewer papers (auto shrink)
    """
    ensure_reasoning_prompts(prompt_dir)

    system_prompt = (prompt_dir / "reasoning_system.txt").read_text(encoding="utf-8")
    user_tmpl = (prompt_dir / "reasoning_user.txt").read_text(encoding="utf-8")
    schema_hint = _schema_hint_reasoning()

    client = GeminiClient.from_env()

    def to_dict(p: Any) -> Dict[str, Any]:
        if hasattr(p, "model_dump"):
            return p.model_dump()
        if isinstance(p, dict):
            return p
        return dict(p)

    papers_dict = [to_dict(p) for p in papers]

    candidate_sizes = []
    n = len(papers_dict)
    if n <= 20:
        candidate_sizes = [n]
    else:
        candidate_sizes = [min(n, 80), min(n, 40), min(n, 20)]

    last_error: Exception | None = None

    for size in candidate_sizes:
        subset = papers_dict[:size]
        user_prompt = _build_user_prompt(query, subset, user_tmpl)

        raw_text, meta = client.generate_json(
            logger=logger,
            stage="reasoning",
            system=system_prompt,
            user=user_prompt,
            schema_hint=schema_hint,
            temperature=0.2,
            max_output_tokens=4096,
        )

        logger.log("stage4", "reasoning_raw_preview", {
            "papers_used": size,
            "preview": (raw_text or "")[:250],
            "text_len": len(raw_text or ""),
            "model": meta.get("model"),
            "latency_ms": meta.get("latency_ms"),
        })

        if not (raw_text or "").strip():
            last_error = RuntimeError("Stage4 returned empty response (possible quota/safety/truncation).")
            logger.log("stage4", "json_parse_failed", {"error": str(last_error), "papers_used": size})
            continue

        candidate = _extract_first_complete_json_object(raw_text)
        try:
            obj = json.loads(candidate)
            return ReasoningResult.model_validate(obj)
        except Exception as e:
            last_error = e
            logger.log("stage4", "json_parse_failed", {
                "error": str(e),
                "papers_used": size,
                "raw_preview": candidate[:250],
            })

        # repair up to 2 times on this size
        repaired_candidate = candidate
        for attempt in (1, 2):
            try:
                repaired = _repair_json_via_llm(client, logger, repaired_candidate, schema_hint=schema_hint)
                repaired_candidate = _extract_first_complete_json_object(repaired)
                obj = json.loads(repaired_candidate)
                return ReasoningResult.model_validate(obj)
            except Exception as e2:
                last_error = e2
                logger.log("stage4", "json_repair_failed", {
                    "attempt": attempt,
                    "papers_used": size,
                    "error": str(e2),
                    "repaired_preview": repaired_candidate[:250],
                })

        # repair still failed -> try smaller size
        logger.log("stage4", "retry_with_smaller_papers", {"from": size})

    # exhausted all sizes
    raise RuntimeError(f"Stage4 reasoning failed after retries: {last_error}")
