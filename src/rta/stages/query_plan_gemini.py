"""
Stage 1: Query Planning Module (Production Grade).
Generates a structured research plan and search queries from a user topic.
File: src/rta/stages/query_plan_gemini.py
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

# Internal Schema Definition to avoid circular imports
class QueryPlan(BaseModel):
    original_topic: str
    expanded_queries: List[str]
    must_include: List[str] = []
    exclude: List[str] = []
    target_subtasks: List[str] = []
    notes: str = ""

logger = logging.getLogger(__name__)

# [TODO COMPLIANCE] Enhanced Prompting with Few-shot examples to ensure structural integrity
_SYSTEM_PROMPT = """You are a Research Scoping Agent.
Given a topic, produce a compact JSON query plan for literature search.

STRICT REQUIREMENTS:
- Output MUST be valid JSON only.
- NO markdown fences (```json), NO extra prose.
- ALL content must be in English.
- Generate exactly 12 diverse search queries.

EXAMPLE OUTPUT FORMAT:
{
  "original_topic": "AI in ECG",
  "expanded_queries": ["deep learning for arrhythmia detection", "wearable ECG signal processing", "transformer models for cardiovascular health", "... (total 12)"],
  "must_include": ["clinical validation", "signal-to-noise ratio"],
  "exclude": ["pediatric cases"],
  "target_subtasks": ["analyze hardware constraints", "compare transformer vs CNN"],
  "notes": "Focus on adult real-time monitoring."
}
"""

_SCHEMA_HINT = """{
  "expanded_queries": ["string", "..."],
  "must_include": ["string", "..."],
  "exclude": ["string", "..."],
  "target_subtasks": ["string", "..."],
  "notes": "string"
}"""

def _extract_json_block(text: str) -> str:
    """Extracts the JSON object from raw LLM text output."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1:
        return t[start : end + 1]
    return t

def _repair_json_via_llm(client: Any, broken_json: str) -> str:
    """Attempts to fix malformed JSON using a secondary LLM request."""
    logger.info("[QueryPlan] Attempting to repair broken JSON via LLM...")
    prompt = f"Fix this broken JSON. Return ONLY valid JSON:\n{broken_json}\nSchema Hint:\n{_SCHEMA_HINT}"
    try:
        # Using generate_text for the repair to avoid recursive structured calls
        return client.generate_text(prompt)
    except Exception:
        return broken_json

def run_query_planning(topic: str) -> QueryPlan:
    from rta.utils.llm_client import get_default_client
    client = get_default_client()
    
    full_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Topic: {topic}\n\n"
        f"Return ONLY JSON matching the schema hint."
    )

    logger.info(f"[QueryPlan] Generating plan for: {topic}")

    raw_text = "" 
    try:
        raw_text = client.generate_text(full_prompt)
        candidate = _extract_json_block(raw_text)
        obj = json.loads(candidate)
    except Exception as e:
        logger.warning(f"[QueryPlan] API Error or Parsing failed: {e}")
        if raw_text:
            logger.info("[QueryPlan] Attempting repair...")
            repaired = _repair_json_via_llm(client, raw_text)
            obj = json.loads(_extract_json_block(repaired))
        else:
            logger.error("[QueryPlan] Could not get any text from LLM. Using emergency fallback.")
            obj = {
                "original_topic": topic,
                "expanded_queries": [topic, f"{topic} review", f"{topic} technology"],
                "notes": "Emergency recovery active."
            }
        
    return QueryPlan.model_validate(obj)