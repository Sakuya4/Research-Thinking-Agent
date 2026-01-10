from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from ..logger import EventLogger

load_dotenv()


def _safe_getattr(obj: Any, path: str, default=None):
    cur = obj
    for p in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, default)
        else:
            cur = getattr(cur, p, default)
    return cur if cur is not None else default


def _extract_text_from_resp(resp: Any) -> str:
    """
    Best-effort extraction for google-genai responses across versions.
    """
    # 1) resp.text (common)
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t

    # 2) candidates[0].content.parts[].text
    cands = getattr(resp, "candidates", None)
    if cands and isinstance(cands, list) and len(cands) > 0:
        content = _safe_getattr(cands[0], "content", None)
        parts = _safe_getattr(content, "parts", None)
        if parts and isinstance(parts, list):
            texts = []
            for part in parts:
                pt = getattr(part, "text", None)
                if isinstance(pt, str) and pt:
                    texts.append(pt)
            if texts:
                return "\n".join(texts)

    return ""


@dataclass
class GeminiClient:
    api_key: str = field(repr=False)
    model: str = "gemini-3-flash"

    @classmethod
    def from_env(cls) -> "GeminiClient":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Please configure .env or environment variables.")
        model = os.getenv("GEMINI_MODEL", "gemini-3-flash").strip()
        return cls(api_key=api_key, model=model)

    def generate_json(
        self,
        logger: EventLogger,
        stage: str,
        system: str,
        user: str,
        schema_hint: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> Tuple[str, Dict[str, Any]]:
        client = genai.Client(api_key=self.api_key)

        prompt = user
        if schema_hint:
            prompt = f"{user}\n\n[SCHEMA_HINT]\n{schema_hint}"

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            system_instruction=system,
        )

        logger.log(stage, "llm_request", {
            "model": self.model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "has_schema_hint": bool(schema_hint),
        })

        t0 = time.time()
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=cfg,
        )
        latency_ms = int((time.time() - t0) * 1000)

        text = _extract_text_from_resp(resp)

        # capture useful meta for debugging
        finish_reason = None
        try:
            cands = getattr(resp, "candidates", None)
            if cands and len(cands) > 0:
                finish_reason = getattr(cands[0], "finish_reason", None)
        except Exception:
            pass

        prompt_feedback = _safe_getattr(resp, "prompt_feedback", None)
        safety_ratings = _safe_getattr(resp, "candidates.0.safety_ratings", None)

        logger.log(stage, "llm_response", {
            "model": self.model,
            "latency_ms": latency_ms,
            "text_len": len(text),
            "finish_reason": str(finish_reason) if finish_reason is not None else None,
            "has_prompt_feedback": prompt_feedback is not None,
            "has_safety_ratings": safety_ratings is not None,
        })

        # IMPORTANT: fail fast if empty output
        if not text.strip():
            # put extra info in log
            logger.log(stage, "llm_empty_text", {
                "finish_reason": str(finish_reason) if finish_reason is not None else None,
                "prompt_feedback": str(prompt_feedback)[:500] if prompt_feedback is not None else None,
                "safety_ratings": str(safety_ratings)[:500] if safety_ratings is not None else None,
            })
            raise RuntimeError(
                "Gemini returned empty text. Possible causes: quota/rate-limit, safety block, or SDK response format."
            )

        return text, {"model": self.model, "latency_ms": latency_ms, "finish_reason": str(finish_reason) if finish_reason else None}
