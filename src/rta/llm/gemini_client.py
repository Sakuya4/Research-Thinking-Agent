from __future__ import annotations
from dataclasses import dataclass, field
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from google import genai  # google-genai SDK
from google.genai import types

from ..logger import EventLogger

load_dotenv()

@dataclass
class GeminiClient:
    """
    Gemini API thin wrapper.

    Firmware-like discipline:
    - minimal side effects
    - explicit inputs/outputs
    - structured logging for each request/response
    """

    api_key: str = field(repr=False)
    model: str = "gemini-flash-latest"
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
        """
        Request JSON output from Gemini, returning (raw_text, meta).

        Notes:
        - Use response_mime_type to encourage strict JSON (JSON mode). :contentReference[oaicite:4]{index=4}
        - Always log model + latency to local run logs for post-mortem debug.
        """
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
        })

        t0 = time.time()
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=cfg,
        )
        latency_ms = int((time.time() - t0) * 1000)

        text = resp.text or ""
        logger.log(stage, "llm_response", {
            "model": self.model,
            "latency_ms": latency_ms,
            "text_len": len(text),
        })

        return text, {"model": self.model, "latency_ms": latency_ms}
