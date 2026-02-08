"""
Core LLM Client
File: src/rta/utils/llm_client.py
"""

import os
import time
import logging
import json
import google.generativeai as genai
from typing import List, Any, Callable
from dotenv import load_dotenv

load_dotenv()

MIN_REQUEST_INTERVAL = 3.0 
last_req_time = 0.0

logger = logging.getLogger(__name__)

class RealGeminiClient:
    def __init__(self, api_keys: List[str]):
        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.current_key_idx = 0
        self.model_candidates = [
            "models/gemini-3-flash-preview",
            "models/gemini-2.5-flash", 
            "models/gemini-2.0-flash",
        ]
        self.current_model_idx = 0

        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self._refresh_client()

    def _refresh_client(self):
        if not self.api_keys:
            raise ValueError("No API Keys found.")
        genai.configure(api_key=self.api_keys[self.current_key_idx])
        self.model = genai.GenerativeModel(
            model_name=self.model_candidates[self.current_model_idx],
            safety_settings=self.safety_settings
        )
        logger.info(f"[Client] Tier 1 | Model: {self.model_candidates[self.current_model_idx]}")

    def get_embedding(self, text: str) -> List[float]:
        """Uses verified embedding model from your previous successful scan."""
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001", 
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"[Embedding] Error: {e}")
            raise e

    def _execute_safe(self, func: Callable) -> Any:
        global last_req_time
        max_attempts = 15
        for attempt in range(max_attempts):
            now = time.time()
            if (now - last_req_time) < MIN_REQUEST_INTERVAL:
                time.sleep(MIN_REQUEST_INTERVAL)
            
            try:
                last_req_time = time.time()
                return func()
            except Exception as e:
                err = str(e)
                if "429" in err:
                    logger.warning(f"[Quota] Cooling 10s (Attempt {attempt+1})...")
                    time.sleep(10)
                    continue
                # For 404/other model errors, rotate to next candidate
                self.current_model_idx = (self.current_model_idx + 1) % len(self.model_candidates)
                self._refresh_client()
                continue
        raise Exception("RTA exhausted all retry paths. Please check API status.")

    def generate_text(self, prompt: str) -> str:
        return self._execute_safe(lambda: self.model.generate_content(prompt).text)

    def generate_structured(self, prompt: str, schema: Any) -> Any:
        try:
            return self._execute_safe(lambda: self._call_structured(prompt, schema))
        except Exception:
            raw_text = self.generate_text(prompt)
            return self._unstructured_fallback(raw_text, schema)

    def _call_structured(self, prompt: str, schema: Any) -> Any:
        res = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema
            )
        )
        return self._auto_fix_data(json.loads(res.text), schema)

    def _auto_fix_data(self, data: dict, schema: Any) -> Any:
        if 'clusters' in data and isinstance(data['clusters'], list):
            for i, cluster in enumerate(data['clusters']):
                if 'cluster_id' not in cluster: cluster['cluster_id'] = f"auto_{i}"
                if 'papers' not in cluster: cluster['papers'] = ["unmapped"]
        filtered_data = {k: v for k, v in data.items() if k in schema.model_fields}
        return schema.model_validate(filtered_data)

    def _unstructured_fallback(self, text: str, schema: Any) -> Any:
        fallback_data = {
            "topic": "Cardiac Research Results",
            "summary": text,
            "clusters": [{"cluster_name": "Analysis", "cluster_id": "fb", "description": "Recovered text", "papers": []}],
            "beginner_roadmap": ["Review final report"]
        }
        valid_data = {k: v for k, v in fallback_data.items() if k in schema.model_fields}
        return schema.model_validate(valid_data)

def get_default_client():
    raw_keys = os.getenv("GEMINI_KEYS", os.getenv("GEMINI_API_KEY", ""))
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    return RealGeminiClient(keys)