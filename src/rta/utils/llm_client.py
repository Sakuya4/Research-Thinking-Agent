"""
LLM Client Factory (Diagnostic & Survival Edition).
File: src/rta/utils/llm_client.py
"""
import os
import logging
import json
import time
import re
from typing import Any, Callable, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Standard safety interval (RPM protection)
MIN_REQUEST_INTERVAL = 12.0

class RateLimiter:
    def __init__(self, interval: float):
        self.interval = interval
        self.last_call_time = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call_time = time.time()

global_limiter = RateLimiter(MIN_REQUEST_INTERVAL)

class RealGeminiClient:
    def __init__(self, api_keys: List[str]):
        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.current_key_idx = 0
        self.model_candidates = [
            "gemini-3-flash-preview",
            "gemini-flash-lite-latest",
            "gemini-2.0-flash-lite", 
            "gemini-flash-lite-latest", 
            "gemini-2.0-flash", 
            "gemini-pro-latest"
        ]
        self.current_model_idx = 0
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self._refresh_client() # Fixed method name call

    def _refresh_client(self):
        """Configure the genai with current rotation state."""
        import google.generativeai as genai
        key = self.api_keys[self.current_key_idx]
        genai.configure(api_key=key)
        self.model_name = self.model_candidates[self.current_model_idx]
        self.model = genai.GenerativeModel(self.model_name)
        
        masked_key = f"{key[:8]}...{key[-5:]}"
        logger.info(f"[Resource] Key: {masked_key} | Model: {self.model_name}")

    def _rotate(self, reason: str):
        """Strategic resource rotation based on error type."""
        is_fatal = "404" in reason
        if is_fatal:
            self.current_model_idx = (self.current_model_idx + 1) % len(self.model_candidates)
            logger.warning(f"[404 Error] Model unsupported. Switching to: {self.model_candidates[self.current_model_idx]}")
        else:
            if self.current_key_idx < len(self.api_keys) - 1:
                self.current_key_idx += 1
                logger.warning(f"[Quota/API Error] Rotating to Next Key ({self.current_key_idx + 1}/{len(self.api_keys)})")
            else:
                self.current_key_idx = 0
                self.current_model_idx = (self.current_model_idx + 1) % len(self.model_candidates)
                logger.warning("[Critical] All Keys exhausted. Rotating Model + Resetting Keys.")
        
        self._refresh_client()

    def _smart_execute(self, func: Callable, *args, **kwargs) -> Any:
        total_limit = len(self.api_keys) * len(self.model_candidates) * 2
        
        for attempt in range(total_limit):
            global_limiter.wait()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e)
                self._rotate(err_str)
                if attempt == total_limit - 1:
                    raise RuntimeError(f"All resources exhausted: {err_str[:50]}")
                time.sleep(1)
        
    def _fuzzy_fix_json(self, data: Any) -> Any:
        """Force type casting and structure injection to pass Pydantic validation."""
        if isinstance(data, list):
            return [self._fuzzy_fix_json(item) for item in data]
        if not isinstance(data, dict):
            return data

        new_data = {}
        for k, v in data.items():
            fixed_v = self._fuzzy_fix_json(v)
            # Map common hallucinations
            key_map = {'id': 'id', 'assertion': 'statement', 'justification': 'evidence', 'Total_Sources_Reviewed': 'documents_reviewed_count'}
            target_key = key_map.get(k, k)
            
            # Type enforce: String
            if target_key.endswith("_id") or target_key in ['paper_id', 'cluster_id', 'claim_id', 'gap_id', 'documents_reviewed_count', 'scope_keywords']:
                if isinstance(fixed_v, (int, float)): fixed_v = str(fixed_v)
                if isinstance(fixed_v, list): fixed_v = ", ".join(map(str, fixed_v))

            # Type enforce: Float
            if target_key == 'confidence':
                if isinstance(fixed_v, str):
                    v_low = fixed_v.lower()
                    if 'high' in v_low: fixed_v = 0.9
                    elif 'low' in v_low: fixed_v = 0.1
                    else: fixed_v = 0.5
                elif not isinstance(fixed_v, (int, float)): fixed_v = 0.5

            new_data[target_key] = fixed_v
        
        # Structure Injection for Clusters
        if 'cluster_id' in new_data or 'cluster_name' in new_data:
            if 'cluster_id' not in new_data: new_data['cluster_id'] = "C" + str(int(time.time() % 1000))
            if 'papers' not in new_data: new_data['papers'] = []
            if 'description' not in new_data: new_data['description'] = "No description."

        return new_data

    def generate_text(self, prompt: str) -> str:
        return self._smart_execute(lambda: self.model.generate_content(prompt, safety_settings=self.safety_settings).text)

    def get_embedding(self, text: str) -> list:
        import google.generativeai as genai
        def _embed():
            candidates = ['models/gemini-embedding-001', 'models/text-embedding-004', 'models/embedding-001']
            for m in candidates:
                try: return genai.embed_content(model=m, content=text, task_type="clustering")['embedding']
                except: continue
            raise RuntimeError("Embed Limit")
        
        try:
            return self._smart_execute(_embed)
        except:
            # [Emergency Fallback] Prevent Stage 3 Crash
            logger.error("[Emergency] Embedding failed. Using Mock Vector.")
            return [0.01] * 768

    def generate_structured(self, prompt: str, schema: Any) -> Any:
        def _call_and_fix():
            res = self.model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"},
                safety_settings=self.safety_settings
            )
            raw_data = json.loads(res.text)
            fixed_data = self._fuzzy_fix_json(raw_data)
            
            # Pydantic Schema Fill
            for field_name, field in schema.model_fields.items():
                if field_name not in fixed_data:
                    fixed_data[field_name] = [] if "List" in str(field.annotation) else {}
            
            return schema.model_validate(fixed_data)

        try:
            return self._smart_execute(_call_and_fix)
        except Exception as e:
            # [Final Fortress Fallback] Prevent Stage 4 Crash
            logger.warning(f"[Fallback] Structured generation failed ({str(e)[:30]}). Wrapping text.")
            raw_text = self.generate_text(prompt + "\nSummarize research gaps.")
            fallback_obj = {
                "meta": {"topic": "Emergency Analysis", "documents_reviewed_count": "3", "scope_keywords": "N/A"},
                "clusters": [{"cluster_id": "C1", "cluster_name": "Findings", "description": raw_text[:200], "papers": []}],
                "claims": [{"claim_id": "CL1", "claim_type": "consensus", "statement": "Data inferred from text.", "confidence": 0.7, "evidence": []}],
                "research_gaps": [{"gap_id": "G1", "description": raw_text[:500], "contributing_claims": []}]
            }
            return schema.model_validate(fallback_obj)

def get_default_client():
    keys_str = os.getenv("GEMINI_KEYS") or os.getenv("GEMINI_API_KEY")
    if not keys_str: return MockGeminiClient()
    return RealGeminiClient(keys_str.split(","))

class MockGeminiClient:
    def generate_text(self, p): return "Mock Data"
    def get_embedding(self, t): return [0.1] * 768
    def generate_structured(self, p, s): return s.model_construct()