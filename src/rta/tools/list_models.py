from __future__ import annotations
import os
from dotenv import load_dotenv
from google import genai

def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")

    client = genai.Client(api_key=api_key)

    # List available models for THIS key
    for m in client.models.list():
        name = getattr(m, "name", "")
        # Some SDK versions expose supported methods; print if present
        methods = getattr(m, "supported_generation_methods", None)
        print(name, methods)

if __name__ == "__main__":
    main()