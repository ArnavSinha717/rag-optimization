import time

from google.genai import types  # Importing the types module from the google.genai package
from app.config import client, GEN_MODEL # Importing the client and GEN_MODEL from the config module

def gemini_chat(system: str, messages: list[dict], temperature: float=0.1, max_retries: int=4)-> str:
    """Send a conversation to gemini and return the response as a string.
    Retries with exponential backoff on transient errors (503 high demand, 429 rate limit)."""
    contents=[{"role": m["role"],"parts":[{"text": m["text"]}]} for m in messages]

    for attempt in range(max_retries):
        try:
            response=client.models.generate_content(
                model=GEN_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                ),
            )
            return response.text or ""   # never None — empty string if model produced no text
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # out of retries — surface the real error to the caller
            wait = 10 * (2 ** attempt)  # 10, 20, 40s — outlasts per-minute rate windows
            print(f"    gemini retry {attempt+1}/{max_retries} ({type(e).__name__}); waiting {wait}s")
            time.sleep(wait)
    