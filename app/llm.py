"""Provider-agnostic chat. ONE constant (LLM_PROVIDER in config) picks gemma4 (local/offline),
a Bedrock-hosted model (BEDROCK_MODEL; prod + build), or Gemini. Existing callers keep the
gemini_chat(system, messages) interface unchanged; the agent uses get_chat_model() directly
so it can .bind_tools(). Embeddings are NOT here — they always stay Gemini (see pipeline.py)."""
import time
from functools import lru_cache

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.config import (LLM_PROVIDER, LOCAL_MODEL, OLLAMA_BASE, BEDROCK_MODEL, BEDROCK_REGION,
                        BEDROCK_AWS_KEY, BEDROCK_AWS_SECRET, GEN_MODEL, GEMINI_API_KEY)


@lru_cache(maxsize=None)
def get_chat_model(provider: str = None, temperature: float = 0.1):
    """One LangChain chat model, cached per (provider, temperature). provider=None uses the
    configured default; pass an explicit provider to override (e.g. the eval judge)."""
    provider = provider or LLM_PROVIDER
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LOCAL_MODEL, base_url=OLLAMA_BASE, temperature=temperature)
    if provider == "bedrock":
        if not BEDROCK_MODEL:
            raise RuntimeError("Set BEDROCK_MODEL to a Bedrock model or inference-profile ID "
                               "to use LLM_PROVIDER=bedrock.")
        import boto3
        from langchain_aws import ChatBedrockConverse
        # Dedicated boto3 client built from the BEDROCK key so it never collides with the S3 key.
        # If the Bedrock vars are unset, session() falls back to the default AWS chain.
        session = boto3.Session(aws_access_key_id=BEDROCK_AWS_KEY,
                                aws_secret_access_key=BEDROCK_AWS_SECRET,
                                region_name=BEDROCK_REGION)
        return ChatBedrockConverse(model=BEDROCK_MODEL,
                                   client=session.client("bedrock-runtime"),
                                   temperature=temperature)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=temperature,
                                      google_api_key=GEMINI_API_KEY)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


_ROLE = {"user": HumanMessage, "model": AIMessage, "assistant": AIMessage}


def gemini_chat(system: str, messages: list[dict], temperature: float = 0.1, max_retries: int = 4) -> str:
    """Send a conversation and return text. Keeps the old name/signature so router.py and chat.py
    are unchanged — but now obeys LLM_PROVIDER. (LangChain's assistant role is 'ai'; our internal
    history uses 'model', so _ROLE translates.) Retries with exponential backoff on transient errors."""
    lc_msgs = [SystemMessage(content=system)] + [_ROLE[m["role"]](content=m["text"]) for m in messages]
    model = get_chat_model(temperature=temperature)
    for attempt in range(max_retries):
        try:
            return model.invoke(lc_msgs).content or ""   # never None
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # out of retries — surface the real error
            wait = 10 * (2 ** attempt)  # 10, 20, 40s — outlasts per-minute rate windows
            print(f"    llm retry {attempt+1}/{max_retries} ({type(e).__name__}); waiting {wait}s")
            time.sleep(wait)
