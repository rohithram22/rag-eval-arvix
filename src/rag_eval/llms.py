"""LLM factory: builds the generator, question generator and judge from config.

All three roles go through this one function, so switching a provider or
model is a one-line config change. (Step 6 adds retries + a disk cache.)
"""
from __future__ import annotations

import os

PROVIDER_ENV_KEYS = {"groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY"}


def _require_key(provider: str) -> str:
    env_name = PROVIDER_ENV_KEYS[provider]
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"{env_name} is not set. Add it as a Codespaces secret (then reload "
            f"the window) or put it in .env"
        )
    return key


def get_llm(section: dict):
    """Return a LlamaIndex LLM for a config section like cfg['generator']."""
    provider = section["provider"]
    model = section["model"]
    temperature = section.get("temperature", 0.0)
    max_tokens = section.get("max_tokens", 512)

    if provider == "groq":
        from llama_index.llms.groq import Groq

        return Groq(
            model=model,
            api_key=_require_key("groq"),
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=section.get("extra", {}),
        )

    if provider == "gemini":
        from llama_index.llms.google_genai import GoogleGenAI

        return GoogleGenAI(
            model=model,
            api_key=_require_key("gemini"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown LLM provider '{provider}'. Expected one of {list(PROVIDER_ENV_KEYS)}")
