"""
core/llm.py — Unified LLM client.

Prefers Gemini (via google-generativeai). Falls back to Groq if Gemini is
unavailable or errors out. Exposes a single `generate()` call so the rest of
the app never needs to know which provider answered.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from core.config import settings

logger = logging.getLogger("sentry.llm")


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMUnavailableError(RuntimeError):
    pass


class LLMClient:
    """Thin wrapper that tries Gemini first, then Groq, based on settings.llm_provider."""

    def __init__(self):
        self._gemini_model = None
        self._groq_client = None
        self._init_gemini()
        self._init_groq()

    def _init_gemini(self):
        if not settings.gemini_api_key:
            return
        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.gemini_api_key)
            self._gemini_model = genai.GenerativeModel(settings.gemini_model)
        except Exception as exc:  # pragma: no cover
            logger.warning("Gemini init failed: %s", exc)
            self._gemini_model = None

    def _init_groq(self):
        if not settings.groq_api_key:
            return
        try:
            from groq import Groq

            self._groq_client = Groq(api_key=settings.groq_api_key)
        except Exception as exc:  # pragma: no cover
            logger.warning("Groq init failed: %s", exc)
            self._groq_client = None

    @property
    def available(self) -> bool:
        return bool(self._gemini_model or self._groq_client)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1500) -> LLMResponse:
        provider_order = []
        if settings.llm_provider == "gemini":
            provider_order = ["gemini"]
        elif settings.llm_provider == "groq":
            provider_order = ["groq"]
        else:  # auto
            provider_order = ["gemini", "groq"]

        last_error: Exception | None = None
        for provider in provider_order:
            try:
                if provider == "gemini" and self._gemini_model:
                    resp = self._gemini_model.generate_content(
                        prompt,
                        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                    )
                    text = (resp.text or "").strip()
                    if text:
                        return LLMResponse(text=text, provider="gemini", model=settings.gemini_model)
                if provider == "groq" and self._groq_client:
                    completion = self._groq_client.chat.completions.create(
                        model=settings.groq_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    text = completion.choices[0].message.content.strip()
                    if text:
                        return LLMResponse(text=text, provider="groq", model=settings.groq_model)
            except Exception as exc:
                logger.warning("Provider %s failed: %s", provider, exc)
                last_error = exc
                continue

        raise LLMUnavailableError(
            f"No LLM provider produced a response. Last error: {last_error}"
            if last_error
            else "No LLM provider is configured. Add GEMINI_API_KEY or GROQ_API_KEY."
        )


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm_client():
    """Call after the user updates API keys in Settings so new keys take effect."""
    global _client
    _client = None
