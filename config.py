"""
core/config.py — Centralized configuration for the SENTRY Threat Intel Assistant.

All secrets are read from environment variables / Streamlit secrets — never hardcoded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import streamlit as st
    _SECRETS = st.secrets if hasattr(st, "secrets") else {}
except Exception:  # pragma: no cover - allows core/ to be imported outside Streamlit
    _SECRETS = {}


def _get(key: str, default: str | None = None) -> str | None:
    """Look up a config value: env var first, then st.secrets, then default."""
    val = os.environ.get(key)
    if val:
        return val
    try:
        if key in _SECRETS:
            return _SECRETS[key]
    except Exception:
        pass
    return default


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = str(DATA_DIR / "chroma_store")
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    # --- LLM providers ---
    gemini_api_key: str | None = field(default_factory=lambda: _get("GEMINI_API_KEY"))
    groq_api_key: str | None = field(default_factory=lambda: _get("GROQ_API_KEY"))
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "auto"))  # "gemini" | "groq" | "auto"

    gemini_model: str = "gemini-2.0-flash"
    groq_model: str = "openai/gpt-oss-120b"

    # --- Embeddings (local, no API needed) ---
    embed_model_name: str = "all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5

    # --- External threat intel APIs ---
    nvd_api_key: str | None = field(default_factory=lambda: _get("NVD_API_KEY"))  # optional, raises rate limit
    nvd_base_url: str = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    kev_url: str = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

    # --- Cache TTLs (seconds) ---
    cve_cache_ttl: int = 6 * 3600
    kev_cache_ttl: int = 6 * 3600

    # --- Vector store ---
    chroma_dir: str = CHROMA_DIR
    cve_collection: str = "cve_intel"
    report_collection: str = "uploaded_reports"

    def has_llm(self) -> bool:
        return bool(self.gemini_api_key or self.groq_api_key)


settings = Settings()
