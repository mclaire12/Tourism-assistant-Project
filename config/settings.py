"""Environment-backed settings.

Secrets are read from the process environment or from Streamlit secrets.
Nothing in this module has a default password, API key, or database URI.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / "data" / "rwanda_places.json"
CHROMA_PATH = ROOT / "data" / "chroma"
HERO_PATH = ROOT / "assets" / "rwanda_hero.png"
LOGO_PATH = ROOT / "assets" / "rwanda_logo.png"

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024


def _in_streamlit() -> bool:
    if "streamlit.runtime.scriptrunner" not in sys.modules:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _read(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()
    if _in_streamlit():
        try:
            import streamlit as st

            secret = st.secrets[name]
            if secret is None:
                return default
            text = str(secret).strip()
            return text or default
        except Exception:
            return default
    return default


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    model_provider: str
    model_api_url: str
    model_api_key: str
    model_name: str
    embedding_provider: str
    embedding_model: str
    request_timeout: float

    @property
    def neo4j_configured(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_username and self.neo4j_password)

    @property
    def model_configured(self) -> bool:
        provider = self.model_provider.lower()
        if provider in {"", "none", "retrieval", "retrieval_only"}:
            return False
        if provider == "local":
            return True
        return bool(self.model_api_key and (self.model_api_url or provider == "huggingface"))


def get_settings() -> Settings:
    provider = _read("MODEL_PROVIDER").lower()
    api_url = _read("MODEL_API_URL")
    if provider == "huggingface" and not api_url:
        api_url = "https://router.huggingface.co/v1"
    return Settings(
        neo4j_uri=_read("NEO4J_URI"),
        neo4j_username=_read("NEO4J_USERNAME"),
        neo4j_password=_read("NEO4J_PASSWORD"),
        model_provider=provider,
        model_api_url=api_url.rstrip("/"),
        model_api_key=_read("MODEL_API_KEY"),
        model_name=_read("MODEL_NAME", DEFAULT_MODEL_NAME),
        embedding_provider=_read("EMBEDDING_PROVIDER", "auto").lower(),
        embedding_model=_read("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        request_timeout=float(_read("REQUEST_TIMEOUT", "45") or "45"),
    )
