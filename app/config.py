"""Centralised, environment-driven application configuration."""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

def _default_vectorstore_path() -> Path:
    if os.environ.get("VERCEL") or os.environ.get("LAMBDA_TASK_ROOT"):
        return Path("/tmp/vectorstore")
    return Path("./vectorstore")

def _default_documents_path() -> Path:
    if os.environ.get("VERCEL") or os.environ.get("LAMBDA_TASK_ROOT"):
        return Path("/tmp/documents")
    return Path("./documents")

class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, repr=False)
    openrouter_api_key: str | None = Field(default=None, repr=False)
    openrouter_model: str = "google/gemini-2.0-flash-exp:free"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o-mini"
    vectorstore_path: Path = Field(default_factory=_default_vectorstore_path)
    documents_path: Path = Field(default_factory=_default_documents_path)
    log_level: str = "INFO"
    max_upload_mb: int = Field(default=50, ge=1, le=500)
    default_top_k: int = Field(default=4, ge=1, le=12)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
