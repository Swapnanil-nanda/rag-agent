"""Centralised, environment-driven application configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, repr=False)
    openrouter_api_key: str | None = Field(default=None, repr=False)
    openrouter_model: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o-mini"
    vectorstore_path: Path = Path("./vectorstore")
    documents_path: Path = Path("./documents")
    log_level: str = "INFO"
    max_upload_mb: int = Field(default=50, ge=1, le=500)
    default_top_k: int = Field(default=4, ge=1, le=12)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
