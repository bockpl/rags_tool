"""Application configuration loading using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SummRAGSettings(BaseSettings):
    """Centralised configuration for the rags_tool service."""

    app_name: str = "rags_tool"
    app_version: str = "0.9.5"

    qdrant_url: str = Field(default="http://127.0.0.1:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_request_timeout: float = Field(default=60.0, alias="QDRANT_TIMEOUT")
    embedding_api_url: str = Field(default="http://127.0.0.1:8000/v1", alias="EMBEDDING_API_URL")
    embedding_api_key: str = Field(default="sk-no-key", alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    summary_api_url: str = Field(default="http://127.0.0.1:8001/v1", alias="SUMMARY_API_URL")
    summary_api_key: str = Field(default="sk-no-key", alias="SUMMARY_API_KEY")
    summary_model: str = Field(default="gpt-4o-mini", alias="SUMMARY_MODEL")
    collection_name: str = Field(default="rags_tool", alias="COLLECTION_NAME")
    debug: bool = Field(default=False, alias="DEBUG")
    vector_store_dir: Path = Field(default=Path(".rags_tool_store"), alias="VECTOR_STORE_DIR")
    # Embedding vector dimension for the chosen embedding model
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")
    # Prefer JSON responses for summaries (OpenAI JSON mode). Fallback to text parser if unsupported.
    summary_json_mode: bool = Field(default=True, alias="SUMMARY_JSON_MODE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> SummRAGSettings:
    """Return cached application settings instance."""

    return SummRAGSettings()
