"""Application configuration loading using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SummRAGSettings(BaseSettings):
    """Centralised configuration for the rags_tool service."""

    app_name: str = "rags_tool"
    app_version: str = "1.3.1"

    qdrant_url: str = Field(default="http://127.0.0.1:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_request_timeout: float = Field(default=60.0, alias="QDRANT_TIMEOUT")
    embedding_api_url: str = Field(default="http://127.0.0.1:8000/v1", alias="EMBEDDING_API_URL")
    embedding_api_key: str = Field(default="sk-no-key", alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    summary_api_url: str = Field(default="http://127.0.0.1:8001/v1", alias="SUMMARY_API_URL")
    summary_api_key: str = Field(default="sk-no-key", alias="SUMMARY_API_KEY")
    summary_model: str = Field(default="gpt-4o-mini", alias="SUMMARY_MODEL")
    summary_system_prompt: str = Field(
        default="Jesteś zwięzłym ekstrakcyjnym streszczaczem.",
        alias="SUMMARY_SYSTEM_PROMPT",
    )
    summary_prompt: str = Field(
        default=(
            "Streść poniższy tekst w maks. 5 zdaniach. Wypisz też sekcje: 'TITLE' (krótki, "
            " jednoznaczny tytuł dokumentu — preferuj pierwszą linię lub numer aktu; "
            " pojedyncza fraza, bez dodatkowego komentarza), 'SIGNATURE' (10–20 lematów "
            " kluczowych), 'ENTITIES' (nazwy własne/ID/zakresy dat) oraz 'REPLACEMENT' "
            " (krótka lista tytułów, zawsze w mianowniku; może składać się jedynie z krótkich "
            " tytułów aktów zastępowanych, jeśli tekst jednolity wypisz wszystkie tytuły aktów "
            " które ujednolica, rozdziel średnikami; wpisz dokładnie 'brak', jeżeli brak danych). "
            " Bez komentarzy.\\n\\nFORMAT:\\nTITLE: ...\\nSUMMARY: ...\\nSIGNATURE: "
            " lemma1, lemma2, ...\\nENTITIES: ...\\nREPLACEMENT: ...\\n\\nTEKST:\\n"
        ),
        alias="SUMMARY_PROMPT",
    )
    summary_prompt_json: str = Field(
        default=(
            "Zwróć wyłącznie poprawny JSON bez komentarzy i bez kodu. Klucze: 'title' (string; "
            " krótki jednoznaczny tytuł, preferuj pierwszą linię lub numer dokumentu; max 200 znaków, "
            " zawsze w mianowniku), 'summary' (string; max 5 zdań po polsku), 'signature' "
            " (lista 10–20 lematów kluczowych jako strings), 'entities' (string z nazwami "
            " własnymi/ID/zakresami dat), 'replacement' (string; krótkie tytuły aktów zastępowanych, "
            " zawsze w mianowniku; może składać się jedynie z krótkich tytułów aktów zastępowanych, "
            " jeśli tekst jednolity wypisz wszystkie tytuły aktów które ujednolica, separator ';'; "
            " wpisz dokładnie 'brak', jeśli brak informacji)."
        ),
        alias="SUMMARY_PROMPT_JSON",
    )
    collection_name: str = Field(default="rags_tool", alias="COLLECTION_NAME")
    summary_collection_name: Optional[str] = Field(
        default=None, alias="SUMMARY_COLLECTION_NAME"
    )
    content_collection_name: Optional[str] = Field(
        default=None, alias="CONTENT_COLLECTION_NAME"
    )
    debug: bool = Field(default=False, alias="DEBUG")
    vector_store_dir: Path = Field(default=Path(".rags_tool_store"), alias="VECTOR_STORE_DIR")
    # Embedding vector dimension for the chosen embedding model
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")
    # Prefer JSON responses for summaries (OpenAI JSON mode). Fallback to text parser if unsupported.
    summary_json_mode: bool = Field(default=True, alias="SUMMARY_JSON_MODE")

    @property
    def qdrant_summary_collection(self) -> str:
        return self.summary_collection_name or f"{self.collection_name}_summaries"

    @property
    def qdrant_content_collection(self) -> str:
        return self.content_collection_name or f"{self.collection_name}_content"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> SummRAGSettings:
    """Return cached application settings instance."""

    return SummRAGSettings()
