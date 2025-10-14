"""Application configuration loading using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SummRAGSettings(BaseSettings):
    """Centralised configuration for the rags_tool service."""

    app_name: str = "rags_tool"
    app_version: str = "2.9.0"

    qdrant_url: str = Field(default="http://127.0.0.1:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_request_timeout: float = Field(default=60.0, alias="QDRANT_TIMEOUT")
    embedding_api_url: str = Field(default="http://127.0.0.1:8000/v1", alias="EMBEDDING_API_URL")
    embedding_api_key: str = Field(default="sk-no-key", alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    # Tokenizer spec controlling chunking + local token limits
    embedding_tokenizer: str = Field(
        default="tiktoken:cl100k_base", alias="EMBEDDING_TOKENIZER"
    )
    # Max tokens per single input to the embedding endpoint (safety cap)
    embedding_max_tokens: int = Field(default=512, alias="EMBEDDING_MAX_TOKENS")
    # (Debugging note) embedding_max_tokens is a local safety cap only; the backend
    # may use different tokenization.
    # Prefixes used by some retrieval models that expect instruction-style inputs
    # Defaults align with sdadas/mmlw-retrieval-roberta-large-v2
    embedding_query_prefix: str = Field(default="query: ", alias="EMBEDDING_QUERY_PREFIX")
    embedding_passage_prefix: str = Field(default="passage: ", alias="EMBEDDING_PASSAGE_PREFIX")
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
            " kluczowych), 'ENTITIES' (nazwy własne/ID/zakresy dat), 'DATE' (data "
            " wprowadzenia/ogłoszenia dokumentu; preferuj format YYYY-MM-DD lub YYYY; wpisz "
            " dokładnie 'brak', jeśli brak informacji) oraz 'REPLACEMENT' (krótka lista tytułów, "
            " zawsze w mianowniku; może składać się jedynie z krótkich tytułów aktów zastępowanych, "
            " jeśli tekst jednolity wypisz wszystkie tytuły aktów które ujednolica, rozdziel „;”; "
            " wpisz dokładnie 'brak', jeżeli brak danych). Bez komentarzy.\\n\\nFORMAT:\\nTITLE: ...\\nSUMMARY: ...\\nSIGNATURE: "
            " lemma1, lemma2, ...\\nENTITIES: ...\\nDATE: ...\\nREPLACEMENT: ...\\n\\nTEKST:\\n"
        ),
        alias="SUMMARY_PROMPT",
    )
    summary_prompt_json: str = Field(
        default=(
            "Zwróć wyłącznie poprawny JSON bez komentarzy i bez kodu. Klucze: 'title' (string; "
            " krótki jednoznaczny tytuł, preferuj pierwszą linię lub numer dokumentu; max 200 znaków, "
            " zawsze w mianowniku), 'summary' (string; max 5 zdań po polsku), 'signature' "
            " (lista 10–20 lematów kluczowych jako strings), 'entities' (string z nazwami "
            " własnymi/ID/zakresami dat), 'doc_date' (string; data wprowadzenia/ogłoszenia dokumentu "
            " w formacie 'YYYY-MM-DD' lub 'YYYY-MM' lub 'YYYY'; jeśli brak informacji wpisz dokładnie 'brak'), "
            " 'replacement' (string; krótkie tytuły aktów zastępowanych, zawsze w mianowniku; może składać się jedynie z krótkich tytułów aktów "
            " zastępowanych, jeśli tekst jednolity wypisz wszystkie tytuły aktów które ujednolica, separator ';'; wpisz dokładnie 'brak', jeśli brak informacji)."
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

    # Chunking defaults (token-based). Tune per embedding model.
    chunk_tokens: int = Field(default=400, alias="CHUNK_TOKENS")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    section_merge_level: str = Field(default="ust", alias="SECTION_MERGE_LEVEL")

    # OpenAPI / tool description used by the /search/query endpoint. Can be overridden
    # via .env to tailor the wording for a specific corpus (e.g., PŁ documents).
    search_tool_description: str = Field(
        default=(
            "Proste wyszukiwanie w korpusie (RAG + reranker). Wywołuj TYLKO w ten sposób: "
            "POST /search/query z JSON: {\n  \"query\": [\n    \"krótkie pytanie\",\n    \"wariant pytania\"\n  ],\n  \"mode\": \"auto\"\n}. "
            "Nie wysyłaj innych pól — limity, wagi i scalanie są ustawione po stronie serwera. "
            "Zwracane jest pole 'blocks' (lista z polami: text, path, ranker_score gdy dostępny). "
            "Używaj do pytań o treści z indeksu; przy braku wyników zawężaj temat lub doprecyzuj daty."
        ),
        alias="SEARCH_TOOL_DESCRIPTION",
    )

    # Globalny przełącznik: pomiń Etap 1 (streszczenia) i szukaj od razu w całym korpusie (chunkach).
    # Sterowany wyłącznie przez admina z .env (brak parametru w API).
    search_skip_stage1_default: bool = Field(
        default=False, alias="SEARCH_SKIP_STAGE1_DEFAULT"
    )

    # --- Hybryda 2‑query (dense + sparse w dwóch zapytaniach) ---
    # Gdy true, Stage 1 i Stage 2 wykonują dwa zapytania: osobno po dense i sparse,
    # a następnie łączą wyniki po stronie aplikacji. Pozwala docelowo usunąć TF‑IDF
    # z payloadów Qdranta (mniejsze rekordy i niższe CPU po stronie serwera).
    search_dual_query_sparse: bool = Field(default=False, alias="SEARCH_DUAL_QUERY_SPARSE")
    dual_query_rrf_k: int = Field(default=60, alias="DUAL_QUERY_RRF_K")
    dual_query_oversample: int = Field(default=2, alias="DUAL_QUERY_OVERSAMPLE")
    dual_query_dense_for_mmr: bool = Field(default=True, alias="DUAL_QUERY_DENSE_FOR_MMR")

    # Redukcja payloadów (wybór pól with_payload). Zalecane pozostawić włączone.
    search_minimal_payload: bool = Field(default=True, alias="SEARCH_MINIMAL_PAYLOAD")

    # Batchowanie sekcji per dokument (jedna kwerenda scroll łącząca sekcje po prefiksach)
    batch_section_fetch: bool = Field(default=True, alias="BATCH_SECTION_FETCH")

    # --- Validators for forgiving .env parsing (blank strings) ---
    @field_validator(
        "search_skip_stage1_default",
        "search_dual_query_sparse",
        "dual_query_dense_for_mmr",
        "search_minimal_payload",
        "batch_section_fetch",
        mode="before",
    )
    @classmethod
    def _coerce_bool_env(cls, v):  # type: ignore[override]
        if isinstance(v, str) and v.strip() == "":
            return False
        return v

    @field_validator("dual_query_rrf_k", mode="before")
    @classmethod
    def _coerce_rrf_k(cls, v):  # type: ignore[override]
        if isinstance(v, str) and v.strip() == "":
            return 60
        return v

    @field_validator("dual_query_oversample", mode="before")
    @classmethod
    def _coerce_oversample(cls, v):  # type: ignore[override]
        if isinstance(v, str) and v.strip() == "":
            return 2
        return v

    # --- Reranker (OpenAI-compatible) minimal configuration ---
    # Pusty BASE_URL lub MODEL oznacza wyłączony ranker i brak rerankingu.
    # K i N są kontrolowane z .env, nie przez publiczne API.
    ranker_base_url: Optional[str] = Field(default=None, alias="RANKER_BASE_URL")
    ranker_api_key: Optional[str] = Field(default=None, alias="RANKER_API_KEY")
    ranker_model: Optional[str] = Field(default=None, alias="RANKER_MODEL")
    rerank_top_n: int = Field(default=50, alias="RERANK_TOP_N")
    return_top_k: int = Field(default=5, alias="RETURN_TOP_K")
    ranker_score_threshold: float = Field(default=0.2, alias="RANKER_SCORE_THRESHOLD")
    # Długość kontekstu dla pojedynczego passage wysyłanego do rankera (znaki, przybliżenie).
    # Jeśli model ma twardy limit tokenów, rekomendujemy ustawić konserwatywnie (np. 2048 znaków).
    ranker_max_length: int = Field(default=2048, alias="RANKER_MAX_LENGTH")

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
