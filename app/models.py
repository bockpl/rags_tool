"""Pydantic request/response models used by the API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.search import (
    DEFAULT_MMR_LAMBDA,
    DEFAULT_PER_DOC_LIMIT,
    DEFAULT_SCORE_NORM,
)
from app.settings import get_settings

settings = get_settings()


class About(BaseModel):
    name: str = settings.app_name
    version: str = settings.app_version
    author: str = "Seweryn Sitarski (seweryn.sitarski@gmail.com) with support from Kat"
    description: str = "Dwustopniowy RAG ze streszczeniami + hybryda dense/TF‑IDF (Qdrant)"


class InitCollectionsRequest(BaseModel):
    collection_name: str = Field(default_factory=lambda: settings.collection_name)
    force_dim_probe: bool = False


class CollectionsExportRequest(BaseModel):
    collection_names: Optional[List[str]] = Field(
        default=None,
        description="Deprecated filter; eksporter zawsze pobiera wszystkie kolekcje.",
    )


class CollectionsImportRequest(BaseModel):
    archive_base64: str = Field(
        ...,
        description=(
            "Base64-encoded tar.gz snapshot bundle (Qdrant + TF-IDF) produced by /collections/export. "
            "Prefer multipart upload with 'archive_file' when using HTTP clients."
        ),
    )
    replace_existing: bool = Field(
        True,
        description="Drop (and recreate) existing collections and TF-IDF artifacts before import. Should stay true in most cases.",
    )


class ScanRequest(BaseModel):
    base_dir: str
    glob: str = "**/*"
    recursive: bool = True


class ScanResponse(BaseModel):
    files: List[str]


class IngestBuildRequest(BaseModel):
    base_dir: str
    glob: str = "**/*"
    recursive: bool = True
    reindex: bool = False
    chunk_tokens: int = Field(default_factory=lambda: settings.chunk_tokens)
    chunk_overlap: int = Field(default_factory=lambda: settings.chunk_overlap)
    language_hint: Optional[str] = None
    collection_name: str = Field(default_factory=lambda: settings.collection_name)
    enable_sparse: bool = True
    rebuild_tfidf: bool = True


class SummariesGenerateRequest(BaseModel):
    files: List[str]


class SearchQuery(BaseModel):
    query: List[str] = Field(
        ...,
        description=(
            "List of focused queries (each 3–12 words). Provide synonyms or variants to improve recall. "
            "Accepted shapes: string | List[str] | List[List[str]] (nested lists are flattened)."
        ),
    )
    top_m: int = Field(
        100,
        description=(
            "Stage‑1 candidate document count (summaries). Default 100; typical 50–200. "
            "Increase for broad topics, decrease for precise queries."
        ),
    )
    top_k: int = Field(
        10,
        description=(
            "Global final result count after Stage‑2 selection. Typical 5–10. "
            "Use per_doc_limit to prevent dominance of a single document."
        ),
    )
    mode: str = Field(
        "auto",
        description=(
            "Retrieval mode: auto|current|archival|all. 'current' filters is_active=true; 'archival' false. "
            "Heuristic in 'auto': queries with 'obowiązując*' → current; with 'archiwaln*' or explicit years → archival."
        ),
    )
    use_hybrid: bool = Field(True, description="Enable hybrid scoring (dense + TF‑IDF) for query.")
    dense_weight: float = Field(0.6, description="Weight of dense similarity in hybrid relevance [0..1].")
    sparse_weight: float = Field(0.4, description="Weight of sparse (TF‑IDF) similarity in hybrid relevance [0..1].")
    mmr_lambda: float = Field(DEFAULT_MMR_LAMBDA, description="MMR relevance-vs-diversity balance [0..1]. Higher = more relevance.")
    per_doc_limit: int = Field(DEFAULT_PER_DOC_LIMIT, description="Max results per single document in Stage-2.")
    score_norm: str = Field(DEFAULT_SCORE_NORM, description="Score normalization: minmax|zscore|none.")
    rep_alpha: Optional[float] = Field(None, description="Redundancy alpha in hybrid MMR (dense contribution). Defaults to dense_weight.")
    mmr_stage1: bool = Field(True, description="Apply hybrid MMR already at Stage-1 (summaries).")
    summary_mode: str = Field("first", description="Document summary duplication: none|first|all. 'first' shows once per doc.")
    merge_chunks: bool = Field(
        False,
        description=(
            "If true, also build merged blocks per (doc_id, section). Note: when result_format='blocks', blocks are built regardless of this flag."
        ),
    )
    merge_group_budget_tokens: int = Field(1200, description="Approx token budget per merged block (~4 chars/token).")
    max_merged_per_group: int = Field(1, description="Max merged blocks to return for each (doc_id, section) group.")
    block_join_delimiter: str = Field("\n\n", description="Delimiter used when concatenating contiguous chunks in a merged block.")
    expand_neighbors: int = Field(0, description="When merging, also try to include up to N missing adjacent chunks from candidates (mmr_pool). 0 disables.")
    result_format: str = Field(
        "blocks",
        description=(
            "Response shape: flat|grouped|blocks. Default 'blocks' (recommended for tools). When 'blocks', merged evidence blocks are returned (text + path + score)."
        ),
    )

    # Walidator wejścia 'query': akceptuje string, listę stringów lub listę list (zagnieżdżenia),
    # a następnie spłaszcza i czyści wartości do List[str]. Dzięki temu żądania typu
    # "query": [["a", "b" ,"c"]] nie kończą się 422 i są interpretowane jako "query": ["a","b","c"].
    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, v):  # type: ignore[override]
        # Funkcja pomocnicza: przekształca dowolną strukturę do listy niepustych stringów.
        def to_list_of_str(x) -> List[str]:
            if x is None:
                return []
            if isinstance(x, str):
                s = x.strip()
                return [s] if s else []
            if isinstance(x, (list, tuple, set)):
                acc: List[str] = []
                for item in x:
                    acc.extend(to_list_of_str(item))
                return acc
            # Dla innych typów (np. liczby) użyj reprezentacji tekstowej
            s = str(x).strip()
            return [s] if s else []

        out = to_list_of_str(v)
        # Zabezpieczenie: jeżeli po konwersji nic nie zostało, przekaż oryginał (pozwoli Pydanticowi zgłosić 422)
        return out if out else v


class SearchHit(BaseModel):
    doc_id: str = Field(..., description="Stable document identifier (sha1 over absolute path).")
    path: str = Field(..., description="Absolute document path (for citation).")
    section: Optional[str] = Field(default=None, description="Optional document section identifier, if present.")
    chunk_id: int = Field(..., description="Chunk index within the document (0-based).")
    score: float = Field(..., description="Hybrid relevance score (normalized according to score_norm).")
    snippet: str = Field(..., description="Short text snippet of the chunk or summary fallback.")
    summary: Optional[str] = Field(default=None, description="Document-level summary (presence controlled by summary_mode).")


class SearchResponse(BaseModel):
    took_ms: int = Field(..., description="Total search latency in milliseconds.")
    hits: List[SearchHit] = Field(..., description="Flat hit list (chunk-level).")
    groups: Optional[List["SearchGroup"]] = Field(default=None, description="Grouped results per document (summary + chunks).")
    blocks: Optional[List["MergedBlock"]] = Field(default=None, description="Merged blocks per (doc_id, section). Prefer for tool use.")


class SearchChunk(BaseModel):
    chunk_id: int = Field(..., description="Chunk index within the document (0-based).")
    score: float = Field(..., description="Hybrid relevance score (normalized).")
    snippet: str = Field(..., description="Short text snippet of the chunk.")


class SearchGroup(BaseModel):
    doc_id: str = Field(..., description="Stable document identifier.")
    path: str = Field(..., description="Absolute document path.")
    summary: Optional[str] = Field(default=None, description="Document-level summary (single copy per document).")
    score: float = Field(..., description="Max score among group's chunks.")
    chunks: List[SearchChunk] = Field(..., description="Chunk-level results belonging to this document.")


class MergedBlock(BaseModel):
    doc_id: str = Field(..., description="Stable document identifier.")
    path: str = Field(..., description="Absolute document path.")
    section: Optional[str] = Field(default=None, description="Optional section identifier.")
    first_chunk_id: int = Field(..., description="First chunk id (inclusive) in this merged block.")
    last_chunk_id: int = Field(..., description="Last chunk id (inclusive) in this merged block.")
    score: float = Field(..., description="Block score = max score among its member chunks.")
    summary: Optional[str] = Field(default=None, description="Document/section summary if requested by summary_mode.")
    text: str = Field(..., description="Merged textual content of the block (joined contiguous chunks).")
    token_estimate: Optional[int] = Field(default=None, description="Heuristic token length (~4 chars/token).")
    # Pola opcjonalne dla rerankera (jeśli włączony):
    ranker_score: Optional[float] = Field(default=None, description="Ocena jakości nadana przez ranker (0..1).")
    ranker_applied: Optional[bool] = Field(default=None, description="Czy zastosowano reranker do wyniku.")
    ranker_model: Optional[str] = Field(default=None, description="Nazwa modelu rankera użytego do oceny.")


# Rebuild forward refs
SearchResponse.model_rebuild()
