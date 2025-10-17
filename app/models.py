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
    # Force regeneration of summaries and overwrite sidecar cache
    force_regen_summary: bool = Field(
        False,
        description="When true, bypass sidecar cache and regenerate LLM summary + dense vector, overwriting .summary/*.json.gz",
    )


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
    # Runtime chunk-merging removed in 2.0.0. Blocks are built directly
    # from section-aware chunks generated at ingest time.
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
    title: Optional[str] = Field(default=None, description="Document title extracted during ingest.")
    doc_date: Optional[str] = Field(default=None, description="Document date (YYYY, YYYY-MM, or YYYY-MM-DD) or 'brak'.")
    is_active: Optional[bool] = Field(default=None, description="Whether the document is marked as current (true) or archival (false).")
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
    title: Optional[str] = Field(default=None, description="Document title extracted during ingest.")
    doc_date: Optional[str] = Field(default=None, description="Document date (YYYY, YYYY-MM, or YYYY-MM-DD) or 'brak'.")
    is_active: Optional[bool] = Field(default=None, description="Whether the document is marked as current (true) or archival (false).")
    summary: Optional[str] = Field(default=None, description="Document-level summary (single copy per document).")
    score: float = Field(..., description="Max score among group's chunks.")
    chunks: List[SearchChunk] = Field(..., description="Chunk-level results belonging to this document.")


class MergedBlock(BaseModel):
    doc_id: str = Field(..., description="Stable document identifier.")
    path: str = Field(..., description="Absolute document path.")
    title: Optional[str] = Field(default=None, description="Document title extracted during ingest.")
    doc_date: Optional[str] = Field(default=None, description="Document date (YYYY, YYYY-MM, or YYYY-MM-DD) or 'brak'.")
    is_active: Optional[bool] = Field(default=None, description="Whether the document is marked as current (true) or archival (false).")
    section: Optional[str] = Field(default=None, description="Optional section identifier.")
    first_chunk_id: int = Field(..., description="First chunk id (inclusive) in this merged block.")
    last_chunk_id: int = Field(..., description="Last chunk id (inclusive) in this merged block.")
    score: float = Field(..., description="Block score = max score among its member chunks.")
    summary: Optional[str] = Field(default=None, description="Document/section summary if requested by summary_mode.")
    text: str = Field(..., description="Merged textual content of the block (joined contiguous chunks).")
    # Pola opcjonalne dla rerankera (jeśli włączony):
    ranker_score: Optional[float] = Field(default=None, description="Ocena jakości nadana przez ranker (0..1).")


# Rebuild forward refs
SearchResponse.model_rebuild()


class SparseQuery(BaseModel):
    indices: List[int]
    values: List[float]


# --- Contradictions analysis models ---

class ContradictionAnalysisRequest(BaseModel):
    title: str = Field(..., description="Tytuł dokumentu referencyjnego (lub jego charakterystyczny nagłówek)")
    doc_id: Optional[str] = Field(
        default=None,
        description="Opcjonalnie znane doc_id dokumentu referencyjnego (przyspiesza identyfikację)",
    )
    mode: str = Field(
        default="current",
        description=(
            "Zakres przeszukiwania: current|archival|all. 'current' filtruje is_active=true; 'archival' false; 'all' bez filtra."
        ),
    )
    section_level: str = Field(
        default="ust",
        description=(
            "Poziom sekcji do raportowania i grupowania. Rekomendowane 'ust'. Dozwolone: chapter|par|ust|pkt|lit."
        ),
    )
    max_candidates_per_section: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maksymalna liczba kandydatów (sekcji z innych dokumentów) rozpatrywanych na sekcję",
    )
    include_archival_conflicts: bool = Field(
        default=False,
        description="Czy raportować konflikty z dokumentami archiwalnymi (is_active=false) jako informacyjne",
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimalny próg pewności LLM dla ujęcia w raporcie",
    )


class ContradictionFinding(BaseModel):
    other_doc_id: str
    other_title: Optional[str] = None
    other_path: Optional[str] = None
    other_doc_date: Optional[str] = None
    other_is_active: Optional[bool] = None
    other_section: Optional[str] = None
    label: str
    confidence: float
    rationale: Optional[str] = None
    quotes_a: Optional[List[str]] = None
    quotes_b: Optional[List[str]] = None
    # Audit fields
    subject_a: Optional[str] = Field(default=None, description="Zidentyfikowany podmiot reguły po stronie A (sekcja referencyjna)")
    subject_b: Optional[str] = Field(default=None, description="Zidentyfikowany podmiot reguły po stronie B (sekcja kandydująca)")
    same_subject: Optional[bool] = Field(default=None, description="Czy A i B dotyczą tego samego aktu/podmiotu")
    rule_type: Optional[str] = Field(default=None, description="Klasyfikacja reguły (np. entry_into_force, deadline, scope, threshold, repeal, other)")


class ContradictionSectionReport(BaseModel):
    section: Optional[str]
    rule: Optional[str] = Field(default=None, description="Zredukowana reguła/teza wyekstrahowana z sekcji A")
    rule_subject: Optional[str] = Field(default=None, description="Podmiot/akt, którego dotyczy reguła sekcji A (zidentyfikowany)")
    rule_type: Optional[str] = Field(default=None, description="Klasyfikacja reguły sekcji A")
    conflicts: List[ContradictionFinding]


class ContradictionAnalysisResponse(BaseModel):
    doc_id: str
    title: Optional[str] = None
    doc_date: Optional[str] = None
    is_active: Optional[bool] = None
    findings: List[ContradictionSectionReport]
    took_ms: int = Field(..., description="Całkowity czas wykonania analizy (ms)")
    # Lista dokumentów, które zostały realnie przetworzone (dla audytu)
    class ProcessedDoc(BaseModel):
        doc_id: str
        title: Optional[str] = None
        doc_date: Optional[str] = None
        is_active: Optional[bool] = None

    processed_docs: List[ProcessedDoc] = Field(
        default_factory=list,
        description="Lista dokumentów uwzględnionych w analizie (zawsze zawiera dokument referencyjny)",
    )


# --- Documents listing (for external tools) ---

class DocListItem(BaseModel):
    """Lightweight document metadata used by tools to enumerate corpus."""

    doc_id: str = Field(..., description="Stable document identifier (sha1 over absolute path).")
    title: Optional[str] = Field(default=None, description="Document title extracted during ingest.")
    doc_date: Optional[str] = Field(default=None, description="Document date (YYYY, YYYY-MM, or YYYY-MM-DD) or 'brak'.")
    is_active: Optional[bool] = Field(default=None, description="Whether the document is marked as current (true) or archival (false).")


class DocsListResponse(BaseModel):
    """Response envelope for documents list endpoint."""

    docs: List[DocListItem]


# --- Multi-query debug models (mirror /search/query with step-by-step outputs) ---

class DebugMultiEmbedRequest(BaseModel):
    # Accept single query or list of queries (flattened)
    query: object
    mode: str = "auto"
    use_hybrid: bool = True
    top_m: int = 100
    top_k: int = 10
    per_doc_limit: int = DEFAULT_PER_DOC_LIMIT
    score_norm: str = DEFAULT_SCORE_NORM
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    rep_alpha: Optional[float] = None
    mmr_stage1: bool = True
    result_format: str = "blocks"
    summary_mode: str = "first"

    @field_validator("query", mode="before")
    @classmethod
    def _coerce_query_multi(cls, v):
        def to_list_of_str(x):
            if x is None:
                return []
            if isinstance(x, str):
                s = x.strip()
                return [s] if s else []
            if isinstance(x, (list, tuple, set)):
                acc = []
                for it in x:
                    acc.extend(to_list_of_str(it))
                return acc
            s = str(x).strip()
            return [s] if s else []
        out = to_list_of_str(v)
        return out if out else v


class DebugMultiEmbedResponse(BaseModel):
    step: str = "embed_multi"
    queries: List[str]
    mode: str
    q_vecs: List[List[float]]
    q_vec_lens: List[int]
    content_sparse_queries: Optional[List[Optional[SparseQuery]]] = None
    summary_sparse_queries: Optional[List[Optional[SparseQuery]]] = None
    _next: Optional[dict] = None


class DebugMultiStage1Request(BaseModel):
    queries: List[str]
    q_vecs: List[List[float]]
    mode: str = "auto"
    use_hybrid: bool = True
    top_m: int = 100
    top_k: int = 10
    per_doc_limit: int = DEFAULT_PER_DOC_LIMIT
    score_norm: str = DEFAULT_SCORE_NORM
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    mmr_stage1: bool = True
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    rep_alpha: Optional[float] = None
    summary_sparse_queries: Optional[List[Optional[SparseQuery]]] = None
    content_sparse_queries: Optional[List[Optional[SparseQuery]]] = None
    # shaping params for parity
    result_format: str = "blocks"
    summary_mode: str = "first"


class DebugMultiStage1Response(BaseModel):
    step: str = "stage1_multi"
    cand_doc_ids_list: List[List[str]]
    doc_maps: List[dict]
    _next: Optional[dict] = None


class DebugMultiStage2Request(BaseModel):
    queries: List[str]
    q_vecs: List[List[float]]
    mode: str = "auto"
    # Stage-1 outputs
    cand_doc_ids_list: Optional[List[List[str]]] = None
    doc_maps: Optional[List[dict]] = None
    # optional sparse from embed
    content_sparse_queries: Optional[List[Optional[SparseQuery]]] = None
    # scoring params
    top_m: int = 100
    top_k: int = 10
    per_doc_limit: int = DEFAULT_PER_DOC_LIMIT
    score_norm: str = DEFAULT_SCORE_NORM
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    rep_alpha: Optional[float] = None
    # shaping params
    result_format: str = "blocks"
    summary_mode: str = "first"


class DebugMultiHit(BaseModel):
    doc_id: str
    path: Optional[str] = None
    section: Optional[str] = None
    chunk_id: int
    score: float
    snippet: Optional[str] = None


class DebugMultiStage2Response(BaseModel):
    step: str = "stage2_multi"
    per_query_hits: List[List[DebugMultiHit]]
    fused_hits: List[DebugMultiHit]
    _next: Optional[dict] = None


class DebugMultiShapeRequest(BaseModel):
    fused_hits: List[DebugMultiHit]
    result_format: str = "blocks"
    summary_mode: str = "first"


class DebugMultiShapeResponse(BaseModel):
    step: str = "shape_multi"
    results: List[SearchHit]
    groups: Optional[List[SearchGroup]] = None
    blocks: Optional[List[MergedBlock]] = None
