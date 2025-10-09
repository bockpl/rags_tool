"""Shared constants for vector and sparse field naming."""

CONTENT_VECTOR_NAME = "content_dense"
SUMMARY_VECTOR_NAME = "summary_dense"
CONTENT_SPARSE_NAME = "content_sparse"
SUMMARY_SPARSE_NAME = "summary_sparse"

# Global toggle for TF‑IDF hybrid support
SPARSE_ENABLED = True

# Sterowanie rerankiem (gdy ranker jest skonfigurowany w .env)
# Etap 1: rerank streszczeń.
RANKER_USE_STAGE1 = True
