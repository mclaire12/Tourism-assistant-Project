"""Vector retrieval. hybrid_retrieval keeps the original signature."""

from __future__ import annotations

_RUNTIME = None


def bind_runtime(runtime) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def hybrid_retrieval(query, top_k=3, metadata_filter=None):
    """Semantic search with optional ChromaDB metadata filter."""
    if _RUNTIME is None:
        raise RuntimeError("The tourism search index is not ready.")
    return _RUNTIME.vector_index.query(query, top_k=top_k, metadata_filter=metadata_filter)
