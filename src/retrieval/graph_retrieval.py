"""Graph retrieval. The Cypher query is the one used in the notebook."""

from __future__ import annotations

import logging

from src.database.neo4j_client import GRAPH_RETRIEVAL_QUERY

logger = logging.getLogger("rwanda_tourism.graph")

_RUNTIME = None


def bind_runtime(runtime) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def neo4j_graph_retrieval(search_term, top_k=5):
    """Retrieve destinations whose graph fields contain the search term."""
    if _RUNTIME is None:
        raise RuntimeError("The tourism knowledge graph is not ready.")
    client = _RUNTIME.neo4j
    if client is not None:
        try:
            return client.search(search_term, top_k=top_k)
        except Exception:
            logger.exception("Neo4j graph retrieval failed")
            if "Neo4j query failed during a question." not in _RUNTIME.warnings:
                _RUNTIME.warnings.append(
                    "Neo4j query failed during a question. Showing the dataset graph instead."
                )
    return _RUNTIME.graph.search(search_term, top_k=top_k)


# Re-exported so the query text stays next to the database client.
__all__ = ["neo4j_graph_retrieval", "GRAPH_RETRIEVAL_QUERY", "bind_runtime"]
