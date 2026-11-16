"""Combine ChromaDB and Neo4j results without dropping either source."""

from __future__ import annotations

from src.retrieval.graph_retrieval import neo4j_graph_retrieval
from src.retrieval.vector_retrieval import hybrid_retrieval


def hybrid_graph_vector_retrieval(query, graph_keyword, top_k=3):
    vector_results = hybrid_retrieval(query=query, top_k=top_k)
    graph_results = neo4j_graph_retrieval(search_term=graph_keyword, top_k=top_k)

    combined_results = {}
    vector_ids = vector_results["ids"][0] if vector_results["ids"] else []
    for index, destination_id in enumerate(vector_ids):
        combined_results[destination_id] = {
            "id": destination_id,
            "source": ["ChromaDB"],
            "document": vector_results["documents"][0][index],
            "metadata": vector_results["metadatas"][0][index],
        }

    for result in graph_results:
        destination_id = result["id"]
        if destination_id in combined_results:
            combined_results[destination_id]["source"].append("Neo4j")
            combined_results[destination_id]["graph_data"] = result
        else:
            combined_results[destination_id] = {
                "id": destination_id,
                "source": ["Neo4j"],
                "graph_data": result,
            }

    return list(combined_results.values())
