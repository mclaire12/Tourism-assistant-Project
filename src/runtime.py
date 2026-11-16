"""Build the shared retrieval and model resources once per process."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import get_settings
from src.agents.tourism_agent import bind_runtime as bind_agent
from src.agents.tourism_agent import compile_agent
from src.database.chroma_client import build_vector_index
from src.database.knowledge_graph import KnowledgeGraph
from src.database.neo4j_client import connect_neo4j
from src.llm.model_provider import build_model_provider
from src.retrieval.embeddings import build_embedder
from src.retrieval.graph_retrieval import bind_runtime as bind_graph
from src.retrieval.vector_retrieval import bind_runtime as bind_vector
from src.utils.documents import build_document, build_metadata, embedding_source_text, load_dataset

logger = logging.getLogger("rwanda_tourism")


@dataclass
class Runtime:
    settings: object
    meta: dict
    places: list
    by_id: dict
    embedder: object
    vector_index: object
    graph: KnowledgeGraph
    neo4j: object | None
    graph_backend: str
    model: object | None
    model_notice: str
    warnings: list[str] = field(default_factory=list)
    agent: object | None = None


def build_runtime() -> Runtime:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    warnings: list[str] = []
    dataset = load_dataset()
    places = dataset.get("places") or []
    if len(places) != 19:
        warnings.append(f"Expected 19 destinations and found {len(places)}.")

    embedder, embedder_notice = build_embedder(settings.embedding_provider, settings.embedding_model)
    if embedder_notice:
        warnings.append(embedder_notice)

    records = []
    for place in places:
        document = build_document(place)
        records.append(
            {
                "id": place["id"],
                "document": document,
                "metadata": build_metadata(place),
                "embedding_text": embedding_source_text(place, document, embedder.name),
            }
        )
    vector_index, vector_notice = build_vector_index(records, embedder)
    if vector_notice:
        warnings.append(vector_notice)

    graph = KnowledgeGraph(places)
    neo4j, neo4j_notice = connect_neo4j(settings, places)
    if neo4j_notice:
        warnings.append(neo4j_notice)

    model, model_notice = build_model_provider(settings)
    if model_notice:
        warnings.append(model_notice)

    runtime = Runtime(
        settings=settings,
        meta=dataset.get("meta") or {},
        places=places,
        by_id={place["id"]: place for place in places},
        embedder=embedder,
        vector_index=vector_index,
        graph=graph,
        neo4j=neo4j,
        graph_backend="Neo4j AuraDB" if neo4j is not None else "Dataset graph",
        model=model,
        model_notice=model_notice,
        warnings=warnings,
    )
    bind_vector(runtime)
    bind_graph(runtime)
    bind_agent(runtime)
    try:
        runtime.agent = compile_agent()
    except Exception:
        logger.exception("LangGraph could not be compiled")
        warnings.append("The LangGraph workflow could not be compiled. Retrieval still runs through the same nodes.")
        runtime.agent = None
    for warning in warnings:
        logger.warning(warning)
    return runtime
