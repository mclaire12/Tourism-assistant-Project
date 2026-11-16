"""ChromaDB index, rebuilt from rwanda_places.json when the saved index does not match."""

from __future__ import annotations

import logging
import os

import numpy as np

from config.settings import CHROMA_PATH
from src.retrieval.embeddings import HashedTokenEmbedder
from src.utils.text import content_tokens, variants

logger = logging.getLogger("rwanda_tourism.chroma")

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")


class VectorIndex:
    backend = "memory"
    embedder_name = ""

    def query(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> dict:
        raise NotImplementedError


class MemoryVectorIndex(VectorIndex):
    """Cosine index with the same result shape as collection.query."""

    backend = "memory"

    def __init__(self, records: list[dict], embedder):
        self.embedder_name = embedder.name
        self.records = records
        matrix = np.array(embedder.embed([record["embedding_text"] for record in records]), dtype=np.float32)
        self.matrix = matrix
        self.embedder = embedder

    def query(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> dict:
        return _rank(
            query=query,
            records=self.records,
            matrix=self.matrix,
            embedder=self.embedder,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )


class ChromaVectorIndex(VectorIndex):
    backend = "chromadb"

    def __init__(self, records: list[dict], embedder):
        import chromadb
        from chromadb.config import Settings

        self.embedder_name = embedder.name
        self.records = records
        self.embedder = embedder
        self._by_id = {record["id"]: record for record in records}
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        signature = f"{embedder.name}:{len(records)}:{_signature(records)}"
        self.collection = self._collection(signature)
        self._ensure_records(signature)

    def query(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> dict:
        if not query or not query.strip():
            return _empty()
        if isinstance(self.embedder, HashedTokenEmbedder) and not _query_tokens(query):
            return _empty()

        count = self.collection.count()
        if count == 0:
            return _empty()
        n_results = min(max(top_k, 1), count)
        kwargs = {
            "query_embeddings": self.embedder.embed([query]),
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        where = _chroma_where(metadata_filter)
        if where:
            kwargs["where"] = where
        try:
            raw = self.collection.query(**kwargs)
        except Exception:
            logger.exception("ChromaDB query failed")
            matrix = np.array(
                self.embedder.embed([record["embedding_text"] for record in self.records]),
                dtype=np.float32,
            )
            return _rank(query, self.records, matrix, self.embedder, top_k, metadata_filter)

        return _filter_hashed(query, raw, self.embedder, self._by_id)

    def _collection(self, signature: str):
        try:
            existing = self.client.get_collection("rwanda_tourism")
            metadata = existing.metadata or {}
            if metadata.get("signature") != signature or existing.count() != len(self.records):
                self.client.delete_collection("rwanda_tourism")
            else:
                return existing
        except Exception:
            pass
        return self.client.get_or_create_collection(
            name="rwanda_tourism",
            metadata={"hnsw:space": "cosine", "signature": signature},
        )

    def _ensure_records(self, signature: str) -> None:
        if self.collection.count() == len(self.records):
            return
        embeddings = self.embedder.embed([record["embedding_text"] for record in self.records])
        self.collection.add(
            ids=[record["id"] for record in self.records],
            documents=[record["document"] for record in self.records],
            metadatas=[record["metadata"] for record in self.records],
            embeddings=embeddings,
        )
        logger.info("Rebuilt ChromaDB collection rwanda_tourism (%s)", signature)


def build_vector_index(records: list[dict], embedder) -> tuple[VectorIndex, str]:
    try:
        return ChromaVectorIndex(records, embedder), ""
    except Exception:
        logger.exception("ChromaDB initialization failed")
        return (
            MemoryVectorIndex(records, embedder),
            "ChromaDB could not be opened. Search is using an in-memory copy of the same vectors.",
        )


def _signature(records: list[dict]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for record in records:
        digest.update(record["id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["document"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _chroma_where(metadata_filter: dict | None):
    if not metadata_filter:
        return None
    return metadata_filter


def _query_tokens(query: str) -> set[str]:
    tokens: set[str] = set()
    for token in content_tokens(query):
        tokens.update(variants(token))
    return tokens


def _filter_hashed(query: str, raw: dict, embedder, by_id: dict) -> dict:
    if not isinstance(embedder, HashedTokenEmbedder):
        return raw
    query_tokens = _query_tokens(query)
    if not query_tokens:
        return _empty()
    ids = []
    documents = []
    metadatas = []
    distances = []
    for index, doc_id in enumerate(raw["ids"][0]):
        record = by_id.get(doc_id)
        text = record["embedding_text"] if record else raw["documents"][0][index]
        if query_tokens.isdisjoint(_query_tokens(text)):
            continue
        ids.append(doc_id)
        documents.append(raw["documents"][0][index])
        metadatas.append(raw["metadatas"][0][index])
        distances.append(raw["distances"][0][index])
    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def _rank(query, records, matrix, embedder, top_k, metadata_filter) -> dict:
    if isinstance(embedder, HashedTokenEmbedder) and not _query_tokens(query):
        return _empty()
    query_vector = np.array(embedder.embed([query])[0], dtype=np.float32)
    if not np.any(query_vector):
        return _empty()
    scores = matrix @ query_vector
    query_tokens = _query_tokens(query)
    ranked = []
    for index, record in enumerate(records):
        if not _metadata_matches(record["metadata"], metadata_filter):
            continue
        if isinstance(embedder, HashedTokenEmbedder) and query_tokens.isdisjoint(
            _query_tokens(record["embedding_text"])
        ):
            continue
        ranked.append((float(scores[index]), index))
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = ranked[:top_k]
    return {
        "ids": [[records[index]["id"] for _, index in chosen]],
        "documents": [[records[index]["document"] for _, index in chosen]],
        "metadatas": [[records[index]["metadata"] for _, index in chosen]],
        "distances": [[1.0 - score for score, _ in chosen]],
    }


def _metadata_matches(metadata: dict, metadata_filter: dict | None) -> bool:
    if not metadata_filter:
        return True
    if "$and" in metadata_filter:
        return all(_metadata_matches(metadata, item) for item in metadata_filter["$and"])
    for key, expected in metadata_filter.items():
        if str(metadata.get(key)) != str(expected):
            return False
    return True


def _empty() -> dict:
    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
