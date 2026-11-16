"""Embedding providers.

BAAI/bge-m3 remains the project embedding model and is used whenever it can
be loaded. Free Streamlit hosting does not have the memory for that model,
so the default deployment uses a 1024-dimension hashed token embedder inside
ChromaDB. The active mode is reported in the interface and is never described
as BGE-M3 unless that model is actually loaded.
"""

from __future__ import annotations

import hashlib
import importlib.util

import numpy as np

from config.settings import EMBEDDING_DIMENSION
from src.utils.text import content_tokens, variants


class Embedder:
    name = "embedder"
    dimension = EMBEDDING_DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashedTokenEmbedder(Embedder):
    """Cosine space based on overlapping tokens. Dimension stays 1024."""

    name = "hashed-token-overlap-1024"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            tokens = _expanded_tokens(text)
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[bucket] += sign
            norm = float(np.linalg.norm(vector))
            if norm:
                vector /= norm
            vectors.append(vector.tolist())
        return vectors


class BGEM3Embedder(Embedder):
    name = "BAAI/bge-m3"

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        array = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if int(array.shape[-1]) != self.dimension:
            raise RuntimeError(
                f"{self.name} returned dimension {array.shape[-1]}, expected {self.dimension}."
            )
        return array.astype(np.float32).tolist()


def build_embedder(provider: str, model_name: str) -> tuple[Embedder, str]:
    choice = (provider or "auto").lower()
    bge_installed = importlib.util.find_spec("sentence_transformers") is not None
    use_bge = choice == "bge-m3" or (choice == "auto" and bge_installed)
    if use_bge:
        try:
            embedder = BGEM3Embedder(model_name or "BAAI/bge-m3")
            return embedder, ""
        except Exception as exc:
            if choice == "bge-m3":
                raise RuntimeError(
                    "BAAI/bge-m3 was requested but could not be loaded. "
                    "Install the local ML requirements on a machine with enough memory, "
                    "or set EMBEDDING_PROVIDER=hashed."
                ) from exc
    notice = (
        "Semantic model BAAI/bge-m3 is not loaded. ChromaDB is using a 1024-dimension "
        "token-overlap embedder so the app can run on free hosting. Set "
        "EMBEDDING_PROVIDER=bge-m3 where memory allows."
    )
    return HashedTokenEmbedder(), notice


def _expanded_tokens(text: str) -> list[str]:
    expanded: list[str] = []
    for token in content_tokens(text):
        # Prefer the longer surface form and a simple singular variant.
        expanded.extend(sorted(variants(token), key=len, reverse=True)[:2])
    return expanded
