from functools import lru_cache
from typing import Protocol

from ..config import get_settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_provider == "voyage":
        from .voyage import VoyageEmbedder

        embedder: Embedder = VoyageEmbedder(s.embedding_model, s.voyage_api_key)
    else:
        from .local import LocalEmbedder

        embedder = LocalEmbedder(s.embedding_model)

    if embedder.dim != s.embedding_dim:
        raise RuntimeError(
            f"Embedding model '{s.embedding_model}' produces {embedder.dim}-dim vectors but "
            f"EMBEDDING_DIM={s.embedding_dim}. Fix EMBEDDING_DIM (and recreate the vector columns)."
        )
    return embedder
