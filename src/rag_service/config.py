from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Postgres / pgvector
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    db_pool_min: int = 1
    db_pool_max: int = 10

    # Graph backend: "postgres" keeps relations in pg tables, "neo4j" stores them in Neo4j
    graph_backend: Literal["postgres", "neo4j"] = "postgres"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    neo4j_database: str = "neo4j"

    # Embeddings
    embedding_provider: Literal["local", "voyage"] = "local"
    # HF model id or a local path (e.g. ./model/bge-small-en-v1.5)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    voyage_api_key: str | None = None

    # LLM (Claude) for entity extraction + answer generation
    anthropic_api_key: str | None = None  # falls back to the SDK's default credential resolution
    llm_model: str = "claude-opus-5-5"
    extraction_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    extraction_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    llm_fallbacks: bool = True  # server-side refusal fallback (Claude API only)
    graph_concurrency: int = 4  # parallel extraction calls during graph indexing

    # Chunking
    chunk_size: int = 1200  # characters
    chunk_overlap: int = 200

    # Retrieval defaults
    top_k_chunks: int = 6
    top_k_entities: int = 8
    graph_hops: int = 2
    max_relations: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
