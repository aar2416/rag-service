from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import get_settings

_pool: ConnectionPool | None = None


def _schema(dim: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT NOT NULL,
    source      TEXT,
    metadata    JSONB NOT NULL DEFAULT '{{}}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index      INT NOT NULL,
    content          TEXT NOT NULL,
    embedding        vector({dim}),
    embedded_at      TIMESTAMPTZ,
    graph_indexed_at TIMESTAMPTZ,
    graph_error      TEXT,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_pending_embed_idx ON chunks (document_id) WHERE embedding IS NULL;
CREATE INDEX IF NOT EXISTS chunks_pending_graph_idx ON chunks (document_id) WHERE graph_indexed_at IS NULL;

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL DEFAULT 'concept',
    description TEXT NOT NULL DEFAULT '',
    embedding   vector({dim}),
    embedded_at TIMESTAMPTZ,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS entities_embedding_idx ON entities USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id  UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS entity_mentions_chunk_idx ON entity_mentions (chunk_id);

-- Used only when GRAPH_BACKEND=postgres
CREATE TABLE IF NOT EXISTS relations (
    id          BIGSERIAL PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    type        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    chunk_id    UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    UNIQUE (source_id, target_id, type, chunk_id)
);
CREATE INDEX IF NOT EXISTS relations_source_idx ON relations (source_id);
CREATE INDEX IF NOT EXISTS relations_target_idx ON relations (target_id);

CREATE TABLE IF NOT EXISTS jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    params      JSONB NOT NULL DEFAULT '{{}}',
    progress    JSONB NOT NULL DEFAULT '{{}}',
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);
"""


def init_db() -> None:
    """Create the extension + tables, then open the shared pool."""
    global _pool
    s = get_settings()
    # The vector extension must exist before register_vector can run on pool connections.
    with psycopg.connect(s.database_url, autocommit=True) as conn:
        conn.execute(_schema(s.embedding_dim))

    def _configure(conn: psycopg.Connection) -> None:
        register_vector(conn)

    _pool = ConnectionPool(
        s.database_url,
        min_size=s.db_pool_min,
        max_size=s.db_pool_max,
        kwargs={"row_factory": dict_row},
        configure=_configure,
        open=True,
    )


def close_db() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    if _pool is None:
        init_db()
    assert _pool is not None
    with _pool.connection() as conn:
        yield conn
