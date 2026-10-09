"""Stage 1: store a document and split it into chunks. No embedding, no graph extraction."""

from psycopg.types.json import Jsonb

from ..config import get_settings
from ..db import get_conn
from ..graph import get_graph_store
from .chunking import chunk_text


def ingest_document(title: str, text: str, source: str | None = None, metadata: dict | None = None) -> dict:
    s = get_settings()
    chunks = chunk_text(text, s.chunk_size, s.chunk_overlap)
    if not chunks:
        raise ValueError("document has no text content")

    with get_conn() as conn, conn.transaction():
        doc = conn.execute(
            "INSERT INTO documents (title, source, metadata) VALUES (%s, %s, %s) RETURNING id::text AS id",
            (title, source, Jsonb(metadata or {})),
        ).fetchone()
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, chunk_index, content) VALUES (%s, %s, %s)",
                [(doc["id"], i, c) for i, c in enumerate(chunks)],
            )
    return {"document_id": doc["id"], "chunks": len(chunks)}


def delete_document(document_id: str) -> bool:
    graph = get_graph_store()
    with get_conn() as conn:
        chunk_ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id::text AS id FROM chunks WHERE document_id = %s", (document_id,)
            ).fetchall()
        ]
    graph.delete_by_chunks(chunk_ids)

    with get_conn() as conn, conn.transaction():
        deleted = conn.execute("DELETE FROM documents WHERE id = %s RETURNING id", (document_id,)).fetchone()
        orphans = conn.execute(
            """
            DELETE FROM entities e
            WHERE NOT EXISTS (SELECT 1 FROM entity_mentions m WHERE m.entity_id = e.id)
            RETURNING e.id
            """
        ).fetchall()
    graph.delete_entities([r["id"] for r in orphans])
    return deleted is not None
