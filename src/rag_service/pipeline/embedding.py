"""Stage 2: embed chunks (and graph entities) that don't have a vector yet."""

from typing import Callable

import numpy as np

from ..config import get_settings
from ..db import get_conn
from ..embeddings import get_embedder

Progress = Callable[[dict], None]


def _entity_text(row: dict) -> str:
    return f"{row['name']} ({row['type']}): {row['description']}".strip()


def run_embedding(
    document_ids: list[str] | None = None,
    reembed: bool = False,
    include_entities: bool = True,
    progress: Progress | None = None,
) -> dict:
    s = get_settings()
    embedder = get_embedder()
    batch = s.embedding_batch_size
    stats = {"chunks_embedded": 0, "entities_embedded": 0}

    if reembed:
        with get_conn() as conn:
            if document_ids:
                conn.execute(
                    "UPDATE chunks SET embedding = NULL, embedded_at = NULL WHERE document_id = ANY(%s::uuid[])",
                    (document_ids,),
                )
            else:
                conn.execute("UPDATE chunks SET embedding = NULL, embedded_at = NULL")
                if include_entities:
                    conn.execute("UPDATE entities SET embedding = NULL, embedded_at = NULL")

    doc_filter = "AND document_id = ANY(%(docs)s::uuid[])" if document_ids else ""
    while True:
        with get_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT id, content FROM chunks
                WHERE embedding IS NULL {doc_filter}
                ORDER BY document_id, chunk_index
                LIMIT %(n)s
                """,
                {"docs": document_ids, "n": batch},
            ).fetchall()
        if not rows:
            break
        vectors = embedder.embed_documents([r["content"] for r in rows])
        with get_conn() as conn, conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET embedding = %s, embedded_at = now() WHERE id = %s",
                [(np.asarray(v, dtype=np.float32), r["id"]) for v, r in zip(vectors, rows)],
            )
        stats["chunks_embedded"] += len(rows)
        if progress:
            progress(stats)

    if include_entities:
        while True:
            with get_conn() as conn:
                rows = conn.execute(
                    "SELECT id, name, type, description FROM entities WHERE embedding IS NULL ORDER BY id LIMIT %s",
                    (batch,),
                ).fetchall()
            if not rows:
                break
            vectors = embedder.embed_documents([_entity_text(r) for r in rows])
            with get_conn() as conn, conn.cursor() as cur:
                cur.executemany(
                    "UPDATE entities SET embedding = %s, embedded_at = now() WHERE id = %s",
                    [(np.asarray(v, dtype=np.float32), r["id"]) for v, r in zip(vectors, rows)],
                )
            stats["entities_embedded"] += len(rows)
            if progress:
                progress(stats)

    return stats
