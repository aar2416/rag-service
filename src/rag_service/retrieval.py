from typing import Literal

import numpy as np

from .config import get_settings
from .db import get_conn
from .embeddings import get_embedder
from .graph import get_graph_store

Mode = Literal["vector", "graph", "hybrid"]

_CHUNK_COLS = """c.id::text AS id, c.document_id::text AS document_id, d.title AS document_title,
                 c.chunk_index, c.content"""


def search_chunks(qvec: np.ndarray, k: int, document_ids: list[str] | None = None) -> list[dict]:
    doc_filter = "AND c.document_id = ANY(%(docs)s::uuid[])" if document_ids else ""
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT {_CHUNK_COLS}, 1 - (c.embedding <=> %(q)s) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL {doc_filter}
            ORDER BY c.embedding <=> %(q)s
            LIMIT %(k)s
            """,
            {"q": qvec, "k": k, "docs": document_ids},
        ).fetchall()
    for r in rows:
        r["via"] = "vector"
    return rows


def search_entities(qvec: np.ndarray, k: int) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT id, name, type, description, 1 - (embedding <=> %(q)s) AS score
            FROM entities WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %(q)s
            LIMIT %(k)s
            """,
            {"q": qvec, "k": k},
        ).fetchall()


def _entities_by_id(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, type, description FROM entities WHERE id = ANY(%s::text[])", (ids,)
        ).fetchall()
    return {r["id"]: r for r in rows}


def graph_retrieve(qvec: np.ndarray, k_entities: int, k_chunks: int, hops: int, max_relations: int,
                   document_ids: list[str] | None = None) -> dict:
    seeds = search_entities(qvec, k_entities)
    seed_ids = [e["id"] for e in seeds]
    relations = get_graph_store().expand(seed_ids, hops, max_relations)

    names = _entities_by_id(list({r["source_id"] for r in relations} | {r["target_id"] for r in relations}))
    for r in relations:
        r["source"] = names.get(r["source_id"], {}).get("name", r["source_id"])
        r["target"] = names.get(r["target_id"], {}).get("name", r["target_id"])

    # Chunks that mention a seed entity or that a traversed edge came from, ranked by similarity
    rel_chunk_ids = list({r["chunk_id"] for r in relations if r.get("chunk_id")})
    doc_filter = "AND c.document_id = ANY(%(docs)s::uuid[])" if document_ids else ""
    with get_conn() as conn:
        chunks = conn.execute(
            f"""
            SELECT {_CHUNK_COLS},
                   CASE WHEN c.embedding IS NULL THEN NULL ELSE 1 - (c.embedding <=> %(q)s) END AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE (c.id IN (SELECT chunk_id FROM entity_mentions WHERE entity_id = ANY(%(seeds)s::text[]))
                   OR c.id = ANY(%(rel_chunks)s::uuid[]))
              {doc_filter}
            ORDER BY score DESC NULLS LAST
            LIMIT %(k)s
            """,
            {"q": qvec, "seeds": seed_ids, "rel_chunks": rel_chunk_ids, "k": k_chunks, "docs": document_ids},
        ).fetchall()
    for c in chunks:
        c["via"] = "graph"
    return {"entities": seeds, "relations": relations, "chunks": chunks}


def retrieve(question: str, mode: Mode = "hybrid", top_k: int | None = None, top_k_entities: int | None = None,
             hops: int | None = None, document_ids: list[str] | None = None) -> dict:
    s = get_settings()
    top_k = top_k or s.top_k_chunks
    qvec = np.asarray(get_embedder().embed_query(question), dtype=np.float32)

    result: dict = {"mode": mode, "chunks": [], "entities": [], "relations": []}
    if mode in ("vector", "hybrid"):
        result["chunks"] = search_chunks(qvec, top_k, document_ids)
    if mode in ("graph", "hybrid"):
        g = graph_retrieve(qvec, top_k_entities or s.top_k_entities, top_k,
                           s.graph_hops if hops is None else hops, s.max_relations, document_ids)
        result["entities"], result["relations"] = g["entities"], g["relations"]
        seen = {c["id"] for c in result["chunks"]}
        result["chunks"] += [c for c in g["chunks"] if c["id"] not in seen]
    return result


def build_context(result: dict) -> str:
    parts: list[str] = []
    if result["chunks"]:
        parts.append("## Document chunks")
        for i, c in enumerate(result["chunks"], 1):
            parts.append(f"[C{i}] (document: {c['document_title']}, chunk {c['chunk_index']})\n{c['content']}")
    if result["relations"]:
        parts.append("## Knowledge graph facts")
        for i, r in enumerate(result["relations"], 1):
            desc = f": {r['description']}" if r["description"] else ""
            parts.append(f"[G{i}] {r['source']} -[{r['type']}]-> {r['target']}{desc}")
    if result["entities"]:
        parts.append("## Relevant entities")
        for e in result["entities"]:
            parts.append(f"- {e['name']} ({e['type']}): {e['description']}")
    return "\n\n".join(parts) if parts else "(no context retrieved)"
