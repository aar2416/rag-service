"""Stage 3: extract entities/relations from chunks with Claude and write them to the graph.

New or changed entities get embedding = NULL, so the embedding stage picks them up afterwards."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from ..config import get_settings
from ..db import get_conn
from ..graph import EntityNode, Relation, get_graph_store
from ..llm import GraphExtraction, extract_graph

log = logging.getLogger(__name__)
Progress = Callable[[dict], None]

_MAX_DESCRIPTION = 1500


def entity_id(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().strip(".,;:'\"").lower())


def _merge_description(existing: str, new: str) -> str:
    new = new.strip()
    if not new or new.lower() in existing.lower():
        return existing
    if not existing:
        return new
    if len(existing) >= _MAX_DESCRIPTION:
        return existing
    return f"{existing} {new}"[:_MAX_DESCRIPTION]


def _write_extraction(chunk_id: str, result: GraphExtraction) -> tuple[int, int]:
    graph = get_graph_store()

    # Dedupe entities within the chunk by normalized id
    entities: dict[str, dict] = {}
    for e in result.entities:
        eid = entity_id(e.name)
        if not eid:
            continue
        if eid in entities:
            entities[eid]["description"] = _merge_description(entities[eid]["description"], e.description)
        else:
            entities[eid] = {"id": eid, "name": e.name.strip(), "type": e.type.strip().lower() or "other",
                             "description": e.description.strip()}

    relations: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for r in result.relations:
        src, tgt = entity_id(r.source), entity_id(r.target)
        rtype = re.sub(r"[^A-Z0-9_]", "_", r.type.strip().upper()) or "RELATED_TO"
        if src not in entities or tgt not in entities or src == tgt or (src, tgt, rtype) in seen:
            continue
        seen.add((src, tgt, rtype))
        relations.append(Relation(source_id=src, target_id=tgt, type=rtype,
                                  description=r.description.strip(), chunk_id=chunk_id))

    with get_conn() as conn, conn.transaction():
        if entities:
            existing = {
                row["id"]: row
                for row in conn.execute(
                    "SELECT id, type, description FROM entities WHERE id = ANY(%s::text[]) FOR UPDATE",
                    (list(entities),),
                ).fetchall()
            }
            for eid, e in entities.items():
                old = existing.get(eid)
                if old is None:
                    conn.execute(
                        "INSERT INTO entities (id, name, type, description) VALUES (%s, %s, %s, %s)",
                        (eid, e["name"], e["type"], e["description"]),
                    )
                    continue
                merged = _merge_description(old["description"], e["description"])
                etype = old["type"] if old["type"] not in ("", "other") else e["type"]
                if merged != old["description"] or etype != old["type"]:
                    # description changed -> clear the vector so the embedding stage refreshes it
                    conn.execute(
                        """UPDATE entities SET description = %s, type = %s, embedding = NULL,
                               embedded_at = NULL, updated_at = now() WHERE id = %s""",
                        (merged, etype, eid),
                    )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO entity_mentions (entity_id, chunk_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    [(eid, chunk_id) for eid in entities],
                )

    graph.upsert_entities([EntityNode(id=e["id"], name=e["name"], type=e["type"]) for e in entities.values()])
    graph.add_relations(relations)

    with get_conn() as conn:
        conn.execute(
            "UPDATE chunks SET graph_indexed_at = now(), graph_error = NULL WHERE id = %s", (chunk_id,)
        )
    return len(entities), len(relations)


def _reset_chunks(chunk_ids: list[str]) -> None:
    get_graph_store().delete_by_chunks(chunk_ids)
    with get_conn() as conn, conn.transaction():
        conn.execute("DELETE FROM entity_mentions WHERE chunk_id = ANY(%s::uuid[])", (chunk_ids,))
        conn.execute(
            "UPDATE chunks SET graph_indexed_at = NULL, graph_error = NULL WHERE id = ANY(%s::uuid[])",
            (chunk_ids,),
        )


def run_graph_indexing(
    document_ids: list[str] | None = None,
    reindex: bool = False,
    retry_failed: bool = True,
    limit: int | None = None,
    progress: Progress | None = None,
) -> dict:
    s = get_settings()
    doc_filter = "AND c.document_id = ANY(%(docs)s::uuid[])" if document_ids else ""

    if reindex:
        with get_conn() as conn:
            ids = [
                r["id"]
                for r in conn.execute(
                    f"SELECT c.id::text AS id FROM chunks c WHERE TRUE {doc_filter}", {"docs": document_ids}
                ).fetchall()
            ]
        _reset_chunks(ids)

    error_filter = "" if retry_failed else "AND c.graph_error IS NULL"
    with get_conn() as conn:
        chunks = conn.execute(
            f"""
            SELECT c.id::text AS id, c.content, d.title
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.graph_indexed_at IS NULL {doc_filter} {error_filter}
            ORDER BY c.document_id, c.chunk_index
            {"LIMIT %(limit)s" if limit else ""}
            """,
            {"docs": document_ids, "limit": limit},
        ).fetchall()

    stats = {"chunks_total": len(chunks), "chunks_done": 0, "chunks_failed": 0, "entities": 0, "relations": 0}
    if progress:
        progress(stats)

    with ThreadPoolExecutor(max_workers=max(1, s.graph_concurrency)) as pool:
        futures = {pool.submit(extract_graph, c["content"], c["title"]): c["id"] for c in chunks}
        for fut in as_completed(futures):
            chunk_id = futures[fut]
            try:
                n_ent, n_rel = _write_extraction(chunk_id, fut.result())
                stats["chunks_done"] += 1
                stats["entities"] += n_ent
                stats["relations"] += n_rel
            except Exception as exc:  # keep going; failed chunks are retried on the next run
                log.exception("graph extraction failed for chunk %s", chunk_id)
                stats["chunks_failed"] += 1
                with get_conn() as conn:
                    conn.execute("UPDATE chunks SET graph_error = %s WHERE id = %s", (str(exc)[:2000], chunk_id))
            if progress:
                progress(stats)

    return stats
