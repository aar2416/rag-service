from ..db import get_conn
from .base import EntityNode, Relation


class PostgresGraphStore:
    """Graph stored as an edge table in Postgres; traversal via recursive CTE."""

    name = "postgres"

    def upsert_entities(self, entities: list[EntityNode]) -> None:
        # Entity rows are already written to the `entities` table by the indexing pipeline.
        return None

    def add_relations(self, relations: list[Relation]) -> None:
        if not relations:
            return
        with get_conn() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO relations (source_id, target_id, type, description, chunk_id)
                VALUES (%(source_id)s, %(target_id)s, %(type)s, %(description)s, %(chunk_id)s)
                ON CONFLICT (source_id, target_id, type, chunk_id)
                DO UPDATE SET description = EXCLUDED.description
                """,
                relations,
            )

    def delete_by_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        with get_conn() as conn:
            conn.execute("DELETE FROM relations WHERE chunk_id = ANY(%s::uuid[])", (chunk_ids,))

    def delete_entities(self, entity_ids: list[str]) -> None:
        # ON DELETE CASCADE from `entities` already removes the edges.
        return None

    def expand(self, seed_ids: list[str], hops: int, limit: int) -> list[Relation]:
        if not seed_ids or hops < 1:
            return []
        with get_conn() as conn:
            rows = conn.execute(
                """
                WITH RECURSIVE walk(entity_id, depth) AS (
                    SELECT unnest(%(seeds)s::text[]), 0
                    UNION
                    SELECT CASE WHEN r.source_id = w.entity_id THEN r.target_id ELSE r.source_id END,
                           w.depth + 1
                    FROM walk w
                    JOIN relations r ON r.source_id = w.entity_id OR r.target_id = w.entity_id
                    WHERE w.depth < %(hops)s
                ),
                reached AS (
                    SELECT entity_id, min(depth) AS depth FROM walk GROUP BY entity_id
                )
                SELECT r.source_id, r.target_id, r.type, r.description, r.chunk_id::text AS chunk_id
                FROM relations r
                JOIN reached a ON a.entity_id = r.source_id
                JOIN reached b ON b.entity_id = r.target_id
                ORDER BY LEAST(a.depth, b.depth), r.id
                LIMIT %(limit)s
                """,
                {"seeds": seed_ids, "hops": hops, "limit": limit},
            ).fetchall()
        return [Relation(**row) for row in rows]

    def relation_count(self) -> int:
        with get_conn() as conn:
            return conn.execute("SELECT count(*) AS n FROM relations").fetchone()["n"]

    def close(self) -> None:
        return None
