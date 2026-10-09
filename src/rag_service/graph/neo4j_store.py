from neo4j import GraphDatabase

from .base import EntityNode, Relation


class Neo4jGraphStore:
    """(:Entity {id, name, type})-[:RELATED {type, description, chunk_id}]->(:Entity)"""

    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.driver.verify_connectivity()
        self.driver.execute_query(
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            database_=self.database,
        )
        self.driver.execute_query(
            "CREATE INDEX related_chunk IF NOT EXISTS FOR ()-[r:RELATED]-() ON (r.chunk_id)",
            database_=self.database,
        )

    def upsert_entities(self, entities: list[EntityNode]) -> None:
        if not entities:
            return
        self.driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {id: row.id})
            SET e.name = row.name, e.type = row.type
            """,
            rows=list(entities),
            database_=self.database,
        )

    def add_relations(self, relations: list[Relation]) -> None:
        if not relations:
            return
        self.driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (a:Entity {id: row.source_id})
            MATCH (b:Entity {id: row.target_id})
            MERGE (a)-[r:RELATED {type: row.type, chunk_id: row.chunk_id}]->(b)
            SET r.description = row.description
            """,
            rows=list(relations),
            database_=self.database,
        )

    def delete_by_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self.driver.execute_query(
            "MATCH ()-[r:RELATED]-() WHERE r.chunk_id IN $ids DELETE r",
            ids=chunk_ids,
            database_=self.database,
        )

    def delete_entities(self, entity_ids: list[str]) -> None:
        if not entity_ids:
            return
        self.driver.execute_query(
            "MATCH (e:Entity) WHERE e.id IN $ids DETACH DELETE e",
            ids=entity_ids,
            database_=self.database,
        )

    def expand(self, seed_ids: list[str], hops: int, limit: int) -> list[Relation]:
        if not seed_ids or hops < 1:
            return []
        hops = max(1, min(int(hops), 5))  # variable-length bounds can't be query parameters
        records, _, _ = self.driver.execute_query(
            f"""
            MATCH (s:Entity) WHERE s.id IN $ids
            MATCH p = (s)-[:RELATED*1..{hops}]-(:Entity)
            UNWIND range(0, length(p) - 1) AS i
            WITH relationships(p)[i] AS r, i
            WITH r, min(i) AS depth
            ORDER BY depth
            LIMIT $limit
            RETURN startNode(r).id AS source_id, endNode(r).id AS target_id,
                   r.type AS type, coalesce(r.description, '') AS description, r.chunk_id AS chunk_id
            """,
            ids=seed_ids,
            limit=limit,
            database_=self.database,
        )
        return [Relation(**rec.data()) for rec in records]

    def relation_count(self) -> int:
        records, _, _ = self.driver.execute_query(
            "MATCH ()-[r:RELATED]->() RETURN count(r) AS n", database_=self.database
        )
        return records[0]["n"]

    def close(self) -> None:
        self.driver.close()
