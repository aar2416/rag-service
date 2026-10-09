from functools import lru_cache

from ..config import get_settings
from .base import EntityNode, GraphStore, Relation

__all__ = ["EntityNode", "GraphStore", "Relation", "get_graph_store"]


@lru_cache
def get_graph_store() -> GraphStore:
    s = get_settings()
    if s.graph_backend == "neo4j":
        from .neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.neo4j_database)

    from .postgres_store import PostgresGraphStore

    return PostgresGraphStore()
