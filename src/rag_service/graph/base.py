from typing import Protocol, TypedDict


class EntityNode(TypedDict):
    id: str
    name: str
    type: str


class Relation(TypedDict):
    source_id: str
    target_id: str
    type: str
    description: str
    chunk_id: str


class GraphStore(Protocol):
    """Holds the relationship graph. Entity rows + their embeddings always live in Postgres;
    the graph store owns edges and traversal."""

    name: str

    def upsert_entities(self, entities: list[EntityNode]) -> None: ...

    def add_relations(self, relations: list[Relation]) -> None: ...

    def delete_by_chunks(self, chunk_ids: list[str]) -> None:
        """Remove edges that were extracted from these chunks."""
        ...

    def delete_entities(self, entity_ids: list[str]) -> None: ...

    def expand(self, seed_ids: list[str], hops: int, limit: int) -> list[Relation]:
        """Return edges reachable within `hops` of the seed entities (undirected walk)."""
        ...

    def relation_count(self) -> int: ...

    def close(self) -> None: ...
