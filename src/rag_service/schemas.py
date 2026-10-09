from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    title: str
    text: str = Field(min_length=1)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestOut(BaseModel):
    document_id: str
    chunks: int


class DocumentOut(BaseModel):
    id: str
    title: str
    source: str | None
    metadata: dict[str, Any]
    created_at: datetime
    chunks: int
    chunks_embedded: int
    chunks_graph_indexed: int
    chunks_graph_failed: int


class EmbedRequest(BaseModel):
    document_ids: list[str] | None = Field(None, description="Limit to these documents; default = everything pending")
    reembed: bool = Field(False, description="Clear and recompute existing vectors")
    include_entities: bool = Field(True, description="Also embed graph entities that have no vector yet")
    run_async: bool = Field(True, description="Run as a background job and return a job id")


class GraphIndexRequest(BaseModel):
    document_ids: list[str] | None = None
    reindex: bool = Field(False, description="Drop existing extractions for these chunks and redo them")
    retry_failed: bool = True
    limit: int | None = Field(None, ge=1, description="Max chunks to process in this run")
    run_async: bool = True


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    params: dict[str, Any]
    progress: dict[str, Any]
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Literal["vector", "graph", "hybrid"] = "hybrid"
    top_k: int | None = Field(None, ge=1, le=50)
    top_k_entities: int | None = Field(None, ge=1, le=50)
    hops: int | None = Field(None, ge=0, le=4)
    document_ids: list[str] | None = None
    generate: bool = Field(True, description="False = return retrieved context only (no LLM call)")


class ChunkHit(BaseModel):
    id: str
    document_id: str
    document_title: str
    chunk_index: int
    content: str
    score: float | None
    via: str


class EntityHit(BaseModel):
    id: str
    name: str
    type: str
    description: str
    score: float | None = None


class RelationHit(BaseModel):
    source: str
    target: str
    type: str
    description: str
    chunk_id: str | None


class QueryOut(BaseModel):
    mode: str
    answer: str | None
    chunks: list[ChunkHit]
    entities: list[EntityHit]
    relations: list[RelationHit]
    context: str
