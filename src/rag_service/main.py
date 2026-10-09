import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import get_settings
from .db import close_db, get_conn, init_db
from .graph import get_graph_store
from .routers import documents, indexing, query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_graph_store()  # fail fast if Neo4j is selected but unreachable
    yield
    get_graph_store().close()
    close_db()


app = FastAPI(title="Graph RAG Service", version="0.1.0", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(indexing.router)
app.include_router(query.router)


@app.get("/health", tags=["health"])
def health():
    s = get_settings()
    with get_conn() as conn:
        conn.execute("SELECT 1")
    return {
        "status": "ok",
        "graph_backend": s.graph_backend,
        "embedding_provider": s.embedding_provider,
        "embedding_model": s.embedding_model,
        "llm_model": s.llm_model,
    }
