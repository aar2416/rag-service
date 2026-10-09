from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..db import get_conn
from ..graph import get_graph_store
from ..jobs import create_job, get_job, is_busy, list_jobs, run_job
from ..pipeline.embedding import run_embedding
from ..pipeline.graph_indexing import run_graph_indexing
from ..schemas import EmbedRequest, GraphIndexRequest, JobOut

router = APIRouter(tags=["indexing"])


def _launch(kind: str, fn, params: dict, run_async: bool, background: BackgroundTasks) -> dict:
    if is_busy(kind):
        raise HTTPException(409, f"a {kind} job is already running")
    job_id = create_job(kind, params)
    if run_async:
        background.add_task(run_job, job_id, kind, fn, **params)
    else:
        run_job(job_id, kind, fn, **params)
    return get_job(job_id)


@router.post("/embeddings/run", response_model=JobOut, status_code=202)
def run_embeddings(req: EmbedRequest, background: BackgroundTasks):
    """Embed chunks (and entities) that have no vector yet."""
    params = req.model_dump(exclude={"run_async"})
    return _launch("embedding", run_embedding, params, req.run_async, background)


@router.post("/graph/index", response_model=JobOut, status_code=202)
def run_graph_index(req: GraphIndexRequest, background: BackgroundTasks):
    """Extract entities + relations from chunks with Claude and write them to the graph store.
    Run /embeddings/run afterwards so new entities get vectors (needed for graph retrieval)."""
    params = req.model_dump(exclude={"run_async"})
    return _launch("graph_index", run_graph_indexing, params, req.run_async, background)


@router.get("/jobs", response_model=list[JobOut])
def jobs(limit: int = 20):
    return list_jobs(limit)


@router.get("/jobs/{job_id}", response_model=JobOut)
def job(job_id: str):
    row = get_job(job_id) if len(job_id) == 36 else None
    if not row:
        raise HTTPException(404, "job not found")
    return row


@router.get("/index/status")
def index_status():
    with get_conn() as conn:
        counts = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM documents) AS documents,
              (SELECT count(*) FROM chunks) AS chunks,
              (SELECT count(*) FROM chunks WHERE embedding IS NULL) AS chunks_pending_embedding,
              (SELECT count(*) FROM chunks WHERE graph_indexed_at IS NULL) AS chunks_pending_graph,
              (SELECT count(*) FROM chunks WHERE graph_error IS NOT NULL AND graph_indexed_at IS NULL)
                  AS chunks_graph_failed,
              (SELECT count(*) FROM entities) AS entities,
              (SELECT count(*) FROM entities WHERE embedding IS NULL) AS entities_pending_embedding
            """
        ).fetchone()
    graph = get_graph_store()
    return {**counts, "graph_backend": graph.name, "relations": graph.relation_count(),
            "embedding_running": is_busy("embedding"), "graph_index_running": is_busy("graph_index")}
