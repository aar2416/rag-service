import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..db import get_conn
from ..pipeline.ingestion import delete_document, ingest_document
from ..schemas import DocumentIn, DocumentOut, IngestOut

router = APIRouter(prefix="/documents", tags=["documents"])

_DOC_QUERY = """
SELECT d.id::text AS id, d.title, d.source, d.metadata, d.created_at,
       count(c.id) AS chunks,
       count(c.embedding) AS chunks_embedded,
       count(c.graph_indexed_at) AS chunks_graph_indexed,
       count(c.graph_error) FILTER (WHERE c.graph_indexed_at IS NULL) AS chunks_graph_failed
FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
{where}
GROUP BY d.id
ORDER BY d.created_at DESC
"""


def _check_uuid(value: str) -> None:
    try:
        uuid.UUID(value)
    except ValueError:
        raise HTTPException(404, "document not found")


@router.post("", response_model=IngestOut, status_code=201)
def create_document(doc: DocumentIn):
    """Store + chunk a document. Embedding and graph indexing are separate steps."""
    try:
        return ingest_document(doc.title, doc.text, doc.source, doc.metadata)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.post("/upload", response_model=IngestOut, status_code=201)
async def upload_document(file: UploadFile = File(...), title: str | None = Form(None)):
    """Upload a UTF-8 text/markdown file."""
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(415, "only UTF-8 text files are supported (.txt, .md, ...)")
    try:
        return ingest_document(title or file.filename or "untitled", text, source=file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("", response_model=list[DocumentOut])
def list_documents():
    with get_conn() as conn:
        return conn.execute(_DOC_QUERY.format(where="")).fetchall()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str):
    _check_uuid(document_id)
    with get_conn() as conn:
        row = conn.execute(_DOC_QUERY.format(where="WHERE d.id = %s"), (document_id,)).fetchone()
    if not row:
        raise HTTPException(404, "document not found")
    return row


@router.get("/{document_id}/chunks")
def get_chunks(document_id: str):
    _check_uuid(document_id)
    with get_conn() as conn:
        return conn.execute(
            """SELECT id::text AS id, chunk_index, content, embedding IS NOT NULL AS embedded,
                      graph_indexed_at, graph_error
               FROM chunks WHERE document_id = %s ORDER BY chunk_index""",
            (document_id,),
        ).fetchall()


@router.delete("/{document_id}", status_code=204)
def remove_document(document_id: str):
    _check_uuid(document_id)
    if not delete_document(document_id):
        raise HTTPException(404, "document not found")
