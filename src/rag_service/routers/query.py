import anthropic
from fastapi import APIRouter, HTTPException

from ..db import get_conn
from ..graph import get_graph_store
from ..llm import LLMRefusal, generate_answer
from ..retrieval import build_context, retrieve
from ..schemas import QueryOut, QueryRequest

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryOut)
def query(req: QueryRequest):
    result = retrieve(req.question, req.mode, req.top_k, req.top_k_entities, req.hops, req.document_ids)
    context = build_context(result)

    answer = None
    if req.generate:
        try:
            answer = generate_answer(req.question, context)
        except LLMRefusal as e:
            raise HTTPException(422, f"model declined to answer: {e}")
        except anthropic.AuthenticationError:
            raise HTTPException(502, "Anthropic authentication failed - check ANTHROPIC_API_KEY")
        except anthropic.RateLimitError:
            raise HTTPException(429, "Anthropic rate limit hit, retry later")
        except anthropic.APIStatusError as e:
            raise HTTPException(502, f"Anthropic API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError:
            raise HTTPException(502, "could not reach the Anthropic API")

    return {**result, "answer": answer, "context": context}


@router.get("/graph/entities")
def find_entities(q: str | None = None, limit: int = 25):
    with get_conn() as conn:
        if q:
            return conn.execute(
                """SELECT id, name, type, description FROM entities
                   WHERE name ILIKE %s OR id ILIKE %s ORDER BY name LIMIT %s""",
                (f"%{q}%", f"%{q}%", limit),
            ).fetchall()
        return conn.execute(
            "SELECT id, name, type, description FROM entities ORDER BY updated_at DESC LIMIT %s", (limit,)
        ).fetchall()


@router.get("/graph/entities/{entity_id}/neighbors")
def neighbors(entity_id: str, hops: int = 1, limit: int = 50):
    with get_conn() as conn:
        entity = conn.execute(
            "SELECT id, name, type, description FROM entities WHERE id = %s", (entity_id,)
        ).fetchone()
    if not entity:
        raise HTTPException(404, "entity not found")
    return {"entity": entity, "relations": get_graph_store().expand([entity_id], max(1, min(hops, 4)), limit)}
