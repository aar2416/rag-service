# Graph RAG Service

FastAPI service for Graph RAG. **pgvector** stores the vectors. The graph can live in **Postgres** (the default) or **Neo4j**. Claude extracts entities and relations and writes the answers.

The pipeline has three separate stages. Each one runs on its own, through the API or the CLI:

| Stage | What it does | API | CLI |
|---|---|---|---|
| 1. Ingest | stores the document and splits it into chunks (no model calls) | `POST /documents`, `POST /documents/upload` | `cli.py ingest` |
| 2. Embed | embeds chunks and graph entities that have no vector yet | `POST /embeddings/run` | `cli.py embed` |
| 3. Graph index | sends each chunk to Claude to extract entities and relations, then writes them to the graph | `POST /graph/index` | `cli.py graph-index` |

Graph indexing creates new entities without vectors. Run **embed again after graph-index** so the entities get vectors. Graph retrieval needs those entity vectors to find its seed entities.

## Layout

```
src/rag_service/
  main.py              FastAPI app
  config.py            settings (env / .env)
  db.py                Postgres pool + schema (pgvector)
  llm.py               Claude: graph extraction (structured output) + answer generation
  retrieval.py         vector / graph / hybrid retrieval + context building
  jobs.py              background job tracking (jobs table)
  embeddings/          local (sentence-transformers) | voyage
  graph/               postgres_store.py | neo4j_store.py
  pipeline/            chunking, ingestion, embedding, graph_indexing
  routers/             documents, indexing, query
scripts/cli.py         run each stage without the server
```

Postgres always holds `documents`, `chunks` (with vectors), `entities` (with vectors) and `entity_mentions`. The graph backend holds the **relations** and runs the traversal:
- `GRAPH_BACKEND=postgres`: a `relations` table, walked with a recursive CTE
- `GRAPH_BACKEND=neo4j`: `(:Entity)-[:RELATED {type, description, chunk_id}]->(:Entity)`, walked with Cypher

## Setup

> **Python version:** your `.venv` runs Python 3.15. `torch`/`sentence-transformers` and `uvicorn[standard]` don't have 3.15 wheels yet. If you want local embeddings, create the venv with Python 3.12 or 3.13. On 3.15 you can still use `EMBEDDING_PROVIDER=voyage`.

```bash
docker compose up -d                       # pgvector
docker compose --profile neo4j up -d       # + Neo4j (optional)

py -3.12 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env                     # set ANTHROPIC_API_KEY, etc.

uvicorn rag_service.main:app --app-dir src --reload
```

Open http://localhost:8000/docs for the Swagger UI. The schema is created on startup.

To use a model stored in `./model`, set `EMBEDDING_MODEL=./model/<folder>` and set `EMBEDDING_DIM` to that model's dimension. The service refuses to start embedding if the two don't match.
**Changing `EMBEDDING_DIM` after tables exist** requires dropping the `chunks.embedding` / `entities.embedding` columns (or the DB) because the vector column dimension is fixed.

## Try it

```bash
# 1. ingest
curl -X POST localhost:8000/documents -H "Content-Type: application/json" \
  -d '{"title": "acme", "text": "Alice is the CTO of Acme Corp. Acme Corp builds Rocket, a database written in Rust. Bob works at Acme on Rocket."}'

# 2. embed chunks
curl -X POST localhost:8000/embeddings/run -H "Content-Type: application/json" -d '{}'

# 3. build graph (calls Claude), then embed the new entities
curl -X POST localhost:8000/graph/index -H "Content-Type: application/json" -d '{}'
curl localhost:8000/jobs                        # poll until succeeded
curl -X POST localhost:8000/embeddings/run -H "Content-Type: application/json" -d '{}'

curl localhost:8000/index/status

# query: mode = vector | graph | hybrid; generate=false skips the LLM and returns context only
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "Who works on Rocket and what language is it in?", "mode": "hybrid"}'

# explore the graph
curl "localhost:8000/graph/entities?q=acme"
curl "localhost:8000/graph/entities/acme%20corp/neighbors?hops=2"
```

All index endpoints take `document_ids` to limit the run, and `run_async: false` to block until the job finishes. `graph/index` also takes `reindex` and `limit`. When an extraction fails, the chunk keeps its `graph_error` and is retried on the next run.

CLI equivalents:

```bash
python scripts/cli.py ingest notes/*.md
python scripts/cli.py embed
python scripts/cli.py graph-index --limit 20
python scripts/cli.py embed
python scripts/cli.py query "..." --mode graph
python scripts/cli.py status
```

## Retrieval modes

- **vector**: top-k chunks by cosine similarity.
- **graph**: finds the top-k entities by similarity, walks `hops` edges in the graph, and collects the relation facts plus the chunks that mention the seed entities or produced the edges.
- **hybrid** (default): returns both, with chunks deduplicated.

Claude answers from that context. It cites chunks as `[C1]` and graph facts as `[G1]`.

## Notes

- Models: `claude-opus-5-5` for both extraction and answers. Extraction runs at `low` effort and answers at `medium`. You can change all of these in `.env`.
- `LLM_FALLBACKS=true` turns on Anthropic's server-side refusal fallback (`fallbacks: "default"`). If Claude declines a request, the same request is retried on a fallback model. Set it to `false` if your account or platform doesn't support it.
- Only one embedding job and one graph-index job can run at a time per process. A second request returns 409.
