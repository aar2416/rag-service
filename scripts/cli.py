"""Run each pipeline stage on its own, without the API server.

    python scripts/cli.py ingest docs/*.md
    python scripts/cli.py embed
    python scripts/cli.py graph-index --limit 50
    python scripts/cli.py embed            # embeds entities created by graph-index
    python scripts/cli.py query "How does X relate to Y?" --mode hybrid
    python scripts/cli.py status
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_service.db import close_db, init_db  # noqa: E402


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_ingest(args) -> None:
    from rag_service.pipeline.ingestion import ingest_document

    paths = [p for pattern in args.paths for p in glob.glob(pattern)] or args.paths
    for p in paths:
        path = Path(p)
        res = ingest_document(args.title or path.stem, path.read_text(encoding="utf-8"), source=str(path))
        print(f"{path}: document {res['document_id']} ({res['chunks']} chunks)")


def cmd_embed(args) -> None:
    from rag_service.pipeline.embedding import run_embedding

    _print(run_embedding(args.doc, reembed=args.reembed, include_entities=not args.chunks_only,
                         progress=lambda p: print(f"  progress: {p}", file=sys.stderr)))


def cmd_graph_index(args) -> None:
    from rag_service.pipeline.graph_indexing import run_graph_indexing

    _print(run_graph_indexing(args.doc, reindex=args.reindex, limit=args.limit,
                              progress=lambda p: print(f"  progress: {p}", file=sys.stderr)))


def cmd_query(args) -> None:
    from rag_service.llm import generate_answer
    from rag_service.retrieval import build_context, retrieve

    result = retrieve(args.question, args.mode, args.top_k, hops=args.hops)
    context = build_context(result)
    if args.no_generate:
        print(context)
    else:
        print(generate_answer(args.question, context))


def cmd_status(args) -> None:
    from rag_service.routers.indexing import index_status

    _print(index_status())


def main() -> None:
    ap = argparse.ArgumentParser(description="Graph RAG pipeline CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="store + chunk text files")
    p.add_argument("paths", nargs="+")
    p.add_argument("--title")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("embed", help="embed pending chunks and entities")
    p.add_argument("--doc", action="append", help="document id (repeatable)")
    p.add_argument("--reembed", action="store_true")
    p.add_argument("--chunks-only", action="store_true")
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("graph-index", help="extract entities/relations into the graph")
    p.add_argument("--doc", action="append", help="document id (repeatable)")
    p.add_argument("--reindex", action="store_true")
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_graph_index)

    p = sub.add_parser("query", help="ask a question")
    p.add_argument("question")
    p.add_argument("--mode", choices=["vector", "graph", "hybrid"], default="hybrid")
    p.add_argument("--top-k", type=int)
    p.add_argument("--hops", type=int)
    p.add_argument("--no-generate", action="store_true", help="print retrieved context only")
    p.set_defaults(fn=cmd_query)

    sub.add_parser("status", help="show index counts").set_defaults(fn=cmd_status)

    args = ap.parse_args()
    init_db()
    try:
        args.fn(args)
    finally:
        close_db()


if __name__ == "__main__":
    main()
