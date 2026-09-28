"""Inspect local evidence, then optionally answer typed questions against it.

    laya-retrieve 'duplicate invoice' --corpus policies.jsonl
    laya-retrieve 'duplicate invoice' --corpus policies.jsonl --questions refund.json --predict

The default lexical inspection path needs no weights or network. Predictions use
Laya's normal checkpoint loading behavior; provision/cache checkpoints first for
an offline deployment. Sentence embedding weights must already be local.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .retrieval import Document, LocalIndex, SentenceTransformerEncoder


def load_corpus(path: str, *, max_chars: int = 10_000_000, max_documents: int = 10_000) -> list[Document]:
    """Read bounded UTF-8 JSONL. No URLs or source labels are followed."""
    documents, total = [], 0
    with open(path, encoding="utf-8") as handle:
        while True:
            # Bound allocation even for a single maliciously large JSONL record.
            line = handle.readline(max_chars - total + 1)
            if not line:
                break
            total += len(line)
            if total > max_chars:
                raise ValueError("corpus exceeds the character limit")
            if not line.strip():
                continue
            if len(documents) >= max_documents:
                raise ValueError("corpus exceeds the document limit")
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or set(record) - {"id", "text", "source"}:
                    raise ValueError("expected only id, text, and optional source fields")
                documents.append(Document(**record))
            except (TypeError, ValueError) as error:
                raise ValueError("invalid corpus record %d: %s" % (len(documents) + 1, error)) from error
    return documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laya-retrieve", description=__doc__)
    parser.add_argument("query", help="text to search for")
    parser.add_argument("--corpus", required=True, help="local JSONL file of id/text/source records")
    parser.add_argument("--backend", choices=("lexical", "dense", "hybrid"), default="lexical")
    parser.add_argument("--embedding-model", help="pre-provisioned local/cached SentenceTransformer model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--max-context-chars", type=int, default=6000,
                        help="serialized evidence budget, NOT the checkpoint token budget")
    parser.add_argument("--document-id", action="append", dest="document_ids",
                        help="restrict document scope; repeat for multiple IDs (not an ACL)")
    parser.add_argument("--min-score", type=float, default=0.0, help="backend-specific ranking threshold")
    parser.add_argument("--predict", action="store_true", help="also run Laya; checkpoint loading may access the Hub")
    parser.add_argument("--questions", help="Laya JSON question file; required with --predict")
    parser.add_argument("--context-key", default="retrieved_context")
    parser.add_argument("--allow-no-context", action="store_true", help="explicitly allow prediction without evidence")
    parser.add_argument("--model", help="Laya checkpoint name/alias")
    parser.add_argument("--lang", help="explicit input language")
    parser.add_argument("--task", help="explicit typed-decision workflow")
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--head-max-len", type=int)
    return parser


def _load_questions(path: str):
    # Reuse the established CLI format and state-key rules, rather than another schema.
    from .cli import load_questions
    return load_questions(path)


def _make_router(device: str):
    from . import Router
    return Router(device=device, preload=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.predict and not args.questions:
            raise ValueError("--predict requires --questions with instructions that read the context field")
        if args.backend == "lexical" and args.embedding_model:
            raise ValueError("--embedding-model requires --backend dense or hybrid")
        if args.backend != "lexical" and not args.embedding_model:
            raise ValueError("dense/hybrid retrieval requires --embedding-model with local weights")
        # Validate files and question schema before allocating an embedding model.
        documents = load_corpus(args.corpus)
        questions, state_key = _load_questions(args.questions) if args.questions else (None, "request")
        original = {state_key: args.query}
        encoder = None
        if args.backend != "lexical":
            encoder = SentenceTransformerEncoder(args.embedding_model, device=args.device)
        index = LocalIndex(documents, backend=args.backend, encoder=encoder,
                           chunk_size=args.chunk_size, overlap=args.overlap)
        prepared = index.prepare(original, query=args.query, k=args.k, context_key=args.context_key,
                                 max_chars=args.max_context_chars, document_ids=args.document_ids,
                                 min_score=args.min_score)
        output = {"state": prepared.state, "retrieval": prepared.report()}
        if args.predict:
            if not prepared.hits and not args.allow_no_context:
                output["prediction"] = None
                output["retrieval"]["abstained"] = True
                print(json.dumps(output, ensure_ascii=False, allow_nan=False))
                return 3
            router = _make_router(args.device)
            # A long English policy must not turn a French request into an English route.
            decision = router.route(original, model=args.model, task=args.task, lang=args.lang)
            overrides = {key: getattr(args, key) for key in ("max_len", "head_max_len")
                         if getattr(args, key) is not None}
            output["prediction"] = router.predict(prepared.state, questions, model=decision["model"],
                                                  task=args.task, lang=args.lang, **overrides)
            output["retrieval"]["pre_context_routing"] = dict(decision)
            output["retrieval"]["abstained"] = False
        print(json.dumps(output, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as error:
        print("laya-retrieve: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
