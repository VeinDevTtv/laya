"""Local retrieval for typed decisions; no model or service is loaded on import.

Retrieval scores are ranking signals, not calibrated probabilities. Build separate
indexes for separate trust domains: ``document_ids`` is a scope filter, not an ACL.
"""
from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

__all__ = ["Document", "SearchHit", "PreparedContext", "LocalIndex", "SentenceTransformerEncoder"]


def _integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return value


def _text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value


def _tokens(text: str) -> list[str]:
    # A deterministic lexical baseline, not a language-specific segmenter/stemmer.
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold(), re.UNICODE)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Document:
    """An application-authorized document. ``source`` is a label, never fetched."""
    id: str
    text: str
    source: str = ""

    def __post_init__(self) -> None:
        _text("document id", self.id)
        _text("document text", self.text)
        if not isinstance(self.source, str):
            raise ValueError("document source must be a string")


@dataclass(frozen=True)
class SearchHit:
    """A ranked passage with exact Python-character offsets in its document."""
    document_id: str
    source: str
    start: int
    end: int
    text: str
    score: float

    def evidence(self) -> dict[str, Any]:
        """The model sees evidence, not a retrieval score masquerading as confidence."""
        return {key: value for key, value in asdict(self).items() if key != "score"}


@dataclass(frozen=True)
class PreparedContext:
    """A copied state and the passages actually packed into it, not every candidate."""
    state: dict[str, Any]
    hits: tuple[SearchHit, ...]
    omitted: int
    context_chars: int
    context_tokens: int | None
    status: str

    def report(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "hits": [asdict(hit) for hit in self.hits],
            "omitted": self.omitted,
            "context_chars": self.context_chars,
            "context_tokens": self.context_tokens,
        }


class SentenceTransformerEncoder:
    """Optional local-only encoder with distinct document/query entry points.

    Provision weights separately. A path or cached model ID is accepted, but this
    adapter never opts into downloads or remote model code. The caller owns the
    encoder lifecycle; its device memory is not managed by Laya's Router.
    """
    def __init__(self, model: str, *, device: str = "cpu", batch_size: int = 32) -> None:
        _text("embedding model", model)
        self.batch_size = _integer("batch_size", batch_size)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError("install laya[retrieval] to use local sentence embeddings") from error
        self.model = SentenceTransformer(model, device=device, local_files_only=True, trust_remote_code=False)

    def encode_documents(self, texts: Sequence[str]) -> Any:
        return self.model.encode_document(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )

    def encode_query(self, text: str) -> Any:
        return self.model.encode_query(
            [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )


class LocalIndex:
    """In-memory BM25, cosine, or reciprocal-rank-fusion retrieval.

    ``encoder`` must expose ``encode_documents(texts)`` and ``encode_query(text)``.
    Both return a finite, nonzero 2-D matrix (the query matrix has one row).
    Corpus vectors are computed once, not on each search. No remote client is
    provided; injected encoders are application code and must respect local-only
    operation. Rebuild the index to update documents or the embedding model.
    """
    def __init__(
        self, documents: Iterable[Document], *, backend: str = "lexical",
        encoder: Any = None, chunk_size: int = 1200, overlap: int = 150,
    ) -> None:
        if backend not in ("lexical", "dense", "hybrid"):
            raise ValueError("backend must be lexical, dense, or hybrid")
        _integer("chunk_size", chunk_size)
        _integer("overlap", overlap, 0)
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if backend != "lexical" and encoder is None:
            raise ValueError("dense and hybrid retrieval require a local encoder")
        if encoder is not None and not all(callable(getattr(encoder, name, None)) for name in
                                           ("encode_documents", "encode_query")):
            raise ValueError("encoder must expose encode_documents and encode_query")
        self.backend, self._encoder = backend, encoder
        docs = tuple(documents)
        if any(not isinstance(doc, Document) for doc in docs):
            raise ValueError("documents must contain Document instances")
        if len({doc.id for doc in docs}) != len(docs):
            raise ValueError("document IDs must be unique")
        passages = []
        for doc in docs:
            for start in range(0, len(doc.text), chunk_size - overlap):
                end = min(start + chunk_size, len(doc.text))
                text = doc.text[start:end]
                if text.strip():
                    passages.append(SearchHit(doc.id, doc.source, start, end, text, 0.0))
                if end == len(doc.text):
                    break
        self._passages = tuple(passages)
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths = []
        for index, passage in enumerate(self._passages):
            terms = Counter(_tokens(passage.text))
            self._lengths.append(sum(terms.values()))
            for term, count in terms.items():
                self._postings[term].append((index, count))
        self._average_length = sum(self._lengths) / max(1, len(self._lengths)) or 1.0
        self._vectors = None
        if backend != "lexical" and self._passages:
            self._vectors = self._normalize(
                encoder.encode_documents([p.text for p in self._passages]), len(self._passages),
            )
            self._vectors.setflags(write=False)

    @staticmethod
    def _normalize(vectors: Any, rows: int, dimension: int | None = None) -> Any:
        import numpy as np

        try:
            raw = np.asarray(vectors)
            if raw.dtype.kind not in "fiu":
                raise ValueError("embeddings must contain real numbers")
            values = np.array(raw, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("embeddings must be a rectangular real-number matrix") from error
        if values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0:
            raise ValueError("embeddings must have shape (%d, positive dimension)" % rows)
        if dimension is not None and values.shape[1] != dimension:
            raise ValueError("query and document embedding dimensions differ")
        if not np.isfinite(values).all():
            raise ValueError("embeddings must be finite")
        scale = np.max(np.abs(values), axis=1, keepdims=True)
        if (scale == 0).any():
            raise ValueError("embeddings must not contain zero vectors")
        # Scaling first avoids overflow even for finite vectors near float64's limit.
        values /= scale
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        return values

    def __len__(self) -> int:
        """Number of indexed passages (not source documents)."""
        return len(self._passages)

    def search(
        self, query: str, *, k: int = 3, document_ids: Iterable[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """Return positive-scoring hits, with stable corpus-order tie breaking.

        ``min_score`` is in the selected backend's units, never a probability.
        An empty scope or blank query returns no hits and does not call the encoder.
        Hybrid fusion ranks the entire eligible set, so changing ``k`` preserves
        the ranking prefix. BM25 statistics are corpus-wide, even with a filter.
        """
        _integer("k", k)
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if isinstance(min_score, bool) or not isinstance(min_score, Real) or not math.isfinite(min_score):
            raise ValueError("min_score must be finite")
        if min_score < 0:
            raise ValueError("min_score must be nonnegative")
        if document_ids is not None:
            if isinstance(document_ids, (str, bytes)):
                raise ValueError("document_ids must be an iterable of IDs, not a string")
            allowed = {_text("scope document id", item) for item in document_ids}
        else:
            allowed = None
        eligible = [i for i, p in enumerate(self._passages) if allowed is None or p.document_id in allowed]
        if not eligible or not query.strip():
            return []
        eligible_set = set(eligible)
        lexical: dict[int, float] = defaultdict(float)
        if self.backend != "dense":
            for term in sorted(set(_tokens(query))):
                postings = self._postings.get(term, ())
                idf = math.log1p((len(self._passages) - len(postings) + 0.5) / (len(postings) + 0.5))
                for index, frequency in postings:
                    if index in eligible_set:
                        norm = 1.2 * (0.25 + 0.75 * self._lengths[index] / self._average_length)
                        lexical[index] += idf * frequency * 2.2 / (frequency + norm)
        dense = {}
        if self.backend != "lexical":
            vector = self._normalize(self._encoder.encode_query(query), 1, self._vectors.shape[1])[0]
            similarities = self._vectors[eligible] @ vector
            dense = {index: min(1.0, float(score)) for index, score in zip(eligible, similarities) if score > 0}
        if self.backend == "hybrid":
            scores: dict[int, float] = defaultdict(float)
            for ranking in (lexical, dense):
                for rank, index in enumerate(sorted(ranking, key=lambda i: (-ranking[i], i)), 1):
                    scores[index] += 1.0 / (60 + rank)
        else:
            scores = lexical if self.backend == "lexical" else dense
        ranked = sorted((i for i in scores if scores[i] > 0 and scores[i] >= min_score),
                        key=lambda i: (-scores[i], i))[:k]
        return [SearchHit(**{**asdict(self._passages[i]), "score": scores[i]}) for i in ranked]

    def prepare(
        self, state: Mapping[str, Any], *, query: str, k: int = 3,
        context_key: str = "retrieved_context", max_chars: int = 6000,
        document_ids: Iterable[str] | None = None, min_score: float = 0.0,
        token_count: Callable[[str], int] | None = None, max_tokens: int | None = None,
    ) -> PreparedContext:
        """Pack whole passages without modifying the caller's state or instructions.

        Bounds include the serialized evidence array and its provenance. Token
        accounting is optional and uses the caller's tokenizer; it covers this
        array only, NOT the complete Laya request or question heads. Oversized
        hits are skipped (not silently cut); smaller later hits can still fit.
        Questions must explicitly read ``context_key``. Inspect ``status`` before
        deciding whether prediction without usable evidence is acceptable.
        """
        if not isinstance(state, Mapping) or any(not isinstance(key, str) for key in state):
            raise ValueError("state must be a mapping with string keys")
        _text("context_key", context_key)
        if context_key in state:
            raise ValueError("context_key already exists in state; refusing to overwrite it")
        _integer("max_chars", max_chars, 2)
        if (token_count is None) != (max_tokens is None):
            raise ValueError("token_count and max_tokens must be supplied together")
        if max_tokens is not None:
            _integer("max_tokens", max_tokens, 0)
            if not callable(token_count):
                raise ValueError("token_count must be callable")

        def measure(text: str) -> int | None:
            if token_count is None:
                return None
            return _integer("token_count result", token_count(text), 0)

        packed: list[SearchHit] = []
        encoded, tokens = "[]", measure("[]")
        if tokens is not None and tokens > max_tokens:
            raise ValueError("max_tokens cannot hold even an empty evidence array")
        candidates = self.search(query, k=k, document_ids=document_ids, min_score=min_score)
        for hit in candidates:
            trial = _json([p.evidence() for p in packed] + [hit.evidence()])
            if len(trial) > max_chars:
                continue
            trial_tokens = measure(trial)
            if trial_tokens is not None and trial_tokens > max_tokens:
                continue
            packed.append(hit)
            encoded, tokens = trial, trial_tokens
        copied = copy.deepcopy(dict(state))
        copied[context_key] = [p.evidence() for p in packed]
        status = "ready" if packed else "budget_exhausted" if candidates else "no_matches"
        return PreparedContext(copied, tuple(packed), len(candidates) - len(packed), len(encoded), tokens, status)
