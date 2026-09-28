# Auditable local retrieval

Use `laya.retrieval` to select local policy passages before a typed decision. It is an
opt-in composition layer: `Router.predict` and ordinary CLI behavior do not change.
The result exposes **what evidence was packed into the state**, not an explanation
of the model's internal reasoning and not proof that the decision is correct.

## Inspect evidence without loading a checkpoint

From a checkout containing this feature:

```sh
python -m pip install -e .
laya-retrieve "duplicate invoice" --corpus examples/retrieval/policies.jsonl --k 2
```

Alternatively, run `python -m laya.retrieval_cli` with the same arguments. Inspection
is the default and lexical search uses no model, hosted service, or new dependency.
`source` is an opaque label: the loader never fetches a URL or opens a referenced file.
The demo policies are synthetic examples, not a real organization's policy.

The JSON output separates `state.retrieved_context` from `retrieval.hits`. Both
identify source documents and exact `[start:end]` Python-character offsets. Only the
report contains ranking scores; they are deliberately excluded from model evidence.
The report includes only passages actually packed, an `omitted` count among the
retrieved candidates, and a `ready`, `no_matches`, or `budget_exhausted` status.

Each corpus line is an object with `id`, `text`, and optional `source`. IDs must be
unique, IDs/text nonblank, and unknown fields are rejected. The CLI loader caps input
at 10 million decoded characters and 10,000 records, including a bounded read for a
single oversized line. Those are input limits, not promises about index memory use.

## Python composition

```python
from laya.retrieval import Document, LocalIndex

index = LocalIndex([
    Document("billing", "Duplicate invoice: ask billing to investigate.", "policy-v1"),
    Document("access", "Password reset: ask account support.", "policy-v1"),
])
state = {"request": "I received a duplicate invoice."}
prepared = index.prepare(state, query=state["request"], k=2, max_chars=1000)
assert "retrieved_context" not in state  # no caller-state mutation
print(prepared.report())

if prepared.status != "ready":
    print("Send this request to a person; no usable evidence was packed.")
else:
    # Pass prepared.state and an explicit question set to your existing runner.
    # The question instructions must refer to `retrieved_context`.
    print(prepared.state)
```

`prepare` deep-copies state. It refuses a pre-existing context key instead of
silently replacing application data. Document and hit records are frozen; the
returned state is a mutable application-owned snapshot. Do not change that snapshot
and then treat the original report as an audit of the changed input.

## Lexical, dense, and hybrid search

The default backend is BM25 with `k1=1.2` and `b=0.75`. Terms are Unicode-normalized
and case-folded. This is a transparent baseline, **not** a language-specific stemmer
or segmenter; it cannot infer synonyms and is limited for languages without word
boundaries. Chunks are overlapping character windows (`chunk_size=1200`,
`overlap=150`), not sentence-aware or token-aware chunks.

For semantic retrieval, provision an embedding checkpoint locally first, then use:

```sh
python -m pip install -e '.[retrieval]'
laya-retrieve "charged twice" --corpus examples/retrieval/policies.jsonl \
  --backend hybrid --embedding-model /path/to/local/embedding-model
```

```python
from laya.retrieval import LocalIndex, SentenceTransformerEncoder

encoder = SentenceTransformerEncoder("/path/to/local/embedding-model", device="cpu")
# `documents` is your application-authorized list of Document objects.
index = LocalIndex(documents, backend="hybrid", encoder=encoder)
hits = index.search("charged twice", k=3)
```

The optional adapter uses distinct document/query encoding methods, sets
`local_files_only=True` and `trust_remote_code=False`, and raises when dependencies
or local weights are absent. It never silently substitutes a hosted embedder.
Injected encoders must expose `encode_documents(texts)` and `encode_query(text)`,
returning finite, nonzero two-dimensional numeric matrices. A query has one row and
must match the corpus dimension. Custom encoder code is the caller's responsibility;
this protocol cannot sandbox it or enforce its network behavior.

Corpus vectors are computed once per index. Dense search uses normalized cosine;
hybrid search fuses lexical and dense rankings with `sum(1 / (60 + rank))` rather
than adding incompatible raw scores. Ties follow corpus order and changing `k`
preserves the ranking prefix. Zero/negative dense similarities are excluded.
`min_score` uses the selected backend's units, **never a probability**. Thresholds
must be evaluated on the caller's corpus; dense search may return weakly related
passages even when no genuinely useful evidence exists.

This is a small in-process index, not an approximate-nearest-neighbor service.
Dense storage uses float64 (roughly `8 * passage_count * embedding_dimension` bytes,
plus text, postings, and transient arrays); a query scans eligible vectors. Rebuild
the index when documents or the embedding model change. The application owns the
encoder lifecycle and device memory independently of the Router's model cache.

## Evidence budgets and missing evidence

`max_chars` bounds the compact JSON serialization of the **evidence array including
provenance**. The packer selects whole passages, skips an oversized passage, and can
still admit a smaller later passage. It does not silently truncate an evidence hit.
`k` bounds the candidate list; the packer does not search more than `k` to refill it.

For an evidence-only token budget, pass the target tokenizer explicitly:

```python
prepared = index.prepare(
    state, query=state["request"],
    token_count=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    max_tokens=128,
)
```

**Neither budget proves that the complete Laya request fits.** State rendering,
other fields, question instructions, and option heads consume additional tokens.
The CLI's character bound is not a token estimate. Choose the checkpoint and its
`max_len` / `head_max_len` budgets deliberately, account for the full rendered
request, and verify real-checkpoint truncation behavior before deploying. The
report proves packing into state, not retention through downstream tokenization.

## Run a typed decision explicitly

After provisioning Laya's prediction checkpoints (normal Router loading may access
the Hugging Face Hub on first use):

```sh
laya-retrieve "duplicate invoice" --corpus examples/retrieval/policies.jsonl \
  --questions examples/retrieval/questions.json --predict --k 1
```

This command calls `route` on the **original request**, then pins that checkpoint
when predicting with enriched state. Otherwise a long policy written in a different
language could change which checkpoint gets chosen. Original routing is reported
under `retrieval.pre_context_routing`. Explicit model, language, task, and token
budget overrides are forwarded; the ordinary Router is not modified.

When no evidence fits, prediction is not attempted and the Router is not constructed:
output contains `prediction: null` and `retrieval.abstained: true`, with exit code
**3**. Success is **0**, invalid input/runtime error is **2**. This is an evidence
availability gate, not calibrated model abstention or a guarantee of relevance.
`--allow-no-context` explicitly opts out. Applications calling `prepare` directly
must implement their own gate. Handle the model's uncertainty and conflicting
policies separately; the demo's `human_review` choice is a question, not enforcement.

## Trust boundaries and validation

Build indexes only from documents the caller is authorized to access. `document_ids`
(and repeatable CLI `--document-id`) scopes candidates before ranking, but is **not an
access-control system**. BM25 corpus statistics remain global even when filtered;
use separate indexes for distinct trust domains. Reports include passage text and
source labels, so redact or restrict logs appropriately.

Keeping evidence in a separate JSON field prevents accidental instruction/schema
rewrites by this code; it does not make the model immune to prompt injection. A
retrieved document can be malicious, obsolete, or incorrect. This layer does not
verify policy authority, resolve conflicts, or prove answer entailment.

Run the offline regression suite:

```sh
python -m unittest discover -s tests -p 'test_retrieval*.py' -v
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
python tests/test_hooks_api.py
```

The targeted suite uses synthetic vectors and stub prediction/embedding adapters;
it verifies algorithms and integration wiring, not checkpoint accuracy. The
retrieval workflow runs without inference dependencies; the existing main CI lane
runs the full API contract suite with its normal dependencies.

For a model-backed evaluation, freeze documents, queries, relevance judgments,
typed labels, model revisions, and hardware. Compare no retrieval, lexical, dense,
and hybrid on the same held-out cases. Report retrieval recall@k separately from
decision accuracy, calibration, no-evidence rate, truncation, index-build cost,
query latency, total prediction latency, and memory. Include irrelevant queries,
conflicting policies, mixed languages, and oversized evidence. Do not infer a
speedup or accuracy gain from these unit tests; this change claims neither.
