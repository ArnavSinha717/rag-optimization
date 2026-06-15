# Deterministic Retrieval Metrics — Reference

All retrieval scoring in this project is done with the code below: pure token-set
arithmetic against hand/CUAD-annotated gold answer spans. **No LLM anywhere in the
scoring path** — same inputs always produce the same numbers. This is the primary
instrument for ranking configurations; RAGAS (LLM-judged) is used only for
generation-quality validation of finalists.

Why it exists: LLM judges proved noisy (±0.03 between identical runs) and once
*inverted* a conclusion (a 3B judge scored reranking as hurting when it helped).
These metrics are the noise-free anchor that caught that.

Lives in: `rag_lib.py` (building blocks), `eval/*_test.py` (per-experiment wiring).

---

## 0. Inputs

Each eval question (`eval/questions_v2.jsonl`) carries:

```json
{
  "id": "q07",
  "question": "Under which state's law is the Chase Affiliate Agreement governed?",
  "expected_answer": "Delaware.",
  "doc_id": "creditcardscominc_..._affiliate_20agreement",
  "evidence": "This Agreement will be governed in all respects by the laws of the State of Delaware",
  "category": "governing_law",
  "difficulty": "easy"
}
```

`evidence` is the gold answer span — the exact contract text that answers the
question. It is the ground truth every metric matches against.

---

## 1. Precondition guard (`rag_lib.assert_corpus_coverage`)

Run before any scoring. A question whose document has no chunks in the variant is
indistinguishable from a retrieval failure and silently corrupts every metric.

```python
def assert_corpus_coverage(questions: list[dict], variant_id: str) -> None:
    """METHODOLOGY GUARD: every in-scope question's doc_id must have chunks in the
    variant. Otherwise a missing document is indistinguishable from a retrieval
    failure and silently corrupts every metric. Refuses to proceed on violation."""
    needed = {q["doc_id"] for q in questions if q.get("doc_id")}
    have = set(chunks.distinct("source.doc_id", {"config.variant_id": variant_id}))
    orphans = sorted(needed - have)
    if orphans:
        raise AssertionError(
            f"{len(orphans)} question doc_ids have NO chunks in variant '{variant_id}'. "
            f"Eval would mis-score them as retrieval failures. First few: {orphans[:5]}")
    print(f"corpus coverage OK: all {len(needed)} question docs present in {variant_id}")
```

---

## 2. Token extraction (`rag_lib.content_tokens`)

Reduces text to its *distinctive* words. The stopword list includes legal
boilerplate — without it, phrases like "governed by the laws of the state of"
match in every contract and produce false positives (a bug we hit and fixed:
naive overlap scored retrieval as working when the actual state name was never
retrieved).

```python
STOPWORDS = set("""a an the this that these those of to in on at by for and or but with as is are was were be been
being shall will may must any all each such other its his her their it they them we you he she i
agreement party parties section clause hereby herein hereof hereto hereunder hereinafter thereof therein
upon under pursuant accordance whereas witnesseth now therefore provided including include includes
between among from into out over per via not no if then than which who whom whose when where while
governed construed effect force law laws state states united right rights obligation obligations term
terms condition conditions set forth made entered date dated effective""".split())

def content_tokens(s: str) -> set:
    toks = re.findall(r"[a-z0-9$]+", s.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 2}
```

Example: `"governed by the laws of the State of California"` → `{"california"}`.

Known limitation: when the answer is a single common token that also appears
elsewhere in the document (e.g. a governing-law state that also appears in the
incorporation clause), overlap can false-positive. Bias is consistent across
configs, so *rankings* between configs remain valid even where absolute values
are slightly optimistic.

---

## 3. Rank of the answer chunk (`answer_rank`)

The core measurement: walk the retrieved chunks in rank order; return the
(1-based) position of the first chunk containing ≥50% of the evidence span's
distinctive tokens. `None` = the answer was not retrieved at all.

```python
def answer_rank(evidence: str, ordered_chunks: list[dict], thr: float = 0.5):
    ev = content_tokens(evidence)
    if not ev:
        return None
    for i, h in enumerate(ordered_chunks):
        if len(ev & content_tokens(h["text"])) / len(ev) >= thr:
            return i + 1
    return None
```

(An earlier binary variant, `passage_hit` in `rag_lib.py` with threshold 0.6,
returns 1/0 for "answer in top-k at all" — superseded by the rank-aware version
for config comparisons; still used by the legacy `evaluate_variant`.)

---

## 4. Aggregation — recall@k and MRR

Every number in the comparison tables is one of these two formulas over the
per-question ranks.

```python
ranks = [answer_rank(q["evidence"], retrieved_for(q)) for q in in_scope_questions]

def recall_at(k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

def mrr() -> float:
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)
```

- `recall@k`: fraction of questions whose answer chunk appears at rank ≤ k.
  recall@5 = "did the answer reach the chunks the LLM actually sees".
- `MRR` (mean reciprocal rank): rank 1 → 1.0, rank 2 → 0.5, rank 4 → 0.25,
  miss → 0. Rewards putting the answer chunk *high*, not merely present.

---

## 5. Refusal / hallucination probe (out-of-scope questions)

For questions whose answer is **not** in the corpus, the only correct behaviour
is the exact refusal string. Scored mechanically:

```python
REFUSAL = "Not found in the provided documents."
score = 1.0 if REFUSAL.lower() in answer.lower() else 0.0   # else = hallucination risk
refusal_rate = mean of scores over out-of-scope probes
```

---

## 6. Bootstrap confidence intervals (for the n≈190 rerun)

Question-sampling uncertainty on any aggregate, with zero additional API calls:
resample the per-question ranks with replacement, recompute the metric, take
percentiles. Used to state "tied" / "worse" statistically rather than by eye.

```python
import random

def bootstrap_ci(ranks, metric_fn, n_boot=10_000, lo=0.025, hi=0.975):
    stats = []
    for _ in range(n_boot):
        sample = random.choices(ranks, k=len(ranks))      # resample WITH replacement
        stats.append(metric_fn(sample))
    stats.sort()
    return stats[int(lo * n_boot)], stats[int(hi * n_boot)]

# e.g. 95% CI on MRR:
ci = bootstrap_ci(ranks, lambda rs: sum(1/r for r in rs if r) / len(rs))
```

For paired config comparisons (A vs B on the same questions), bootstrap the
per-question *difference* instead — cancels question difficulty and is the more
sensitive test. Note: near-duplicate questions (tagged `near_dup_group` in the
question file) are not independent observations; run the bootstrap with and
without them collapsed.

---

## 7. Wiring a config comparison

Every experiment differs only in how the ordered chunk list is produced; the
yardstick (steps 2–4) is identical. Skeleton from `eval/multiquery_test.py`:

```python
assert_corpus_coverage(questions, VARIANT)              # step 1: gate

ranks = []
for q in in_scope_questions:
    qv   = cached_query_embedding(q)                    # Gemini RETRIEVAL_QUERY, cached to disk
    hits = vector_search(qv, VARIANT, n=20)             # <- the only part that varies
    scores = reranker.predict([(q["question"], h["text"]) for h in hits])
    hits = [h for _, h in sorted(zip(scores, hits), key=lambda z: z[0], reverse=True)][:5]
    ranks.append(answer_rank(q["evidence"], hits))      # same yardstick, every config

print(f"rec@1={recall_at(1):.3f}  rec@5={recall_at(5):.3f}  MRR={mrr():.3f}")
```

Query embeddings are cached in `eval/query_embed_cache.json` keyed by question
id, so reruns cost zero Gemini calls and stay byte-identical.

---

## Properties worth stating in any writeup

| Property | Value |
|---|---|
| LLM calls in scoring | 0 |
| Run-to-run variance | 0 for deterministic configs (vector/BM25/hybrid + rerank); HyDE/multi-query vary via their generation step, not via scoring |
| Cost per full sweep | ~0 (cached embeds) |
| Covers | retrieval only — generation quality requires RAGAS / manual reading |
| Known bias | token-overlap can false-positive on repeated distinctive terms; consistent across configs, so rankings are trustworthy, absolutes slightly optimistic |
