# Legal RAG Optimization Study

A systematic, measured evaluation of nine RAG optimization techniques on a legal-contract
corpus — including the negative results. Final production pipeline: **recursive chunking →
vector search → cross-encoder reranking → strict concise generation.**

## Setup

| Component | Choice |
|---|---|
| Corpus | 11 CUAD contracts (sponsorship + affiliate agreements), PDFs in S3 |
| Vector DB | MongoDB Atlas (M0), `$vectorSearch`, scalar quantization |
| Embeddings | `gemini-embedding-001`, 768-dim, `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types |
| Generation | Gemini 2.5 Flash (production) / qwen2.5-7B local (evaluation phase) |
| Reranker | `BAAI/bge-reranker-base` cross-encoder, local GPU |
| Eval set | 59 in-scope questions + 8 out-of-scope refusal probes, hand-written with evidence spans |

## Evaluation methodology (the foundation everything rests on)

Two-tier system, adopted after early failures with LLM judges:

1. **Deterministic retrieval metric (iteration loop).** For each question, find the rank of the
   answer-bearing chunk in the final top-5 — matched by stopword-filtered token overlap against a
   hand-annotated evidence span. Reported as recall@1 / recall@5 / MRR. Zero LLM calls, zero
   noise, perfectly reproducible.
2. **RAGAS (milestones only).** context_recall, context_precision, faithfulness,
   answer_relevancy — judged by a local qwen2.5-7B-instruct (a 3B judge proved unusable, see
   Lessons). Run only to validate final configurations.

Three methodology bugs were caught and fixed along the way; they shaped everything after:

- **Document-level recall was trivially 1.0** (any chunk of the right contract counted) and hid
  the real retrieval problem. Passage-level matching replaced it.
- **Naive token overlap false-positived on legal boilerplate** ("governed by the laws of the
  state of…" matches everywhere). Stopword/boilerplate filtering fixed it.
- **The 3B LLM judge inverted a conclusion** (see Reranking below) — caught because the
  deterministic metric disagreed.

---

## Results by technique

### Chunking (ingestion-time)

| Config | passage_recall@5 (32Q) | Notes |
|---|---|---|
| fixed 2000/200 (v01, baseline) | 0.844 | |
| fixed 1000/100 (v02) | 0.750 | **regression** + introduced a hallucination (fabricated CEO with fake citation) |
| recursive 2000/200 (v04) | — | MRR 0.625 → 0.663 vs v01 on dev set; **adopted** |
| markdown/structure-aware via pymupdf4llm (v07) | — | rec@5 0.932 vs 0.949 (59Q); parser still extracted footer noise; **rejected** |

**Why recursive won:** boundary-aware splitting stopped atomic facts (fee tables, license-term
lists) from being cut in half — the v02 failure mode. Smaller chunks at fixed k *halved* the
total retrieved text and increased misses: chunk size and k are coupled.

### Reranking (query-time) — the headline win

Deterministic, dev set, single-variable:

| Config | rec@1 | rec@5 | MRR |
|---|---|---|---|
| v01 vector only | 0.533 | 0.733 | 0.625 |
| v01 + rerank | 0.600 | 0.800 | 0.693 |
| v04 recursive + rerank | 0.667 | 0.800 | 0.733 |

Combined chunking + reranking: **MRR 0.625 → 0.733 (+17%)**. The cross-encoder reads
(query, chunk) jointly and rescues buried-but-relevant chunks (one HQ-address chunk moved from
rank 10 → 1).

**The judge-inversion incident:** RAGAS with a 3B judge scored reranking as *hurting*
context_precision (0.400 → 0.273). The deterministic metric proved the opposite (MRR up on every
cut). The weak judge wasn't merely noisy — it pointed the wrong direction. All subsequent RAGAS
used the 7B judge, and the deterministic metric remained the arbiter for variant ranking.

### Generation hardening

RAGAS (7B judge), 32Q, retrieval held constant:

| Generator | faithfulness (answered-only) | answer_relevancy | Hallucinations |
|---|---|---|---|
| 3B, loose prompt | 0.863 | 0.724 | 2 (fabricated acronym expansion; inverted either/neither logic) |
| 7B, strict anti-inference prompt | 0.890 | 0.728 | **0** |
| + conciseness rule (59Q) | — | **0.742** | 0 |

The strict prompt (never infer, never expand acronyms not in sources, exact refusal string, read
negations carefully) eliminated both observed hallucinations. Aggregate faithfulness *appears*
flat (0.706 → 0.695) because RAGAS scores refusals as 0 and the stricter model refuses more —
splitting answered-vs-refused reveals the real gain. **Trade-off accepted deliberately: more
refusals, zero confident-wrong answers — the right operating point for legal.**

### Query transformation: HyDE and Multi-Query — the small-eval-set trap

| Eval set | HyDE vs baseline (rec@5) |
|---|---|
| 15-question dev set (failure-enriched) | 0.800 → 0.933 **(+13 pts — looked like a major win)** |
| 59-question full set | 0.949 → 0.949 **(zero)** |

RAGAS on 59Q, generation held constant — statistically tied on every metric (recall 0.908/0.908,
precision 0.827/0.835, faithfulness 0.799/0.828, relevancy 0.742/0.733; 7B judge noise is ±0.03).
Manual reading of the diffing answers found HyDE *slightly worse*: its stochastic hypothetical
(temp 0.3) sometimes steered retrieval to a plausible-but-wrong clause (extension clause instead
of termination; wrong contract's governing law), producing confidently wrong answers. Note
**faithfulness ≠ correctness**: HyDE answered *faithfully from the wrong context*, so its
faithfulness score was high while its answers were worse.

Multi-Query (7B rephrasings, union, rerank): 0.831 / 0.949 / 0.887 — exactly tied with baseline.

**Verdict: query transformation rejected.** Equal-or-worse quality, +1 LLM call per query,
non-deterministic. The apparent dev-set win was a small-sample artifact.

### Hybrid search (vector + BM25, RRF fusion)

| Config (59Q) | rec@1 | rec@5 | MRR |
|---|---|---|---|
| vector + rerank | 0.831 | 0.949 | 0.890 |
| BM25 only | 0.847 | 0.949 | 0.888 |
| hybrid + rerank | 0.831 | 0.949 | 0.890 |

BM25 alone nearly matches the full vector pipeline on legal text (exact terms dominate), and
fusing them changes nothing — both retrievers already surface the same chunks at rec@5 ≈ 0.95.
BM25 *did* uniquely solve one pathological case (an acronym defined once, invisible to
embeddings: vector rank >20, BM25 rank 1) — but stacking hybrid onto HyDE *hurt* (dev rec@5
0.933 → 0.800): **techniques interfere; combinations must be measured, never assumed additive.**

### Contextual retrieval (Anthropic-style chunk blurbs)

| Config (59Q) | rec@1 | rec@5 | MRR |
|---|---|---|---|
| v08 blurbs + recursive | 0.746 | **0.966** | 0.850 |
| v09 blurbs + markdown | 0.831 | 0.949 | 0.876 |

Mixed: best rec@5 of any config (+1 question into the top-5) but rec@1 collapsed — the prepended
blurbs make same-contract sibling chunks more similar, blurring precise ranking. Net ≈ tied; not
worth the re-ingestion pipeline. Did not rescue the markdown parser either.

### Metadata filtering (LLM auto-picks the contract, hard doc_id filter)

| Config (59Q) | rec@1 | rec@5 | MRR |
|---|---|---|---|
| auto-filtered | 0.729 | 0.847 | **0.788 — worst result in the study** |

The 7B picked the correct contract only 50/59 (85%) — and every wrong pick is a *guaranteed*
miss, since the filter excludes the right document entirely. **Hard filters amplify router
errors.** The technique belongs where the contract is named with high confidence (by the user or
conversation history in the chatbot), not auto-inferred per query.

---

## Final scoreboard (deterministic, 59 questions)

| Config | rec@1 | rec@5 | MRR | Verdict |
|---|---|---|---|---|
| **recursive + vector + rerank (CHAMPION)** | 0.831 | 0.949 | **0.890** | shipped |
| + HyDE | 0.864 | 0.949 | 0.902* | rejected (cost, non-determinism, worse answers) |
| BM25 only | 0.847 | 0.949 | 0.888 | tied |
| + hybrid | 0.831 | 0.949 | 0.890 | tied |
| + multi-query | 0.831 | 0.949 | 0.887 | tied |
| markdown parser | 0.864 | 0.932 | 0.895 | worse rec@5 |
| + contextual blurbs | 0.746 | 0.966 | 0.850 | mixed, net tied |
| auto metadata filter | 0.729 | 0.847 | 0.788 | worst |

\* within HyDE's own ±0.03 run-to-run variance (temp-0.3 hypotheticals are non-deterministic).

**Champion validated by RAGAS (7B judge, 59Q):** context_recall 0.908, context_precision 0.827,
faithfulness 0.799, answer_relevancy 0.742, refusal rate on out-of-scope probes 1.0.

## Why everything tied: the ceiling

The answer chunk reaches the top-5 on ~56/59 questions in *every* configuration. The 3 remaining
misses are pathological (e.g., a company-specific acronym defined once inside a dense definitions
preamble — proven retrieval-bound: given the chunk, the generator answers correctly). Each
technique that could fix one of them regressed something else. On this corpus, retrieval is at
its measured ceiling and the bottleneck moved to generation — which the prompt/model hardening
then addressed.

Several rejected techniques (hybrid, contextual retrieval, metadata filtering) are expected to
pay off on larger, noisier, more heterogeneous corpora; the negative results are claims about
*this* corpus, not the techniques in general.

## Lessons that generalize

1. **Build a deterministic metric before touching an LLM judge.** It is free, noise-free, and it
   caught our judge lying.
2. **Judge quality is load-bearing.** A 3B judge inverted a conclusion; the 7B was needed for
   claim-decomposition metrics. Validate any judge against hand-graded examples first.
3. **Small eval sets exaggerate.** A +13-point "win" on 15 questions vanished at 59. Confirm
   every win on the full set before adopting.
4. **Faithfulness ≠ correctness.** A model can answer faithfully from the wrong context.
5. **Aggregate metrics hide refusal-rate shifts.** Split answered-vs-refused before reading
   faithfulness.
6. **Techniques interfere.** HyDE+hybrid < HyDE. Measure combinations.
7. **Hard filters amplify upstream errors** — an 85%-accurate router becomes a 15% guaranteed
   failure rate.
8. **Read the actual outputs.** Manual reading caught the judge inversion, a fabricated citation,
   and inverted contract logic — none visible in any aggregate score.
9. **The biggest wins were unglamorous:** boundary-aware chunking, a reranker, and a stricter
   prompt — not the fashionable techniques.

---

# Part 2: At Scale — 50 contracts, 189 questions

Part 1's central claim ("retrieval is at its ceiling; everything ties") was suspected to be an
artifact of a tiny, homogeneous corpus. Part 2 tests that directly: a **6.4× larger, deliberately
heterogeneous corpus**, re-running the retrieval configs with the same deterministic metric plus
bootstrap confidence intervals. The headline: **the ceiling was an artifact, and two Part-1
conclusions invert.**

## Setup changes

| Dimension | Part 1 | Part 2 |
|---|---|---|
| Contracts | 11 (sponsorship + affiliate only) | **48** (license, distributor, development, co-branding, service, endorsement, manufacturing, supply, franchise, + the original 11) |
| Chunks (recursive variant) | 220 | **1,402** |
| In-scope questions | 59 | **189** |
| Question source | hand-written | derived from CUAD's expert clause annotations, spot-checked (~8% dropped as mis-annotated), categorized by clause type |
| Largest single contract | ~30 chunks | 157 chunks (Array Biopharma license) |
| Methodology guard | — | `assert_corpus_coverage()` — refuses to score if any question's document is absent (a missing doc is indistinguishable from a retrieval failure) |

Variant: `v10_corpus50_recursive`. Stats: bootstrap 95% CIs, 10,000 resamples; paired
champion-vs-config differences reported both naive (per-question) and clustered (resampling the
near-duplicate question groups together, which shrinks effective n).

## Retrieval scoreboard (deterministic, n=189, 95% CI)

| Config | rec@1 | rec@5 | 95% CI (rec@5) | MRR | 95% CI (MRR) |
|---|---|---|---|---|---|
| **vector + rerank (CHAMPION)** | 0.598 | 0.815 | [0.757, 0.868] | 0.691 | [0.633, 0.748] |
| vector + rerank, k=10 | 0.598 | 0.815 | [0.757, 0.868] | 0.699 | [0.642, 0.755] |
| vector only, k=5 | 0.545 | 0.799 | [0.741, 0.857] | 0.638 | [0.577, 0.698] |
| hybrid (vec+BM25 RRF) + rerank | 0.587 | 0.810 | [0.751, 0.862] | 0.682 | [0.623, 0.740] |
| vector + rerank, k=3 | 0.598 | 0.783 | [0.725, 0.841] | 0.684 | [0.624, 0.743] |
| HyDE + rerank | 0.614 | 0.820 | [0.762, 0.873] | 0.699 | [0.640, 0.756] |
| multi-query + rerank | 0.598 | 0.815 | [0.757, 0.868] | 0.690 | [0.631, 0.747] |
| BM25 + rerank | 0.471 | 0.619 | [0.550, 0.688] | 0.530 | [0.463, 0.597] |
| BM25 only | 0.323 | 0.487 | [0.418, 0.561] | 0.382 | [0.319, 0.448] |

## Paired significance vs champion (MRR difference, CI excludes 0 ⇒ real)

| Config | ΔMRR | naive 95% CI | clustered 95% CI | Verdict |
|---|---|---|---|---|
| vector + rerank k=10 | −0.008 | [−0.013, −0.004] | [−0.013, −0.004] | k=10 marginally better (catches rank 6–10) |
| **vector only** | **+0.053** | **[+0.001, +0.109]** | **[−0.004, +0.111]** | **reranking helps — significant naive, BORDERLINE clustered** |
| hybrid + rerank | +0.009 | [−0.009, +0.029] | [−0.009, +0.030] | **tie** — hybrid is dead weight |
| vector + rerank k=3 | +0.007 | [+0.002, +0.013] | [+0.002, +0.013] | k=3 loses the tail |
| HyDE + rerank | −0.007 | [−0.020, +0.004] | [−0.020, +0.004] | **tie** (faint nominal edge, not significant) |
| multi-query + rerank | +0.002 | [−0.012, +0.018] | [−0.012, +0.018] | **tie** (most inert — 94% identical to champion) |
| BM25 + rerank | +0.161 | [+0.104, +0.221] | [+0.105, +0.220] | champion massively better |
| BM25 only | +0.309 | [+0.244, +0.377] | [+0.240, +0.376] | champion massively better |

## Finding 1 — The ceiling was an artifact. Headroom is back.

rec@5 fell from **0.949 → 0.815**; the champion now misses **35 of 189 questions**. Part 1's "every
technique ties because retrieval is solved" was a small-corpus illusion. With 1,402 chunks
competing, retrieval differences are measurable again — which is what makes the rest of Part 2
possible.

## Finding 2 — BM25 didn't flip positive; it COLLAPSED (the most important result).

Part 1 predicted hybrid/lexical search might *help* on a heterogeneous corpus. The opposite
happened: BM25-only rec@5 fell **0.949 → 0.487**, and even rerank-rescued BM25 (0.619) trails plain
vector search by a mile (champion beats it by ΔMRR +0.161, CI [+0.104, +0.221] — unambiguous).

**Mechanism, verified by inspection:** asked about the *Bellicum Pharmaceuticals* supply
agreement's termination, BM25's top hit was a termination clause from *Revolution Medicines* — a
different contract. Every contract contains "termination," "governed by," "shall"; lexical
matching surfaces that boilerplate from whichever document has the highest term-frequency, with no
signal for *which* contract the user meant. Dense embeddings carry entity/semantic context that
disambiguates. **Lexical retrieval degrades as a corpus grows more homogeneous in vocabulary —
inverting the popular "hybrid search always helps" guidance.** It helps on small or jargon-sparse
corpora; it actively hurts here. (BM25 index health was verified: 20 hits/query, correctly scoped,
zero empty returns — the collapse is real, not a misconfiguration.)

## Finding 2b — HyDE's verdict was STABLE across scale (unlike BM25).

HyDE was the other technique expected to behave differently at scale. It didn't. At 59 questions
it tied the champion; at 189 it ties again — rec@5 0.820 vs 0.815, MRR 0.699 vs 0.691, paired
ΔMRR −0.007 with CI [−0.020, +0.004] straddling zero. A faint nominal edge (rec@1 0.614 vs 0.598),
never significant, still not worth +1 LLM call and a non-deterministic query vector per request.

**Why HyDE stayed stable while BM25 collapsed** is itself instructive: HyDE only changes the
*query vector* — retrieval still happens in the same dense space that carries entity/semantic
signal, so it never enters the boilerplate-disambiguation regime that destroyed BM25. Techniques
that operate on the **query representation** generalized across scale; the technique that operates
on a **different (lexical) index** did not. Verdict stability is not uniform — it depends on which
part of the pipeline a technique touches.

**Reading the HyDE answers (not just the aggregate).** The tie is *offsetting, not inert*. Diffing
HyDE against the champion per question: both hit 152, both miss 32, HyDE **rescued 3** (champion
missed → HyDE found) and **broke 2** (champion had it → HyDE lost it), net +1 — well inside the
noise band. Reading the generated hypotheticals explains the mechanism:

- **Rescues** come when the hypothetical supplies clause-level *vocabulary* the terse question
  lacked. q104 (non-compete) → "prohibit … from engaging in competitive activities"; q177
  (anti-assignment) → "assignment … subject to prior written consent" — phrasing that matched the
  real clause better than the bare question, pulling the right chunk into the pool.
- **Breaks** come when the hypothetical fabricates a specific *value* that steers retrieval to the
  wrong document. The cleanest case, q135 (governing_law): HyDE confidently asserted *"The laws of
  the State of New York shall govern …"* when the real clause says **"Commonwealth"** (not New York
  at all). That hallucinated jurisdiction pulled retrieval toward New-York-law clauses in *other*
  contracts and knocked the correct chunk — which the champion ranked **#2** — out of the top-5
  entirely. (The 7B also fabricated the identical date "March 15, 2023" in both date questions; it
  was harmless noise in one and broke a rank-5 hit in the other.)

So HyDE's fabrications *amplify* the exact cross-contract boilerplate confusion that already limits
BM25 and the reranker, and they do it most on governing-law and dates — the pipeline's weakest
categories. This is a sharper rejection than "it ties": HyDE doesn't just fail to help, it
introduces a hallucination-driven failure mode precisely where the system is already fragile. The
faithfulness≠correctness lesson from Part 1, now visible in the retrieval stage itself.

## Finding 2c — Multi-query is the most inert technique: paraphrases are redundant in dense space.

Multi-query (original + 3 local-7B rephrasings, union the top-10s, rerank against the real
question) produced **identical rec@1 (0.598) and rec@5 (0.815)** to the champion and tied on MRR
(ΔMRR +0.002, CI [−0.012, +0.018]). It is the most inert technique in the study: **94% of
questions got the exact same rank** as single-query, and the confusion is net zero (2 rescued, 2
broke).

The mechanism, from reading the rephrasings: **paraphrases collapse to nearly the same embedding.**
"What is the term of X?" and "What duration does X have?" land in almost the same place in dense
space, so unioning their results mostly reproduces the single-query set — there is little new to
find. The two rescues (q177 anti-assignment, q221 term) came when a rephrasing happened to add
clause vocabulary ("transfer or assignment", "duration/timeframe") — the *same* vocabulary-rescue
mechanism as HyDE, minus the hallucination risk.

The two breaks are the more interesting half, and they indict the union itself: **q170 went from
rank #1 to out of top-5.** Adding more query variants pulled more candidate chunks into the union,
and the reranker — already the bottleneck (Finding 3) — had *more distractors to mis-rank*. Union
expansion doesn't just fail to help; on the categories where the reranker is weak, feeding it a
larger, noisier candidate pool actively hurts. **Query expansion and a struggling reranker are at
cross-purposes.**

Combined verdict on query-side techniques at scale: HyDE and multi-query both *tie* and both stay
stable across scale (query-representation techniques, Finding 2b) — but neither earns its +1
LLM-call-per-query cost, and both can *hurt* the weak reranker categories (HyDE via hallucinated
values, multi-query via union dilution). The single dense query is not just sufficient; it is
cleaner.

## Finding 3 — The bottleneck moved from retrieval to RERANKING.

Diagnosing all 35 champion misses by whether the evidence chunk was even in the top-20 vector
candidate pool:

| Miss type | Count | Meaning |
|---|---|---|
| **Rerank-bound** (evidence in top-20 pool, reranker dropped it below 5) | **24 / 35** | the cross-encoder is the limiting factor |
| Retrieval-bound (evidence not in top-20 pool at all) | 11 / 35 | vector search never surfaced it |

**Two-thirds of failures are the reranker's fault, not retrieval's** — a reversal from Part 1,
where the reranker was the hero. In several cases the reranker actively *demoted* a correct chunk
that vector search had ranked #1–#3 (q177 pool_rank 1→out; q107, q109, q212 pool_rank 2–3→out).

**Why:** the cross-encoder suffers the *same* disambiguation failure as BM25, for the same reason.
At 48 contracts the top-20 pool for "which state's law governs the Newegg agreement" contains
near-identical "governed by the laws of the State of X" clauses from a dozen contracts. The
reranker scores (query, chunk) textual relevance — and every one of those clauses is textually
relevant to the question. It has no entity-level signal for which contract is *Newegg's*, so it
sometimes ranks the wrong contract's boilerplate above the right one. Reranking and lexical search
share a blind spot: **formulaic clauses that every contract has.**

## Finding 4 — Failure is concentrated by clause type, not spread evenly.

Per-category rec@5 (champion) splits cleanly into "distinctive content" vs "every-contract
boilerplate":

| Strong (distinctive content) | rec@5 | | Weak (formulaic boilerplate) | rec@5 |
|---|---|---|---|---|
| parties | 1.000 | | cap_on_liability | **0.429** |
| factual | 1.000 | | anti-assignment | 0.556 |
| effective_date | 1.000 | | term | 0.571 |
| license_grant | 1.000 | | exclusivity | 0.714 |
| payment_terms | 1.000 | | expiration_date | 0.714 |
| termination (named) | 1.000 | | governing_law | 0.750 |
| | | | termination_for_convenience | 0.750 |

The weak categories are exactly the clauses that read nearly identically across all 48 contracts
(`"NEITHER PARTY SHALL BE LIABLE FOR ANY..."`, `"Neither party may assign this Agreement..."`,
`"governed by the laws of the State of..."`). The strong categories carry contract-specific
entities (party names, dollar amounts, product names) that both the embedder and reranker can
latch onto. **This is the single most actionable diagnostic in the study:** it tells you *what
kind of question* the pipeline fails, and points the fix.

## Finding 5 — Latency and cost: the rejected techniques are also more expensive.

Quality verdicts alone understate the case. Measuring per-stage latency (median wall-clock on the
dev machine: 6GB GPU, Atlas M0, Gemini API over WAN) and per-query cost (measured token counts ×
published rates — Gemini 2.5 Flash $0.30/$2.50 per 1M in/out, gemini-embedding-001 $0.15/1M;
prices May 2026) makes the rejections decisive — each tied-or-worse technique is also slower:

| Config | retrieval (ms) | + answer gen (ms) | **total (ms)** | $/query | $/1k queries |
|---|---|---|---|---|---|
| BM25 only | 48 | 1441 | 1489 | 0.00080 | 0.80 |
| vector only | 455 | 1441 | 1897 | 0.00080 | 0.80 |
| **vector + rerank (champion)** | 930 | 1441 | **2371** | 0.00080 | 0.80 |
| hybrid + rerank | 978 | 1441 | 2419 | 0.00080 | 0.80 |
| HyDE + rerank | 2371 | 1441 | 3812 | 0.00098 | 0.98 |
| multi-query + rerank | 4317 | 1441 | 5758 | 0.01040* | 1.04 |

Stage medians: query embed **406 ms** (a WAN round-trip — the dominant retrieval cost), vector
search 49 ms, BM25 49 ms, rerank 20 candidates **474 ms**, rerank 45 candidates **1053 ms**,
Flash answer generation **1441 ms**.

**Two findings:**

1. **Cost is dominated by answer generation, not retrieval.** The 5-source generation prompt is
   ~2,162 input tokens; the query embed is ~18 tokens ($0.0000027). So every retrieval technique
   costs essentially the same **$0.80 per 1,000 queries** — the differences between them are rounding
   error. *The lever for cost is how many/large the sources you stuff into the generation prompt,
   not which retriever you choose.* Reranking, hybrid, BM25 are all cost-equivalent.

2. **Latency is the real axis — and it confirms every rejection.** The query-transformation
   techniques that merely *tied* on quality are **1.6×–2.4× slower**: HyDE adds 1,441 ms (a second
   LLM call for the hypothetical), and multi-query adds **3,387 ms** (4 embed round-trips at 406 ms
   each + a 2.2× larger rerank pool at 1,053 ms). Multi-query nearly *triples* end-to-end latency
   (5.8 s vs 2.4 s) for a statistical dead heat. The reranker itself scales with pool size
   (474 ms → 1,053 ms from 20 → 45 candidates), so multi-query's union is penalized twice: it
   mis-ranks (Finding 2c) *and* pays double rerank latency.

The champion's 930 ms retrieval breaks down as embed (406) + vector search (49) + rerank (474):
the reranker buys its +0.05 MRR for ~474 ms, the one cost-for-quality trade in the pipeline that
pays off. BM25-only is by far the fastest retrieval (48 ms, no embed round-trip, no rerank) — but
at rec@5 0.487 it is the worst by a wide margin; speed cannot rescue it.

**Net:** on quality the rejected techniques tie-or-lose; on cost they tie; on latency they lose
outright. There is no axis on which HyDE, multi-query, or hybrid beats the plain champion at scale.

## Finding 6 — Generation scorecard at scale: the bottleneck is retrieval, not the generator.

RAGAS on the champion (vector + rerank), 189 questions, local 7B generation + 7B judge, with the
new `answer_correctness` metric (factual + semantic match to the gold answer):

| Metric | Part 1 (59Q) | Part 2 (189Q) | Read |
|---|---|---|---|
| context_recall | 0.908 | **0.786** | ↓ tracks the retrieval drop (rec@5 0.949 → 0.815) |
| context_precision | 0.827 | **0.635** | ↓ more distractors in the top-5 at scale |
| faithfulness (aggregate) | 0.799 | 0.726 | ↓ but dragged down by more refusals (below) |
| faithfulness (answered only) | 0.890 | **0.850** | ≈ held — when it answers, it stays grounded |
| answer_relevancy | 0.742 | **0.731** | ≈ held |
| answer_correctness | — | 0.536 | new; first end-to-end correctness number |
| refusal_rate (in-scope) | — | 0.185 | = 35/189 — refuses exactly on the retrieval misses |

1. **Retrieval-grounded metrics fell in lockstep with deterministic retrieval.** context_recall
   0.908 → 0.786 and context_precision 0.827 → 0.635 mirror the rec@5 drop — the LLM judge,
   scoring an entirely different way, independently reproduces Finding 1. This is a
   retrieval-difficulty signal, not a generation regression.
2. **Generation quality held where it matters.** `faithfulness_answered_only` 0.850 and
   answer_relevancy 0.731 are within noise of Part 1 — the strict prompt still keeps answered
   responses grounded and on-topic at 4.4× scale.
3. **The refusal rate is the system working as designed.** 0.185 = **35/189 — exactly the 35
   retrieval misses** (Finding 3). On every question where retrieval failed, the champion said
   *"Not found in the provided documents"* instead of hallucinating. That is why
   `faithfulness_answered_only` (0.850) far exceeds aggregate faithfulness (0.726): RAGAS scores a
   refusal as unfaithful even though refusing is the *correct* action when the context lacks the
   answer. Same answered-vs-refused artifact flagged in Part 1, now confirmed at scale — and the
   legal-safety operating point (refuse rather than guess) holds.
4. **`answer_correctness` 0.536 is a floor, not the correct-when-answered rate.** It is depressed
   by the 18.5% refusals (correctness ≈ 0 on a refusal, even a *correct* refusal) and the harder
   corpus; correctness conditional on answering is materially higher. *Caveat:* a few items hit a
   judge output-parsing error (the 7B's claim-classification occasionally malformed) and were
   dropped to NaN, adding minor noise to this one metric.

**Why the absolute numbers look low — and why it's mostly measurement, not quality.** Stripping
the 35 correct refusals lifts every metric, and three further effects depress the absolutes:

| Metric | All 189 | Answered-only (154) |
|---|---|---|
| answer_correctness | 0.536 | 0.628 |
| faithfulness | 0.726 | 0.850 |
| context_recall | 0.786 | 0.848 |
| context_precision | 0.635 | 0.700 |
| answer_relevancy | 0.731 | 0.774 |

1. **Correct refusals scored as failures.** The 18.5% "Not found" responses score ~0 on
   faithfulness and correctness even though refusing is the *right* action — this is the entire gap
   between the two columns.
2. **A deliberately weak judge.** `qwen2.5:7b` was chosen to avoid Gemini quota, not for accuracy;
   it systematically under-scores versus a frontier judge and even threw output-parsing errors on
   `answer_correctness`. These absolutes are **floors** — the trustworthy signal is the *comparison*
   (Part 1 vs Part 2, config vs config), not the level. RAGAS absolute values are judge-dependent.
3. **Concise answers vs verbose references.** We engineered 1–2 sentence answers; the gold spans
   are long legal text, so the semantic-similarity component of correctness/relevancy penalizes the
   brevity we deliberately chose.
4. **Fixed k=5 caps precision.** Always returning 5 chunks counts the 4 non-answer chunks as
   imprecise when the answer is localized to one.

The only genuinely low metric is `context_recall` (0.848 answered-only ≈ deterministic rec@5
0.815) — the real retrieval-difficulty signal, which is already the study's headline.

**Bottom line:** at 4.4× scale the limiter is retrieval, not generation. The generator is faithful
(0.85) and relevant (0.77) on what it can ground, and refuses cleanly on what it can't — exactly
the operating point the strict prompt was built for; the low absolutes are dominated by
refusal-scoring and a weak judge, not by wrong answers. Every lever worth pulling next is on the
retrieval/disambiguation side (Findings 3–4), which is what motivates the agent.

## What this means for the product (the bridge to the chatbot/agent)

The boilerplate-disambiguation failure has an obvious fix that the eval *cannot* apply but the
product can: **if you know which contract, the problem disappears.** Filtering the candidate pool
to one contract collapses "20 near-identical governing-law clauses from 20 contracts" to "the one
governing-law clause in this contract." Part 1 *rejected* automatic metadata filtering because the
LLM picked the right contract only 85% of the time and a hard filter turns every wrong pick into a
guaranteed miss. Part 2 explains *why filtering matters so much* and *where the contract name must
come from*: not auto-inferred per query, but supplied with confidence by the user or the
conversation — i.e. the chatbot's contextualizer, or an agent that asks. The negative result and
the product design now point the same direction.

## Updated lessons (additions from scale)

10. **"At ceiling" is corpus-size-dependent.** A solved-looking retrieval problem on 11 documents
    had 18.5% headroom on 48. Never conclude "good enough" without testing at target scale.
11. **Lexical search scales *down*, not up, with corpus homogeneity.** The more documents share a
    vocabulary, the worse term-matching disambiguates them. "Add hybrid search" is not free advice.
12. **Reranking and BM25 share a blind spot.** Both rank textual relevance; neither disambiguates
    *which document* a boilerplate clause belongs to. Adding a reranker does not fix the failure
    mode that adding BM25 causes.
13. **Bottlenecks move.** The same pipeline that was retrieval-bound at small scale became
    rerank-bound at large scale (24/35 misses). Re-diagnose after every scale change.
14. **Significance can hinge on how you count.** Reranking's MRR gain over plain vector was
    significant under naive bootstrap [+0.001, +0.109] but borderline once near-duplicate questions
    were clustered [−0.004, +0.111]. Report the clustered interval — it is the honest one.
15. **Per-category analysis beats a single aggregate.** rec@5 = 0.815 hides a 0.429–1.000 spread by
    clause type. The aggregate says "decent"; the breakdown says "it fails specifically on
    formulaic clauses," which is a fixable, nameable problem.
