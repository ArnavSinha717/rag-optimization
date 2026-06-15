# Evaluation suite

Every number in [`../OPTIMIZATION_STUDY.md`](../OPTIMIZATION_STUDY.md) is produced by a script in
this folder. Nothing here depends on the (un-shipped) scratch notebook — these `.py` files are the
evaluation, standalone and re-runnable.

## How the evaluation works (two tiers)

1. **Deterministic retrieval metric** — for each question, find the rank of the answer-bearing
   chunk in the retrieved set by stopword-filtered token overlap against a hand-/CUAD-annotated
   evidence span. Reports `rec@1`, `rec@5`, `MRR`. **Zero LLM calls, noise-free, reproducible** —
   this is the iteration loop and the arbiter for ranking configs.
2. **RAGAS** (milestones only) — `context_recall`, `context_precision`, `faithfulness`,
   `answer_relevancy`, `answer_correctness`, judged by a local `qwen2.5:7b-instruct`. Used to score
   *generation* quality on the final config(s), which the deterministic metric cannot see.

See the study doc for *why* this split (an LLM judge once inverted a conclusion; the deterministic
anchor caught it).

## Prerequisites

- `.env` with `GEMINI_API_KEY`, `MONGO_URI` (and `S3_*` only if re-ingesting from S3). **Never
  committed.** All scripts read these via `rag_lib.py` — no secrets live in code.
- MongoDB Atlas with a `$vectorSearch` index named `chunks_vector_index` and a BM25 `$search`
  index over the `chunks` collection.
- The [CUAD v1 corpus](https://github.com/TheAtticusProject/cuad) for (re-)ingestion. The repo
  ships the *questions and results*, not the contract PDFs (see `.gitignore`).
- Local [Ollama](https://ollama.com) with `qwen2.5:7b-instruct` (generation + judge) and
  `nomic-embed-text` (RAGAS embeddings). A 6GB GPU is enough if phases run sequentially.
- `rag_lib.py` (repo root) is the shared library: Gemini + Mongo clients, embedders, chunkers,
  retrievers, the deterministic scorer, and `assert_corpus_coverage()` (a guard that refuses to
  score if any question's document is missing from the corpus).

## The scaled study (Part 2 — 48 contracts, 189 questions)

Run in order:

| # | Script | Produces |
|---|---|---|
| 1 | `ingest_corpus50.py` | ingests the 50-contract corpus as variant `v10_corpus50_recursive` (pdfplumber → recursive 2000/200 → Gemini embeddings → Mongo). Checkpointed per document; safe to re-run after a quota wall. |
| 2 | `sweep_v10.py` | deterministic sweep of all retrieval configs (vector ± rerank, BM25 ± rerank, hybrid RRF, k-sweep, HyDE). Writes per-question ranks to `../results/retrieval_ranks.json`. |
| 3 | `sweep_v10_mq.py` | multi-query config (local-7B rephrasings → union → rerank), appended to the ranks file. Phased to avoid GPU contention. |
| 4 | `finish_hyde.py` | standalone HyDE rerank pass from cached hypothetical embeds (used to recompute after a GPU eviction). |
| 5 | `analyze_v10.py` | **bootstrap 95% CIs** (naive + near-dup-clustered), paired significance vs champion, per-category `rec@5`, and the 35-miss diagnosis (retrieval-bound vs rerank-bound). |
| 6 | `measure_latency_cost.py` | per-stage latency medians + token counts → the latency/cost table (Finding 5). |
| 7 | `ragas_v10_champion.py` | RAGAS generation scorecard for the champion at scale (5 metrics, local 7B gen + judge). Phased for 6GB VRAM; checkpoints contexts/answers so the slow judge phase can resume. |

## The original study (Part 1 — 11 contracts, 59 questions)

`phase2_sweep.py` (deterministic), `ragas_champion.py` / `ragas_winner.py` /
`ragas_winner_strict.py` / `ragas_retrieval_compare.py` (RAGAS baseline vs variants), and the
technique probes: `metadata_filter_test.py`, `multiquery_test.py`, `contextual_ingest.py` +
`contextual_eval.py`, `rerank_inspect.py`. These back the Part-1 tables.

## Results (committed evidence)

`../results/` holds the small, durable outputs: `*_ledger.json` (aggregate scores),
`retrieval_ranks.json` (per-question ranks), `champion_misses.json` (failure diagnosis),
`latency_cost.json`, and the RAGAS `*_perq.json` per-question judgments.

**Naming convention:** descriptive names are committed evidence; files prefixed `cache_` (or
suffixed `_contexts` / `_answers`) are large regenerable intermediates and are git-ignored —
re-running the scripts reproduces them. Part-1 files keep their original variant-coded names
(e.g. `v01_pdfplumber_fixed2000` = variant 01, pdfplumber parser, fixed 2000-char chunks).
