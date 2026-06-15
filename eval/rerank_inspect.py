"""Deterministic reranker analysis — no LLM judge, no noise.

For each question: retrieve top-20 by vector, find the RANK of the answer-bearing
chunk (via the evidence quote) in vector order vs cross-encoder rerank order.
Caches Gemini query embeddings to disk so we never re-spend quota.
"""
import os, json, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import (embed, chunks, VECTOR_INDEX, eval_questions, get_reranker,
                     _content_tokens)

CACHE = "results/query_embed_cache.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
_calls = [0]

def cached_query_vec(qid: str, question: str) -> list[float]:
    if qid in cache:
        return cache[qid]
    v = embed([question], "RETRIEVAL_QUERY")[0].tolist()
    cache[qid] = v
    _calls[0] += 1
    json.dump(cache, open(CACHE, "w"))
    return v

def answer_rank(evidence: str, ordered_chunks: list[dict], thr: float = 0.6):
    """Rank (1-based) of the first chunk containing the answer; None if absent."""
    ev = _content_tokens(evidence)
    if not ev:
        return None
    for i, h in enumerate(ordered_chunks):
        toks = _content_tokens(h["text"])
        if len(ev & toks) / len(ev) >= thr:
            return i + 1
    return None

def vsearch(qv, variant, n=20):
    pipe = [
        {"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
                           "filter": {"config.variant_id": variant},
                           "numCandidates": 100, "limit": n}},
        {"$project": {"_id": 0, "text": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(chunks.aggregate(pipe))

VARIANT = "v01_pdfplumber_fixed2000"
dev = [q for q in eval_questions if q.get("dev") and q["doc_id"]]
ce = get_reranker()

print(f"Deterministic reranker analysis on {len(dev)} dev questions (variant {VARIANT})\n")
print(f"{'id':<5}{'vec_rank':>9}{'rerank_rank':>12}   verdict")
print("-" * 50)

rows = []
for q in dev:
    qv = cached_query_vec(q["id"], q["question"])
    hits = vsearch(qv, VARIANT, n=20)
    vec_rank = answer_rank(q["evidence"], hits)               # rank in vector order
    scores = ce.predict([(q["question"], h["text"]) for h in hits])
    reranked = [h for _, h in sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)]
    rr_rank = answer_rank(q["evidence"], reranked)            # rank after rerank
    rows.append((q["id"], vec_rank, rr_rank))
    def fmt(r): return "miss" if r is None else str(r)
    if vec_rank and rr_rank:
        verdict = "up" if rr_rank < vec_rank else ("down" if rr_rank > vec_rank else "same")
    elif rr_rank and not vec_rank:
        verdict = "RESCUED (was >20)"
    elif vec_rank and not rr_rank:
        verdict = "LOST"
    else:
        verdict = "miss both"
    print(f"{q['id']:<5}{fmt(vec_rank):>9}{fmt(rr_rank):>12}   {verdict}")

# ---- aggregate, deterministic ----
def recall_at(rows, k, idx):
    got = [r for r in rows if r[idx] is not None and r[idx] <= k]
    return len(got) / len(rows)
def mrr(rows, idx):
    return sum((1.0 / r[idx]) for r in rows if r[idx] is not None) / len(rows)

print("\n" + "=" * 50)
print(f"{'metric':<16}{'vector':>10}{'rerank':>10}")
print("-" * 36)
for k in (1, 3, 5):
    print(f"recall@{k:<10}{recall_at(rows,k,1):>10.3f}{recall_at(rows,k,2):>10.3f}")
print(f"{'MRR':<16}{mrr(rows,1):>10.3f}{mrr(rows,2):>10.3f}")
print(f"\nGemini embed calls used this run: {_calls[0]} (rest served from cache)")
