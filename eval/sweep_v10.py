"""Phase-2 deterministic sweep on v10_corpus50_recursive, 189 questions.
Quota-ordered: (0) cache query embeds, (1) vector/BM25/hybrid/k-sweep (no new
embeds), (2) HyDE (+189 embeds), (3) Multi-Query (+~570 embeds, runs last).
Per-question ranks saved to results/retrieval_ranks.json for bootstrap CIs."""
import sys, os, json, subprocess, time
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import (chunks, VECTOR_INDEX, embed, embed_query_retry, content_tokens,
                     get_reranker, ollama_chat, text_search, assert_corpus_coverage)

V = "v10_corpus50_recursive"
qs = [json.loads(l) for l in open('eval/questions_v2.jsonl') if l.strip()]
inscope = [q for q in qs if q.get("doc_id")]
assert_corpus_coverage(inscope, V)           # the gate

# ---- 0. query embed cache ----
CACHE = "results/cache_query_embeddings.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
old = json.load(open("results/query_embed_cache.json"))
for q in inscope:                             # reuse old cache where ids match
    if q["id"] in old and q["id"] not in cache:
        cache[q["id"]] = old[q["id"]]
missing = [q for q in inscope if q["id"] not in cache]
print(f"caching {len(missing)} new query embeds")
for q in missing:
    cache[q["id"]] = embed_query_retry(q["question"])
    json.dump(cache, open(CACHE, "w"))
print("cache complete")

def vsearch(qv, n=20):
    return list(chunks.aggregate([
        {"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
                           "filter": {"config.variant_id": V}, "numCandidates": 150, "limit": n}},
        {"$project": {"_id": 0, "text": 1, "source": 1}}]))

def arank(ev, hits, thr=0.5):
    e = content_tokens(ev)
    if not e: return None
    for i, h in enumerate(hits):
        if len(e & content_tokens(h["text"])) / len(e) >= thr: return i + 1
    return None

ce = get_reranker()
def rerank_top(q, hits, k=5):
    if not hits: return []
    sc = ce.predict([(q, h["text"]) for h in hits])
    return [h for _, h in sorted(zip(sc, hits), key=lambda z: z[0], reverse=True)][:k]

RANKS_PATH = "results/retrieval_ranks.json"
ranks = json.load(open(RANKS_PATH)) if os.path.exists(RANKS_PATH) else {}

def run_config(name, fn):
    if name in ranks and len(ranks[name]) == len(inscope):
        print(f"[skip] {name} (cached)"); return
    rs = []
    for q in inscope:
        try: rs.append(fn(q))
        except Exception as ex:
            print(f"  {q['id']} ERR {type(ex).__name__}"); rs.append(None)
    ranks[name] = rs
    json.dump(ranks, open(RANKS_PATH, "w"))
    n = len(rs)
    r1 = sum(1 for r in rs if r and r <= 1)/n; r5 = sum(1 for r in rs if r and r <= 5)/n
    mrr = sum(1/r for r in rs if r)/n
    print(f"=== {name:<24} rec@1={r1:.3f} rec@5={r5:.3f} MRR={mrr:.3f} (n={n})")

# ---- 1. core configs (cached embeds only) ----
run_config("vector_only_k5",   lambda q: arank(q["evidence"], vsearch(cache[q["id"]], 5)))
run_config("vector_rerank",    lambda q: arank(q["evidence"], rerank_top(q["question"], vsearch(cache[q["id"]], 20))))
run_config("bm25_only",        lambda q: arank(q["evidence"], text_search(q["question"], V, 20)[:5]))
run_config("bm25_rerank",      lambda q: arank(q["evidence"], rerank_top(q["question"], text_search(q["question"], V, 20))))
def hybrid(q, n=20):
    vec, txt = vsearch(cache[q["id"]], n), text_search(q["question"], V, n)
    sc, docs = {}, {}
    for lst in (vec, txt):
        for r, h in enumerate(lst):
            k = (h["source"]["doc_id"], h["source"]["chunk_index"])
            sc[k] = sc.get(k, 0) + 1/(60+r+1); docs[k] = h
    return [docs[k] for k in sorted(docs, key=lambda k: sc[k], reverse=True)][:n]
run_config("hybrid_rerank",    lambda q: arank(q["evidence"], rerank_top(q["question"], hybrid(q))))
# k-sweep on the champion
for kk in (3, 10):
    run_config(f"vector_rerank_k{kk}", lambda q, kk=kk: arank(q["evidence"], rerank_top(q["question"], vsearch(cache[q["id"]], 20), k=kk)))

# ---- 2. HyDE (7B local hypotheticals; +189 embeds) ----
HYDE_P = "Write a brief plausible answer to this question as it might appear in a legal contract (1-2 sentences). Guessing is fine. Question: {q}\n\nHypothetical excerpt:"
HYDE_CACHE = "results/cache_hyde_hypotheses.json"
hyp = json.load(open(HYDE_CACHE)) if os.path.exists(HYDE_CACHE) else {}
for q in inscope:
    if q["id"] not in hyp:
        hyp[q["id"]] = ollama_chat("", HYDE_P.format(q=q["question"]), 0.3, model="qwen2.5:7b-instruct")
        json.dump(hyp, open(HYDE_CACHE, "w"))
subprocess.run(["ollama", "stop", "qwen2.5:7b-instruct"], capture_output=True)
HYDE_EMB = "results/cache_hyde_embeddings.json"
hemb = json.load(open(HYDE_EMB)) if os.path.exists(HYDE_EMB) else {}
for q in inscope:
    if q["id"] not in hemb:
        hemb[q["id"]] = embed_query_retry(hyp[q["id"]])
        json.dump(hemb, open(HYDE_EMB, "w"))
run_config("hyde_rerank",      lambda q: arank(q["evidence"], rerank_top(q["question"], vsearch(hemb[q["id"]], 20))))

print("\nSWEEP CORE+HYDE COMPLETE (multi-query deferred — run sweep_v10_mq.py if quota allows)")
