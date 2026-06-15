"""Phase-2 deterministic sweep: all retrieval configs on the new corpus.
Gates on corpus coverage. Caches query embeds. Outputs the Part-2 table + per-question
ranks (results/phase2_ranks.json) for later bootstrap CIs."""
import sys, os, json, subprocess, time
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import (chunks, VECTOR_INDEX, embed, content_tokens, get_reranker,
                     ollama_chat, text_search, assert_corpus_coverage)

V = "v10_corpus50_recursive"
qs = [json.loads(l) for l in open("eval/questions_v2.jsonl") if l.strip()]
inscope = [q for q in qs if q.get("doc_id")]

# ---- gate ----
assert_corpus_coverage(inscope, V)

# ---- cache query embeds ----
CACHE = "results/cache_query_embeddings.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
missing = [q for q in inscope if q["id"] not in cache]
print(f"caching {len(missing)} query embeds...")
for q in missing:
    for attempt in range(5):
        try:
            cache[q["id"]] = embed([q["question"]], "RETRIEVAL_QUERY")[0].tolist()
            break
        except Exception:
            time.sleep(10 * (2 ** attempt))
    json.dump(cache, open(CACHE, "w"))
print(f"cache: {len(cache)} entries")

def vsearch(qv, n=20):
    return list(chunks.aggregate([
        {"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
                           "filter": {"config.variant_id": V}, "numCandidates": 100, "limit": n}},
        {"$project": {"_id": 0, "text": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}}}]))

def arank(ev, hits, thr=0.5):
    e = content_tokens(ev)
    if not e: return None
    for i, h in enumerate(hits):
        if len(e & content_tokens(h["text"])) / len(e) >= thr: return i + 1
    return None

ce = get_reranker()
def rr(q, hits, k=5):
    if not hits: return None
    s = ce.predict([(q["question"], h["text"]) for h in hits])
    return arank(q["evidence"], [h for _, h in sorted(zip(s, hits), key=lambda z: z[0], reverse=True)][:k])

HYDE_PROMPT = "Write a brief plausible answer to this legal-contract question as it might appear in a contract (1-2 sentences). Output ONLY that text.\n\nQuestion: {q}"

def cfg_vector_rerank(q):  return rr(q, vsearch(cache[q["id"]], 20))
def cfg_vector_only(q):    return arank(q["evidence"], vsearch(cache[q["id"]], 5))
def cfg_bm25(q):           return arank(q["evidence"], text_search(q["question"], V, 20)[:5])
def cfg_bm25_rerank(q):    return rr(q, text_search(q["question"], V, 20))
def cfg_hybrid_rerank(q):
    vec, txt = vsearch(cache[q["id"]], 20), text_search(q["question"], V, 20)
    sc, docs = {}, {}
    for lst in (vec, txt):
        for r, h in enumerate(lst):
            k = (h["source"]["doc_id"], h["source"]["chunk_index"])
            sc[k] = sc.get(k, 0) + 1 / (60 + r + 1); docs[k] = h
    fused = [docs[k] for k in sorted(docs, key=lambda k: sc[k], reverse=True)][:20]
    return rr(q, fused)
def cfg_hyde_rerank(q):
    hyp = ollama_chat("", HYDE_PROMPT.format(q=q["question"]), 0.3, model="qwen2.5:3b")
    for attempt in range(5):
        try:
            qv = embed([hyp], "RETRIEVAL_QUERY")[0].tolist(); break
        except Exception:
            time.sleep(10 * (2 ** attempt))
    return rr(q, vsearch(qv, 20))

CONFIGS = [("vector top5 (no rerank)", cfg_vector_only),
           ("vector + rerank", cfg_vector_rerank),
           ("BM25 top5 (no rerank)", cfg_bm25),
           ("BM25 + rerank", cfg_bm25_rerank),
           ("hybrid + rerank", cfg_hybrid_rerank),
           ("HyDE + rerank", cfg_hyde_rerank)]

all_ranks = {}
for name, fn in CONFIGS:
    ranks = []
    for q in inscope:
        try:
            ranks.append(fn(q))
        except Exception as e:
            print(f"  [{q['id']}] {name} ERR {type(e).__name__}"); ranks.append(None)
    all_ranks[name] = ranks
    r1 = sum(1 for r in ranks if r and r <= 1) / len(ranks)
    r5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)
    mrr = sum(1 / r for r in ranks if r) / len(ranks)
    print(f"{name:<28} rec@1={r1:.3f}  rec@5={r5:.3f}  MRR={mrr:.3f}")
    json.dump({"question_ids": [q["id"] for q in inscope], "ranks": all_ranks},
              open("results/phase2_ranks.json", "w"), indent=2)

print("\nDONE — per-question ranks saved for bootstrap CIs")
