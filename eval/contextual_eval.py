"""Deterministic eval of the contextual variants (vector+rerank, 59 Qs)."""
import sys, os, json
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import eval_questions, chunks, VECTOR_INDEX, content_tokens, get_reranker

V = sys.argv[1]   # v08_ctx_recursive | v09_ctx_markdown
cache = json.load(open("results/query_embed_cache.json"))
inscope = [q for q in eval_questions if q["doc_id"]]
ce = get_reranker()

def vsearch_qv(qv, n=20):
    return list(chunks.aggregate([
        {"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
                           "filter": {"config.variant_id": V}, "numCandidates": 100, "limit": n}},
        {"$project": {"_id": 0, "text": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}}}]))

def arank(ev_s, hits, thr=0.5):
    ev = content_tokens(ev_s)
    if not ev: return None
    for i, h in enumerate(hits):
        if len(ev & content_tokens(h["text"])) / len(ev) >= thr: return i + 1
    return None

ranks = []
for q in inscope:
    hits = vsearch_qv(cache[q["id"]], 20)
    sc = ce.predict([(q["question"], h["text"]) for h in hits])
    reranked = [h for _, h in sorted(zip(sc, hits), key=lambda z: z[0], reverse=True)]
    ranks.append(arank(q["evidence"], reranked[:5]))

r1 = sum(1 for r in ranks if r and r <= 1) / len(ranks)
r5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)
mrr = sum(1 / r for r in ranks if r) / len(ranks)
print(f"=== {V} vector+rerank (n={len(ranks)}):  rec@1={r1:.3f}  rec@5={r5:.3f}  MRR={mrr:.3f} ===")
