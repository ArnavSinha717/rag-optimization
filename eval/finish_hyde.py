"""Standalone HyDE rerank pass — uses cached hyde embeds (results/cache_hyde_embeddings.json).
No quota needed. Writes hyde_rerank into sweep_v10_ranks.json + bootstrap CI."""
import sys, os, json, random
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import chunks, VECTOR_INDEX, content_tokens, get_reranker

V = "v10_corpus50_recursive"
qs = [json.loads(l) for l in open("eval/questions_v2.jsonl") if l.strip()]
inscope = [q for q in qs if q.get("doc_id")]
hemb = json.load(open("results/cache_hyde_embeddings.json"))
ce = get_reranker()

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
def rerank_top(q, hits, k=5):
    if not hits: return []
    sc = ce.predict([(q, h["text"]) for h in hits])
    return [h for _, h in sorted(zip(sc, hits), key=lambda z: z[0], reverse=True)][:k]

rs = []
for i, q in enumerate(inscope):
    # HyDE: retrieve with the hypothetical's embedding, rerank against the REAL question
    rs.append(arank(q["evidence"], rerank_top(q["question"], vsearch(hemb[q["id"]], 20))))
    if (i+1) % 40 == 0: print(f"  {i+1}/{len(inscope)}")

ranks = json.load(open("results/retrieval_ranks.json"))
ranks["hyde_rerank"] = rs
json.dump(ranks, open("results/retrieval_ranks.json", "w"))

n = len(rs)
def mrr(x): return sum(1/r for r in x if r)/len(x)
def rec(x,k): return sum(1 for r in x if r and r<=k)/len(x)
random.seed(20260614)
def ci(x, stat, nb=10000):
    v=sorted(stat([x[random.randrange(len(x))] for _ in x]) for _ in range(nb)); return v[int(.025*nb)],v[int(.975*nb)]
# paired vs champion
champ = ranks["vector_rerank"]
def pdiff(nb=10000):
    d=sorted(mrr([champ[i] for i in idx])-mrr([rs[i] for i in idx]) for idx in ([random.randrange(n) for _ in range(n)] for _ in range(nb)))
    return d[int(.025*nb)],d[int(.975*nb)]
r5l,r5h = ci(rs, lambda s: rec(s,5)); ml,mh = ci(rs, mrr); dl,dh = pdiff()
print("\n=== HyDE + rerank (n=%d) ===" % n)
print(f"rec@1={rec(rs,1):.3f}  rec@5={rec(rs,5):.3f} [{r5l:.3f},{r5h:.3f}]  MRR={mrr(rs):.3f} [{ml:.3f},{mh:.3f}]")
print(f"champion ΔMRR={mrr(champ)-mrr(rs):+.3f}  CI[{dl:+.3f},{dh:+.3f}]  => {'champion REAL better' if dl>0 else ('HyDE REAL better' if dh<0 else 'TIE')}")
