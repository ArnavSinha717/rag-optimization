"""Multi-query retrieval on v10, 189 questions. Phased to avoid GPU contention:
  P1 generate 3 rephrasings/question via local 7B (free)  -> cache, then `ollama stop`
  P2 batch-embed original + rephrasings via Gemini (25/call, checkpointed)
  P3 vector-search each query, UNION by (doc,chunk), rerank union vs REAL question, top-5
  P4 metrics + bootstrap CI + paired-vs-champion, write hyde-style row into ranks file
All phases checkpoint to disk; safe to kill/resume (also survives a quota wall in P2)."""
import sys, os, json, subprocess, random, time
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import (chunks, VECTOR_INDEX, embed, embed_query_retry, content_tokens,
                     get_reranker, ollama_chat)

V = "v10_corpus50_recursive"
qs = [json.loads(l) for l in open("eval/questions_v2.jsonl") if l.strip()]
inscope = [q for q in qs if q.get("doc_id")]
N = len(inscope)
R = "results"

# ---- P1: rephrasings (local 7B) ----
MQ_PROMPT = ("Rewrite this question about a legal contract in 3 different ways that keep the same "
             "meaning but vary wording (synonyms, clause-style phrasing). One per line, no numbering.\n\n"
             "Question: {q}\n\nThree rewrites:")
RP = f"{R}/mq_rephrasings_v10.json"
reph = json.load(open(RP)) if os.path.exists(RP) else {}
todo = [q for q in inscope if q["id"] not in reph]
print(f"P1 rephrasings: {len(reph)} cached, {len(todo)} to generate")
for q in todo:
    out = ollama_chat("", MQ_PROMPT.format(q=q["question"]), 0.7, model="qwen2.5:7b-instruct")
    lines = [l.strip(" -•\t").strip() for l in out.splitlines() if l.strip()]
    reph[q["id"]] = lines[:3] if lines else [q["question"]]
    json.dump(reph, open(RP, "w"))
subprocess.run(["ollama", "stop", "qwen2.5:7b-instruct"], capture_output=True)
print("P1 done; 7B stopped")

# ---- P2: batch-embed original + rephrasings ----
EP = f"{R}/mq_embeds_v10.json"
emb = json.load(open(EP)) if os.path.exists(EP) else {}
# reuse cached original-question embeds
qcache = json.load(open("results/cache_query_embeddings.json"))
texts = {}  # text -> None (dedup)
for q in inscope:
    texts[q["question"]] = None
    for r in reph[q["id"]]:
        texts[r] = None
need = [t for t in texts if t not in emb]
print(f"P2 embeds: {len(emb)} cached, {len(need)} new (batched 25/call)")
for i in range(0, len(need), 25):
    batch = need[i:i+25]
    for attempt in range(6):
        try:
            vs = embed(batch, "RETRIEVAL_QUERY")
            for t, v in zip(batch, vs): emb[t] = v.tolist()
            json.dump(emb, open(EP, "w")); break
        except Exception as e:
            w = 5*(2**attempt); print(f"  embed wall ({type(e).__name__}); sleep {w}s"); time.sleep(w)
    else:
        print("P2 HIT QUOTA WALL — rerun after reset to resume"); sys.exit(2)
    print(f"  embedded {min(i+25,len(need))}/{len(need)}")
print("P2 done")

# ---- P3: union retrieve + rerank ----
ce = get_reranker()
def vsearch(qv, n=10):
    return list(chunks.aggregate([
        {"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
                           "filter": {"config.variant_id": V}, "numCandidates": 100, "limit": n}},
        {"$project": {"_id": 0, "text": 1, "source": 1}}]))
def arank(ev, hits, thr=0.5):
    e = content_tokens(ev)
    if not e: return None
    for i, h in enumerate(hits):
        if len(e & content_tokens(h["text"])) / len(e) >= thr: return i+1
    return None
ranks = []
for j, q in enumerate(inscope):
    queries = [q["question"]] + reph[q["id"]]
    union = {}
    for t in queries:
        for h in vsearch(emb[t], 10):
            union[(h["source"]["doc_id"], h["source"]["chunk_index"])] = h
    pool = list(union.values())
    sc = ce.predict([(q["question"], h["text"]) for h in pool]) if pool else []
    top = [h for _, h in sorted(zip(sc, pool), key=lambda z: z[0], reverse=True)][:5]
    ranks.append(arank(q["evidence"], top))
    if (j+1) % 40 == 0: print(f"  P3 {j+1}/{N}")

# ---- P4: metrics + CI ----
allr = json.load(open(f"{R}/sweep_v10_ranks.json"))
allr["multiquery_rerank"] = ranks
json.dump(allr, open(f"{R}/sweep_v10_ranks.json", "w"))
def mrr(x): return sum(1/r for r in x if r)/len(x)
def rec(x,k): return sum(1 for r in x if r and r<=k)/len(x)
random.seed(20260614)
def ci(x, stat, nb=10000):
    v=sorted(stat([x[random.randrange(len(x))] for _ in x]) for _ in range(nb)); return v[int(.025*nb)],v[int(.975*nb)]
champ=allr["vector_rerank"]
def pdiff(a,b,nb=10000):
    d=sorted(mrr([a[i] for i in idx])-mrr([b[i] for i in idx]) for idx in ([random.randrange(N) for _ in range(N)] for _ in range(nb)))
    return d[int(.025*nb)],d[int(.975*nb)]
r5l,r5h=ci(ranks,lambda s:rec(s,5)); ml,mh=ci(ranks,mrr); dl,dh=pdiff(champ,ranks)
# confusion vs champion
resc=sum(1 for c,h in zip(champ,ranks) if (h and h<=5) and not (c and c<=5))
brok=sum(1 for c,h in zip(champ,ranks) if (c and c<=5) and not (h and h<=5))
print("\n=== Multi-Query + rerank (n=%d) ===" % N)
print(f"rec@1={rec(ranks,1):.3f}  rec@5={rec(ranks,5):.3f} [{r5l:.3f},{r5h:.3f}]  MRR={mrr(ranks):.3f} [{ml:.3f},{mh:.3f}]")
print(f"champion ΔMRR={mrr(champ)-mrr(ranks):+.3f}  CI[{dl:+.3f},{dh:+.3f}]  => {'champion REAL better' if dl>0 else ('MQ REAL better' if dh<0 else 'TIE')}")
print(f"vs champion top-5: rescued={resc} broke={brok}")
