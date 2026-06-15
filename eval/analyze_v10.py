"""Phase-2 analysis on saved ranks (CPU/Atlas only — no GPU, safe alongside sweep).
1. Bootstrap 95% CIs for every config (MRR, rec@5, rec@1).
2. Paired bootstrap: champion vs each other config — is the gap real (CI excludes 0)?
   Both naive (per-question) and clustered (near-dup groups) resampling.
3. Per-category rec@5 for the champion.
4. Diagnostic dump of every champion miss: was the evidence chunk even in the
   top-20 vector candidate pool? (separates retrieval-bound from rerank-bound)."""
import sys, os, json, random
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import chunks, VECTOR_INDEX, content_tokens

random.seed(20260614)
V = "v10_corpus50_recursive"
qs = [json.loads(l) for l in open("eval/questions_v2.jsonl") if l.strip()]
inscope = [q for q in qs if q.get("doc_id")]
ranks = json.load(open("results/retrieval_ranks.json"))
cache = json.load(open("results/cache_query_embeddings.json"))
N = len(inscope)

def mrr(rs):  return sum(1/r for r in rs if r)/len(rs)
def rec(rs,k): return sum(1 for r in rs if r and r<=k)/len(rs)

def boot_ci(rs, stat, nb=10000):
    idx = range(len(rs))
    vals = []
    for _ in range(nb):
        s = [rs[random.choice(idx)] for _ in idx]
        vals.append(stat(s))
    vals.sort()
    return vals[int(.025*nb)], vals[int(.975*nb)]

print("="*72)
print(f"BOOTSTRAP 95% CIs  (n={N}, 10k resamples)")
print("="*72)
print(f"{'config':<22}{'MRR':>7}{'  95% CI':>16}{'rec@5':>8}{'  95% CI':>16}")
order = ["vector_rerank","vector_rerank_k10","vector_only_k5","hybrid_rerank",
         "vector_rerank_k3","bm25_rerank","bm25_only","hyde_rerank"]
for c in order:
    if c not in ranks: continue
    rs = ranks[c]
    ml,mh = boot_ci(rs, mrr); r5l,r5h = boot_ci(rs, lambda s: rec(s,5))
    print(f"{c:<22}{mrr(rs):>7.3f}  [{ml:.3f},{mh:.3f}]{rec(rs,5):>8.3f}  [{r5l:.3f},{r5h:.3f}]")

print("\n"+"="*72)
print("PAIRED BOOTSTRAP vs champion (vector_rerank): MRR difference")
print("CI excludes 0  => real difference.  CI includes 0 => statistical tie.")
print("="*72)
champ = ranks["vector_rerank"]
# cluster map for clustered bootstrap
clusters = {}
for i,q in enumerate(inscope):
    clusters.setdefault(q.get("near_dup_group", q["id"]), []).append(i)
clist = list(clusters.values())
def paired_diff_ci(a, b, clustered=False, nb=10000):
    diffs = []
    for _ in range(nb):
        if clustered:
            idx = [i for cl in (random.choice(clist) for _ in clist) for i in cl]
        else:
            idx = [random.randrange(len(a)) for _ in a]
        diffs.append(mrr([a[i] for i in idx]) - mrr([b[i] for i in idx]))
    diffs.sort()
    return diffs[int(.025*nb)], diffs[int(.975*nb)]
print(f"{'config':<22}{'dMRR':>8}{'  naive 95% CI':>20}{'  clustered 95% CI':>22}{'  verdict':>10}")
for c in order:
    if c=="vector_rerank" or c not in ranks: continue
    d = mrr(champ)-mrr(ranks[c])
    nl,nh = paired_diff_ci(champ, ranks[c])
    cl_,ch = paired_diff_ci(champ, ranks[c], clustered=True)
    verdict = "REAL" if nl>0 else "tie"
    print(f"{c:<22}{d:>+8.3f}  [{nl:+.3f},{nh:+.3f}]   [{cl_:+.3f},{ch:+.3f}]   {verdict:>8}")

print("\n"+"="*72)
print("PER-CATEGORY rec@5 (champion vector_rerank)")
print("="*72)
cats = {}
for q,r in zip(inscope, champ):
    cats.setdefault(q["category"], []).append(r)
for cat,rs in sorted(cats.items(), key=lambda x:-len(x[1])):
    if len(rs)>=4:
        print(f"  {cat:<22} n={len(rs):>3}  rec@5={rec(rs,5):.3f}  rec@1={rec(rs,1):.3f}")

print("\n"+"="*72)
print("CHAMPION MISSES — candidate-pool diagnosis")
print("(pool_rank = rank of evidence chunk in top-20 VECTOR candidates;")
print(" None => retrieval-bound (reranker cannot help); a number => rerank-bound)")
print("="*72)
def vsearch(qv,n=20):
    return list(chunks.aggregate([
        {"$vectorSearch":{"index":VECTOR_INDEX,"path":"embedding","queryVector":qv,
         "filter":{"config.variant_id":V},"numCandidates":150,"limit":n}},
        {"$project":{"_id":0,"text":1,"source":1}}]))
def arank(ev,hits,thr=0.5):
    e=content_tokens(ev)
    if not e: return None
    for i,h in enumerate(hits):
        if len(e&content_tokens(h["text"]))/len(e)>=thr: return i+1
    return None
retr_bound=rerank_bound=0
dump=[]
for q,r in zip(inscope, champ):
    if r is None:
        pool = vsearch(cache[q["id"]],20)
        pr = arank(q["evidence"], pool)
        if pr is None: retr_bound+=1
        else: rerank_bound+=1
        dump.append({"id":q["id"],"cat":q["category"],"pool_rank":pr,
                     "doc":q["doc_id"][:40],"q":q["question"][:65],"ev":q["evidence"][:75]})
print(f"total champion misses: {len(dump)}/{N}")
print(f"  retrieval-bound (evidence NOT in top-20 pool): {retr_bound}")
print(f"  rerank-bound (in pool but reranker dropped it): {rerank_bound}")
for d in dump:
    print(f"\n[{d['id']}] {d['cat']} pool_rank={d['pool_rank']}")
    print(f"   Q: {d['q']}")
    print(f"   evidence: {d['ev']}")
json.dump(dump, open("results/champion_misses.json","w"), indent=2)
print(f"\n\nsaved {len(dump)} misses to champion_misses_v10.json")
