"""TEST 1 — Multi-Query Retrieval (7B rephrasings), phased for 6GB VRAM.
Phase 1: 7B generates 3 rephrasings per question (all 59, checkpointed).
Phase 2: unload 7B -> embed rephrasings (Gemini) -> union search -> rerank -> score."""
import sys, os, json, subprocess
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import (eval_questions, chunks, VECTOR_INDEX, embed, content_tokens,
                     get_reranker, ollama_chat, embed_query_retry)

V = "v04_pdfplumber_recursive"
cache = json.load(open("results/query_embed_cache.json"))
inscope = [q for q in eval_questions if q["doc_id"]]

MQ_PROMPT = """Generate exactly 3 diverse rephrasings of this question about legal contracts. Vary vocabulary and phrasing; keep the meaning. Output ONLY the 3 rephrasings, one per line, no numbering, no preamble.

Question: {q}"""

# ---- Phase 1: rephrasings via 7B (checkpointed) ----
REPHRASE_PATH = "results/multiquery_rephrasings.json"
if os.path.exists(REPHRASE_PATH):
    rephrasings = json.load(open(REPHRASE_PATH))
    print(f"RESUME: {len(rephrasings)} rephrasing sets")
else:
    rephrasings = {}
for q in inscope:
    if q["id"] in rephrasings:
        continue
    raw = ollama_chat("", MQ_PROMPT.format(q=q["question"]), temperature=0.7,
                      model="qwen2.5:7b-instruct")
    rs = [l.strip("-• \t") for l in raw.splitlines() if l.strip()][:3]
    rephrasings[q["id"]] = rs
    json.dump(rephrasings, open(REPHRASE_PATH, "w"), indent=2)
    print(f"  [{q['id']}] {len(rs)} rephrasings")

# ---- free 7B VRAM before reranker ----
subprocess.run(["ollama", "stop", "qwen2.5:7b-instruct"], capture_output=True)
print("Unloaded 7B; loading reranker...")
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

# ---- Phase 2: union search + rerank + score ----
ranks = []
for q in inscope:
    queries = [None] + rephrasings[q["id"]]   # None = original (cached vector)
    seen, pooled = set(), []
    for i, qq in enumerate(queries):
        qv = cache[q["id"]] if i == 0 else embed_query_retry(qq)
        for h in vsearch_qv(qv, 20):
            key = (h["source"]["doc_id"], h["source"]["chunk_index"])
            if key not in seen:
                seen.add(key); pooled.append(h)
    sc = ce.predict([(q["question"], h["text"]) for h in pooled])
    reranked = [h for _, h in sorted(zip(sc, pooled), key=lambda z: z[0], reverse=True)]
    r = arank(q["evidence"], reranked[:5])
    ranks.append(r)
    print(f"  [{q['id']}] pool={len(pooled)} rank={r}")

r1 = sum(1 for r in ranks if r and r <= 1) / len(ranks)
r5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)
mrr = sum(1 / r for r in ranks if r) / len(ranks)
print(f"\n=== MULTI-QUERY 7B (n={len(ranks)}):  rec@1={r1:.3f}  rec@5={r5:.3f}  MRR={mrr:.3f} ===")
print("ref  v04 vector+rerank:  rec@1=0.831  rec@5=0.949  MRR=0.890")
print("ref  v04 HyDE+rerank:    rec@1=0.864  rec@5=0.949  MRR=0.902")
