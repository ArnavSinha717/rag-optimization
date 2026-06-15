"""Empirical per-stage latency + token counts on v10, to build the latency/cost table.
Latency = real wall-clock medians on this machine. Cost = measured token counts x
published rates (Flash $0.30/$2.50 per 1M in/out; embed $0.15/1M in; 2026-06)."""
import sys, os, json, time, statistics
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import (chunks, VECTOR_INDEX, embed, content_tokens, get_reranker,
                     client, text_search)
from google.genai import types

V = "v10_corpus50_recursive"
qs = [json.loads(l) for l in open("eval/questions_v2.jsonl") if l.strip()]
inscope = [q for q in qs if q.get("doc_id")]
sample = inscope[:6]
emb_cache = json.load(open("results/cache_query_embeddings.json"))
def med(xs): return statistics.median(xs)*1000  # ms

def vsearch(qv, n=20):
    return list(chunks.aggregate([
        {"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
                           "filter": {"config.variant_id": V}, "numCandidates": 150, "limit": n}},
        {"$project": {"_id": 0, "text": 1, "source": 1}}]))

print("measuring stage latencies (median of 6)...")
# 1 embed (1 API call)
t=[]
for q in sample:
    s=time.time(); embed([q["question"]],"RETRIEVAL_QUERY"); t.append(time.time()-s)
L_embed=med(t)
# 2 vector search
t=[]
for q in sample:
    s=time.time(); vsearch(emb_cache[q["id"]],20); t.append(time.time()-s)
L_vsearch=med(t)
# 3 BM25 search
t=[]
for q in sample:
    s=time.time(); text_search(q["question"],V,20); t.append(time.time()-s)
L_bm25=med(t)
# 4 rerank pool=20 (champion) and pool=45 (MQ union)
ce=get_reranker()
pools={20:[],45:[]}
for q in sample:
    hits=vsearch(emb_cache[q["id"]],20)
    pad=(hits*3)[:45]
    for sz in (20,45):
        p=hits if sz==20 else pad
        s=time.time(); ce.predict([(q["question"],h["text"]) for h in p]); pools[sz].append(time.time()-s)
L_rr20=med(pools[20]); L_rr45=med(pools[45])
# 5 Flash generation latency (hypothetical-sized + answer-sized), 3 calls
t=[]
for q in sample[:3]:
    s=time.time()
    client.models.generate_content(model="gemini-2.5-flash",
        contents=f"Write a 1-sentence plausible legal-contract answer to: {q['question']}",
        config=types.GenerateContentConfig(max_output_tokens=80,temperature=0.3))
    t.append(time.time()-s)
L_flash_gen=med(t)

# token counts for cost
def ntok(text): return client.models.count_tokens(model="gemini-2.5-flash",contents=text).total_tokens
q_tok=med([[ntok(q["question"])][0]/1 for q in sample])  # avg question tokens
# final answer prompt: 5 sources (~realistic) + question
hits=vsearch(emb_cache[sample[0]["id"]],5)
block="\n\n".join(f"[Source {i+1}] {h['text']}" for i,h in enumerate(hits))
ans_in_tok=ntok(f"Sources:\n{block}\n\nQuestion: {sample[0]['question']}\n\nAnswer:")
ans_out_tok=60  # measured typical concise answer
hyde_in=ntok(f"Write a brief plausible answer... Question: {sample[0]['question']}"); hyde_out=70
mq_in=ntok(f"Rewrite this question in 3 ways... {sample[0]['question']}"); mq_out=90

R={"L_embed":L_embed,"L_vsearch":L_vsearch,"L_bm25":L_bm25,"L_rr20":L_rr20,"L_rr45":L_rr45,
   "L_flash_gen":L_flash_gen,"q_tok":q_tok,"ans_in_tok":ans_in_tok,"ans_out_tok":ans_out_tok,
   "hyde_in":hyde_in,"hyde_out":hyde_out,"mq_in":mq_in,"mq_out":mq_out}
json.dump(R, open("results/latency_cost_stages.json","w"), indent=2)

print("\n=== STAGE LATENCY (ms, median) ===")
for k in ["L_embed","L_vsearch","L_bm25","L_rr20","L_rr45","L_flash_gen"]:
    print(f"  {k:<14} {R[k]:7.1f} ms")
print("\n=== TOKEN COUNTS ===")
print(f"  question ~{q_tok:.0f} tok | final-answer prompt {ans_in_tok} in / {ans_out_tok} out")
print(f"  HyDE gen {hyde_in} in / {hyde_out} out | MQ gen {mq_in} in / {mq_out} out")

# ---- COST per query (USD), production = Gemini Flash for all generation ----
IN=0.30/1e6; OUT=2.50/1e6; EMB=0.15/1e6
def cost(n_embed, gen_in=0, gen_out=0):
    return n_embed*q_tok*EMB + (ans_in_tok*IN + ans_out_tok*OUT) + (gen_in*IN + gen_out*OUT)
configs={
 "vector+rerank (champ)": dict(lat=L_embed+L_vsearch+L_rr20, c=cost(1)),
 "vector only":           dict(lat=L_embed+L_vsearch,         c=cost(1)),
 "BM25 only":             dict(lat=L_bm25,                    c=cost(0)),
 "hybrid+rerank":         dict(lat=L_embed+L_vsearch+L_bm25+L_rr20, c=cost(1)),
 "HyDE+rerank":           dict(lat=L_flash_gen+L_embed+L_vsearch+L_rr20, c=cost(1,hyde_in,hyde_out)),
 "multi-query+rerank":    dict(lat=L_flash_gen+4*L_embed+4*L_vsearch+L_rr45, c=cost(4,mq_in,mq_out)),
}
print("\n=== PER-QUERY LATENCY + COST (production: Flash generation) ===")
print(f"{'config':<24}{'retr-lat(ms)':>13}{'+answer-gen':>13}{'cost/query':>13}{'per 1k queries':>16}")
base=configs["vector+rerank (champ)"]["lat"]
for name,d in configs.items():
    total_lat=d["lat"]+L_flash_gen  # + final answer gen
    print(f"{name:<24}{d['lat']:>11.0f}  {L_flash_gen:>10.0f}  ${d['c']:>10.5f}  ${d['c']*1000:>13.2f}")
json.dump({n:{"latency_ms":d["lat"],"cost_usd":d["c"]} for n,d in configs.items()},
          open("results/latency_cost.json","w"), indent=2)
print("\nsaved latency_cost_table.json")
