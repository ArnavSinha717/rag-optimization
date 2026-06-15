"""RAGAS scorecard for the HyDE contender (HyDE retrieval + rerank) on v10, 189 questions.
Same generation + judge as the champion run — isolates whether HyDE's hallucination-steering
(Finding 2b) propagates into worse faithfulness/correctness. Reuses cached HyDE query embeds.
Phased for 6GB VRAM; all phases checkpoint."""
import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
import sys, json, time
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag_lib
from rag_lib import (chunks, VECTOR_INDEX, get_reranker, ollama_chat,
                     SYSTEM_INSTRUCTION, GEN_OLLAMA_MODEL, REFUSAL)

V = "v10_corpus50_recursive"; OUT = "ragas_v10_hyde"
qs = [json.loads(l) for l in open("eval/questions_v2.jsonl") if l.strip()]
inscope = [q for q in qs if q.get("doc_id")]
hyde_emb = json.load(open("results/cache_hyde_embeddings.json"))   # HyDE hypothetical embeddings
ctx_path = "results/cache_ragas_hyde_contexts.json"; ans_path = "results/cache_ragas_hyde_answers.json"
perq_path = "results/ragas_hyde_perq.json"
print(f"=== RAGAS v10 HyDE — {len(inscope)} Q, retrieval=HyDE+rerank, gen+judge=7B ===")

def vsearch(qv, n=20):
    return list(chunks.aggregate([
        {"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
                           "filter": {"config.variant_id": V}, "numCandidates": 150, "limit": n}},
        {"$project": {"_id": 0, "text": 1, "source": 1}}]))

# ---- P1 retrieve via HyDE embeddings, rerank vs the REAL question ----
if os.path.exists(ctx_path):
    ctxs = json.load(open(ctx_path)); print(f"RESUME contexts ({len(ctxs)})")
else:
    ce = get_reranker(); ctxs = []
    for q in inscope:
        pool = vsearch(hyde_emb[q["id"]], 20)
        sc = ce.predict([(q["question"], h["text"]) for h in pool]) if pool else []
        top = [h["text"] for _, h in sorted(zip(sc, pool), key=lambda z: z[0], reverse=True)][:5]
        ctxs.append({"id": q["id"], "question": q["question"], "contexts": top,
                     "reference": q["expected_answer"]})
        json.dump(ctxs, open(ctx_path, "w"))
    print(f"P1 done ({len(ctxs)} retrieved)")

# ---- P2 free reranker VRAM ----
import torch, gc
try: get_reranker(); rag_lib._reranker = None
except Exception: pass
gc.collect(); torch.cuda.empty_cache()
print("P2 freed reranker VRAM")

# ---- P3 generate (7B concise-strict) ----
if os.path.exists(ans_path):
    recs = json.load(open(ans_path)); print(f"RESUME answers ({len(recs)})")
else:
    recs = []
    for r in ctxs:
        if r["contexts"]:
            block = "\n\n".join(f"[Source {i+1}] {c}" for i, c in enumerate(r["contexts"]))
            try:
                ans = ollama_chat(SYSTEM_INSTRUCTION,
                                  f"Sources:\n{block}\n\nQuestion: {r['question']}\n\nAnswer:",
                                  0.1, model=GEN_OLLAMA_MODEL)
            except Exception as e:
                print(f"  [{r['id']}] GEN ERR {type(e).__name__}"); ans = REFUSAL
        else:
            ans = REFUSAL
        recs.append({**r, "answer": ans})
        json.dump(recs, open(ans_path, "w"))
        if len(recs) % 30 == 0: print(f"  gen {len(recs)}/{len(ctxs)}")
    print("P3 done")

# ---- P4 RAGAS judge (5 metrics) ----
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.metrics import (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference,
                           LLMContextRecall, AnswerCorrectness)
llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:7b-instruct", temperature=0))
emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
metrics = [LLMContextRecall(llm=llm), LLMContextPrecisionWithReference(llm=llm),
           Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb),
           AnswerCorrectness(llm=llm, embeddings=emb)]
ds = EvaluationDataset(samples=[
    SingleTurnSample(user_input=r["question"], response=r["answer"],
                     retrieved_contexts=r["contexts"], reference=r["reference"]) for r in recs])
print(f"\nP4 scoring {len(recs)} x 5 metrics (7B judge)...")
t = time.time()
res = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb,
               run_config=RunConfig(max_workers=1, timeout=300, max_retries=1))
print(f"done in {(time.time()-t)/60:.1f} min")
df = res.to_pandas()
meta = {"user_input","response","retrieved_contexts","reference","reference_contexts","rubrics"}
scores = {c: float(df[c].mean()) for c in df.columns if c not in meta}
if "llm_context_precision_with_reference" in scores:
    scores["context_precision"] = scores.pop("llm_context_precision_with_reference")
df.to_json(perq_path, orient="records", indent=2)
ans_mask = [not r["answer"].startswith("Not found") for r in recs]
import statistics as st
faith_col = next((c for c in df.columns if "faith" in c), None)
if faith_col:
    answered = [df[faith_col][i] for i in range(len(recs)) if ans_mask[i] and df[faith_col][i]==df[faith_col][i]]
    scores["faithfulness_answered_only"] = float(st.mean(answered)) if answered else None
    scores["refusal_rate"] = 1 - sum(ans_mask)/len(ans_mask)
led = json.load(open("results/ragas_ledger.json")) if os.path.exists("results/ragas_ledger.json") else {}
led[OUT] = {"variant_id": OUT, "n_questions": len(recs), "retrieval": "hyde+rerank",
            "generator": "qwen2.5:7b (concise)", "judge": "qwen2.5:7b", **scores}
json.dump(led, open("results/ragas_ledger.json", "w"), indent=2)
print("\n" + "="*50 + f"\nRAGAS v10 HyDE (n={len(recs)})")
for k in ["context_recall","context_precision","faithfulness","faithfulness_answered_only",
          "answer_relevancy","answer_correctness","refusal_rate"]:
    if k in scores and scores[k] is not None: print(f"  {k:<28} {scores[k]:.3f}")
# inline champion comparison
ch = led.get("ragas_v10_champion", {})
if ch:
    print(f"\n  {'metric':<28}{'champion':>10}{'hyde':>8}")
    for k in ["context_recall","context_precision","faithfulness_answered_only","answer_relevancy","answer_correctness","refusal_rate"]:
        print(f"  {k:<28}{ch.get(k,float('nan')):>10.3f}{scores.get(k,float('nan')):>8.3f}")
