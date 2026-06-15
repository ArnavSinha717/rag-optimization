"""Full RAGAS validation of the CHAMPION config:
  v04 recursive chunks  ->  HyDE + cross-encoder rerank  ->  7B strict-prompt answer
Scored on all 32 in-scope questions, 4 metrics, qwen2.5:7b-instruct judge.

Phased to fit 6GB VRAM: (1) all retrieval (HyDE-3B + reranker) -> save contexts,
(2) free reranker, (3) all generation (7B), (4) RAGAS judge (7B).
Resumable: contexts and answers checkpoint to disk.
"""
import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
import sys, json, time
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag_lib
from rag_lib import (eval_questions, search_hyde, ollama_chat, SYSTEM_INSTRUCTION,
                     GEN_OLLAMA_MODEL, REFUSAL, get_reranker)

VARIANT = "v04_pdfplumber_recursive"
OUT_ID = "champion_hyde_rerank_FULL"
ctx_path = f"results/{OUT_ID}_contexts.json"
ans_path = f"results/{OUT_ID}_answers.json"
inscope = [q for q in eval_questions if q["doc_id"]]

# ---- Phase 1: HyDE + rerank retrieval for all questions (reranker on GPU) ----
if os.path.exists(ctx_path):
    ctxs = json.load(open(ctx_path))
    print(f"RESUME: loaded contexts for {len(ctxs)} questions")
else:
    print(f"Phase 1: HyDE+rerank retrieval, {len(inscope)} questions")
    ctxs = []
    for q in inscope:
        try:
            hits = search_hyde(q["question"], VARIANT, retrieve_n=20, final_k=5)
            c = [h["text"] for h in hits]
        except Exception as e:
            print(f"  [{q['id']}] RETR ERR {type(e).__name__}"); c = []
        ctxs.append({"id": q["id"], "question": q["question"],
                     "contexts": c, "reference": q["expected_answer"]})
        json.dump(ctxs, open(ctx_path, "w"), indent=2)
        print(f"  [{q['id']}] {len(c)} ctx")

# ---- Phase 2: free reranker VRAM before loading 7B ----
import torch, gc
try:
    get_reranker(); rag_lib._reranker = None
except Exception:
    pass
gc.collect(); torch.cuda.empty_cache()
print("Freed reranker VRAM.")

# ---- Phase 3: generate answers with 7B strict (7B on GPU) ----
if os.path.exists(ans_path):
    records = json.load(open(ans_path))
    print(f"RESUME: loaded {len(records)} answers")
else:
    print(f"Phase 3: generating answers with {GEN_OLLAMA_MODEL} (strict prompt)")
    records = []
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
        records.append({**r, "answer": ans})
        json.dump(records, open(ans_path, "w"), indent=2)
        print(f"  [{r['id']}] {ans[:65]}")

# ---- Phase 4: RAGAS judge (7B), 4 metrics ----
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.metrics import (Faithfulness, ResponseRelevancy,
                           LLMContextPrecisionWithReference, LLMContextRecall)

llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:7b-instruct", temperature=0))
emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
metrics = [LLMContextRecall(llm=llm), LLMContextPrecisionWithReference(llm=llm),
           Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb)]
ds = EvaluationDataset(samples=[
    SingleTurnSample(user_input=r["question"], response=r["answer"],
                     retrieved_contexts=r["contexts"], reference=r["reference"])
    for r in records])

print(f"\nPhase 4: scoring {len(records)} x 4 metrics with qwen2.5:7b-instruct...")
t = time.time()
result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb,
                  run_config=RunConfig(max_workers=1, timeout=300, max_retries=1))
print(f"Done in {(time.time()-t)/60:.1f} min")

df = result.to_pandas()
meta = {"user_input", "response", "retrieved_contexts", "reference", "reference_contexts", "rubrics"}
scores = {c: float(df[c].mean()) for c in df.columns if c not in meta}
if "llm_context_precision_with_reference" in scores:
    scores["context_precision"] = scores.pop("llm_context_precision_with_reference")
df.to_json(f"results/{OUT_ID}_perq.json", orient="records", indent=2)

ledger_path = "results/ragas_ledger.json"
ledger = json.load(open(ledger_path)) if os.path.exists(ledger_path) else {}
ledger[OUT_ID] = {"variant_id": OUT_ID, "retrieval": "v04 recursive + HyDE + rerank",
                  "generator": GEN_OLLAMA_MODEL, "judge": "qwen2.5:7b-instruct",
                  "n_questions": len(records), **scores}
json.dump(ledger, open(ledger_path, "w"), indent=2)

print("\n" + "=" * 60)
print(f"CHAMPION RAGAS ({OUT_ID}, n={len(records)})")
prev = ledger.get("v04_recursive_rerank_FULL", {})
print(f"{'metric':<22}{'prev (no HyDE)':>16}{'champion (HyDE)':>17}")
for k in ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]:
    print(f"{k:<22}{prev.get(k, float('nan')):>16.3f}{scores.get(k, float('nan')):>17.3f}")
