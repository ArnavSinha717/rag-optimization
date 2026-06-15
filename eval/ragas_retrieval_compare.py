"""RAGAS baseline-vs-winner on the EXPANDED 59-question set, generation held
constant (7B + concise-strict prompt) so ONLY retrieval differs.

Usage (notebook):  %run eval/ragas_retrieval_compare.py baseline
                   %run eval/ragas_retrieval_compare.py hyde

  baseline = v04 vector + rerank
  winner   = v04 HyDE   + rerank
"""
import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
import sys, json, time
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag_lib
from rag_lib import (eval_questions, search_reranked, search_hyde, ollama_chat,
                     SYSTEM_INSTRUCTION, GEN_OLLAMA_MODEL, REFUSAL, get_reranker)

MODE = (sys.argv[1] if len(sys.argv) > 1 else "baseline").lower()
assert MODE in ("baseline", "hyde"), "arg must be 'baseline' or 'hyde'"
VARIANT = "v04_pdfplumber_recursive"
OUT_ID = f"ragas59_{MODE}"
retrieve = (lambda q: search_hyde(q, VARIANT, 20, 5)) if MODE == "hyde" \
    else (lambda q: search_reranked(q, VARIANT, 20, 5))

ctx_path = f"results/{OUT_ID}_contexts.json"
ans_path = f"results/{OUT_ID}_answers.json"
inscope = [q for q in eval_questions if q["doc_id"]]
print(f"=== RAGAS59 {MODE.upper()} — {len(inscope)} in-scope, retrieval={MODE}, gen=7B concise ===")

# ---- Phase 1: retrieval (reranker on GPU) ----
if os.path.exists(ctx_path):
    ctxs = json.load(open(ctx_path)); print(f"RESUME contexts ({len(ctxs)})")
else:
    ctxs = []
    for q in inscope:
        try:
            c = [h["text"] for h in retrieve(q["question"])]
        except Exception as e:
            print(f"  [{q['id']}] RETR ERR {type(e).__name__}"); c = []
        ctxs.append({"id": q["id"], "question": q["question"], "contexts": c,
                     "reference": q["expected_answer"]})
        json.dump(ctxs, open(ctx_path, "w"), indent=2)
        print(f"  [{q['id']}] {len(c)} ctx")

# ---- Phase 2: free reranker VRAM ----
import torch, gc
try:
    get_reranker(); rag_lib._reranker = None
except Exception:
    pass
gc.collect(); torch.cuda.empty_cache()
print("Freed reranker VRAM.")

# ---- Phase 3: generate answers (7B concise-strict) ----
if os.path.exists(ans_path):
    records = json.load(open(ans_path)); print(f"RESUME answers ({len(records)})")
else:
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
        print(f"  [{r['id']}] {ans[:60]}")

# ---- Phase 4: RAGAS judge ----
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

print(f"\nScoring {len(records)} x 4 metrics with qwen2.5:7b-instruct...")
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

ledger = json.load(open("results/ragas_ledger.json")) if os.path.exists("results/ragas_ledger.json") else {}
ledger[OUT_ID] = {"variant_id": OUT_ID, "retrieval": MODE, "n_questions": len(records),
                  "generator": "qwen2.5:7b-instruct (concise)", "judge": "qwen2.5:7b-instruct", **scores}
json.dump(ledger, open("results/ragas_ledger.json", "w"), indent=2)

print("\n" + "=" * 55)
print(f"RAGAS59 {MODE.upper()} (n={len(records)})")
for k in ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]:
    print(f"  {k:<22} {scores.get(k, float('nan')):.3f}")
# show comparison if both done
other = "ragas59_hyde" if MODE == "baseline" else "ragas59_baseline"
if other in ledger:
    print(f"\n  {'metric':<22}{'baseline':>10}{'hyde':>10}")
    b = ledger['ragas59_baseline']; h = ledger['ragas59_hyde']
    for k in ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]:
        print(f"  {k:<22}{b.get(k,float('nan')):>10.3f}{h.get(k,float('nan')):>10.3f}")
