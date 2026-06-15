"""Full RAGAS eval of the WINNER config (v04 recursive + rerank), 32 in-scope
questions, all 4 metrics, judged by local qwen2.5:7b-instruct (no Gemini quota).

Pipeline per question: v04 vector top-20 -> bge rerank top-5 -> Ollama 3B answer.
Then free reranker VRAM, load 7B judge, score 4 RAGAS metrics.
Saves answers + per-question scores to results/ for manual review.
"""
import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"   # before ragas import
import sys, json, time
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import (eval_questions, search_reranked, ask_variant, ollama_chat,
                     SYSTEM_INSTRUCTION, REFUSAL, get_reranker)

WINNER_VARIANT = "v04_pdfplumber_recursive"
RETRIEVE_N, FINAL_K = 20, 5
OUT_ID = "v04_recursive_rerank_FULL"

# ---- 1. Build dataset: rerank-retrieve + generate answers (local, free) ----
# RESUME: if answers already built (e.g. a prior crashed run), reuse them and
# skip straight to scoring. Delete the *_answers.json to force a rebuild.
answers_path = f"results/{OUT_ID}_answers.json"
inscope = [q for q in eval_questions if q["doc_id"]]
if os.path.exists(answers_path):
    records = json.load(open(answers_path))
    print(f"RESUME: loaded {len(records)} cached answers from {answers_path} (skipping build)")
else:
    print(f"Building dataset: {len(inscope)} in-scope questions, {WINNER_VARIANT} + rerank")
    records = []
    for i, q in enumerate(inscope):
        try:
            hits = search_reranked(q["question"], WINNER_VARIANT, RETRIEVE_N, FINAL_K)
            contexts = [h["text"] for h in hits]
            if hits:
                ctx = "\n\n".join(f"[Source {j+1}] {h['source']['doc_title']}\n{h['text']}"
                                  for j, h in enumerate(hits))
                answer = ollama_chat(SYSTEM_INSTRUCTION,
                                     f"Sources:\n{ctx}\n\nQuestion: {q['question']}\n\nAnswer:", 0.1)
            else:
                answer = REFUSAL
        except Exception as e:
            print(f"  [{q['id']}] BUILD ERROR ({type(e).__name__}); using refusal")
            contexts, answer = [], REFUSAL
        records.append({"id": q["id"], "question": q["question"], "answer": answer,
                        "contexts": contexts, "reference": q["expected_answer"]})
        json.dump(records, open(answers_path, "w"), indent=2)   # checkpoint each
        print(f"  [{q['id']}] answered ({len(contexts)} ctx)")
    print(f"Saved answers -> {answers_path}")

# ---- 2. Free reranker VRAM before loading the 7B judge ----
import torch, gc
try:
    ce = get_reranker()
    import rag_lib
    rag_lib._reranker = None
    del ce
except Exception:
    pass
gc.collect(); torch.cuda.empty_cache()
print("Freed reranker VRAM.")

# ---- 3. RAGAS with 7B judge, all 4 metrics ----
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

print(f"\nScoring {len(records)} questions x 4 metrics with qwen2.5:7b-instruct (slow)...")
t = time.time()
result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb,
                  run_config=RunConfig(max_workers=1, timeout=300, max_retries=1))
print(f"Done in {(time.time()-t)/60:.1f} min")

# ---- 4. Save + report ----
df = result.to_pandas()
meta = {"user_input", "response", "retrieved_contexts", "reference", "reference_contexts", "rubrics"}
scores = {c: float(df[c].mean()) for c in df.columns if c not in meta}
if "llm_context_precision_with_reference" in scores:
    scores["context_precision"] = scores.pop("llm_context_precision_with_reference")

# per-question scores for manual review
df.to_json(f"results/{OUT_ID}_perq.json", orient="records", indent=2)

ledger_path = "results/ragas_ledger.json"
ledger = json.load(open(ledger_path)) if os.path.exists(ledger_path) else {}
ledger[OUT_ID] = {"variant_id": OUT_ID, "judge": "qwen2.5:7b-instruct",
                  "n_questions": len(records), **scores}
json.dump(ledger, open(ledger_path, "w"), indent=2)

print("\n" + "=" * 50)
print(f"RAGAS (7B judge) — {OUT_ID}, n={len(records)}")
for k, v in scores.items():
    print(f"  {k:<26} {v:.3f}")
print(f"\nSaved -> {ledger_path} + {OUT_ID}_perq.json + _answers.json")
