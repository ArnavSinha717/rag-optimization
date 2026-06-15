"""Re-eval of the WINNER with the GENERATION FIX: strict anti-inference prompt
+ qwen2.5:7b-instruct generator. Retrieval is held IDENTICAL by reusing last
night's reranked contexts -> isolates the generation change alone.

Compares against v04_recursive_rerank_FULL (loose prompt, 3B generator).
"""
import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
import sys, json, time
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import SYSTEM_INSTRUCTION, GEN_OLLAMA_MODEL, ollama_chat, REFUSAL

PREV = "results/v04_recursive_rerank_FULL_answers.json"   # reuse its contexts
OUT_ID = "v04_rerank_strict7b_FULL"
answers_path = f"results/{OUT_ID}_answers.json"

# ---- 1. Regenerate answers with strict prompt + 7B, REUSING contexts ----
if os.path.exists(answers_path):
    records = json.load(open(answers_path))
    print(f"RESUME: loaded {len(records)} cached answers (skipping regen)")
else:
    prev = json.load(open(PREV))
    print(f"Regenerating {len(prev)} answers with strict prompt + {GEN_OLLAMA_MODEL} (contexts reused)")
    records = []
    for r in prev:
        ctx = r["contexts"]
        if ctx:
            block = "\n\n".join(f"[Source {i+1}] {c}" for i, c in enumerate(ctx))
            try:
                ans = ollama_chat(SYSTEM_INSTRUCTION,
                                  f"Sources:\n{block}\n\nQuestion: {r['question']}\n\nAnswer:",
                                  0.1, model=GEN_OLLAMA_MODEL)
            except Exception as e:
                print(f"  [{r['id']}] GEN ERROR {type(e).__name__}; refusal"); ans = REFUSAL
        else:
            ans = REFUSAL
        records.append({"id": r["id"], "question": r["question"], "answer": ans,
                        "contexts": ctx, "reference": r["reference"]})
        json.dump(records, open(answers_path, "w"), indent=2)
        print(f"  [{r['id']}] {ans[:70]}")
    print(f"Saved -> {answers_path}")

# ---- 2. RAGAS score with 7B judge (no reranker loaded -> no VRAM juggling) ----
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

ledger_path = "results/ragas_ledger.json"
ledger = json.load(open(ledger_path)) if os.path.exists(ledger_path) else {}
ledger[OUT_ID] = {"variant_id": OUT_ID, "judge": "qwen2.5:7b-instruct",
                  "generator": GEN_OLLAMA_MODEL, "prompt": "strict", "n_questions": len(records), **scores}
json.dump(ledger, open(ledger_path, "w"), indent=2)

print("\n" + "=" * 60)
print(f"GENERATION FIX RESULT ({OUT_ID}, n={len(records)})")
print(f"{'metric':<22}{'before (3B,loose)':>18}{'after (7B,strict)':>18}")
prev_scores = ledger.get("v04_recursive_rerank_FULL", {})
for k in ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]:
    b = prev_scores.get(k, float('nan'))
    print(f"{k:<22}{b:>18.3f}{scores.get(k, float('nan')):>18.3f}")
print("\n(retrieval metrics should be ~unchanged — contexts identical; faithfulness/relevancy should rise)")
