"""Push our benchmark questions to LangSmith as a Golden Dataset.

A Golden Dataset stores curated test cases + ground-truth answers so Experiments
(eval/langsmith_eval.py) can score any agent version against them in the LangSmith UI.
Source of truth is eval/questions_v2.jsonl (the same 197 Qs our RAGAS/regression evals use).

Idempotent: reuses the dataset if it exists and only seeds examples when it's empty, so
re-running is safe. Needs LANGCHAIN_API_KEY in .env; does NOT need MongoDB.

  PYTHONPATH=. .venv/bin/python eval/langsmith_push_dataset.py
"""
import json, os
from dotenv import load_dotenv
load_dotenv()
from langsmith import Client

DATASET = os.getenv("LS_DATASET", "legal-contract-eval-197")
QS = [json.loads(l) for l in open("eval/questions_v2.jsonl")]

client = Client()

if client.has_dataset(dataset_name=DATASET):
    ds = client.read_dataset(dataset_name=DATASET)
    print(f"dataset '{DATASET}' already exists ({ds.id})")
else:
    ds = client.create_dataset(
        dataset_name=DATASET,
        description="CUAD legal-contract QA benchmark: 197 questions with ground-truth answers, "
                    "supporting evidence, source doc_id and clause category.",
    )
    print(f"created dataset '{DATASET}' ({ds.id})")

# Only seed when empty so re-runs don't duplicate. To refresh: delete examples in the UI and re-run.
if next(client.list_examples(dataset_id=ds.id, limit=1), None):
    print("dataset already has examples — not re-adding (delete in UI + re-run to refresh).")
else:
    client.create_examples(
        dataset_id=ds.id,
        examples=[{
            "inputs":  {"question": q["question"]},
            "outputs": {"ground_truth": q.get("expected_answer", ""),
                        "evidence": q.get("evidence", ""),
                        "doc_id": q.get("doc_id", "")},
            "metadata": {"id": q["id"], "category": q.get("category", ""),
                         "difficulty": q.get("difficulty", "")},
        } for q in QS],
    )
    print(f"added {len(QS)} examples to '{DATASET}'")
