"""Run a LangSmith Experiment: the v2 agent over the golden dataset, graded by LLM-judge
evaluators, so LangSmith shows an aggregate scorecard you can compare across configs.

Maps our v1-vs-v2 work onto LangSmith's native comparison view: each run here becomes an
"experiment" tagged by model/config, and the UI diffs their Correctness/latency/token metrics
side-by-side. The judge is our own model (JUDGE_PROVIDER) — same HIT/MISS rubric
the offline evals use — kept independent from the agent so we never self-grade.

Needs: MongoDB up (the target calls the live agent) + LANGCHAIN_API_KEY. Run the dataset push first.

  # cheap smoke on gemma (free), 5 examples:
  LIMIT=5 LLM_PROVIDER=ollama JUDGE_PROVIDER=ollama PYTHONPATH=. .venv/bin/python eval/langsmith_eval.py
  # full experiment on Bedrock, gemma judge:
  LLM_PROVIDER=bedrock JUDGE_PROVIDER=ollama PYTHONPATH=. .venv/bin/python eval/langsmith_eval.py
"""
import os, uuid
from dotenv import load_dotenv
load_dotenv()
from langsmith import Client, evaluate
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import LLM_PROVIDER, JUDGE_PROVIDER
from app.llm import get_chat_model
from app.agent import ask

DATASET = os.getenv("LS_DATASET", "legal-contract-eval-197")
LIMIT = int(os.getenv("LIMIT", "0"))
_judge = get_chat_model(provider=JUDGE_PROVIDER, temperature=0)


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return str(c)


# --- target: the system under test. One fresh thread per example so memory never bleeds across Qs.
def target(inputs: dict) -> dict:
    return {"answer": ask(inputs["question"], thread_id=f"ls_{uuid.uuid4().hex}")}


# --- evaluators: LLM-as-judge, (run, example) signature. Return {key, score} in [0,1]. ------------
_CORRECT_SYS = """You grade a contract-QA answer against a reference. Reply EXACTLY one word:
HIT if the answer states the reference fact correctly, or MISS if it is wrong, missing, or refuses.
Ignore wording, formatting, and extra detail — judge only whether the key fact matches."""

_RELEVANT_SYS = """You judge whether an answer directly ADDRESSES the question that was asked
(independent of whether it is factually correct). Reply EXACTLY one word: YES or NO."""


def correctness(run, example) -> dict:
    ans = run.outputs.get("answer", "")
    ref = example.outputs.get("ground_truth", "")
    ev = example.outputs.get("evidence", "")
    prompt = (f"Question: {example.inputs['question']}\nReference answer: {ref}\n"
              f"Supporting evidence: {ev}\nAgent answer: {ans}\n\nHIT or MISS?")
    verdict = _text(_judge.invoke([SystemMessage(content=_CORRECT_SYS), HumanMessage(content=prompt)]))
    return {"key": "correctness", "score": 1.0 if "HIT" in verdict.upper() else 0.0}


def relevancy(run, example) -> dict:
    ans = run.outputs.get("answer", "")
    prompt = f"Question: {example.inputs['question']}\nAnswer: {ans}\n\nYES or NO?"
    verdict = _text(_judge.invoke([SystemMessage(content=_RELEVANT_SYS), HumanMessage(content=prompt)]))
    return {"key": "relevancy", "score": 1.0 if "YES" in verdict.upper() else 0.0}


# NOTE: no faithfulness evaluator — true faithfulness needs the RETRIEVED CONTEXT, which the agent
# hides behind its tools. It could be added later by having the agent surface its tool outputs.

if __name__ == "__main__":
    client = Client()
    data = list(client.list_examples(dataset_name=DATASET, limit=LIMIT)) if LIMIT else DATASET
    results = evaluate(
        target,
        data=data,
        evaluators=[correctness, relevancy],
        experiment_prefix=f"v2-agent-{LLM_PROVIDER}",
        metadata={"agent_provider": LLM_PROVIDER, "judge_provider": JUDGE_PROVIDER},
        max_concurrency=2,   # keep low to avoid model-API rate limits
    )
    print(f"\nExperiment done — open it in LangSmith (dataset '{DATASET}').")
    print(f"experiment name: {getattr(results, 'experiment_name', '(see UI)')}")
