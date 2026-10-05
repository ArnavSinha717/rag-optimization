"""Replay the 162 questions v1 got RIGHT through the v2 agent and count REGRESSIONS.
v2 recovers old misses (scoped_recovery.py) — this proves it doesn't BREAK old wins.
Agent runs on LLM_PROVIDER (e.g. bedrock); an INDEPENDENT judge (JUDGE_PROVIDER; the reported
runs used a local Gemma via Ollama) grades HIT/MISS so we never self-grade. A MISS here is a regression.

  LIMIT=5 LLM_PROVIDER=ollama  PYTHONPATH=. .venv/bin/python eval/regression_check.py  # smoke
          LLM_PROVIDER=bedrock PYTHONPATH=. .venv/bin/python eval/regression_check.py  # full
"""
import json, os
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import JUDGE_PROVIDER, LLM_PROVIDER
from app.llm import get_chat_model
from app.agent import ask

MISS_IDS = {m["id"] for m in json.load(open("results/champion_misses.json"))}
QS = [json.loads(l) for l in open("eval/questions_v2.jsonl")]
HITS = [q for q in QS if q["id"] not in MISS_IDS]          # the 162 v1 got right
ONLY = set(filter(None, os.getenv("IDS", "").split(",")))  # re-run a specific subset (e.g. the regressed)
if ONLY:
    HITS = [q for q in HITS if q["id"] in ONLY]
LIMIT = int(os.getenv("LIMIT", "0")) or len(HITS)

JUDGE_SYS = """You grade a contract-QA answer. Reply with EXACTLY one word: HIT if the agent's
answer correctly states the expected fact, or MISS if it is wrong, missing, or refuses. Ignore
wording, formatting, and extra detail — judge only whether the key fact matches."""

_judge = get_chat_model(provider=JUDGE_PROVIDER, temperature=0)


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return str(c)


def judge(question, expected, evidence, answer) -> str:
    prompt = (f"Question: {question}\nExpected answer: {expected}\nSupporting evidence: {evidence}\n"
              f"Agent answer: {answer}\n\nHIT or MISS?")
    out = _judge.invoke([SystemMessage(content=JUDGE_SYS), HumanMessage(content=prompt)])
    return "HIT" if "HIT" in _text(out.content).upper() else "MISS"


if __name__ == "__main__":
    results = []
    for i, q in enumerate(HITS[:LIMIT]):
        qid = q["id"]
        question, expected = q["question"], q.get("expected_answer", "")
        evidence, cat = q.get("evidence", ""), q.get("category", "?")
        try:
            ans = ask(question, thread_id=f"reg_{qid}")
            verdict = judge(question, expected, evidence, ans)
        except Exception as e:
            ans, verdict = f"ERROR: {type(e).__name__}: {e}", "MISS"
        results.append({"id": qid, "cat": cat, "verdict": verdict,
                        "question": question, "expected": expected, "evidence": evidence,
                        "answer": ans})
        flag = "  <-- REGRESSION" if verdict == "MISS" else ""
        print(f"[{i+1}/{LIMIT}] {qid:5} {cat:18} {verdict}{flag}")

    kept = sum(r["verdict"] == "HIT" for r in results)
    regressed = len(results) - kept
    print(f"\nKEPT {kept}/{len(results)}   REGRESSED {regressed}   "
          f"(agent={LLM_PROVIDER}, judge={JUDGE_PROVIDER})")
    out = {"agent": LLM_PROVIDER, "judge": JUDGE_PROVIDER, "kept": kept,
           "regressed": regressed, "total": len(results), "results": results}
    path = "results/v2_regression_recheck.json" if ONLY else "results/v2_regression.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"saved -> {path}")
