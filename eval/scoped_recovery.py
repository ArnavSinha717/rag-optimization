"""Replay the 35 v1 champion misses through the v2 agent and count how many it RECOVERS.
The agent runs on LLM_PROVIDER; correctness is graded by an INDEPENDENT model (JUDGE_PROVIDER,
default gemini) so we never self-grade. Optional LIMIT env for cheap dry-runs.

  LIMIT=3 LLM_PROVIDER=ollama  .venv/bin/python eval/scoped_recovery.py   # free smoke test
        LLM_PROVIDER=bedrock   .venv/bin/python eval/scoped_recovery.py   # full real run
"""
import json, os
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import JUDGE_PROVIDER, LLM_PROVIDER
from app.llm import get_chat_model
from app.agent import ask

MISSES = json.load(open("results/champion_misses.json"))
QS = {json.loads(l)["id"]: json.loads(l) for l in open("eval/questions_v2.jsonl")}
LIMIT = int(os.getenv("LIMIT", "0")) or len(MISSES)

JUDGE_SYS = """You grade a contract-QA answer. Reply with EXACTLY one word: HIT if the agent's
answer correctly states the expected fact, or MISS if it is wrong, missing, or refuses. Ignore
wording, formatting, and extra detail — judge only whether the key fact matches."""

# Judge uses an explicit provider (NOT gemini_chat, which follows LLM_PROVIDER) -> stays independent.
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
    for i, m in enumerate(MISSES[:LIMIT]):
        qid = m["id"]
        q = QS.get(qid, {})
        question = q.get("question", m["q"])
        expected = q.get("expected_answer", "")
        evidence = q.get("evidence", m.get("ev", ""))
        try:
            ans = ask(question, thread_id=f"rec_{qid}")
            verdict = judge(question, expected, evidence, ans)
        except Exception as e:
            ans, verdict = f"ERROR: {type(e).__name__}: {e}", "MISS"
        results.append({"id": qid, "cat": m["cat"], "verdict": verdict,
                        "question": question, "expected": expected, "evidence": evidence,
                        "answer": ans})
        print(f"[{i+1}/{LIMIT}] {qid:5} {m['cat']:18} {verdict}")

    hit = sum(r["verdict"] == "HIT" for r in results)
    print(f"\nRECOVERED {hit}/{len(results)} v1 misses   (agent={LLM_PROVIDER}, judge={JUDGE_PROVIDER})")
    out = {"agent": LLM_PROVIDER, "judge": JUDGE_PROVIDER, "recovered": hit,
           "total": len(results), "results": results}
    json.dump(out, open("results/v2_scoped_recovery.json", "w"), indent=2)
    print("saved -> results/v2_scoped_recovery.json")
