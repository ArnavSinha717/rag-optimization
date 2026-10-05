"""Step 3: validate the gemma judge against an independent Gemini judge on a SAMPLE.
Re-judges N saved results with gemini-2.5-flash and reports agreement, so we know whether to
trust the free local judge. Kept small to stay under the Gemini free-tier limit.

  N=10 .venv/bin/python eval/spotcheck_judge.py
"""
import json, os, time
from langchain_core.messages import SystemMessage, HumanMessage
from app.llm import get_chat_model

N = int(os.getenv("N", "10"))
data = json.load(open("results/v2_scoped_recovery.json"))
rows = data["results"][:N]

JUDGE_SYS = """You grade a contract-QA answer. Reply with EXACTLY one word: HIT if the agent's
answer correctly states the expected fact, or MISS if it is wrong, missing, or refuses. Ignore
wording, formatting, and extra detail — judge only whether the key fact matches."""

_gemini = get_chat_model(provider="gemini", temperature=0)


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return str(c)


def gemini_judge(r) -> str:
    prompt = (f"Question: {r['question']}\nExpected answer: {r['expected']}\n"
              f"Supporting evidence: {r['evidence']}\nAgent answer: {r['answer']}\n\nHIT or MISS?")
    for attempt in range(5):
        try:
            out = _gemini.invoke([SystemMessage(content=JUDGE_SYS), HumanMessage(content=prompt)])
            return "HIT" if "HIT" in _text(out.content).upper() else "MISS"
        except Exception as e:
            if attempt == 4:
                return f"ERR({type(e).__name__})"
            time.sleep(8 * (2 ** attempt))   # 8,16,32,64s — rides out 503 demand spikes


if __name__ == "__main__":
    agree = scored = 0
    print(f"{'id':6} {'gemma':6} {'gemini':6} {'agree'}")
    for r in rows:
        g = gemini_judge(r)
        if g.startswith("ERR"):
            print(f"{r['id']:6} {r['verdict']:6} {g:6} -")
            continue
        scored += 1
        ok = g == r["verdict"]
        agree += ok
        print(f"{r['id']:6} {r['verdict']:6} {g:6} {'OK' if ok else 'DISAGREE'}")
    print(f"\nGemini agrees with gemma on {agree}/{scored} scored verdicts "
          f"({100*agree//max(scored,1)}%) — higher = trust the free local judge.")
