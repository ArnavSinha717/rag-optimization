"""Benchmark local models for the agent dev role: generation throughput + latency.
Hits Ollama's API and reads its own timing fields, so tokens/sec is measured, not guessed.
Two prompts that mirror the agent's real work:
  1. GROUNDED ANSWER  — system + sources + question (the slow, token-heavy part)
  2. TOOL DECISION     — a short reasoning turn that picks a tool (latency-sensitive)
Run after the models are pulled:  .venv/bin/python eval/bench_local_models.py
"""
import json, time, urllib.request

MODELS = ["qwen2.5:7b-instruct", "gemma4:e4b-it-qat", "qwen3.5:4b"]
OLLAMA = "http://localhost:11434/api/generate"

GROUNDED = """You are a legal assistant. Answer ONLY from the sources. If absent, say "Not found in the provided documents."

Sources:
[Source 1] This Agreement shall be governed by, and construed and enforced in accordance with, the laws of the State of New York, without regard to its conflict-of-laws principles.
[Source 2] Neither party shall be liable to the other for any special, consequential, incidental, or indirect damages or lost profits arising out of this Agreement.

Question: Which state's law governs this agreement, and is there a cap on liability?

Answer:"""

TOOL_DECISION = """You can call: search_contracts(query, contract_name), lookup_clause(contract_name, clause_type), list_contracts(), find_contracts_with_clause(clause_type).
User: "Compare the governing law of the Chase agreement and the BellRing agreement."
Briefly state which tool calls you would make, in order. Be concise."""

PROMPTS = {"grounded_answer": GROUNDED, "tool_decision": TOOL_DECISION}


def run(model, prompt):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.1}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    wall = time.time() - t0
    ev, ed = r.get("eval_count", 0), r.get("eval_duration", 1) / 1e9          # tokens, sec
    load = r.get("load_duration", 0) / 1e9
    pe, ped = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 1) / 1e9
    return {"wall_s": wall, "load_s": load, "gen_tok": ev, "gen_tok_s": ev / ed if ed else 0,
            "prompt_tok": pe, "prompt_tok_s": pe / ped if ped else 0, "text": r.get("response", "")}


if __name__ == "__main__":
    for model in MODELS:
        print(f"\n{'='*70}\n{model}\n{'='*70}")
        try:
            run(model, "warmup")                                              # load into VRAM first
        except Exception as e:
            print(f"  SKIP — not available ({type(e).__name__}: {e})"); continue
        for name, prompt in PROMPTS.items():
            m = run(model, prompt)
            print(f"\n[{name}]  load {m['load_s']:.1f}s | wall {m['wall_s']:.1f}s | "
                  f"GEN {m['gen_tok']} tok @ {m['gen_tok_s']:.1f} tok/s | "
                  f"prompt {m['prompt_tok']} tok @ {m['prompt_tok_s']:.0f} tok/s")
            print(f"  -> {m['text'][:280].strip()}")
