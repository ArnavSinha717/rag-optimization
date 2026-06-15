"""TEST 2 — Metadata filtering: 7B identifies WHICH contract the question is about,
retrieval is scoped to that doc_id. Phased for VRAM. Targets q44-type cross-doc confusion."""
import sys, os, json, subprocess
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import (eval_questions, chunks, VECTOR_INDEX, content_tokens,
                     get_reranker, ollama_chat)

V = "v04_pdfplumber_recursive"
cache = json.load(open("results/query_embed_cache.json"))
inscope = [q for q in eval_questions if q["doc_id"]]

# the 11 contracts with human-readable names the LLM can pick from
CONTRACTS = {
    "alliedesportsentertainmentinc_20190815_8-k_ex-10_34_11788308_ex-10_34_sponsorshi": "Newegg / Allied Esports event sponsorship agreement (HyperX Arena)",
    "arcgroupinc_20171211_8-k_ex-10_1_10976103_ex-10_1_sponsorship_20agreement": "Jacksonville Jaguars / ARC Group (Dick's Wings) sponsorship agreement",
    "creditcardscominc_20070810_s-1_ex-10_33_362297_ex-10_33_affiliate_20agreement": "Chase Affiliate Agreement (creditcards.com)",
    "cybergyholdingsinc_20140520_10-q_ex-10_27_8605784_ex-10_27_affiliate_20agreement": "Birch First Global / Mount Knowledge marketing affiliate agreement",
    "digitalcinemadestinationscorp_20111220_s-1_ex-10_10_7346719_ex-10_10_affiliate_2": "National CineMedia (NCM) / Digital Cinema Destinations network affiliate agreement",
    "linkpluscorp_20050802_8-k_ex-10_3240252_ex-10_affiliate_20agreement": "Link Plus (LKPL) / Axiometric affiliate agreement",
    "southernstarenergyinc_20051202_sb-2a_ex-9_801890_ex-9_affiliate_20agreement": "element 5 GmbH affiliate program terms",
    "steelvaultcorp_20081224_10-k_ex-10_16_3074935_ex-10_16_affiliate_20agreement": "Equidata / National Credit Report marketing affiliate agreement",
    "tubemediacorp_20060310_8-k_ex-10_1_513921_ex-10_1_affiliate_20agreement": "TUBE Music Network / Tribune Broadcasting affiliation agreement",
    "uniondentalholdingsinc_20050204_8-ka_ex-10_3345577_ex-10_affiliate_20agreement": "Union Dental / Dr. George Green business affiliate agreement",
    "usioinc_20040428_sb-2_ex-10_11_1723988_ex-10_11_affiliate202": "Network 1 Financial / Payment Data Systems affiliate office agreement",
}
MENU = "\n".join(f"{i+1}. {name}" for i, name in enumerate(CONTRACTS.values()))
IDS = list(CONTRACTS.keys())

PICK_PROMPT = """Which ONE contract is this question about? Reply with ONLY the number (1-11), or 0 if unclear.

Contracts:
""" + MENU + """

Question: {q}"""

# ---- Phase 1: 7B picks the contract per question (checkpointed) ----
PICKS_PATH = "results/metadata_picks.json"
picks = json.load(open(PICKS_PATH)) if os.path.exists(PICKS_PATH) else {}
for q in inscope:
    if q["id"] in picks:
        continue
    reply = ollama_chat("", PICK_PROMPT.format(q=q["question"]), temperature=0.0,
                        model="qwen2.5:7b-instruct").strip()
    num = next((int(t) for t in reply.replace(".", " ").split() if t.isdigit()), 0)
    picks[q["id"]] = IDS[num - 1] if 1 <= num <= 11 else None
    json.dump(picks, open(PICKS_PATH, "w"), indent=2)
    ok = "OK " if picks[q["id"]] == q["doc_id"] else ("-- " if picks[q["id"]] is None else "WRONG")
    print(f"  [{q['id']}] {ok} picked={str(picks[q['id']])[:30]}")

correct = sum(1 for q in inscope if picks[q["id"]] == q["doc_id"])
nopick = sum(1 for q in inscope if picks[q["id"]] is None)
print(f"\nPick accuracy: {correct}/{len(inscope)} correct, {nopick} abstained")

subprocess.run(["ollama", "stop", "qwen2.5:7b-instruct"], capture_output=True)
print("Unloaded 7B; loading reranker...")
ce = get_reranker()

def vsearch_qv(qv, doc_id=None, n=20):
    flt = {"config.variant_id": V}
    if doc_id:
        flt = {"$and": [{"config.variant_id": V}, {"source.doc_id": doc_id}]}
    return list(chunks.aggregate([
        {"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding",
                           "queryVector": qv, "filter": flt,
                           "numCandidates": 100, "limit": n}},
        {"$project": {"_id": 0, "text": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}}}]))

def arank(ev_s, hits, thr=0.5):
    ev = content_tokens(ev_s)
    if not ev: return None
    for i, h in enumerate(hits):
        if len(ev & content_tokens(h["text"])) / len(ev) >= thr: return i + 1
    return None

# ---- Phase 2: filtered search + rerank + score ----
ranks = []
for q in inscope:
    qv = cache[q["id"]]
    hits = vsearch_qv(qv, doc_id=picks[q["id"]], n=20)   # None -> unfiltered fallback
    sc = ce.predict([(q["question"], h["text"]) for h in hits])
    reranked = [h for _, h in sorted(zip(sc, hits), key=lambda z: z[0], reverse=True)]
    r = arank(q["evidence"], reranked[:5])
    ranks.append(r)
    print(f"  [{q['id']}] filtered={'yes' if picks[q['id']] else 'no '} rank={r}")

r1 = sum(1 for r in ranks if r and r <= 1) / len(ranks)
r5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)
mrr = sum(1 / r for r in ranks if r) / len(ranks)
print(f"\n=== METADATA-FILTERED (n={len(ranks)}):  rec@1={r1:.3f}  rec@5={r5:.3f}  MRR={mrr:.3f} ===")
print("ref  v04 vector+rerank:  rec@1=0.831  rec@5=0.949  MRR=0.890")
