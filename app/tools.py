"""The 2 tools the LangGraph agent can call. Each @tool's DOCSTRING is what the LLM reads to
decide WHEN to use it, so the wording is load-bearing.

Design is evidence-driven (see results/champion_misses.json): 34/35 v1 misses were named-contract
single-clause questions where the right clause was drowned by near-identical boilerplate across
all 48 contracts. The fix is SCOPING the search to the named contract — proven to lift the gold
chunk from "not in top-5 corpus-wide" to rank 1. So:
  - search_contracts  : semantic search, optionally SCOPED to one contract (the v1-miss fix)
  - list_contracts    : the known title set — for enumeration AND name disambiguation
No CSV / annotation tools: exact-match would have recovered only 1/35 misses, not worth it."""
from collections import Counter

from langchain_core.tools import tool

from app.config import chunks, VECTOR_INDEX, VARIANT_ID, RETRIEVAL_N, FINAL_K
from app.pipeline import embed_query, get_reranker

_norm = lambda s: "".join(c for c in s.lower() if c.isalnum())

_contracts = None
def _all_contracts() -> list[dict]:
    """[{doc_id, title}] for every contract in the variant. Cached."""
    global _contracts
    if _contracts is None:
        ids = chunks.distinct("source.doc_id", {"config.variant_id": VARIANT_ID})
        _contracts = []
        for d in ids:
            one = chunks.find_one(
                {"config.variant_id": VARIANT_ID, "source.doc_id": d}, {"source.doc_title": 1}
            )
            _contracts.append({"doc_id": d, "title": one["source"]["doc_title"]})
    return _contracts


# Generic words a user/LLM tacks onto a contract reference — ignored so "Array Biopharma license
# agreement" still resolves to the title "Array BioPharma Inc. - LICENSE, DEVELOPMENT...".
_STOP = {"the", "a", "an", "agreement", "contract", "deal", "inc", "llc", "corp", "co", "ltd",
         "company", "of", "and"}


def _matches(name: str) -> list[dict]:
    """ALL corpus contracts that resolve from a user-typed name. Two-stage:
      1. fast path — the whole normalized name is a contiguous substring (e.g. "Array Biopharma");
      2. token path — every DISTINCTIVE token (filler like 'agreement' dropped) appears in the
         title, order-independent (e.g. "Array Biopharma license agreement").
    Lets the caller distinguish 0 (unknown), 1 (scope it), >1 (ambiguous -> disambiguate)."""
    contracts = _all_contracts()
    n = _norm(name)
    if not n:
        return []
    whole = [c for c in contracts if n in _norm(c["title"]) or n in _norm(c["doc_id"])]
    if whole:
        return whole
    toks = [t for t in "".join(ch if ch.isalnum() else " " for ch in name.lower()).split()
            if len(t) > 2 and t not in _STOP]
    if not toks:
        return []
    return [c for c in contracts
            if all(t in _norm(c["title"]) or t in _norm(c["doc_id"]) for t in toks)]


def _match(name: str) -> dict | None:
    """Best single resolution (first match), or None. Convenience for callers that don't care
    about ambiguity."""
    ms = _matches(name)
    return ms[0] if ms else None


def invalidate_contract_cache() -> None:
    """Drop the cached contract list so a freshly-ingested document is searchable immediately
    (called after an upload). The next _all_contracts() re-reads MongoDB."""
    global _contracts
    _contracts = None


def _resolve_by_text(name: str) -> tuple[str, list[dict]]:
    """Rung 2 of name resolution. The name matched no TITLE — but in this corpus the title is the
    SEC filer (e.g. 'LinkPlusCorp...') while users name the OTHER party or the program ('Equidata').
    That name still lives in the contract TEXT, so: vector-search the name, rerank, and see which
    contract owns the best chunks. Returns one of:
      ("scope", [{doc_id,title}])   one contract clearly owns the name -> scope the real query there
      ("ask",   [{doc_id,title}...]) several contracts compete         -> let the agent ask the user
      ("none",  [])                  nothing names it                  -> caller searches unscoped
    """
    qv = embed_query(name)
    hits = list(chunks.aggregate([
        {"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
                           "filter": {"config.variant_id": VARIANT_ID},
                           "numCandidates": 150, "limit": 10}},
        {"$project": {"_id": 0, "text": 1, "source": 1}},
    ]))
    if not hits:
        return "none", []
    ce = get_reranker()
    scores = ce.predict([(name, h["text"]) for h in hits])
    ranked = [h for _, h in sorted(zip(scores, hits), key=lambda z: z[0], reverse=True)][:5]
    title_of = {h["source"]["doc_id"]: h["source"]["doc_title"] for h in ranked}
    by_doc = Counter(h["source"]["doc_id"] for h in ranked)
    (best_doc, best_n), = by_doc.most_common(1)
    if best_n >= 3:                       # a majority of the top-5 chunks -> confident single owner
        return "scope", [{"doc_id": best_doc, "title": title_of[best_doc]}]
    return "ask", [{"doc_id": d, "title": title_of[d]} for d, _ in by_doc.most_common()]


@tool
def list_contracts() -> str:
    """List every contract available to search, by title. Call this when the user asks
    'which contract...', wants to compare/sweep across contracts, OR when search_contracts reports
    that a contract name was ambiguous or not found — so you can see the exact valid titles."""
    return "\n".join(f"- {c['title']}" for c in _all_contracts())


@tool
def search_contracts(query: str, contract_name: str = "") -> str:
    """Semantic search over contract TEXT — your main tool. ALWAYS pass contract_name when the
    question is about a specific contract: scoping to that one document is what makes short clauses
    (governing law, dates, renewal/termination terms, liability caps) findable — unscoped, they get
    buried under identical boilerplate from other contracts. To COMPARE a clause across contracts,
    call this once per contract. For open-ended questions with no specific contract, leave
    contract_name empty to search the whole corpus.

    Returns top excerpts, each labeled with its contract. Pass the name the user uses (a party or
    program name is fine — it doesn't have to be the exact title); the tool resolves it. If the name
    genuinely points at several different contracts, this returns a short notice asking which one —
    relay that to the user. Otherwise you always get results; read the contract labels to confirm
    what was found."""
    flt = {"config.variant_id": VARIANT_ID}
    scoped = ""
    if contract_name:
        ms = _matches(contract_name)
        if len(ms) > 1:                                    # rung 1: name matches several TITLES
            names = "\n".join(f"- {c['title']}" for c in ms)
            return (f"'{contract_name}' is ambiguous — it matches {len(ms)} contracts:\n{names}\n"
                    f"Ask the user which one, then retry with a more specific name.")
        if len(ms) == 1:                                   # rung 1: one title match -> scope
            flt["source.doc_id"] = ms[0]["doc_id"]
            scoped = f" (in {ms[0]['title']})"
        else:                                              # rung 2: no title match -> resolve by TEXT
            kind, cands = _resolve_by_text(contract_name)
            if kind == "scope":                            # one contract owns the name -> scope there
                flt["source.doc_id"] = cands[0]["doc_id"]
                scoped = f" (in {cands[0]['title']})"
            elif kind == "ask":                            # rung 3: several compete -> ask the user
                names = "\n".join(f"- {c['title']}" for c in cands)
                return (f"'{contract_name}' isn't a contract title, and its text appears across "
                        f"several contracts:\n{names}\nAsk the user which one they mean, then retry.")
            # kind == "none": nothing names it -> fall through to an unscoped corpus search.
    qv = embed_query(query)
    hits = list(chunks.aggregate([
        {"$vectorSearch": {"index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
                           "filter": flt, "numCandidates": 150, "limit": RETRIEVAL_N}},
        {"$project": {"_id": 0, "text": 1, "source": 1}},
    ]))
    if not hits:
        return f"No results found{scoped}."
    ce = get_reranker()
    scores = ce.predict([(query, h["text"]) for h in hits])
    hits = [h for _, h in sorted(zip(scores, hits), key=lambda z: z[0], reverse=True)][:FINAL_K]
    return "\n\n".join(f"[{h['source']['doc_title']}] {h['text']}" for h in hits)
