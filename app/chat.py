"""Orchestrates chat: Contextualize->Route->(RAG retrieve | Direct) -> Answer"""
from app.llm import gemini_chat
from app.router import contextualize, route
from app.pipeline import retrieve

RAG_SYSTEM="""You are a legal research assistant. Answer the user's question using ONLY the provided source excerpts.
Rules:
- Use ONLY facts explicitly written in the sources. Never infer,deduce,guess,or rely on outside knowledge.
-Do NOT expand acronyms or abbreviations unless the full form is explicitly written in the sources.
-If the sources do not explicitly contain the answer, reply EXACTLY: "Not found in the provided documents." Do not attempt a partial or inferred answer.
-Read conditions, negations, and "either/neither" carefully; state precisely what the source says about who may or may not act.
-Lead with the direct answer in ONE sentence. Cite the supporting source as [Source N] at the end. Keep the whole answer to 1-2 sentences."""

DIRECT_SYSTEM="""You are a friendly legal contract assistant. You have a database of sponsorship and affiliate agreements that you can search when asked about specific contracts.
For general conversation,questions about your capabilities, or general knowledge, answer naturally and concisely. Do not invent details about specific contracts - If asked about contract contents, suggest the user ask directly so you can search the documents."""

def build_sources_block(hits:list[dict])->str:
    """Formats retrieved sources into a string block for LLM input."""
    return "\n\n".join([f"[Source {i+1}]\n{h['source']['doc_title']}\n{h['text']}" for i, h in enumerate(hits)])

def chat(history: list[dict],question:str)->dict:
    """One conversational turn. Returns the answer plus metadata for the UI.
        history: prior turns, OUR format:
    [{"role":"user"|"model","text":...}]
        question: the user's newest message(not yet in history)
    """
    #Step 1. rewrite follow-ups into a standalone question (uses history)
    standalone=contextualize(history,question)
    #Step 2. Decide: Search the contracts or answer directly
    decision=route(standalone)
    if decision=="rag":
        #3a. retrival:embed->vector search->rerank0>top-5
        hits=retrieve(standalone)
        sources_block=build_sources_block(hits)
        final_user_msg=f"Sources:\n{sources_block}\n\nQuestion:{standalone}\n\nAnswer:"
        messages=history+[{"role":"user","text":final_user_msg}]
        answer=gemini_chat(RAG_SYSTEM,messages,temperature=0.1)
        sources=[
            {"n":i+1,
             "doc":h["source"]["doc_title"],
             "score":h["rerank_score"]}
             for i,h in enumerate(hits)
        ]
    else:
        #3b.Direct: no retrieval, just converse with history
        messages=history+[{"role":"user","text":question}]
        answer=gemini_chat(DIRECT_SYSTEM,messages,temperature=0.7)
        sources=[]
    return {
        "answer":answer,
        "sources":sources,
        "route":decision,
        "standalone_question": standalone,
    }