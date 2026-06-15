"""Decide when to use RAG and when to use LLM only."""

from app.llm import gemini_chat
#System generates a standalone question if the chat history contains references to specific contracts, parties, or other contextual information that is necessary to understand the user's latest question. If the user's latest question is already standalone and does not require any additional context from the chat history, the system will return it unchanged.
CONTEXTUALIZE_SYSTEM="""Given the conversation history, rewrite the user's latest question as a single standalone question that contains all the context needed to understand it (e.g, which contract or party it refers to). If then question is already standalone, return it unchanged. Output ONLY the rewritten question. No explanations, no preamble."""
ROUTER_SYSTEM="""You are a router for a legal contract assistant. The assistant has a database of contract documents ( sponsorship and affiliate agreements). Decide whether answering the user's question requires searching the contract database.
Reply with exactly ONE word:
-RAG if the question asks about the contents of specific contracts(parties,terms,fees,clauses,dates,definitions).
-DIRECT if it is small talk, a question about your capabilities, or general knowledge not tied to the documents.
Reply with ONLY the single word RAG or DIRECT."""


def contextualize(history: list[dict],question: str)-> str:
    """Use history to rewrite the question as a standalone question with all necessary context."""
    if not history:
        return question #If no history, return the original question
    messages=history+[{"role": "user", "text": question}]
    rewritten = gemini_chat(CONTEXTUALIZE_SYSTEM, messages,temperature=0.0).strip()
    return rewritten if rewritten else question  # empty rewrite -> keep the original

def route(question:str)->str:
    """Decice 'rag' or 'direct' for the standalone question"""
    messages=[{"role": "user", "text": question}]
    reply=gemini_chat(ROUTER_SYSTEM, messages, temperature=0.0).strip().upper()
    return "rag" if "RAG" in reply else "direct"