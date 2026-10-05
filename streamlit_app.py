"""Streamlit chat UI for the legal-contract RAG agent (Phase 3).

Talks to the LangGraph tool-calling agent (app.agent.ask_detailed): it decides when to search,
scopes to the right contract (by title OR party/program name), and sweeps/compares across
documents. Every answer shows the agent's tool-call trace, and the sidebar lists what's in the
knowledge base and lets you upload a new contract straight into the live vector DB — no S3,
searchable immediately.
"""
import os
import uuid

# Which model generates in the UI, via UI_PROVIDER:
#   ollama  (default) — local gemma, free/offline, but slow (~100s/query) — not great live
#   gemini            — fast + free-tier, good for a snappy demo
#   bedrock           — model set by BEDROCK_MODEL, best quality + fast (spends the Bedrock credential)
# MUST run before any `app.*` import: app.config reads LLM_PROVIDER at import time, and its
# load_dotenv() won't override an env var we've already set here.
os.environ["LLM_PROVIDER"] = os.getenv("UI_PROVIDER", "ollama")

import streamlit as st

from app.agent import ask_detailed
from app.ingest import ingest_upload
from app.tools import _all_contracts
from app.config import LLM_PROVIDER

st.set_page_config(page_title="Legal Contract Assistant", page_icon="⚖️", layout="centered")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem;}
  .agent-step {font-size: 0.86rem; color: #6b7280; margin: 0.1rem 0;}
  div[data-testid="stChatMessage"] {padding-top: 0.15rem; padding-bottom: 0.15rem;}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Legal Contract Assistant")
st.caption("A tool-calling agent over a corpus of legal contracts — it finds the right document, "
           "scopes its search, and cites what it used. Ask about one contract or compare several.")

# --- per-tab state: a display log + a stable thread_id so the agent keeps its own multi-turn memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex


def render_steps(steps: list[dict]) -> None:
    """Show WHAT the agent did — its tool calls, in order — so the answer isn't a black box."""
    if not steps:
        return        st.caption("_answered from memory — no new search_")

    lines = []
    for s in steps:
        if s["name"] == "list_contracts":
            lines.append("📋 Listed all contracts")
        elif s.get("contract"):
            lines.append(f'🔍 Searched **{s["contract"]}** — “{s["query"]}”')
        else:
            lines.append(f'🔍 Searched all contracts — “{s["query"]}”')
    with st.expander(f"🔧 What the agent did · {len(steps)} tool call{'s' if len(steps) != 1 else ''}"):
        for ln in lines:
            st.markdown(f"<div class='agent-step'>{ln}</div>", unsafe_allow_html=True)


# --- replay the conversation so far (Streamlit reruns top-to-bottom on every interaction) ---
for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["text"])
        if m["role"] == "model" and m.get("steps") is not None:
            render_steps(m["steps"])

# --- input box; walrus assigns AND tests in one line ---
if question := st.chat_input("Ask about the contracts..."):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the contracts..."):
            try:
                result = ask_detailed(question, thread_id=st.session_state.thread_id)
            except Exception as e:
                result = {"answer": f"⚠️ Error: {type(e).__name__}: {e}", "steps": []}
        st.markdown(result["answer"])
        render_steps(result["steps"])
    st.session_state.messages.append({"role": "user", "text": question})
    st.session_state.messages.append({"role": "model", "text": result["answer"], "steps": result["steps"]})

# --- sidebar: knowledge base + upload + controls ---
with st.sidebar:
    contracts = _all_contracts()
    st.header(f"📚 Knowledge base · {len(contracts)}")
    st.caption("Documents currently searchable.")
    with st.expander("View all documents"):
        for c in sorted(contracts, key=lambda c: c["title"].lower()):
            st.markdown(f"<div class='agent-step'>• {c['title']}</div>", unsafe_allow_html=True)

    st.divider()
    st.header("📄 Add a contract")
    st.caption("Upload a document to make it searchable right away (PDF, .txt, .md).")
    upload = st.file_uploader("Choose a file", type=["pdf", "txt", "md"], label_visibility="collapsed")
    if upload is not None and st.button("Add to knowledge base", width="stretch"):
        with st.spinner(f"Ingesting {upload.name}..."):
            try:
                res = ingest_upload(upload.getvalue(), upload.name)
            except Exception as e:
                res = {"ok": False, "doc_id": upload.name, "reason": f"{type(e).__name__}: {e}"}
        if res["ok"]:
            extra = f" (replaced {res['replaced']} old chunks)" if res.get("replaced") else ""
            st.success(f"Added **{res['doc_id']}** — {res['chunks']} chunks{extra}. Ask about it now.")
            st.rerun()   # refresh the knowledge-base list so the new doc shows up
        else:
            st.error(f"Couldn't ingest {res['doc_id']}: {res.get('reason')}")

    st.divider()
    st.header("About")
    st.markdown(
        "Tool-calling agent:\n"
        "1. **Resolve** which contract you mean (title *or* party/program name)\n"
        "2. **Scope** the semantic search to that document\n"
        "3. **Search** → vector search + cross-encoder rerank\n"
        "4. **Answer** — grounded and cited; compares across contracts when asked"
    )
    st.caption(f"Model: `{LLM_PROVIDER}`  ·  Turns: {len(st.session_state.messages) // 2}")
    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.thread_id = uuid.uuid4().hex   # fresh agent memory too
        st.rerun()
