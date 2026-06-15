"""Streamlit chat UI for the legal-contract RAG chatbot."""

import streamlit as st
from app.chat import chat

st.set_page_config(page_title="Legal Contract Assistant", page_icon="⚖️")
st.title("⚖️ Legal Contract Assistant")
st.caption("Ask about the sponsorship & affiliate agreements — answers are cited from the documents.")

# --- memory: one list in session state, survives reruns within this browser tab ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- replay the conversation so far (Streamlit reruns top-to-bottom on every interaction) ---
for m in st.session_state.messages:
    role = "user" if m["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(m["text"])

# --- input box; walrus operator assigns AND tests in one line ---
if question := st.chat_input("Ask about the contracts..."):
    with st.chat_message("user"):
        st.markdown(question)

    # run the full backend (contextualize -> route -> retrieve/direct -> answer)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chat(st.session_state.messages, question)
            except Exception as e:
                result = {"answer": f"⚠️ Error: {type(e).__name__}: {e}",
                          "sources": [], "route": "error", "standalone_question": question}
        st.markdown(result["answer"])

        if result["route"] == "rag" and result["sources"]:
            with st.expander(f"🔍 Searched the contracts — {len(result['sources'])} sources"):
                st.caption(f"Standalone query: *{result['standalone_question']}*")
                for s in result["sources"]:
                    st.markdown(f"**[Source {s['n']}]** {s['doc']}  · rerank score {s['score']:+.2f}")
        elif result["route"] == "direct":
            st.caption("💬 answered directly (no contract search)")

    # persist the turn: original short question, NOT the sources-stuffed prompt
    st.session_state.messages.append({"role": "user", "text": question})
    st.session_state.messages.append({"role": "model", "text": result["answer"]})

# --- sidebar: reset + transparency ---
with st.sidebar:
    st.header("About")
    st.markdown(
        "Production RAG pipeline:\n"
        "1. **Contextualize** follow-ups → standalone question\n"
        "2. **Route** → search contracts vs answer directly\n"
        "3. **Retrieve** → vector search + cross-encoder rerank (top 5)\n"
        "4. **Generate** → grounded, cited, refuses if not in sources"
    )
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Turns in memory: {len(st.session_state.messages)//2}")
