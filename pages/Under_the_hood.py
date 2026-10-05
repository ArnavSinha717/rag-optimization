"""Second Streamlit page: the stack we settled on (from the optimization study) + the agent graph.
Deliberately static — no app.* imports — so it renders instantly and works even if Mongo is paused."""
import streamlit as st

st.set_page_config(page_title="Under the hood", page_icon="⚙️", layout="centered")

st.title("⚙️ Under the hood")
st.caption("The pipeline, and why each piece — decisions validated by the retrieval study (n=189).")

st.subheader("What we're using")
st.markdown("""
| Stage | Choice | Why |
|---|---|---|
| Corpus | 48 legal contracts · 1,402 chunks (CUAD) | — |
| Parse | `pdfplumber` | cleaner than markdown/pymupdf (footer noise) |
| Chunk | recursive **2000/200** | beat fixed-size — fixed-1000 regressed 0.844→0.750 *and* hallucinated |
| Embed | Gemini `gemini-embedding-001` (768-d) | query & document sides must match |
| Store | MongoDB Atlas Vector Search (cosine) | — |
| **Retrieve** | top-20 vector → **cross-encoder rerank** → top-5 | **the headline win** |
| Reranker | `BAAI/bge-reranker-base` | — |
| Generate | LLM on **AWS Bedrock** (model set via env/config) | swappable: Gemini / local gemma |
| Agent | **LangGraph** ReAct · 2 tools | scoping + party-name resolution + multi-doc |
| Observability | LangSmith | tracing + benchmark experiments |
""")

c1, c2, c3 = st.columns(3)
c1.metric("recall@5 (rerank)", "0.815", "+0.016 vs vector-only")
c2.metric("MRR", "0.691", "+0.053 vs vector-only")
c3.metric("recall@5 (BM25 + rerank)", "0.619", "-0.196", delta_color="inverse")
st.caption("Deterministic, n=189, bootstrap 95% CIs. BM25 alone (no rerank): recall@5 0.487. "
           "Reranking is what moved the needle.")

st.divider()

st.subheader("The agent (LangGraph)")
st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; node [fontname="sans-serif", fontsize=11];
  start [shape=circle, label="START", style=filled, fillcolor="#e5e7eb", width=0.5];
  agent [shape=box, style="rounded,filled", fillcolor="#dbeafe", label="agent\\n(LLM + tools bound)"];
  tools [shape=box, style="rounded,filled", fillcolor="#dcfce7", label="tools\\nsearch_contracts · list_contracts"];
  end   [shape=doublecircle, label="END", width=0.45];
  start -> agent;
  agent -> tools [label="  needs a tool"];
  tools -> agent [label="  results"];
  agent -> end   [label="  final answer"];
}
""", width="stretch")
st.caption("ReAct loop: the agent reasons, calls a tool, reads the result, and loops until it can "
           "answer — so one question can trigger several scoped searches (that's how comparisons work).")
