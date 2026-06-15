"""
rag_lib.py — reusable helpers for the legal-RAG project.

After a kernel restart, restore the entire working environment with ONE line:

    from rag_lib import *

This replaces having to re-run cells 1/3/6/11/14/15/16 in the right order.
It initializes the Gemini client, MongoDB connection, and S3 client on import,
and exposes every helper function used by ingestion, retrieval, evaluation,
and persistence.

Requires the same .env (GEMINI_API_KEY, MONGO_URI, S3_BUCKET, AWS_* ).
"""

import os
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # RAGAS telemetry off (WSL has no DNS; fully local)
import io
import re
import json
import time
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlparse

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# ===========================================================================
# Connections / config (initialized on import)
# ===========================================================================
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-embedding-001"
DIMS = 768

from pymongo import MongoClient
from pymongo.server_api import ServerApi

mongo = MongoClient(os.environ["MONGO_URI"], server_api=ServerApi("1"))
db = mongo["rag"]
chunks = db["chunks"]
VECTOR_INDEX = "chunks_vector_index"

import boto3

_s3 = boto3.client("s3")

# Local Ollama (generation = 3b; RAGAS judge = 7b set in the notebook)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"
REFUSAL = "Not found in the provided documents."

SYSTEM_INSTRUCTION = """You are a legal research assistant. Answer the user's question using ONLY the provided source excerpts.

Rules:
- Use ONLY facts explicitly written in the sources. Never infer, deduce, guess, or rely on outside knowledge.
- Do NOT expand acronyms or abbreviations unless the full form is explicitly written in the sources.
- If the sources do not explicitly contain the answer, reply EXACTLY: "Not found in the provided documents." Do not attempt a partial or inferred answer.
- Read conditions, negations, and "either/neither" carefully; state precisely what the source says about who may or may not act.
- Lead with the direct answer in ONE sentence. Do not restate the question or add preamble. Cite the supporting source as [Source N] at the end. Keep the whole answer to 1-2 sentences."""

# Generation model. 3B is fast; 7B reasons better (cleaner refusals, fewer
# inverted-logic errors) — chosen for the legal safety margin.
GEN_OLLAMA_MODEL = "qwen2.5:7b-instruct"

JUDGE_PROMPT = """You are grading an answer from a legal research assistant. Compare the ACTUAL answer to the EXPECTED answer.

QUESTION: {question}
EXPECTED ANSWER: {expected}
ACTUAL ANSWER: {actual}

Scoring:
- 1.0 = the actual answer contains the same key fact(s) as expected (wording may differ; extra detail is fine).
- 0.5 = partially correct: right topic but missing or slightly wrong details.
- 0.0 = wrong, hallucinated, or it refused ("Not found...") when the answer was expected.

Be lenient about wording and extra detail. "governed by Delaware law" matches "Delaware".
A verbatim-correct answer with extra surrounding text is still 1.0.
Output ONLY one number: 1.0, 0.5, or 0.0"""

STOPWORDS = set("""a an the this that these those of to in on at by for and or but with as is are was were be been
being shall will may must any all each such other its his her their it they them we you he she i
agreement party parties section clause hereby herein hereof hereto hereunder hereinafter thereof therein
upon under pursuant accordance whereas witnesseth now therefore provided including include includes
between among from into out over per via not no if then than which who whom whose when where while
governed construed effect force law laws state states united right rights obligation obligations term
terms condition conditions set forth made entered date dated effective""".split())

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# S3
# ===========================================================================
def fetch(uri: str) -> bytes:
    p = urlparse(uri)
    obj = _s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return obj["Body"].read()


# ===========================================================================
# Embeddings
# ===========================================================================
def embed(texts: list[str], task_type: str) -> list[np.ndarray]:
    res = client.models.embed_content(
        model=MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=DIMS),
    )
    return [np.array(e.values) for e in res.embeddings]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_with_retry(texts: list[str], task_type: str, max_retries: int = 5) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            return [v.tolist() for v in embed(texts, task_type)]
        except Exception as e:
            wait = 5 * (2 ** attempt)
            print(f"    embed retry {attempt + 1}/{max_retries} ({type(e).__name__}: {e}); sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"embed failed after {max_retries} retries")


def embed_query_retry(text: str, max_retries: int = 5) -> list[float]:
    for attempt in range(max_retries):
        try:
            return embed([text], "RETRIEVAL_QUERY")[0].tolist()
        except Exception as e:
            wait = 5 * (2 ** attempt)
            print(f"    embed retry {attempt + 1}/{max_retries} ({type(e).__name__}); sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError("query embed failed after retries")


# ===========================================================================
# PDF parsers + chunkers (registries — add variants here)
# ===========================================================================
import pdfplumber


def parse_pdf_pdfplumber(pdf_bytes: bytes) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def clean_pdf_text(text: str) -> str:
    """Strip PDF page artifacts that pollute chunk embeddings: page-footer lines
    ('Source: COMPANY, FORM, DATE'), bare page-number lines, and excess whitespace."""
    text = re.sub(r"\n?Source: [A-Z][^\n]*\n?", "\n", text)   # SEC page footers
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)            # bare page numbers
    text = re.sub(r"\n{3,}", "\n\n", text)                     # collapse blank runs
    text = re.sub(r"[ \t]{2,}", " ", text)                     # collapse spaces
    return text.strip()


def parse_pdf_pdfplumber_clean(pdf_bytes: bytes) -> str:
    return clean_pdf_text(parse_pdf_pdfplumber(pdf_bytes))


def parse_pdf_pymupdf4llm(pdf_bytes: bytes) -> str:
    """PyMuPDF4LLM -> Markdown: cleaner extraction than pdfplumber, preserves
    section headings (## ...) which enables structure-aware chunking."""
    import pymupdf
    import pymupdf4llm
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    return pymupdf4llm.to_markdown(doc)


PDF_PARSERS: dict[str, Callable[[bytes], str]] = {
    "pdfplumber": parse_pdf_pdfplumber,
    "pdfplumber_clean": parse_pdf_pdfplumber_clean,
    "pymupdf4llm": parse_pdf_pymupdf4llm,
}


def chunk_fixed_chars(text: str, size: int, overlap: int) -> list[dict]:
    """Fixed-size character chunking with overlap. Returns dicts with text + char offsets."""
    out = []
    step = max(1, size - overlap)
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        piece = text[start:end].strip()
        if piece:
            out.append({"text": piece, "char_start": start, "char_end": end})
        if end == n:
            break
        start += step
    return out


def chunk_recursive(text: str, size: int = 2000, overlap: int = 200) -> list[dict]:
    """Recursive/boundary-aware splitting: breaks at paragraph -> line -> sentence
    -> word boundaries instead of blind fixed-size cuts, keeping semantic units
    (clauses, definitions) intact. Same size as fixed for a fair comparison."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    out = []
    for d in splitter.create_documents([text]):
        t = d.page_content.strip()
        if t:
            start = d.metadata.get("start_index", 0)
            out.append({"text": t, "char_start": start, "char_end": start + len(d.page_content)})
    return out


def chunk_markdown(text: str, size: int = 2000, overlap: int = 200) -> list[dict]:
    """Structure-aware chunking for Markdown: prefers splitting on headings (##),
    then paragraphs/sentences, bounded by size. Keeps sections/clauses intact."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=size, chunk_overlap=overlap, add_start_index=True)
    out = []
    for d in splitter.create_documents([text]):
        t = d.page_content.strip()
        if t:
            start = d.metadata.get("start_index", 0)
            out.append({"text": t, "char_start": start, "char_end": start + len(d.page_content)})
    return out


CHUNKERS: dict[str, Callable[[str], list[dict]]] = {
    "fixed_2000_200": lambda t: chunk_fixed_chars(t, 2000, 200),
    "fixed_1000_100": lambda t: chunk_fixed_chars(t, 1000, 100),
    "fixed_500_50": lambda t: chunk_fixed_chars(t, 500, 50),
    "recursive_2000_200": lambda t: chunk_recursive(t, 2000, 200),
    "markdown_2000_200": lambda t: chunk_markdown(t, 2000, 200),
    "markdown_1000_100": lambda t: chunk_markdown(t, 1000, 100),
}


# ===========================================================================
# Ingestion
# ===========================================================================
def ingest_variant(variant_id: str, config: dict, embed_batch: int = 25, batch_pause: float = 5.0) -> dict:
    parser = PDF_PARSERS[config["pdf_parser"]]
    chunker = CHUNKERS[config["chunk_strategy"]]

    cleared = chunks.delete_many({"config.variant_id": variant_id}).deleted_count
    print(f"Cleared {cleared} existing chunks for variant '{variant_id}'\n")

    manifest_uri = f"s3://{os.environ['S3_BUCKET']}/manifest/cuad.json"
    manifest = json.loads(fetch(manifest_uri))
    print(f"Manifest: {len(manifest)} contracts\n")

    now = datetime.now(timezone.utc)
    total_chunks = 0
    per_doc = []

    for entry in manifest:
        doc_id, title, uri = entry["doc_id"], entry["title"], entry["source_uri"]
        pdf_bytes = fetch(uri)
        text = parser(pdf_bytes)
        raw_chunks = chunker(text)
        if not raw_chunks:
            print(f"  {doc_id[:60]:<60}  SKIPPED (no text extracted)")
            continue

        vectors: list[list[float]] = []
        for i in range(0, len(raw_chunks), embed_batch):
            batch = [c["text"] for c in raw_chunks[i: i + embed_batch]]
            vectors.extend(embed_with_retry(batch, "RETRIEVAL_DOCUMENT"))
            time.sleep(batch_pause)

        docs = []
        for i, (c, v) in enumerate(zip(raw_chunks, vectors)):
            docs.append({
                "text": c["text"],
                "embedding": v,
                "source": {
                    "doc_id": doc_id, "doc_title": title, "source_uri": uri,
                    "chunk_index": i, "char_start": c["char_start"], "char_end": c["char_end"],
                },
                "config": {
                    "variant_id": variant_id,
                    "pdf_parser": config["pdf_parser"],
                    "chunk_strategy": config["chunk_strategy"],
                    "embedding_model": MODEL, "embedding_dims": DIMS,
                    "embedding_task_type": "RETRIEVAL_DOCUMENT",
                },
                "created_at": now,
            })
        chunks.insert_many(docs)
        total_chunks += len(docs)
        per_doc.append((doc_id, len(docs)))
        print(f"  {doc_id[:60]:<60}  {len(docs):>3} chunks  ({len(text):>7,} chars)")

    print(f"\nVariant '{variant_id}' -> {total_chunks} chunks across {len(per_doc)} docs")
    return {"variant_id": variant_id, "config": config, "chunk_count": total_chunks, "per_doc": per_doc}


# ===========================================================================
# Generation (local Ollama)
# ===========================================================================
def ollama_chat(system: str, user: str, temperature: float = 0.1, max_retries: int = 3,
                model: str = None) -> str:
    model = model or OLLAMA_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    for attempt in range(max_retries):
        try:
            r = requests.post(
                OLLAMA_URL,
                json={"model": model, "messages": messages, "stream": False,
                      "options": {"temperature": temperature}},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            print(f"    ollama retry {attempt + 1}/{max_retries} ({type(e).__name__}); sleeping 3s")
            time.sleep(3)
    raise RuntimeError("ollama failed after retries")


# ===========================================================================
# Retrieval + RAG
# ===========================================================================
def search_variant(query_text: str, variant_id: str, k: int = 5) -> list[dict]:
    qv = embed_query_retry(query_text)
    pipeline = [
        {"$vectorSearch": {
            "index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
            "filter": {"config.variant_id": variant_id},
            "numCandidates": max(50, k * 10), "limit": k,
        }},
        {"$project": {"_id": 0, "text": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(chunks.aggregate(pipeline))


_reranker = None  # lazy-loaded cross-encoder (bge-reranker-base)


def get_reranker():
    """Lazy-load the cross-encoder reranker (only when first used)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    return _reranker


def search_reranked(query_text: str, variant_id: str, retrieve_n: int = 20, final_k: int = 5) -> list[dict]:
    """Stage 1: vector top-N (Gemini bi-encoder). Stage 2: cross-encoder rerank -> top-k."""
    qv = embed_query_retry(query_text)
    pipeline = [
        {"$vectorSearch": {
            "index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
            "filter": {"config.variant_id": variant_id},
            "numCandidates": max(100, retrieve_n * 5), "limit": retrieve_n,
        }},
        {"$project": {"_id": 0, "text": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    hits = list(chunks.aggregate(pipeline))
    if not hits:
        return []
    ce = get_reranker()
    ce_scores = ce.predict([(query_text, h["text"]) for h in hits])
    for h, s in zip(hits, ce_scores):
        h["rerank_score"] = float(s)
    hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return hits[:final_k]


TEXT_INDEX = "chunks_text_index"   # BM25 full-text Atlas Search index


def text_search(query_text: str, variant_id: str, k: int = 20) -> list[dict]:
    """BM25 keyword search via Atlas $search. Catches exact terms (acronyms,
    names, numbers) that vector search misses."""
    pipeline = [
        {"$search": {
            "index": TEXT_INDEX,
            "compound": {
                "must": [{"text": {"query": query_text, "path": "text"}}],
                "filter": [{"equals": {"path": "config.variant_id", "value": variant_id}}],
            },
        }},
        {"$limit": k},
        {"$project": {"_id": 0, "text": 1, "source": 1, "score": {"$meta": "searchScore"}}},
    ]
    return list(chunks.aggregate(pipeline))


def _chunk_key(h: dict) -> tuple:
    s = h["source"]
    return (s["doc_id"], s["chunk_index"])


def hybrid_search(query_text: str, variant_id: str, k_each: int = 20,
                  rrf_k: int = 60, final_n: int = 20) -> list[dict]:
    """Hybrid retrieval: vector + BM25, fused with Reciprocal Rank Fusion.
    RRF score for a chunk = sum over each ranked list of 1/(rrf_k + rank).
    A chunk ranked high in EITHER list (or both) rises to the top."""
    vec = search_variant(query_text, variant_id, k=k_each)
    txt = text_search(query_text, variant_id, k=k_each)
    scores, docs = {}, {}
    for ranked in (vec, txt):
        for rank, h in enumerate(ranked):
            key = _chunk_key(h)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            docs[key] = h
    fused = sorted(docs.keys(), key=lambda key: scores[key], reverse=True)
    out = []
    for key in fused[:final_n]:
        h = docs[key]
        h["rrf_score"] = scores[key]
        out.append(h)
    return out


def search_hybrid_reranked(query_text: str, variant_id: str, k_each: int = 20,
                           final_k: int = 5) -> list[dict]:
    """Hybrid (vector+BM25, RRF) -> cross-encoder rerank -> top-k."""
    fused = hybrid_search(query_text, variant_id, k_each=k_each, final_n=k_each)
    if not fused:
        return []
    ce = get_reranker()
    sc = ce.predict([(query_text, h["text"]) for h in fused])
    for h, s in zip(fused, sc):
        h["rerank_score"] = float(s)
    fused.sort(key=lambda h: h["rerank_score"], reverse=True)
    return fused[:final_k]


HYDE_PROMPT = """Write a brief, plausible answer to this question as it might appear in a legal contract. Be specific and concise (1-2 sentences). Make a reasonable guess even if unsure — accuracy is not required, only plausible legal phrasing.

Question: {q}

Hypothetical contract excerpt:"""


def hyde_query(question: str, model: str = "qwen2.5:3b") -> str:
    """Generate a hypothetical answer to embed instead of the raw question.
    The guess need not be correct — it just needs to look like the target text."""
    return ollama_chat("", HYDE_PROMPT.format(q=question), temperature=0.3, model=model)


def search_hyde(question: str, variant_id: str, retrieve_n: int = 20, final_k: int = 5,
                rerank: bool = True) -> list[dict]:
    """HyDE retrieval: embed a hypothetical answer (not the question) for vector
    search. Optionally rerank the candidates with the ORIGINAL question."""
    hyp = hyde_query(question)
    qv = embed([hyp], "RETRIEVAL_QUERY")[0].tolist()
    pipeline = [
        {"$vectorSearch": {
            "index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
            "filter": {"config.variant_id": variant_id},
            "numCandidates": max(100, retrieve_n * 5), "limit": retrieve_n,
        }},
        {"$project": {"_id": 0, "text": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    hits = list(chunks.aggregate(pipeline))
    if not hits:
        return []
    if rerank:
        ce = get_reranker()
        sc = ce.predict([(question, h["text"]) for h in hits])   # rerank by REAL question
        for h, s in zip(hits, sc):
            h["rerank_score"] = float(s)
        hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return hits[:final_k]


def ask_variant(question: str, variant_id: str, k: int = 5) -> tuple[str, list[dict]]:
    hits = search_variant(question, variant_id, k=k)
    if not hits:
        return REFUSAL, []
    context = "\n\n".join(
        f"[Source {i+1}] {h['source']['doc_title']}\n{h['text']}" for i, h in enumerate(hits)
    )
    user = f"Sources:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    answer = ollama_chat(SYSTEM_INSTRUCTION, user, temperature=0.1, model=GEN_OLLAMA_MODEL)
    return answer, hits


def judge_answer(question: str, expected: str, actual: str) -> float:
    txt = ollama_chat("", JUDGE_PROMPT.format(question=question, expected=expected, actual=actual),
                      temperature=0.0)
    for token in txt.replace(",", " ").replace("*", " ").split():
        try:
            v = float(token)
            if v in (0.0, 0.5, 1.0):
                return v
        except ValueError:
            continue
    return 0.0


# ===========================================================================
# Our hand-rolled metrics
# ===========================================================================
def content_tokens(s: str) -> set:
    toks = re.findall(r"[a-z0-9$]+", s.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 2}


def passage_hit(evidence: str, hits: list[dict], threshold: float = 0.6):
    """Passage-level recall on distinctive (non-boilerplate) tokens. None if no evidence."""
    if not evidence:
        return None
    ev = content_tokens(evidence)
    if not ev:
        return None
    joined = content_tokens(" ".join(h["text"] for h in hits))
    overlap = len(ev & joined) / len(ev)
    return 1 if overlap >= threshold else 0


def evaluate_variant(variant_id: str, questions: list[dict], k: int = 5, pause: float = 0.5) -> dict:
    per_q = []
    for q in questions:
        try:
            answer, hits = ask_variant(q["question"], variant_id, k=k)
        except Exception as e:
            print(f"  [{q['id']}] ASK ERROR: {e}")
            continue

        retrieved_doc_ids = [h["source"]["doc_id"] for h in hits]

        if q["category"] == "out_of_scope":
            score = 1.0 if REFUSAL.lower() in answer.lower() else 0.0
            doc_recall = None
            psg_recall = None
        else:
            doc_recall = 1 if q["doc_id"] in retrieved_doc_ids else 0
            psg_recall = passage_hit(q.get("evidence", ""), hits)
            try:
                score = judge_answer(q["question"], q["expected_answer"], answer)
            except Exception as e:
                print(f"  [{q['id']}] JUDGE ERROR: {e}")
                score = 0.0

        per_q.append({
            "id": q["id"], "category": q["category"], "difficulty": q["difficulty"],
            "question": q["question"], "expected_answer": q["expected_answer"],
            "expected_doc": q["doc_id"], "retrieved_docs": retrieved_doc_ids,
            "answer": answer, "doc_recall@k": doc_recall, "passage_recall@k": psg_recall,
            "answer_score": score,
        })

        flag = "OK" if score >= 0.5 else "XX"
        pr = "-" if psg_recall is None else psg_recall
        print(f"  {flag} [{q['id']}] {q['category']:<12} doc_rec={doc_recall} psg_rec={pr} score={score}  {q['question'][:46]}")
        time.sleep(pause)

    in_scope = [r for r in per_q if r["category"] != "out_of_scope"]
    out_scope = [r for r in per_q if r["category"] == "out_of_scope"]
    psg = [r["passage_recall@k"] for r in in_scope if r["passage_recall@k"] is not None]
    metrics = {
        "n_questions": len(per_q),
        "doc_recall_at_k": sum(r["doc_recall@k"] for r in in_scope) / len(in_scope) if in_scope else 0,
        "passage_recall_at_k": sum(psg) / len(psg) if psg else 0,
        "answer_correctness": sum(r["answer_score"] for r in in_scope) / len(in_scope) if in_scope else 0,
        "refusal_rate": sum(r["answer_score"] for r in out_scope) / len(out_scope) if out_scope else 0,
    }
    return {"variant_id": variant_id, "k": k, "metrics": metrics, "per_question": per_q}


def print_failures(result: dict, width: int = 240):
    import textwrap
    fails = [r for r in result["per_question"]
             if r["category"] != "out_of_scope" and r["answer_score"] < 1.0]
    print(f"\n{'=' * 70}\n{len(fails)} in-scope answers scored < 1.0\n{'=' * 70}")
    for r in fails:
        print(f"\n[{r['id']}] score={r['answer_score']}  doc_rec={r['doc_recall@k']}  psg_rec={r['passage_recall@k']}")
        print(f"  Q:        {r['question']}")
        print(f"  EXPECTED: {r['expected_answer']}")
        print(f"  ACTUAL:   {textwrap.shorten(r['answer'], width=width)}")


# ===========================================================================
# Persistence
# ===========================================================================
def load_eval_questions(path: str = "eval/questions.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_result(result: dict) -> None:
    vid = result["variant_id"]
    detail_path = os.path.join(RESULTS_DIR, f"{vid}.json")
    with open(detail_path, "w") as f:
        json.dump(result, f, indent=2)

    ledger_path = os.path.join(RESULTS_DIR, "ledger.json")
    ledger = json.load(open(ledger_path)) if os.path.exists(ledger_path) else {}
    ledger[vid] = {"variant_id": vid, "k": result["k"], **result["metrics"]}
    with open(ledger_path, "w") as f:
        json.dump(ledger, f, indent=2)
    print(f"Saved {detail_path}")
    print(f"Updated {ledger_path} ({len(ledger)} variants)")


def show_ledger() -> None:
    ledger_path = os.path.join(RESULTS_DIR, "ledger.json")
    if not os.path.exists(ledger_path):
        print("No ledger yet.")
        return
    ledger = json.load(open(ledger_path))
    rows = sorted(ledger.values(), key=lambda r: r["variant_id"])
    print(f"{'variant_id':<30} {'psg_recall':>11} {'correctness':>12} {'refusal':>8} {'doc_recall':>11}")
    print("-" * 75)
    for r in rows:
        print(f"{r['variant_id']:<30} {r['passage_recall_at_k']:>11.3f} "
              f"{r['answer_correctness']:>12.3f} {r['refusal_rate']:>8.3f} {r['doc_recall_at_k']:>11.3f}")


def assert_corpus_coverage(questions: list[dict], variant_id: str) -> None:
    """METHODOLOGY GUARD: every in-scope question's doc_id must have chunks in the
    variant. Otherwise a missing document is indistinguishable from a retrieval
    failure and silently corrupts every metric. Refuses to proceed on violation."""
    needed = {q["doc_id"] for q in questions if q.get("doc_id")}
    have = set(chunks.distinct("source.doc_id", {"config.variant_id": variant_id}))
    orphans = sorted(needed - have)
    if orphans:
        raise AssertionError(
            f"{len(orphans)} question doc_ids have NO chunks in variant '{variant_id}'. "
            f"Eval would mis-score them as retrieval failures. First few: {orphans[:5]}")
    print(f"corpus coverage OK: all {len(needed)} question docs present in {variant_id}")


# convenience: load the eval set on import
eval_questions = load_eval_questions() if os.path.exists("eval/questions.jsonl") else []

print(f"rag_lib loaded: Gemini + Mongo ({chunks.count_documents({})} chunks) + S3 ready. "
      f"{len(eval_questions)} eval questions.")
