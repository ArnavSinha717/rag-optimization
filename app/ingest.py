"""Ingest a user-uploaded document into the SAME vector collection/variant the agent searches, so
it's queryable immediately — no S3, no re-running the offline pipeline. Mirrors rag_lib.ingest_variant
exactly (recursive 2000/200 chunks, Gemini RETRIEVAL_DOCUMENT embeddings, identical chunk schema) but
reads bytes straight from the upload instead of fetching from S3."""
import io
import time
from datetime import datetime, timezone

from google.genai import types

from app.config import client, chunks, VARIANT_ID, EMBED_MODEL, EMBED_DIMS
from app.tools import invalidate_contract_cache

# Must match the corpus the index was built on (rag_lib CHUNKERS["recursive_2000_200"]).
CHUNK_SIZE, CHUNK_OVERLAP = 2000, 200


def _extract_text(data: bytes, filename: str) -> str:
    """PDF -> text via pdfplumber; .txt/.md decoded directly. Same parser family as ingestion."""
    if filename.lower().endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return data.decode("utf-8", errors="ignore")


def _chunk(text: str) -> list[dict]:
    """Recursive/boundary-aware split — identical settings to the offline variant."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""], add_start_index=True,
    )
    out = []
    for d in splitter.create_documents([text]):
        t = d.page_content.strip()
        if t:
            start = d.metadata.get("start_index", 0)
            out.append({"text": t, "char_start": start, "char_end": start + len(d.page_content)})
    return out


def _embed(texts: list[str], batch: int = 25) -> list[list[float]]:
    """Gemini RETRIEVAL_DOCUMENT embeddings — MUST match the query side and the index dims."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch):
        res = client.models.embed_content(
            model=EMBED_MODEL, contents=texts[i: i + batch],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT", output_dimensionality=EMBED_DIMS),
        )
        vectors.extend(list(e.values) for e in res.embeddings)
        if i + batch < len(texts):
            time.sleep(1)
    return vectors


def ingest_upload(data: bytes, filename: str) -> dict:
    """Parse -> chunk -> embed -> upsert into the live collection under the searched variant.
    Re-uploading the same filename replaces its old chunks. Returns a small summary for the UI."""
    title = filename.rsplit(".", 1)[0]          # doc_id == title == filename stem (matches corpus style)
    text = _extract_text(data, filename)
    raw = _chunk(text)
    if not raw:
        return {"ok": False, "doc_id": title, "chunks": 0, "reason": "no extractable text"}

    vectors = _embed([c["text"] for c in raw])
    now = datetime.now(timezone.utc)
    docs = [{
        "text": c["text"], "embedding": v,
        "source": {"doc_id": title, "doc_title": title, "source_uri": f"upload://{filename}",
                   "chunk_index": i, "char_start": c["char_start"], "char_end": c["char_end"]},
        "config": {"variant_id": VARIANT_ID, "pdf_parser": "upload", "chunk_strategy": "recursive_2000_200",
                   "embedding_model": EMBED_MODEL, "embedding_dims": EMBED_DIMS,
                   "embedding_task_type": "RETRIEVAL_DOCUMENT"},
        "created_at": now,
    } for i, (c, v) in enumerate(zip(raw, vectors))]

    replaced = chunks.delete_many({"config.variant_id": VARIANT_ID, "source.doc_id": title}).deleted_count
    chunks.insert_many(docs)
    invalidate_contract_cache()                 # so list_contracts / name-resolution see it now
    return {"ok": True, "doc_id": title, "chunks": len(docs), "replaced": replaced}
