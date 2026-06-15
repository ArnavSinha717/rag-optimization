"""Ingest the 50-contract LOCAL corpus (data/corpus50/) as variant v10_corpus50_recursive.
Local PDFs -> pdfplumber -> recursive 2000/200 -> Gemini embed (paced) -> Mongo.
Checkpointed per-document: re-running skips docs already inserted."""
import sys, os, re, time
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import chunks, parse_pdf_pdfplumber, chunk_recursive, embed_with_retry, MODEL, DIMS
from datetime import datetime, timezone

V = "v10_corpus50_recursive"
SRC = "data/corpus50"

def slug(name):
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()
    return s[:80] or "untitled"

done_docs = set(chunks.distinct("source.doc_id", {"config.variant_id": V}))
print(f"{V}: {len(done_docs)} docs already ingested (resume mode)")

now = datetime.now(timezone.utc)
files = sorted(f for f in os.listdir(SRC) if f.lower().endswith(".pdf"))
total = 0
for fname in files:
    stem = os.path.splitext(fname)[0]
    doc_id = slug(stem)
    if doc_id in done_docs:
        continue
    try:
        text = parse_pdf_pdfplumber(open(f"{SRC}/{fname}", "rb").read())
        raw = chunk_recursive(text, 2000, 200)
        if not raw:
            print(f"  {doc_id[:55]:<55} SKIP (no text)"); continue
        vecs = []
        for i in range(0, len(raw), 25):
            vecs.extend(embed_with_retry([c["text"] for c in raw[i:i+25]], "RETRIEVAL_DOCUMENT"))
            time.sleep(5)
        docs = [{
            "text": c["text"], "embedding": v,
            "source": {"doc_id": doc_id, "doc_title": stem, "source_uri": f"local://{SRC}/{fname}",
                       "chunk_index": i, "char_start": c["char_start"], "char_end": c["char_end"]},
            "config": {"variant_id": V, "pdf_parser": "pdfplumber", "chunk_strategy": "recursive_2000_200",
                       "embedding_model": MODEL, "embedding_dims": DIMS,
                       "embedding_task_type": "RETRIEVAL_DOCUMENT"},
            "created_at": now,
        } for i, (c, v) in enumerate(zip(raw, vecs))]
        chunks.insert_many(docs)
        total += len(docs)
        print(f"  {doc_id[:55]:<55} {len(docs):>3} chunks")
    except Exception as e:
        print(f"  {doc_id[:55]:<55} ERROR {type(e).__name__}: {e}")
print(f"\nDONE: inserted {total} new chunks; variant total = {chunks.count_documents({'config.variant_id': V})}")
