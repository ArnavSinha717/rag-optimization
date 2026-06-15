"""TEST 3 — Contextual Retrieval (Anthropic-style): prepend a 7B-generated
situating blurb to each chunk, re-embed, store as new variants.
Usage: python eval/contextual_ingest.py v04 | v07
Phased: blurbs via 7B (checkpointed) -> unload 7B -> Gemini embed -> insert."""
import sys, os, json, subprocess, time
sys.modules.pop("rag_lib", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lib import chunks, ollama_chat, embed_with_retry, MODEL, DIMS
from datetime import datetime, timezone

BASE = {"v04": "v04_pdfplumber_recursive", "v07": "v07_pymupdf_markdown"}[sys.argv[1]]
NEW = {"v04": "v08_ctx_recursive", "v07": "v09_ctx_markdown"}[sys.argv[1]]
BLURB_PATH = f"results/blurbs_{NEW}.json"

BLURB_PROMPT = """This chunk is from the contract titled: "{title}"

Chunk:
{chunk}

Write ONE short sentence (max 25 words) situating this chunk: name the contract/parties and what this part covers. Output ONLY that sentence."""

src = list(chunks.find({"config.variant_id": BASE},
                       {"text": 1, "source": 1, "config": 1}).sort([("source.doc_id", 1), ("source.chunk_index", 1)]))
print(f"Base {BASE}: {len(src)} chunks -> new variant {NEW}")

# ---- Phase 1: blurbs via 7B (checkpointed by index) ----
blurbs = json.load(open(BLURB_PATH)) if os.path.exists(BLURB_PATH) else {}
for i, c in enumerate(src):
    key = str(i)
    if key in blurbs:
        continue
    blurbs[key] = ollama_chat("", BLURB_PROMPT.format(
        title=c["source"]["doc_title"], chunk=c["text"][:1500]),
        temperature=0.1, model="qwen2.5:7b-instruct").strip()
    if i % 10 == 0:
        json.dump(blurbs, open(BLURB_PATH, "w"))
        print(f"  blurbs {i+1}/{len(src)}")
json.dump(blurbs, open(BLURB_PATH, "w"))
print(f"Blurbs done: {len(blurbs)}")

subprocess.run(["ollama", "stop", "qwen2.5:7b-instruct"], capture_output=True)

# ---- Phase 2: embed contextualized text (Gemini, batched) + insert ----
cleared = chunks.delete_many({"config.variant_id": NEW}).deleted_count
print(f"Cleared {cleared} old {NEW} chunks")
now = datetime.now(timezone.utc)
B = 25
for start in range(0, len(src), B):
    batch = src[start:start + B]
    texts = [blurbs[str(start + j)] + "\n\n" + c["text"] for j, c in enumerate(batch)]
    vecs = embed_with_retry(texts, "RETRIEVAL_DOCUMENT")
    docs = []
    for j, (c, v, t) in enumerate(zip(batch, vecs, texts)):
        docs.append({
            "text": t,                      # blurb + original text (what the LLM will see)
            "embedding": v,
            "source": c["source"],
            "config": {**c["config"], "variant_id": NEW, "contextual": True},
            "created_at": now,
        })
    chunks.insert_many(docs)
    print(f"  embedded+inserted {min(start+B, len(src))}/{len(src)}")
    time.sleep(5)
print(f"DONE {NEW}: {chunks.count_documents({'config.variant_id': NEW})} chunks")
