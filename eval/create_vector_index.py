"""(Re)create the Atlas Vector Search index the app queries — needed after a cluster rebuild.

Matches what the code filters on: vector field `embedding` (768-dim cosine) + filter fields
`config.variant_id` (pipeline.py / every eval) and `source.doc_id` (tools.py scoped search).
Idempotent: skips if an index of the same name already exists. Polls until queryable.

  PYTHONPATH=. .venv/bin/python eval/create_vector_index.py
"""
import time
from pymongo.operations import SearchIndexModel
from app.config import chunks, VECTOR_INDEX, EMBED_DIMS

DEFN = {
    "fields": [
        {"type": "vector", "path": "embedding", "numDimensions": EMBED_DIMS, "similarity": "cosine"},
        {"type": "filter", "path": "config.variant_id"},
        {"type": "filter", "path": "source.doc_id"},
    ]
}

existing = {ix["name"] for ix in chunks.list_search_indexes()}
if VECTOR_INDEX in existing:
    print(f"index '{VECTOR_INDEX}' already exists — nothing to do.")
else:
    chunks.create_search_index(
        SearchIndexModel(definition=DEFN, name=VECTOR_INDEX, type="vectorSearch"))
    print(f"created '{VECTOR_INDEX}' ({EMBED_DIMS}-dim cosine + filters). Waiting for build...")
    for _ in range(60):                       # up to ~5 min
        ix = next((i for i in chunks.list_search_indexes() if i["name"] == VECTOR_INDEX), None)
        if ix and ix.get("queryable"):
            print("index is QUERYABLE ✅"); break
        time.sleep(5)
    else:
        print("still building — check Atlas UI; queries will work once it's queryable.")
