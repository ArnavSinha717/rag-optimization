from app.pipeline import retrieve, get_reranker

hits = retrieve("Which state's law governs the Chase Affiliate Agreement?")
for h in hits:
    print(f"[{h['rerank_score']:+.2f}] {h['source']['doc_id'][:40]} :: {h['text'][:75].strip()}")

# also verify bug 4's fix: second call must NOT return None
assert get_reranker() is not None, "singleton broke on 2nd call"
print("\nsecond get_reranker() call OK (singleton works)")
