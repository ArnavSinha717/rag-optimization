"""RAG Retrieval core: embed the query, vector-search MongoDB, rerank,return top chunks"""

from google.genai import types
from app.config import(client,chunks,VECTOR_INDEX,VARIANT_ID,EMBED_MODEL,EMBED_DIMS,RETRIEVAL_N,FINAL_K,RERANKER_MODEL)
_reranker=None
def get_reranker():
    """Load the cross-encoder once and reuse it."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker=CrossEncoder(RERANKER_MODEL,max_length=512)
    return _reranker
    
def embed_query(question:str)-> list[float]:
    """Embed the question with Gemini. MUST match the model/dims/task_type
    used at ingestion, or the vectors live in different spaces."""
    result=client.models.embed_content(
        model=EMBED_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIMS,
        ),
    )
    return list(result.embeddings[0].values)

def vector_search(query_vector: list[float],n:int=RETRIEVAL_N)-> list[dict]:
    """Top-n most similar"""
    cursor=chunks.aggregate([
      {"$vectorSearch":{
          "index":VECTOR_INDEX,
          "path":"embedding",
          "queryVector":query_vector,
          "filter":{"config.variant_id":VARIANT_ID},
          "numCandidates":n*5,#Over-retrieve for reranking
          "limit":n,
      }},
      {"$project":{"_id":0,"text":1,"source":1, "score":{"$meta":"vectorSearchScore"}}},
    ])
    return list(cursor)

def rerank(question:str, hits:list[dict],k:int=FINAL_K)->list[dict]:
    """Using the cross encoder to rerank and keep the best k"""
    if not hits:
        return []
    ce=get_reranker()
    scores=ce.predict([(question,h["text"]) for h in hits])
    for h,s in zip(hits,scores):
        h["rerank_score"]=float(s)
    hits.sort(key=lambda h:h["rerank_score"],reverse=True)
    return hits[:k]

def retrieve(question:str)->list[dict]:
    """Full retrieval: embed->Vector search->Rerank->top-k"""
    qv=embed_query(question)
    candidates=vector_search(qv)
    return rerank(question,candidates)