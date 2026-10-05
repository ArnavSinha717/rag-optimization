"""This module is responsible for loading environment variables and initializing the GenAI client."""

import os
from dotenv import load_dotenv 
from google import genai
from pymongo import MongoClient
#Loading environment variables from .env file
load_dotenv()
#Initializing the Gemini client with the API key from environment variables
client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEN_MODEL=os.getenv("GEN_MODEL","gemini-2.5-flash")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

#--LLM provider: one constant picks who generates. Embeddings ALWAYS stay Gemini (must match index).
#  "ollama"  -> gemma4 local, free/offline fallback
#  "bedrock" -> model set by BEDROCK_MODEL (prod + build); needs AWS creds in env
#  "gemini"  -> gemini-2.5-flash
LLM_PROVIDER=os.getenv("LLM_PROVIDER","ollama")
LOCAL_MODEL=os.getenv("LOCAL_MODEL","gemma4:e4b-it-qat")
OLLAMA_BASE=os.getenv("OLLAMA_BASE","http://localhost:11434")
#  BEDROCK_MODEL is required when LLM_PROVIDER=bedrock: a Bedrock model or inference-profile ID.
BEDROCK_MODEL=os.getenv("BEDROCK_MODEL")
# Bedrock uses its OWN key AND its OWN region (both separate from the S3 key/region that are the
# env default) so the two never collide. If unset, we fall back to the default AWS chain.
# Keep secrets in .env, never in code.
BEDROCK_REGION=os.getenv("BEDROCK_REGION","us-east-1")
BEDROCK_AWS_KEY=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID")
BEDROCK_AWS_SECRET=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY")

#--Eval judge: kept SEPARATE from the agent model so we never self-grade (a model grading its own answers).
#  Default Gemini = independent cross-family judge for trustworthy RAGAS scores.
JUDGE_PROVIDER=os.getenv("JUDGE_PROVIDER","gemini")
JUDGE_MODEL=os.getenv("JUDGE_MODEL","gemini-2.5-flash")

#--MongoDB--
mongo=MongoClient(os.environ["MONGO_URI"])
chunks=mongo["rag"]["chunks"]

#--Retrival Settings--
VECTOR_INDEX="chunks_vector_index"
VARIANT_ID="v10_corpus50_recursive"
EMBED_MODEL='gemini-embedding-001'
EMBED_DIMS=768
RERANKER_MODEL="BAAI/bge-reranker-base"
RERANKER_DEVICE=os.getenv("RERANKER_DEVICE","cpu")
RETRIEVAL_N=20
FINAL_K=5
 
