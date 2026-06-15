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

#--MongoDB--
mongo=MongoClient(os.environ["MONGO_URI"])
chunks=mongo["rag"]["chunks"]

#--Retrival Settings--
VECTOR_INDEX="chunks_vector_index"
VARIANT_ID="v10_corpus50_recursive"
EMBED_MODEL='gemini-embedding-001'
EMBED_DIMS=768
RERANKER_MODEL="BAAI/bge-reranker-base"
RETRIEVAL_N=20
FINAL_K=5
