import os
import sys
import argparse
from dataclasses import dataclass
import asyncio
import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
import asyncio
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Load environment variables from .env file
dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

WORKING_DIR = "RAG_DataBase"


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    client = genai.Client(api_key=gemini_api_key)
    if history_messages is None:
        history_messages = []
    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"
    for msg in history_messages:
        combined_prompt += f"{msg['role']}: {msg['content']}\n"
    combined_prompt += f"user: {prompt}"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[combined_prompt],
        config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1),
    )
    return response.text


async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings



async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    return rag



def main():
    rag = asyncio.run(initialize_rag())
    user_query = input("Ask your medical question: ")
    response = rag.query(
        query=user_query,
        param=QueryParam(mode="mix", top_k=5, response_type="single line"),
    )
    print("\nAnswer:\n", response)

if __name__ == "__main__":
    main()