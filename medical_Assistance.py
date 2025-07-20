import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import numpy as np
import nest_asyncio

nest_asyncio.apply()
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
WORKING_DIR = "RAG_DataBase"

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, response_type="multi line", **kwargs
) -> str:
    client = genai.Client(api_key=gemini_api_key)
    if history_messages is None:
        history_messages = []
    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"
    for msg in history_messages:
        combined_prompt += f"{msg['role']}: {msg['content']}\n"
    combined_prompt += f"user: {prompt}\n"
    combined_prompt += f"Please answer in {response_type} format."
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

async def get_rag():
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

async def main():
    st.title("Doctor-Patient Medical Chat")
    st.markdown("#### Welcome to your medical assistant. Chat with the doctor below.")

    # Session state for chat history and rag
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag" not in st.session_state:
        st.session_state.rag = await get_rag()

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("patient"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("doctor"):
                st.markdown(msg["content"])

    # Chat input (no response type selector)
    user_input = st.chat_input("Describe your symptoms or ask a medical question...")

    response_type = "multi line"  # Always single line

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("patient"):
            st.markdown(user_input)

        with st.chat_message("doctor"):
            message_placeholder = st.empty()
            rag = st.session_state.rag
            response = rag.query(
                query=user_input,
                param=QueryParam(mode="mix", top_k=5, response_type=response_type),
            )
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "doctor", "content": response})

if __name__ == "__main__":
    asyncio.run(main())