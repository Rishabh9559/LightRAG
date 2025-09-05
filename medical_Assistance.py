import os
import sys
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
from PIL import Image


# Windows fix for event loop
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

nest_asyncio.apply()

# Load environment
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
WORKING_DIR = "RAG_DataBase"

# Load avatars
doctor_avatar = Image.open("images/doctor.png")
patient_avatar = Image.open("images/image.png")

# Gemini LLM function
async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, response_type="single line", **kwargs) -> str:
    client = genai.Client(api_key=gemini_api_key)
    if prompt.lower().strip() in ["who are you", "who are you?", "what are you?", "identify yourself"]:
        return "I am your personal AI doctor."
    if history_messages is None:
        history_messages = []
    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"
    for msg in history_messages:
        combined_prompt += f"{msg['role']}: {msg['content']}\n"
    combined_prompt += f"user: {prompt}\n"
    # combined_prompt += f"Please answer in {response_type} format."
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[combined_prompt],
        config=types.GenerateContentConfig(max_output_tokens=500, temperature=1),
    )
    return response.text

# Cache the embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Embedding function
async def embedding_func(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True)

# Initialize LightRAG
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

# Render message bubble

def render_message(content, sender="doctor"):
    if sender == "doctor":
        bg_color = "#e0f7fa"
        text_color = "#003333"
        align = "left"
        border_radius = "20px 20px 20px 0"
    else:
        bg_color = "#e8f5e9"
        text_color = "#003300"
        align = "right"
        border_radius = "20px 20px 0 20px"

    # Escape any raw HTML but allow markdown (like *, _, etc.)
    # sanitized = html.escape(content)
    sanitized=content

    st.markdown(
        f"""
        <div style='
            background-color: {bg_color};
            color: {text_color};
            padding: 12px 16px;
            border-radius: {border_radius};
            margin: 8px 0;
            text-align: left;
            max-width: 80%;
            float: {align};
            clear: both;
            word-wrap: break-word;
        '>
            {sanitized}
        
        """,
        unsafe_allow_html=True,
    )



def main():
    st.title("🩺 Doctor-Patient Medical Chat")
    st.markdown("#### Welcome to your medical assistant. Chat with the doctor below.")


    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag" not in st.session_state:
        st.session_state.rag = asyncio.run(get_rag())

    # Display message history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("patient", avatar=patient_avatar):
                render_message(msg["content"], sender="patient")
        else:
            with st.chat_message("doctor", avatar=doctor_avatar):
                render_message(msg["content"], sender="doctor")

    # Input
    user_input = st.chat_input("Describe your symptoms or ask a medical question...")
    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("patient", avatar=patient_avatar):
            render_message(user_input, sender="patient")

        # Get response
        rag = st.session_state.rag
        response = rag.query(
            query=user_input,
            param=QueryParam(mode="mix", top_k=5, response_type="single line"), # resopnse type change multi or single line
        )

        # Show doctor reply
        with st.chat_message("doctor", avatar=doctor_avatar):
            render_message(response, sender="doctor")
        st.session_state.messages.append({"role": "doctor", "content": response})

# Entry point
if __name__ == "__main__":
    main()
