"""
rag.py: the "brain" of the Flowly Help Bot.

RAG = Retrieval-Augmented Generation:
  1. RETRIEVE: find the 3 document chunks closest in meaning to the question.
  2. AUGMENT:  put those chunks into the prompt.
  3. GENERATE: ask Gemini to answer ONLY from them (or say "I don't know").
"""

import json
import os
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding   # small ONNX embedding engine, no torch, fits in ~1 GB RAM

# ---------- Settings ----------
INDEX_FILE = Path(__file__).parent / "index.json"     # chunks + their embeddings, built in advance
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # SAME model used to build index.json
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")  # current free-tier Flash model
TOP_K = 3

# ---------- Load the index once ----------
CHUNKS = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
EMBEDDINGS = np.array([c["embedding"] for c in CHUNKS], dtype=np.float32)  # one row of 384 numbers per chunk
_embedder = None  # created on first use (downloads ~90 MB once)


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder


# ---------- 1. Retrieve ----------
def search(question, top_k=TOP_K):
    """Turn the question into numbers and return the closest chunks with their scores."""
    q = np.array(list(get_embedder().embed([question]))[0], dtype=np.float32)
    q = q / np.linalg.norm(q)                 # length 1, so dot product = cosine similarity
    scores = EMBEDDINGS @ q                   # one similarity score per chunk (higher = closer)
    best = np.argsort(scores)[::-1][:top_k]   # positions of the highest scores
    return [{**CHUNKS[i], "score": float(scores[i])} for i in best]


# ---------- 2. Augment: build the prompt ----------
RULES = (
    "You are Flowly's internal help bot. Answer using ONLY the context below. "
    "Be short (1-3 sentences) and end with the source file in brackets, e.g. [product_faq.md]. "
    "If the context does not contain the answer, reply exactly: I don't know. "
    "Never guess and never use outside knowledge."
)


def build_prompt(question, chunks):
    context = "\n\n".join(f"--- Source: {c['source']} ---\n{c['text']}" for c in chunks)
    return f"{RULES}\n\nContext:\n{context}\n\nQuestion: {question}"


# ---------- 3. Generate ----------
def get_api_key():
    """Read the key from Streamlit secrets (on the cloud) or an environment variable (locally)."""
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def call_gemini(prompt):
    from google import genai             # official current SDK: pip install google-genai
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()


def answer(question, llm=call_gemini):
    """Full RAG pipeline. `llm` can be swapped for a fake one when testing without a key."""
    chunks = search(question)
    return llm(build_prompt(question, chunks)), chunks
