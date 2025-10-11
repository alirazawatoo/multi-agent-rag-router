from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import requests

from agents.graph import invoke_graph
from agents.llm import stream_answer_with_citations  # optional direct use if needed

# ---- Env and Config ----
load_dotenv()
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "data/chroma")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_NAME = "pwb_docs"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ---- Lazy singletons ----
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def get_chroma():
    client = chromadb.PersistentClient(path=str(Path(CHROMA_DB_DIR)))
    return client.get_or_create_collection(COLLECTION_NAME)

def embed_query(texts: List[str]):
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()
# ---- UI ----
st.set_page_config(page_title="PWB Agentic POC", layout="wide")
st.title("PWB Agentic POC — Knowledge & Actions (RAG Preview)")

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Top-K retrieved", 2, 8, 4)
    style = st.selectbox("Response style", ["detailed", "concise"], index=0)
    temperature = st.slider("Temperature", 0.0, 0.8, 0.2, 0.05)

if "history" not in st.session_state:
    st.session_state.history = []  # [{role, content}]
if "decision_log" not in st.session_state:
    st.session_state.decision_log = []  # list of strings for now

col1, col2 = st.columns([2, 1])

# Chat-style input
prompt = st.chat_input("Ask a question about finance, ops, onboarding, or platforms")
if prompt:
    # Run LangGraph pipeline with style/temperature controls
    st.session_state.decision_log = []  # reset per turn for clarity
    result = invoke_graph(query=prompt, top_k=top_k, style=style, temperature=temperature)

    # Chat transcript: show user message immediately
    st.session_state.history.append({"role": "user", "content": prompt})

    # Stream the assistant response for live typing feel
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""
        # Reuse retrieved contexts from result to drive streamed answering if available
        contexts = result.get("contexts", [])
        if contexts:
            for chunk in stream_answer_with_citations(
                prompt,
                contexts,
                temperature=temperature,
                style=style,
            ):
                streamed += chunk
                placeholder.markdown(streamed)
        else:
            # Fallback to the non-streamed answer if no contexts
            streamed = result.get("answer", "")
            placeholder.markdown(streamed)

    # Persist the final assistant message and logs
    st.session_state.history.append({"role": "assistant", "content": streamed})
    st.session_state.decision_log.extend(result.get("log", []))
    st.session_state["last_result"] = {**result, "answer": streamed}

with col1:
    st.subheader("Chat")
    if st.session_state.history:
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        st.info("Ask a question to get started.")

with col2:
    st.subheader("Decision Log")
    if st.session_state.decision_log:
        for step in st.session_state.decision_log:
            st.code(step)

    st.subheader("Retrieved Sources")
    last = st.session_state.get("last_result")
    if last and last.get("contexts"):
        for i, ctx in enumerate(last.get("contexts", []), start=1):
            meta = ctx.get("metadata", {})
            src = meta.get("filename", meta.get("source", ""))
            page = meta.get("page", "?")
            st.markdown(f"**[{i}] {src} (p{page}) — distance={ctx.get('distance', 0):.4f}**")
            st.caption(meta.get("title", ""))
            st.write(ctx.get("document", ""))

    # Action approval flow (stub)
    st.subheader("Proposed Action")
    last = st.session_state.get("last_result")
    proposed = (last or {}).get("proposed_action")
    if proposed and proposed.get("type") != "unknown":
        st.json(proposed)
        colA, colB = st.columns(2)
        with colA:
            approve = st.button("Approve Action", type="primary")
        with colB:
            deny = st.button("Deny Action")
        if approve:
            ops_url = os.environ.get("OPS_API_URL", "http://localhost:8001")
            operator = os.environ.get("OPERATOR_NAME", "demo_user")
            try:
                endpoint = {
                    "pause_item": "/pause_item",
                    "unpause_item": "/unpause_item",
                    "update_hours": "/update_hours",
                }.get(proposed.get("type"), "/unknown")
                payload = {**proposed, "operator": operator}
                r = requests.post(f"{ops_url}{endpoint}", json=payload, timeout=10)
                if r.ok:
                    st.success("Action executed and audited.")
                else:
                    st.error(f"Action failed: {r.status_code} {r.text}")
            except Exception as e:
                st.error(f"Action error: {e}")
    else:
        st.caption("No action proposed.")
