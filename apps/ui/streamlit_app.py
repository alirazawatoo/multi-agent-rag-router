from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

from agents.graph import invoke_graph
from ingestion.vectorstore import get_embedder, open_collection

load_dotenv()
ENDPOINTS = {"pause_item": "/pause_item", "unpause_item": "/unpause_item", "update_hours": "/update_hours"}

st.set_page_config(page_title="Document routing agent", layout="wide")
st.title("Document questions and proposed actions")


embedder_name = get_embedder().name
provider = os.environ.get("LLM_PROVIDER", "stub")
st.caption(
    f"Embeddings: {embedder_name} · Answers: {provider} · "
    "Citation formatting is checked; factual grounding is not verified."
)

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Passages to retrieve", 2, 8, 4)
    style = st.selectbox("Response style", ["detailed", "concise"], index=0)
    temperature = st.slider("Temperature (Groq only)", 0.0, 0.8, 0.2, 0.05, disabled=provider != "groq")

st.session_state.setdefault("history", [])
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("action_outcome", None)

if open_collection(get_embedder()).count() == 0:
    st.warning(
        "No documents are indexed yet, so questions will return no sources. "
        "Run `python -m scripts.generate_pdfs` and `python -m ingestion.build_index`, then reload."
    )

prompt = st.chat_input("Ask about finance, menus, onboarding or platforms, or request a pause/hours change")
if prompt:
    with st.spinner("Routing and retrieving…"):
        result = invoke_graph(query=prompt, top_k=top_k, style=style, temperature=temperature)
    st.session_state.history += [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": result.get("answer", "")},
    ]
    st.session_state.last_result = result
    st.session_state.action_outcome = None

chat_col, side_col = st.columns([2, 1], gap="large")

with chat_col:
    st.subheader("Conversation")
    if not st.session_state.history:
        st.info(
            "Try “How do I re-send an invoice?” for a document answer, "
            "or “Pause 'Chicken Burger' until 6pm” for an action proposal."
        )
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

last = st.session_state.last_result or {}

with side_col:
    st.subheader("Workflow trace")
    if last.get("log"):
        st.code("\n".join(last["log"]), language=None)
    else:
        st.caption("The node-by-node path of the last request appears here.")

    if last.get("intent") == "knowledge":
        st.subheader("Retrieved sources")
        if not last.get("contexts"):
            st.caption("No passages were retrieved.")
        for i, ctx in enumerate(last.get("contexts", []), start=1):
            meta = ctx.get("metadata", {})
            label = f"[{i}] {meta.get('filename', '')} p{meta.get('page', '?')} · distance {ctx.get('distance', 0):.3f}"
            with st.expander(label):
                st.write(ctx.get("document", ""))

    if last.get("intent") == "action":
        st.subheader("Proposed action")
        proposed = last.get("proposed_action") or {}
        if proposed.get("type", "unknown") == "unknown":
            st.caption("Nothing actionable was extracted from the request.")
        else:
            st.json(proposed)
            outcome = st.session_state.action_outcome
            approve_col, deny_col = st.columns(2)
            approve = approve_col.button("Approve", type="primary", disabled=outcome is not None)
            deny = deny_col.button("Deny", disabled=outcome is not None)
            if approve:
                ops_url = os.environ.get("OPS_API_URL", "http://localhost:8001")
                payload = {**proposed, "operator": os.environ.get("OPERATOR_NAME", "demo_user")}
                try:
                    r = requests.post(f"{ops_url}{ENDPOINTS[proposed['type']]}", json=payload, timeout=10)
                    outcome = ("success", r.json()["message"]) if r.ok else ("error", f"{r.status_code}: {r.text}")
                except requests.RequestException as exc:
                    outcome = ("error", f"Could not reach the ops connector at {ops_url}: {exc}")
                st.session_state.action_outcome = outcome
                st.rerun()
            if deny:
                st.session_state.action_outcome = ("info", "Denied. Nothing was sent to the ops connector.")
                st.rerun()
            if outcome:
                kind, message = outcome
                getattr(st, kind)(message)
