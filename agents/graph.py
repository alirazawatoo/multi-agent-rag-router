from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

from langgraph.graph import StateGraph, START, END

from agents.llm import answer_with_citations

# ---- Env ----
load_dotenv()
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "data/chroma")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_NAME = "pwb_docs"


# ---- Caches ----
_embedder: Optional[SentenceTransformer] = None
_collection = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(Path(CHROMA_DB_DIR)))
        _collection = client.get_or_create_collection(COLLECTION_NAME)
    return _collection


# ---- State ----
class AgentState(TypedDict, total=False):
    query: str
    intent: str
    top_k: int
    style: str
    temperature: float
    contexts: List[Dict]
    answer: str
    proposed_action: Dict
    log: List[str]


# ---- Nodes ----
def router_node(state: AgentState) -> AgentState:
    q = (state.get("query") or "").lower()
    log = state.get("log", [])

    action_keywords = ["pause", "unpause", "resume", "update hours", "change hours", "set hours"]
    if any(k in q for k in action_keywords):
        intent = "action"
    else:
        intent = "knowledge"

    log.append(f"Router → {intent.title()}")
    state["intent"] = intent
    state["log"] = log
    return state


def retriever_node(state: AgentState) -> AgentState:
    top_k = state.get("top_k", 4)
    query = state["query"]
    model = get_embedder()
    coll = get_collection()

    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    res = coll.query(query_embeddings=q_emb, n_results=top_k, include=["distances", "metadatas", "documents"])  # type: ignore

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    contexts: List[Dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        contexts.append({"document": doc, "metadata": meta, "distance": dist})

    log = state.get("log", [])
    log.append("Retriever → Compose")

    state["contexts"] = contexts
    state["log"] = log
    return state


def compose_node(state: AgentState) -> AgentState:
    query = state["query"]
    contexts = state.get("contexts", [])
    style = state.get("style", "detailed")
    temp = state.get("temperature", 0.2)
    answer = answer_with_citations(query, contexts, temperature=temp, style=style)

    log = state.get("log", [])
    log.append("Compose → Critic")

    state["answer"] = answer
    state["log"] = log
    return state


def critic_node(state: AgentState) -> AgentState:
    # Simple grounding check: require at least one [filename pX] pattern
    answer = state.get("answer", "")
    grounded = bool(re.search(r"\[[^\]]+ p\d+\]", answer))

    log = state.get("log", [])
    log.append(f"Critic → END (grounded={grounded})")
    state["log"] = log
    return state


# --- Action extraction ---
action_item_re = re.compile(r"(pause|unpause|resume)\s+'?\"?([^'\"]+?)'?\"?\b", re.IGNORECASE)
action_hours_re = re.compile(r"(update|set|change)\s+(opening\s+)?hours(.*)", re.IGNORECASE)


def action_node(state: AgentState) -> AgentState:
    q = state.get("query", "")
    proposed: Dict = {"type": "unknown", "original_query": q}

    m_item = action_item_re.search(q)
    if m_item:
        verb = m_item.group(1).lower()
        item = m_item.group(2).strip()
        if verb == "pause":
            proposed["type"] = "pause_item"
        else:
            proposed["type"] = "unpause_item"
        proposed["item"] = item
    else:
        m_hours = action_hours_re.search(q)
        if m_hours:
            proposed["type"] = "update_hours"
            proposed["details"] = (m_hours.group(0)).strip()

    log = state.get("log", [])
    log.append(f"Action → END (proposed={proposed.get('type')})")

    state["proposed_action"] = proposed
    state["answer"] = (
        f"I identified an action request: {proposed.get('type')}. "
        "Review the proposed parameters below and approve to execute."
    )
    state["log"] = log
    return state


# ---- Graph compile ----
_graph = None

def get_graph():
    global _graph
    if _graph is not None:
        return _graph

    sg = StateGraph(AgentState)
    sg.add_node("router", router_node)
    sg.add_node("retriever", retriever_node)
    sg.add_node("compose", compose_node)
    sg.add_node("critic", critic_node)
    sg.add_node("action", action_node)

    sg.add_edge(START, "router")

    # Conditional branching from router
    def route_selector(state: AgentState):
        return state.get("intent", "knowledge")

    sg.add_conditional_edges(
        "router",
        route_selector,
        {
            "knowledge": "retriever",
            "action": "action",
        },
    )

    # Knowledge path
    sg.add_edge("retriever", "compose")
    sg.add_edge("compose", "critic")
    sg.add_edge("critic", END)

    # Action path ends
    sg.add_edge("action", END)

    _graph = sg.compile()
    return _graph


def invoke_graph(query: str, top_k: int = 4, style: str | None = None, temperature: float = 0.2) -> AgentState:
    graph = get_graph()
    init: AgentState = {"query": query, "top_k": top_k, "log": []}
    if style:
        init["style"] = style
    init["temperature"] = temperature
    result: AgentState = graph.invoke(init)  # type: ignore
    return result
