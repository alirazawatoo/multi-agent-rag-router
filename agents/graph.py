from __future__ import annotations

import re
from typing import Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agents.llm import answer_with_citations
from ingestion.vectorstore import get_embedder, open_collection

load_dotenv()


class AgentState(TypedDict, total=False):
    query: str
    intent: str
    top_k: int
    style: str
    temperature: float
    contexts: List[Dict]
    answer: str
    citation_shape_present: bool
    proposed_action: Dict
    log: List[str]


# ---- Routing ----
_ACTION_RE = re.compile(
    r"\b(?:un)?pause\b|\bresume\b|\b(?:update|set|change)\s+(?:the\s+)?(?:opening\s+)?hours\b", re.IGNORECASE
)
# "How do I pause an item?" asks for the procedure; "Can you pause X?" asks for the action.
_POLITE_REQUEST_RE = re.compile(r"^(?:please|can you|could you|would you|will you)\b", re.IGNORECASE)
_INTERROGATIVES = {"how", "what", "why", "when", "where", "which", "who", "is", "are", "do", "does", "should", "can"}


def classify_intent(query: str) -> str:
    q = query.strip()
    if not _ACTION_RE.search(q):
        return "knowledge"
    if _POLITE_REQUEST_RE.match(q):
        return "action"
    first_word = q.split(maxsplit=1)[0].lower() if q else ""
    if q.endswith("?") or first_word in _INTERROGATIVES:
        return "knowledge"
    return "action"


# ---- Action extraction ----
_ITEM_RE = re.compile(
    r"\b(unpause|pause|resume)\s+(?:the\s+)?(?:item\s+)?(?:\"([^\"]+)\"|'([^']+)'|(.+))", re.IGNORECASE
)
# Trailing qualifiers such as "until 5pm" are not part of the item name.
_ITEM_TAIL_RE = re.compile(r"\s+(?:for|until|till|because|due to|on|at)\b.*$|[.,;!?]+.*$", re.IGNORECASE)
_HOURS_RE = re.compile(r"\b(update|set|change)\s+(?:the\s+)?(?:opening\s+)?hours\b.*", re.IGNORECASE)


def parse_action(query: str) -> Dict:
    proposed: Dict = {"type": "unknown", "original_query": query}
    m_item = _ITEM_RE.search(query)
    if m_item:
        verb = m_item.group(1).lower()
        quoted = m_item.group(2) or m_item.group(3)
        item = quoted if quoted else _ITEM_TAIL_RE.sub("", m_item.group(4))
        item = re.sub(r"\s+item$", "", item.strip(), flags=re.IGNORECASE)
        if item:
            proposed["type"] = "pause_item" if verb == "pause" else "unpause_item"
            proposed["item"] = item
        return proposed
    m_hours = _HOURS_RE.search(query)
    if m_hours:
        proposed["type"] = "update_hours"
        proposed["details"] = m_hours.group(0).strip().rstrip(".!")
    return proposed


def has_citation_shape(answer: str) -> bool:
    return bool(re.search(r"\[[^\]]+ p\d+\]", answer))


# ---- Nodes ----
def router_node(state: AgentState) -> AgentState:
    intent = classify_intent(state.get("query") or "")
    return {"intent": intent, "log": [*state.get("log", []), f"Router → {intent.title()}"]}


def retriever_node(state: AgentState) -> AgentState:
    embedder = get_embedder()
    collection = open_collection(embedder)
    contexts: List[Dict] = []
    if collection.count():
        res = collection.query(
            query_embeddings=embedder.embed([state["query"]]),
            n_results=min(state.get("top_k", 4), collection.count()),
            include=["distances", "metadatas", "documents"],
        )
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            contexts.append({"document": doc, "metadata": meta, "distance": dist})
    return {"contexts": contexts, "log": [*state.get("log", []), f"Retriever → Compose ({len(contexts)} passages)"]}


def compose_node(state: AgentState) -> AgentState:
    answer = answer_with_citations(
        state["query"],
        state.get("contexts", []),
        temperature=state.get("temperature", 0.2),
        style=state.get("style", "detailed"),
    )
    return {"answer": answer, "log": [*state.get("log", []), "Compose → Critic"]}


def critic_node(state: AgentState) -> AgentState:
    # Citation-shape check only; this does not verify source identity or factual support.
    present = has_citation_shape(state.get("answer", ""))
    return {
        "citation_shape_present": present,
        "log": [*state.get("log", []), f"Critic → END (citation_shape_present={present})"],
    }


def action_node(state: AgentState) -> AgentState:
    proposed = parse_action(state.get("query", ""))
    if proposed["type"] == "unknown":
        answer = "This looks like an action request, but no item or opening-hours change could be extracted."
    else:
        answer = (
            f"I identified an action request: {proposed['type']}. "
            "Review the proposed parameters below and approve to execute."
        )
    return {
        "proposed_action": proposed,
        "answer": answer,
        "log": [*state.get("log", []), f"Action → END (proposed={proposed['type']})"],
    }


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
    sg.add_conditional_edges(
        "router",
        lambda state: state.get("intent", "knowledge"),
        {"knowledge": "retriever", "action": "action"},
    )
    sg.add_edge("retriever", "compose")
    sg.add_edge("compose", "critic")
    sg.add_edge("critic", END)
    sg.add_edge("action", END)

    _graph = sg.compile()
    return _graph


def invoke_graph(query: str, top_k: int = 4, style: str | None = None, temperature: float = 0.2) -> AgentState:
    init: AgentState = {"query": query, "top_k": top_k, "log": [], "temperature": temperature}
    if style:
        init["style"] = style
    return get_graph().invoke(init)  # type: ignore[return-value]
