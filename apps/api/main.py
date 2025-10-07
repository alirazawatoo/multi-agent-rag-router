"""HTTP front door to the routing graph.

``POST /ask`` runs one request through the graph and returns the answer, the retrieved
sources and, for action requests, the proposed action. It never executes an action:
approval stays a separate, explicit call to the ops connector.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agents.graph import invoke_graph
from ingestion.vectorstore import get_embedder, open_collection

app = FastAPI(title="Document routing agent", version="0.2.0")


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    style: Literal["detailed", "concise"] = "detailed"


class Source(BaseModel):
    filename: str
    page: int
    distance: float
    excerpt: str


class AskResponse(BaseModel):
    intent: str
    answer: str
    citation_shape_present: Optional[bool] = None
    sources: List[Source] = []
    proposed_action: Optional[Dict] = None
    trace: List[str]


@app.get("/health")
def health():
    collection = open_collection(get_embedder())
    return {"ok": True, "indexed_chunks": collection.count()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = invoke_graph(request.query, top_k=request.top_k, style=request.style)
    sources = [
        Source(
            filename=str(c["metadata"].get("filename", "")),
            page=int(c["metadata"].get("page", 0)),
            distance=round(float(c["distance"]), 4),
            excerpt=c["document"][:240],
        )
        for c in result.get("contexts", [])
    ]
    return AskResponse(
        intent=result["intent"],
        answer=result.get("answer", ""),
        citation_shape_present=result.get("citation_shape_present"),
        sources=sources,
        proposed_action=result.get("proposed_action"),
        trace=result.get("log", []),
    )
