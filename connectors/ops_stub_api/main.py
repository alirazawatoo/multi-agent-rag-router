from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "ops_state.json"
AUDIT_FILE = DATA_DIR / "audit.log"


class ActionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")  # accept flexible fields from UI

    type: str
    operator: str
    item: Optional[str] = None
    details: Optional[str] = None
    original_query: Optional[str] = None


class OpsState(BaseModel):
    model_config = ConfigDict(extra="allow")

    items: Dict[str, str] = {}  # item -> status ("paused"|"active")
    hours_note: str = ""


def load_state() -> OpsState:
    if STATE_FILE.exists():
        try:
            return OpsState(**json.loads(STATE_FILE.read_text()))
        except Exception:
            pass
    return OpsState()


def save_state(state: OpsState) -> None:
    STATE_FILE.write_text(state.model_dump_json(indent=2))


def append_audit(entry: Dict[str, Any]) -> None:
    entry["ts"] = datetime.utcnow().isoformat() + "Z"
    with AUDIT_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


app = FastAPI(title="Ops Stub API", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/pause_item")
def pause_item(payload: ActionPayload):
    if not payload.item:
        raise HTTPException(status_code=400, detail="Missing 'item' in payload")
    state = load_state()
    state.items[payload.item] = "paused"
    save_state(state)
    append_audit({"action": "pause_item", **payload.model_dump()})
    return {"ok": True, "message": f"Paused item '{payload.item}'", "state": state.model_dump()}


@app.post("/unpause_item")
def unpause_item(payload: ActionPayload):
    if not payload.item:
        raise HTTPException(status_code=400, detail="Missing 'item' in payload")
    state = load_state()
    state.items[payload.item] = "active"
    save_state(state)
    append_audit({"action": "unpause_item", **payload.model_dump()})
    return {"ok": True, "message": f"Unpaused item '{payload.item}'", "state": state.model_dump()}


@app.post("/update_hours")
def update_hours(payload: ActionPayload):
    state = load_state()
    note = payload.details or payload.original_query or "updated hours"
    state.hours_note = note
    save_state(state)
    append_audit({"action": "update_hours", **payload.model_dump()})
    return {"ok": True, "message": "Hours updated", "note": note, "state": state.model_dump()}


@app.get("/state")
def get_state():
    return load_state()


@app.get("/audit")
def get_audit():
    entries = []
    if AUDIT_FILE.exists():
        with AUDIT_FILE.open() as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return {"count": len(entries), "entries": entries}
