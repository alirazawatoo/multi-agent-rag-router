from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


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


def data_dir() -> Path:
    return Path(os.environ.get("OPS_DATA_DIR", "data"))


def state_file() -> Path:
    return data_dir() / "ops_state.json"


def audit_file() -> Path:
    return data_dir() / "audit.log"


def load_state() -> OpsState:
    path = state_file()
    if not path.exists():
        return OpsState()
    return OpsState.model_validate_json(path.read_text())


def save_state(state: OpsState) -> None:
    data_dir().mkdir(parents=True, exist_ok=True)
    state_file().write_text(state.model_dump_json(indent=2))


def append_audit(entry: Dict[str, Any]) -> None:
    data_dir().mkdir(parents=True, exist_ok=True)
    entry["ts"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with audit_file().open("a") as f:
        f.write(json.dumps(entry) + "\n")


def require_type(payload: ActionPayload, expected: str) -> None:
    # The UI posts the proposal as-is; a mismatch means the wrong endpoint was chosen.
    if payload.type != expected:
        raise HTTPException(status_code=400, detail=f"Payload type '{payload.type}' does not match /{expected}")


app = FastAPI(title="Ops Stub API", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/pause_item")
def pause_item(payload: ActionPayload):
    require_type(payload, "pause_item")
    if not payload.item:
        raise HTTPException(status_code=400, detail="Missing 'item' in payload")
    state = load_state()
    state.items[payload.item] = "paused"
    save_state(state)
    append_audit({"action": "pause_item", **payload.model_dump()})
    return {"ok": True, "message": f"Paused item '{payload.item}'", "state": state.model_dump()}


@app.post("/unpause_item")
def unpause_item(payload: ActionPayload):
    require_type(payload, "unpause_item")
    if not payload.item:
        raise HTTPException(status_code=400, detail="Missing 'item' in payload")
    state = load_state()
    state.items[payload.item] = "active"
    save_state(state)
    append_audit({"action": "unpause_item", **payload.model_dump()})
    return {"ok": True, "message": f"Unpaused item '{payload.item}'", "state": state.model_dump()}


@app.post("/update_hours")
def update_hours(payload: ActionPayload):
    require_type(payload, "update_hours")
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
    path = audit_file()
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return {"count": len(entries), "entries": entries}
