"""
core/storage.py — Lightweight local persistence.

Investigation history is stored as JSON-lines so it survives app restarts
(no external DB required). Generated PDF/DOCX reports are written to
data/reports/ and indexed the same way.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from core.config import DATA_DIR

HISTORY_FILE = DATA_DIR / "investigation_history.jsonl"
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def log_investigation(record: dict[str, Any]) -> str:
    record = {"id": uuid.uuid4().hex[:10], "timestamp": time.time(), **record}
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record["id"]


def load_history(limit: int = 200) -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    lines = HISTORY_FILE.read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return list(reversed(records))[:limit]


def save_report_file(filename: str, content: bytes) -> Path:
    path = REPORTS_DIR / filename
    path.write_bytes(content)
    return path


def list_report_files() -> list[Path]:
    return sorted(REPORTS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
