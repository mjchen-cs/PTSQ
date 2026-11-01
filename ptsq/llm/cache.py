"""Persistent response cache (SQLite).

Every LLM response (device, edge and cloud) is cached under a key derived from the
backend identity and the exact request. Re-running an experiment therefore costs nothing
and is reproducible even if a commercial API model is updated later. Ship the cache file
with the artifact.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


class ResponseCache:
    def __init__(self, path: str | Path | None):
        self.path = str(path) if path else ":memory:"
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._db.commit()

    @staticmethod
    def key(backend_id: str, payload: dict) -> str:
        blob = json.dumps({"b": backend_id, "p": payload}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, k: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)",
                             (k, json.dumps(v, ensure_ascii=False)))
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
