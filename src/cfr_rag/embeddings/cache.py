"""SQLite-backed embedding cache keyed by a hash of (model, max_length, kind, text)."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np


def cache_key(model: str, max_length: int, kind: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x1f{max_length}\x1f{kind}\x1f{text}".encode()).hexdigest()


class EmbeddingCache:
    """Stores full-dimension float32 vectors; projection to a smaller dim happens on read."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")

    @classmethod
    def for_model(cls, cache_dir: str | Path, model_name: str) -> EmbeddingCache:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
        return cls(Path(cache_dir) / f"embeddings-{safe}.sqlite")

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 900):
            batch = keys[i:i + 900]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})"
            for key, blob in self.conn.execute(q, batch):
                out[key] = np.frombuffer(blob, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        rows = [(k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()]
        self.conn.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", rows)
        self.conn.commit()

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM emb").fetchone()[0]
