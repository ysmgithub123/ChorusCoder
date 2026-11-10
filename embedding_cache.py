import hashlib
import sqlite3
from pathlib import Path

import numpy as np


SQLITE_PARAMETER_LIMIT = 900


class SQLiteEmbeddingCache:
    """Persistent, lazy cache for fixed-size float32 embeddings."""

    def __init__(self, path, namespace):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        self.connection.commit()

    def key(self, text):
        payload = f"{self.namespace}\0{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get_many(self, keys):
        connection = self.connection
        if connection is None:
            raise RuntimeError("Embedding cache is closed")
        unique_keys = list(dict.fromkeys(keys))
        values = {}
        for start in range(0, len(unique_keys), SQLITE_PARAMETER_LIMIT):
            chunk = unique_keys[start : start + SQLITE_PARAMETER_LIMIT]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT cache_key, dimension, vector FROM embeddings "
                f"WHERE cache_key IN ({placeholders})",
                chunk,
            )
            for cache_key, dimension, payload in rows:
                vector = np.frombuffer(payload, dtype=np.float32)
                if vector.size != dimension:
                    raise ValueError(
                        f"Corrupt embedding cache entry {cache_key}: "
                        f"expected {dimension} values, found {vector.size}"
                    )
                values[cache_key] = vector.copy()
        return values

    def put_many(self, values):
        connection = self.connection
        if connection is None:
            raise RuntimeError("Embedding cache is closed")
        if not values:
            return

        def rows():
            for cache_key, vector in values.items():
                array = np.asarray(vector, dtype=np.float32).reshape(-1)
                yield cache_key, int(array.size), array.tobytes(order="C")

        connection.executemany(
            "INSERT OR REPLACE INTO embeddings(cache_key, dimension, vector) "
            "VALUES (?, ?, ?)",
            rows(),
        )
        connection.commit()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
