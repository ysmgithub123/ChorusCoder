"""Transactional RLRetriever vectors with per-entry checksums and no overwrites."""
import hashlib

import numpy as np

from embedding_cache import SQLiteEmbeddingCache


class CheckedVectorCache(SQLiteEmbeddingCache):
    def __init__(self, path, namespace):
        super().__init__(path, namespace)
        self.connection.execute('CREATE TABLE IF NOT EXISTS checksums (cache_key TEXT PRIMARY KEY, checksum TEXT NOT NULL)')
        self.connection.commit()

    def checksum(self, key, payload):
        return hashlib.sha256((self.namespace + '\0' + key + '\0').encode() + payload).hexdigest()

    def get_many(self, keys):
        values = super().get_many(keys)
        for key, vector in values.items():
            row = self.connection.execute('SELECT checksum FROM checksums WHERE cache_key=?', (key,)).fetchone()
            if row is None or row[0] != self.checksum(key, vector.tobytes()):
                raise ValueError(f'RLCoder vector checksum mismatch: {key}')
        return values

    def put_many(self, values):
        # A vector and its checksum become visible in the same SQLite transaction.
        existing = self.get_many(values)
        with self.connection:
            for key, vector in values.items():
                array = np.asarray(vector, dtype=np.float32).reshape(-1)
                if key in existing:
                    if not np.array_equal(existing[key], array):
                        raise ValueError('Refusing to overwrite an existing RLCoder vector')
                    continue
                payload = array.tobytes()
                self.connection.execute('INSERT INTO embeddings VALUES (?, ?, ?)', (key, len(array), payload))
                self.connection.execute('INSERT INTO checksums VALUES (?, ?)', (key, self.checksum(key, payload)))
