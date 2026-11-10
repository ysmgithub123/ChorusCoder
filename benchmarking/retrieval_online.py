"""Explicit repository preparation and question-local online computation.

Persistent storage is only accessible to repository preparation/lookup. Query
and support encodings never consult it, including when their text is identical.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np

from choruscoder import atomic_json, digest, load_json
from benchmarking.choruscoder_runtime_cache import BoundedLRU

ONLINE_TIMING = 'repository_preprocessed_online_v1'


def online(config):
    return config.get('retrieval_timing') == ONLINE_TIMING


def validate_runtime():
    import inspect
    import choruscoder_far
    from benchmarking.choruscoder_timing import finish_retrieval
    if (getattr(choruscoder_far, 'TASK_VECTOR_PROTOCOL', None) != 'question_only_v1' or
            'config' not in inspect.signature(finish_retrieval).parameters):
        raise RuntimeError('Online retrieval files are incompatible. Synchronize the complete online update, '
            'including choruscoder_far.py and benchmarking/choruscoder_timing.py. Loaded support encoder: '
            + str(choruscoder_far.__file__))


def local_preparation_complete(output, fingerprint):
    """Avoid loading an encoder when continuing a fully prepared local run."""
    root = Path(output).resolve()
    path = root/'prepared_manifest.json'
    if not path.exists():
        return False
    manifest = load_json(path)
    if manifest.get('config_hash') != fingerprint or not manifest.get('tasks'):
        return False
    files = [Path(t['file']).resolve() for t in manifest['tasks']]
    return (len(set(files)) == len(files) and
            all(p.parent == root/'prepared' and p.is_file() for p in files))


def clear_support_memos():
    # These caches contain visible-prefix or question-selected API computations.
    for module_name, names in (
        ('choruscoder_api_support_interfaces', ('raw_prefix',)),
        ('choruscoder_api_support_combinations', ('parameters',)),
        ('choruscoder_compact_apis', ('source_info',)),
        ('choruscoder_compact_support', ('module_apis',)),
    ):
        module = sys.modules.get(module_name)
        for name in names:
            fn = getattr(module, name, None)
            if hasattr(fn, 'cache_clear'):
                fn.cache_clear()


def task_encode(backend, texts, statistics=None):
    memo = backend._task_vectors
    unique = list(dict.fromkeys(texts))
    missing = [text for text in unique if text not in memo]
    if statistics is not None:
        statistics.update(texts=len(texts), unique_texts=len(unique),
                          cache_hits=len(unique)-len(missing), encoded_texts=len(missing))
    if missing:
        started = time.perf_counter()
        values = backend.encode(missing)
        values = values.numpy() if hasattr(values, 'numpy') else np.asarray(values)
        if len(values) != len(missing) or not np.isfinite(values).all():
            raise ValueError('Invalid question-local encoder output')
        memo.update(zip(missing, values))
        backend.task_encoding_seconds += time.perf_counter()-started
        backend.task_encoded_texts += len(missing)
    return np.stack([memo[text] for text in texts])


class RepositoryStore:
    """Checksummed, bounded JSON storage; online misses never invoke builders."""
    def __init__(self, root, identity, memory_bytes=64*1024**2):
        self.root, self.identity = Path(root), identity
        self.memory = BoundedLRU(memory_bytes)
        self.hits = self.builds = 0

    def get(self, key, build=None):
        identity = dict(format=ONLINE_TIMING, implementation=self.identity, source=key)
        name = digest(identity)
        path = self.root/(name+'.json')
        # Check disk revision before reusing a validated memory entry.
        from benchmarking.choruscoder_runtime_cache import file_revision
        revision = file_revision(path)
        cached = self.memory.get(name)
        if cached is not None and cached[0] == revision and revision is not None:
            self.hits += 1
            return deepcopy(cached[1])
        if path.exists():
            record = load_json(path)
            if record.get('identity') != identity or record.get('checksum') != digest(record.get('value')):
                raise ValueError('Repository preprocessing checksum/identity mismatch: '+str(path))
            value = record['value']
            self.hits += 1
        elif build is None:
            raise ValueError('Missing repository preprocessing artifact: '+str(path))
        else:
            value = build()
            atomic_json(path, dict(identity=identity, checksum=digest(value), value=value))
            self.builds += 1
        self.memory.put(name, (file_revision(path), value))
        return deepcopy(value)


def preprocess_dataset(cli, adapter, output, request_builder=None):
    """Scan only allowed inputs. All work finishes before online selection."""
    from benchmarking.common import iter_parquet_batches
    if request_builder is None:
        from benchmarking.choruscoder_runner import safe_request
        request_builder = lambda rows: [safe_request(row) for row in rows]
    count, seconds = 0, 0.0
    for rows in iter_parquet_batches(cli.parquet,
            ('task_id', 'path', 'left_context', 'crossfile_context'), cli.batch_size, cli.limit):
        started = time.perf_counter()
        adapter.preprocess(request_builder(rows))
        seconds += time.perf_counter()-started
        count += len(rows)
        print(f'[repository-preprocessing] samples={count}; seconds={seconds:.3f}', flush=True)
    stats = dict(version=ONLINE_TIMING, samples=count, seconds=seconds,
                 excluded_from_retrieval=True, **adapter.preprocess_stats())
    atomic_json(Path(output)/'preprocessing_stats.json', stats)
    return stats


class OnlineDenseIndex:
    """Unchanged model/chunker and ranking, with a prepared read-only corpus."""
    def __init__(self, backend, root):
        self.backend = backend
        identity = dict(chunks=backend.chunk_identity, encoder=backend.cache.namespace)
        self.store = RepositoryStore(Path(root)/'online_repository', identity)
        self.seen = set()
        self.preprocess_encoding_seconds = 0.0
        self.preprocess_chunk_seconds = 0.0
        self.preprocess_vector_lookup_seconds = 0.0
        self.preprocess_encoded_texts = 0
        self.cache_query_vectors = True
        self.encoder_parked = False
        self.begin_question()

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def set_encoder_device(self, device):
        """Move existing weights at phase boundaries, without reloading them."""
        if device not in ('cpu', 'cuda'):
            raise ValueError('Unsupported encoder phase device')
        parked = device == 'cpu'
        if self.encoder_parked == parked:
            return
        self.backend.model.to(device)
        self.encoder_parked = parked
        if parked:
            self.backend.torch.cuda.empty_cache()

    def begin_question(self):
        clear_support_memos()
        self._task_vectors = {}
        self.task_encoding_seconds = 0.0
        self.task_encoded_texts = 0
        self.last_index_seconds = self.last_chunk_seconds = 0.0
        self.last_query_encoding_seconds = 0.0
        self.last_query_encoding_cache = {}
        self.last_repository_runtime_timings = {}
        self.last_repository_chunks = {}
        self._question_files = None
        self._question_repository = None

    def end_question(self):
        self._task_vectors.clear()
        self._question_files = None
        self._question_repository = None
        clear_support_memos()

    @staticmethod
    def file_key(file):
        return dict(path=file['path'], source_hash=digest(file['text']))

    def preprocess(self, requests):
        for request in requests:
            for file in request['files']:
                key = self.file_key(file)
                token = digest(key)
                if token in self.seen:
                    continue
                def build():
                    chunks = self.backend.build_chunks([file])
                    return dict(chunks=chunks, diagnostics=self.backend.last_repository_chunks)
                tick = time.perf_counter()
                record = self.store.get(key, build)
                self.preprocess_chunk_seconds += time.perf_counter()-tick
                texts = dict.fromkeys(c['text'] for c in record['chunks'])
                keys = {self.cache.key(text): text for text in texts}
                tick = time.perf_counter()
                cached = self.cache.get_many(keys)
                self.preprocess_vector_lookup_seconds += time.perf_counter()-tick
                missing = [k for k in keys if k not in cached]
                # Bound temporary encoder/vector allocations even for large files.
                for start in range(0, len(missing), 32):
                    batch = missing[start:start+32]
                    tick = time.perf_counter()
                    vectors = self.backend.encode([keys[k] for k in batch])
                    vectors = vectors.numpy() if hasattr(vectors, 'numpy') else np.asarray(vectors)
                    self.cache.put_many(dict(zip(batch, vectors)))
                    self.preprocess_encoding_seconds += time.perf_counter()-tick
                    self.preprocess_encoded_texts += len(batch)
                self.seen.add(token)

    def preprocess_stats(self):
        return dict(repository_files=len(self.seen), artifact_hits=self.store.hits,
                    artifact_builds=self.store.builds, repository_encoded_texts=self.preprocess_encoded_texts,
                    repository_chunks_and_artifact_seconds=self.preprocess_chunk_seconds,
                    repository_vector_lookup_seconds=self.preprocess_vector_lookup_seconds,
                    repository_encoding_seconds=self.preprocess_encoding_seconds)

    def _repository_vectors(self, keys):
        """Only prepared corpus keys may reach the cross-question vector LRU."""
        from benchmarking.choruscoder_runtime_cache import embedding_revision
        unique = list(dict.fromkeys(keys))
        memory = getattr(self.backend, '_repository_vector_cache', None)
        values = {}
        if memory is not None:
            revision = embedding_revision(self.cache)
            if getattr(self, '_repository_vector_revision', revision) != revision:
                memory.clear()
            self._repository_vector_revision = revision
            for key in unique:
                vector = memory.get(key)
                if vector is not None:
                    values[key] = vector
        needed = [key for key in unique if key not in values]
        tick = time.perf_counter()
        loaded = self.cache.get_many(needed) if needed else {}
        self._last_vector_lookup = dict(repository_vector_memory_hits=len(values),
            repository_vector_disk_keys=len(needed), vector_disk_read_seconds=time.perf_counter()-tick)
        if any(key not in loaded for key in needed):
            raise ValueError('Missing prepared repository vectors; rerun preprocessing')
        if memory is not None:
            for key, vector in loaded.items():
                stored = np.array(vector, dtype=np.float32, copy=True)
                stored.flags.writeable = False
                memory.put(key, stored)
        values.update(loaded)
        return values

    def retrieve(self, queries, files, quotas):
        if self.encoder_parked:
            raise RuntimeError('Restore the repository encoder before online retrieval')
        if len(queries) != len(quotas):
            raise ValueError('One quota required per query')
        self.last_index_seconds = self.last_chunk_seconds = 0.0
        started = time.perf_counter()
        self.last_query_encoding_seconds = 0.0
        self.last_query_encoding_cache = {}
        self._last_vector_lookup = dict(repository_vector_memory_hits=0,
                                       repository_vector_disk_keys=0, vector_disk_read_seconds=0.0)
        chunks, parts = [], []
        if files is self._question_files:
            chunks, matrix, diagnostic = self._question_repository
        else:
            for file in files:
                record = self.store.get(self.file_key(file))
                chunks.extend(record['chunks'])
                parts.append(record['diagnostics'])
            diagnostic = self._diagnostic(parts, chunks)
            keys = [self.cache.key(c['text']) for c in chunks]
            values = self._repository_vectors(keys)
            matrix = np.stack([values[key] for key in keys]) if chunks else None
            if matrix is not None and not np.isfinite(matrix).all():
                raise ValueError('Invalid prepared repository vectors')
            self._question_files = files
            self._question_repository = (chunks, matrix, diagnostic)
        self.last_repository_chunks = deepcopy(diagnostic)
        if not chunks:
            self.last_repository_runtime_timings = {}
            return [[] for _ in queries]
        lookup = time.perf_counter()-started
        tick = time.perf_counter()
        self.last_query_encoding_cache = {}
        qvectors = task_encode(self, [q['text'] for q in queries], self.last_query_encoding_cache)
        self.last_query_encoding_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        result = []
        for vector, quota in zip(qvectors, quotas):
            scores = matrix @ vector
            order = np.lexsort((np.arange(len(chunks)), -scores))[:quota]
            result.append([dict(chunks[int(i)], score=float(scores[int(i)])) for i in order])
        self.last_repository_runtime_timings = dict(vector_lookup_seconds=lookup,
            query_encoding_seconds=self.last_query_encoding_seconds,
            scoring_sorting_seconds=time.perf_counter()-tick, repository_encoded_texts=0,
            repository_unique_texts=len({c['text'] for c in chunks}), repository_encoding_seconds=0.0,
            **self._last_vector_lookup)
        return result

    def _diagnostic(self, parts, chunks):
        diagnostic = deepcopy(parts[0]) if parts else dict(
            configuration=asdict(self.repository_config),
            identity=self.chunk_identity, files=[], token_counts=[], degraded=False)
        diagnostic.update(files=[f for p in parts for f in p['files']], chunk_count=len(chunks),
                          token_counts=[c['token_count'] for c in chunks],
                          degraded=any(p['degraded'] for p in parts), chunks_seconds=0.0)
        return diagnostic

    def close(self):
        self.end_question()
        self.backend.close()
