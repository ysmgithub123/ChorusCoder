"""Pinned RLCoder inference with bounded, content-addressed persistent caches.

Only pure upstream definitions are executed: importing its main.py would load
training data and importing datasets.py normally would shadow Hugging Face datasets.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

from choruscoder import atomic_json, digest, load_json


SOURCE_MANIFEST = Path(__file__).with_name('rlcoder_upstream.json')
STOP_DESCRIPTION = "Don't need cross file context for completion"
ENCODING_VERSION = 'rlcoder_upstream_masked_mean_float32_v1'


def upstream(root):
    """Validate a pinned snapshot and expose the pure inference definitions."""
    manifest = load_json(SOURCE_MANIFEST)
    root = Path(root)
    for name, expected in manifest['sha256'].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Missing or modified pinned RLCoder source: {path}')
    namespace = {}
    for filename, symbol in (('datasets.py', 'CodeBlock'), ('bm25.py', 'split_into_smaller_blocks'),
                             ('retriever.py', 'tokenize')):
        tree = ast.parse((root / filename).read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == symbol)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(root / filename), 'exec'), namespace)
    tree = ast.parse((root / 'generator.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CustomDataset')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'construct_prompts')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(root / 'generator.py'), 'exec'), namespace)
    return manifest, namespace


def configure(cli, config, policy):
    manifest, _ = upstream(cli.rlcoder_root)
    revision = cli.rlcoder_revision or policy.revision
    policy = replace(policy, revision=revision)
    if manifest['commit'] != policy.upstream_commit:
        raise ValueError('RLCoder upstream commit mismatch')
    if cli.embedding_model != policy.model:
        raise ValueError(f'RLCoder requires the trained checkpoint {policy.model}, not original UniXCoder')
    config.update(rlcoder_config=asdict(policy), rlcoder_source=manifest,
                  dataloader_prefetch_factor=cli.dataloader_prefetch_factor,
                  retrieval_backend='rlcoder', window_tokens=None,
                  query_source=f'last_{policy.query_lines}_nonempty_visible_lines',
                  probe_model=None, probe_tokens=0, probe_score='none',
                  decoding={'temperature': 0, 'top_p': 1},
                  retrieval_mode=f'bm25_{policy.bm25_topk}_then_rlretriever_with_stop')
    for source in (Path(__file__), SOURCE_MANIFEST, Path(__file__).with_name('rlcoder_models.py'),
                   Path(__file__).with_name('rlcoder_cache.py')):
        config['source_hashes'][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()


def query_text(prefix, policy):
    return '\n'.join([line for line in prefix.split('\n') if line.strip()][-policy['query_lines']:])


class JsonCache:
    """Checksummed JSON, bounded by serialized size; no executable pickle cache."""
    def __init__(self, root, max_bytes):
        self.root = Path(root)
        self.limit = max_bytes
        self.memory = OrderedDict()
        self.size = 0
        self.hits = self.misses = self.builds = 0

    def _remember(self, key, value):
        from .choruscoder_runtime_cache import retained_bytes
        size = retained_bytes(value) + len(key) + 256
        if size > self.limit:
            return
        if key in self.memory:
            self.size -= self.memory.pop(key)[1]
        while self.memory and self.size + size > self.limit:
            _, (_, old_size) = self.memory.popitem(last=False)
            self.size -= old_size
        self.memory[key] = (value, size)
        self.size += size

    def get(self, kind, identity, build):
        key = digest([kind, identity])
        if key in self.memory:
            value, _ = self.memory[key]
            self.memory.move_to_end(key)
            self.hits += 1
            return value
        path = self.root / kind / (key + '.json')
        if path.exists():
            envelope = load_json(path)
            if (envelope.get('identity') != identity or
                    envelope.get('checksum') != digest(envelope.get('value'))):
                raise ValueError(f'RLCoder {kind} cache identity/checksum mismatch: {path}')
            value = envelope['value']
            self.hits += 1
        else:
            self.misses += 1
            value = build()
            self.builds += 1
            atomic_json(path, {'identity': identity, 'value': value, 'checksum': digest(value)})
        self._remember(key, value)
        return value


def split_file(file, definitions):
    CodeBlock = definitions['CodeBlock']
    original = CodeBlock(file['path'], '', file['text'], 'python', '')
    blocks = definitions['split_into_smaller_blocks'](original, False)
    lines = [(i + 1, line) for i, line in enumerate(file['text'].splitlines()) if line.strip()]
    offset, result = 0, []
    for ordinal, block in enumerate(blocks):
        count = len(block.code_content.split('\n'))
        actual = lines[offset:offset + count]
        if '\n'.join(line for _, line in actual) != block.code_content:
            raise ValueError('Official RLCoder chunk cannot be mapped to allowed source')
        offset += count
        result.append({'id': 'RL' + digest([file['path'], digest(file['text']), ordinal])[:24],
                       'path': file['path'], 'text': block.code_content,
                       'description': block.description, 'encoded_text': str(block),
                       'source_hash': digest(file['text']), 'source_lines': [i for i, _ in actual],
                       'ordinal': ordinal, 'type': block._type})
    return result


def stop_block(definitions):
    block = definitions['CodeBlock']('', STOP_DESCRIPTION, '', 'python', '')
    return {'id': 'STOP', 'encoded_text': str(block), 'stop': True}


def rank_candidates(candidates, scores):
    import numpy as np
    if len(candidates) != len(scores) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError('Invalid RLCoder ranking scores')
    order = np.asarray(scores).argsort()[::-1]
    ranked = [dict(candidates[i], candidate_index=int(i), score=float(scores[i])) for i in order]
    stop_position = next((i for i, block in enumerate(ranked) if block.get('stop')), None)
    if stop_position is None or sum(bool(b.get('stop')) for b in ranked) != 1:
        raise ValueError('RLCoder ranking must contain exactly one stop candidate')
    return ranked, stop_position


def returned_evidence(value):
    returned = value['returned']
    stop = next((i for i, block in enumerate(returned) if block.get('stop')), len(returned))
    return [dict(block, query_ids=['Q0']) for block in returned[:stop]]


class RLCoderRetriever:
    def __init__(self, cache_root, config, tokenizer, definitions, encoder_factory=None):
        from .rlcoder_cache import CheckedVectorCache
        from .rlcoder_models import Encoder, tokenizer_identity
        self.policy = config['rlcoder_config']
        self.definitions = definitions
        self.tokenizer = tokenizer
        self.encoder_factory = encoder_factory or Encoder
        self.encoder = None
        self.cache = JsonCache(Path(cache_root) / 'rlcoder', self.policy['index_cache_bytes'])
        self.encoding = dict(version=ENCODING_VERSION, model=config['embedding_model'],
                             revision=self.policy['revision'], tokenizer=tokenizer_identity(tokenizer))
        namespace = json.dumps(self.encoding, sort_keys=True)
        self.vectors = CheckedVectorCache(Path(cache_root) / 'rlcoder' /
            ('vectors_' + digest(namespace)[:24] + '.sqlite3'), namespace)
        self.vector_memory = OrderedDict()
        self.vector_bytes = 0
        self.index_memory = OrderedDict()
        self.index_bytes = 0
        self.encoding_calls = self.encoded_inputs = self.model_loads = 0
        self.vector_hits = self.vector_misses = self.index_builds = 0
        self.model_load_seconds = 0.0
        from .retrieval_online import online, RepositoryStore
        self.online_mode = online(config)
        if self.online_mode:
            self.repository_store = RepositoryStore(Path(cache_root)/'rlcoder'/'online_corpus',
                dict(upstream=self.policy['upstream_commit'], split='official_split_v1',
                     bm25='rank_bm25.BM25Okapi_0.2.2_lower_corpus_original_query'))
        self.task_vectors = {}

    def ensure_encoder(self):
        if self.encoder is None:
            started = time.perf_counter()
            self.encoder = self.encoder_factory(self.policy)
            self.model_load_seconds += time.perf_counter()-started
            self.model_loads += 1

    def preprocess(self, requests):
        from rank_bm25 import BM25Okapi
        for request in requests:
            key = [dict(path=f['path'], source_hash=digest(f['text'])) for f in request['files']]
            def build():
                blocks = []
                for file in request['files']:
                    identity = dict(path=file['path'], source_hash=digest(file['text']),
                        upstream=self.policy['upstream_commit'], version='official_split_v1')
                    blocks.extend(self.cache.get('chunks', identity,
                        lambda f=file: split_file(f, self.definitions)))
                state = BM25Okapi([b['text'].lower().split() for b in blocks]).__dict__ if blocks else None
                return dict(blocks=blocks, bm25=state)
            self.repository_store.get(key, build)

    def preprocess_stats(self):
        return dict(artifact_hits=self.repository_store.hits, artifact_builds=self.repository_store.builds,
                    repository_encoded_texts=0, model_load_seconds=self.model_load_seconds)

    def _retrieve_online(self, request):
        from rank_bm25 import BM25Okapi
        if self.encoder is None:
            raise ValueError('RLCoder online model must be loaded before retrieval')
        self.task_vectors = {}
        before = self.encoded_inputs
        try:
            preparing = time.perf_counter()
            key = [dict(path=f['path'], source_hash=digest(f['text'])) for f in request['files']]
            corpus = self.repository_store.get(key)
            blocks = corpus['blocks']
            index = None
            if blocks:
                index = BM25Okapi.__new__(BM25Okapi)
                index.__dict__.update(corpus['bm25'])
            preprocessing_seconds = time.perf_counter()-preparing
            started = time.perf_counter()
            query = query_text(request['prefix'], self.policy)
            if blocks:
                scores = index.get_scores(query.split())
                selected = sorted(range(len(blocks)), key=lambda i: (-float(scores[i]), i))[:self.policy['bm25_topk']]
                candidates = [dict(blocks[i], bm25_score=float(scores[i])) for i in selected]
                candidates.append(stop_block(self.definitions))
                q = self._vectors([query], self.policy['query_length'])[0]
                matrix = self._vectors([b['encoded_text'] for b in candidates], self.policy['candidate_length'])
                ranked, stop = rank_candidates(candidates, matrix @ q)
                topk = self.policy['retrieval_topk']
                value = dict(query=query, bm25_ids=[blocks[i]['id'] for i in selected], ranked=ranked,
                    returned=ranked[:topk], stop_position=stop,
                    returned_stop_position=stop if stop < topk else None,
                    stop_state='in_top10' if stop < topk else 'outside_top10')
            else:
                value = dict(query=query, bm25_ids=[], ranked=[], returned=[], stop_position=None,
                             returned_stop_position=None, stop_state='empty_pool')
            value['logical_seconds'] = time.perf_counter()-started
            return value, dict(retrieval_cache_hit=False, observed_seconds=value['logical_seconds'],
                encoded_inputs=self.encoded_inputs-before, task_cache='question_only',
                preprocessing_seconds=preprocessing_seconds)
        finally:
            self.task_vectors.clear()

    def _vectors(self, texts, length):
        import numpy as np
        from .rlcoder_models import encode_inputs
        inputs = [encode_inputs(text, self.tokenizer, length, self.definitions) for text in texts]
        keys = [self.vectors.key(json.dumps(row, separators=(',', ':'))) for row in inputs]
        unique = dict(zip(keys, inputs))
        if self.online_mode:
            missing = [key for key in unique if key not in self.task_vectors]
            if missing:
                self.encoding_calls += 1
                self.encoded_inputs += len(missing)
                fresh = np.asarray(self.encoder.encode([unique[key] for key in missing]), dtype=np.float32)
                if (fresh.shape != (len(missing), 768) or not np.isfinite(fresh).all() or
                        not np.allclose(np.linalg.norm(fresh, axis=1), 1, atol=1e-4)):
                    raise ValueError('Invalid online RLCoder vectors')
                self.task_vectors.update(zip(missing, fresh))
            return np.stack([self.task_vectors[key] for key in keys])
        values = {key: self.vector_memory[key] for key in unique if key in self.vector_memory}
        for key in values:
            self.vector_memory.move_to_end(key)
        missing_memory = [key for key in unique if key not in values]
        values.update(self.vectors.get_many(missing_memory))
        missing = [key for key in unique if key not in values]
        self.vector_hits += len(unique) - len(missing)
        self.vector_misses += len(missing)
        if missing:
            if self.encoder is None:
                started = time.perf_counter()
                self.encoder = self.encoder_factory(self.policy)
                self.model_load_seconds += time.perf_counter() - started
                self.model_loads += 1
            self.encoding_calls += 1
            self.encoded_inputs += len(missing)
            fresh = np.asarray(self.encoder.encode([unique[key] for key in missing]), dtype=np.float32)
            if (fresh.shape != (len(missing), 768) or not np.isfinite(fresh).all() or
                    not np.allclose(np.linalg.norm(fresh, axis=1), 1, atol=1e-4)):
                raise ValueError('RLRetriever returned invalid vectors')
            additions = dict(zip(missing, fresh))
            self.vectors.put_many(additions)
            values.update(additions)
        for key, vector in values.items():
            if vector.shape != (768,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
                raise ValueError('Invalid cached RLRetriever vector')
            if key not in self.vector_memory:
                cost = vector.nbytes + len(key) + 256
                while self.vector_memory and self.vector_bytes + cost > self.policy['vector_cache_bytes']:
                    old_key, old = self.vector_memory.popitem(last=False)
                    self.vector_bytes -= old.nbytes + len(old_key) + 256
                if cost <= self.policy['vector_cache_bytes']:
                    self.vector_memory[key] = vector
                    self.vector_bytes += cost
        return np.stack([values[key] for key in keys])

    def retrieve(self, request):
        from importlib.metadata import version
        from rank_bm25 import BM25Okapi
        if version('rank-bm25') != '0.2.2':
            raise ValueError('RLCoder requires rank-bm25==0.2.2 for pinned BM25 behavior')
        if self.online_mode:
            return self._retrieve_online(request)
        started = time.perf_counter()
        model_load_before = self.model_load_seconds
        blocks, file_keys = [], []
        for file in request['files']:
            identity = dict(path=file['path'], source_hash=digest(file['text']),
                            upstream=self.policy['upstream_commit'], version='official_split_v1')
            file_blocks = self.cache.get('chunks', identity, lambda f=file: split_file(f, self.definitions))
            blocks.extend(file_blocks)
            file_keys.append(digest(identity))
        query = query_text(request['prefix'], self.policy)
        identity = dict(files=file_keys, query=query, encoding=self.encoding,
                        query_length=self.policy['query_length'], candidate_length=self.policy['candidate_length'],
                        topk=self.policy['bm25_topk'], returned_topk=self.policy['retrieval_topk'],
                        stop=STOP_DESCRIPTION, version=self.policy['version'])

        def build():
            if not blocks:
                return dict(query=query, bm25_ids=[], ranked=[], returned=[], stop_position=None,
                            stop_state='empty_pool', returned_stop_position=None,
                            logical_seconds=time.perf_counter() - started)
            pool_identity = dict(files=file_keys, upstream=self.policy['upstream_commit'],
                                 bm25='rank_bm25.BM25Okapi_0.2.2_lower_corpus_original_query')

            def build_index():
                index = BM25Okapi([block['text'].lower().split() for block in blocks])
                return index.__dict__

            state = self.cache.get('bm25', pool_identity, build_index)
            index_key = digest(pool_identity)
            if index_key in self.index_memory:
                index, cost = self.index_memory[index_key]
                self.index_memory.move_to_end(index_key)
            else:
                index = BM25Okapi.__new__(BM25Okapi)
                index.__dict__.update(state)
                from .choruscoder_runtime_cache import retained_bytes
                cost = retained_bytes(state) + 256
                while self.index_memory and self.index_bytes + cost > self.policy['index_cache_bytes']:
                    _, (_, old_cost) = self.index_memory.popitem(last=False)
                    self.index_bytes -= old_cost
                if cost <= self.policy['index_cache_bytes']:
                    self.index_memory[index_key] = (index, cost)
                    self.index_bytes += cost
                self.index_builds += 1
            scores = index.get_scores(query.split())
            selected = sorted(range(len(blocks)), key=lambda i: (-float(scores[i]), i))[:self.policy['bm25_topk']]
            candidates = [dict(blocks[i], bm25_score=float(scores[i])) for i in selected]
            candidates.append(stop_block(self.definitions))
            q = self._vectors([query], self.policy['query_length'])[0]
            matrix = self._vectors([block['encoded_text'] for block in candidates], self.policy['candidate_length'])
            ranked, stop_position = rank_candidates(candidates, matrix @ q)
            topk = self.policy['retrieval_topk']
            return dict(query=query, bm25_ids=[blocks[i]['id'] for i in selected], ranked=ranked,
                        returned=ranked[:topk], stop_position=stop_position,
                        returned_stop_position=stop_position if stop_position < topk else None,
                        stop_state='in_top10' if stop_position < topk else 'outside_top10',
                        logical_seconds=max(0.0, time.perf_counter() - started - (self.model_load_seconds - model_load_before)))

        before = self.cache.builds
        value = self.cache.get('retrieval', identity, build)
        source_blocks = {block['id']: block for block in blocks}
        if value['query'] != query or len(value['bm25_ids']) > self.policy['bm25_topk']:
            raise ValueError('RLCoder cached retrieval does not match its source/query')
        for block in value['ranked']:
            if block.get('stop'):
                if any(block.get(k) != v for k, v in stop_block(self.definitions).items()):
                    raise ValueError('RLCoder cached stop candidate differs from upstream')
            elif block['id'] not in source_blocks or any(block.get(k) != v for k, v in source_blocks[block['id']].items()):
                raise ValueError('RLCoder cached candidate is not in the allowed source pool')
        observed = time.perf_counter() - started
        return value, dict(retrieval_cache_hit=self.cache.builds == before,
                          observed_seconds=observed, cache_hits=self.cache.hits,
                          cache_misses=self.cache.misses, vector_hits=self.vector_hits,
                          vector_misses=self.vector_misses, encoded_inputs=self.encoded_inputs,
                          model_loads=self.model_loads, index_restorations=self.index_builds)

    def close(self):
        self.vectors.close()
        self.vector_memory.clear()
        self.index_memory.clear()
        if self.encoder is not None:
            self.encoder.close()


def record_checksum(record):
    return digest({key: value for key, value in record.items() if key != 'rlcoder_checksum'})


def validate_record(record, config, tokenizer=None, definitions=None):
    from .choruscoder_inputs import validate_input, INPUT_VERSION
    if (record.get('rlcoder_checksum') != record_checksum(record) or
            record.get('rlcoder_config') != config['rlcoder_config'] or
            record.get('context_profile') != config['context_profile'] or
            record.get('evidence_hash') != digest(record['evidence']) or
            record.get('input_budget_version') != config.get('input_budget_version', INPUT_VERSION)):
        raise ValueError('RLCoder prepared record checksum/configuration mismatch')
    value = record['diagnostics']['rlcoder_result']
    ranked, stop = value['ranked'], value['stop_position']
    if len(value['bm25_ids']) > config['rlcoder_config']['bm25_topk'] or len(set(value['bm25_ids'])) != len(value['bm25_ids']):
        raise ValueError('RLCoder exceeds its real-candidate budget')
    if ranked:
        originals = sorted(ranked, key=lambda block: block['candidate_index'])
        if [block['candidate_index'] for block in originals] != list(range(len(originals))):
            raise ValueError('RLCoder original ranking indices mismatch')
        expected, expected_stop = rank_candidates(originals, [block['score'] for block in originals])
        if (expected != ranked or stop != expected_stop or len(ranked) != len(value['bm25_ids']) + 1 or
                len({b['id'] for b in ranked}) != len(ranked) or
                set(b['id'] for b in ranked if not b.get('stop')) != set(value['bm25_ids'])):
            raise ValueError('RLCoder ranking/stop mismatch')
    elif stop is not None or value['bm25_ids']:
        raise ValueError('Invalid empty RLCoder retrieval')
    topk = config['rlcoder_config']['retrieval_topk']
    expected_state = 'empty_pool' if not ranked else 'in_top10' if stop < topk else 'outside_top10'
    expected_returned_stop = stop if ranked and stop < topk else None
    if (value['returned'] != ranked[:topk] or value['stop_state'] != expected_state or
            value['returned_stop_position'] != expected_returned_stop):
        raise ValueError('RLCoder Top-10 return/stop mismatch')
    allowed = returned_evidence(value)
    source_files = {file['path']: file['source_hash'] for file in record['source_files']}
    if record['path'] in source_files or any(
            block.get('path') not in source_files or block.get('source_hash') != source_files[block['path']]
            for block in ranked if not block.get('stop')):
        raise ValueError('RLCoder candidates escape the source whitelist')
    by_id = {block['id']: block for block in allowed}
    for block in record['evidence']:
        original = by_id.get(block['id'])
        if original is None or any(block.get(k) != original.get(k) for k in ('path', 'text', 'source_lines', 'source_hash')):
            raise ValueError('RLCoder evidence does not precede the stop signal')
    if value['query'] != query_text(record['prefix'], config['rlcoder_config']):
        raise ValueError('RLCoder query differs from visible prefix')
    if record['queries'] != [{'id': 'Q0', 'text': value['query'], 'segment_ids': []}]:
        raise ValueError('RLCoder query metadata mismatch')
    if record['evidence'] != allowed:
        raise ValueError('RLCoder evidence differs from returned Top-10 before STOP')
    payload = validate_input(record, tokenizer, definitions)
    if record['evidence_tokens'] != len(payload['crossfile_token_ids']):
        raise ValueError('RLCoder actual cross-file token count mismatch')


def prepare(cli, config, output, cache):
    from .choruscoder_inputs import build_input, INPUT_VERSION
    from transformers import AutoTokenizer
    from .common import iter_parquet_batches
    from .choruscoder_runner import safe_request, identity, ProgressReporter
    from .rlcoder_models import tokenizer_identity
    _, definitions = upstream(cli.rlcoder_root)
    output, cache = Path(output), Path(cache)
    prepared_dir = output / 'prepared'
    prepared_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = digest(config)
    generator_tokenizer = AutoTokenizer.from_pretrained(cli.model)
    backend = None
    tasks, hits, misses = [], 0, 0
    task_ids = set()
    stats = []
    started = time.perf_counter()
    progress = ProgressReporter(output)
    from .retrieval_online import online, preprocess_dataset, local_preparation_complete
    if online(config) and not local_preparation_complete(output, fingerprint):
        tokenizer = AutoTokenizer.from_pretrained(cli.embedding_model,
            revision=config['rlcoder_config']['revision'], use_fast=False)
        backend = RLCoderRetriever(cache, config, tokenizer, definitions)
        try:
            backend.ensure_encoder()
            progress.update('repository_preprocessing', 0, force=True)
            preprocess_dataset(cli, backend, output)
        except BaseException:
            backend.close()
            raise
    try:
        for batch in iter_parquet_batches(cli.parquet, ('task_id', 'path', 'left_context', 'crossfile_context'), cli.batch_size, cli.limit):
            for row in batch:
                request = safe_request(row)
                if request['task_id'] in task_ids:
                    raise ValueError('Duplicate RLCoder task ID')
                task_ids.add(request['task_id'])
                key = identity(request)
                location = cache / 'rlcoder' / 'prepared' / fingerprint / (key + '.json')
                local = prepared_dir / (key + '.json')
                if online(config):
                    location = local
                if local.exists() or location.exists():
                    record = load_json(local if local.exists() else location)
                    validate_record(record, config, generator_tokenizer, definitions)
                    from .choruscoder_timing import validate_timing
                    validate_timing(record, config)
                    if record['request_hash'] != key or record['config_hash'] != fingerprint:
                        raise ValueError('RLCoder prepared request mismatch')
                    expected_sources = [dict(path=f['path'], source_hash=digest(f['text'])) for f in request['files']]
                    if (record['source_files'] != expected_sources or record['prefix'] != request['prefix'] or
                            record['path'] != request['path'] or record['task_id'] != request['task_id']):
                        raise ValueError('RLCoder prepared source mismatch')
                    hits += 1
                else:
                    if backend is None:
                        tokenizer = AutoTokenizer.from_pretrained(cli.embedding_model,
                            revision=config['rlcoder_config']['revision'], use_fast=False)
                        backend = RLCoderRetriever(cache, config, tokenizer, definitions)
                    retrieval_started = time.perf_counter()
                    value, observed = backend.retrieve(request)
                    evidence = returned_evidence(value)
                    record = dict(task_id=request['task_id'], path=request['path'], prefix=request['prefix'],
                        language=cli.language, request_hash=key, config_hash=fingerprint,
                        context_profile=config['context_profile'], rlcoder_config=config['rlcoder_config'],
                        input_budget_version=INPUT_VERSION,
                        source_files=[dict(path=f['path'], source_hash=digest(f['text'])) for f in request['files']],
                        queries=[dict(id='Q0', text=value['query'], segment_ids=[])],
                        selected_segments=[], eligible_supports=0, fallback=None if evidence else 'infile_only',
                        probe_seconds=0.0, retrieval_seconds=value['logical_seconds'],
                        evidence=evidence, evidence_hash=digest(evidence),
                        diagnostics=dict(rlcoder_result=value, rlcoder_cache=observed,
                                         tokenizer_identity=tokenizer_identity(backend.tokenizer)))
                    record['generation_input'] = build_input(record, generator_tokenizer, definitions)
                    record['evidence_tokens'] = len(record['generation_input']['crossfile_token_ids'])
                    record['diagnostics']['context_usage'] = record['generation_input']['usage']
                    if online(config):
                        from .choruscoder_timing import finish_retrieval
                        record['selection_seconds'] = 0.0
                        record['diagnostics']['index_build_seconds'] = 0.0
                        finish_retrieval(record, retrieval_started + observed['preprocessing_seconds'], config)
                    record['rlcoder_checksum'] = record_checksum(record)
                    validate_record(record, config, generator_tokenizer, definitions)
                    atomic_json(location, record)
                    misses += 1
                    stats.append(dict(task_id=request['task_id'], **observed))
                if not local.exists():
                    atomic_json(local, record)
                tasks.append(dict(key=key, file=str(local.resolve())))
                progress.update('rlcoder_preparation', len(tasks), task_id=request['task_id'],
                                details=dict(prepared_cache_hits=hits, prepared_cache_misses=misses))
    finally:
        if backend is not None:
            backend.close()
    if len({task['key'] for task in tasks}) != len(tasks):
        raise ValueError('Duplicate RLCoder requests')
    atomic_json(output / 'prepared_manifest.json', dict(config_hash=fingerprint, tasks=tasks))
    atomic_json(output / 'prepare_stats.json', dict(wall_seconds=time.perf_counter() - started,
        prepared_cache_hits=hits, prepared_cache_misses=misses, tasks=stats,
        model_load_seconds=backend.model_load_seconds if backend else 0.0,
        note='Observed work and cached logical cold cost are recorded separately.'))
    progress.update('prepare_complete', len(tasks), len(tasks), force=True)
