"""Explicit, checked recovery of one interrupted Parquet retrieval run.

Only complete keyword stages are recoverable. Completed records stay byte-for-
byte intact, including their original measured times and source configuration.
"""
import importlib.metadata
import json
import math
import re
import time
import uuid
from .common import iter_parquet_batches
from .greprag_backend import request
from .greprag_upstream import ROOT, file_hash, load_definitions
from greprag_profiles import ONLINE_TIMING, BOUNDED_METHOD
from choruscoder import atomic_json, digest, load_json
ERROR_POLICY = 'official_best_effort_no_regex_rewrite_or_retry'
KEYWORD_COMMAND_POLICY = 'explicit_pattern_for_dash_keywords_v1'
RUNTIME_PACKAGES = ('torch', 'transformers', 'tokenizers', 'numpy', 'rank-bm25', 'tiktoken', 'pyarrow')

def runtime_versions():
    return {p: importlib.metadata.version(p) for p in RUNTIME_PACKAGES}

def keyword_requests(cli):
    with open(cli.reference, encoding='utf8') as stream:
        refs = (json.loads(line) for line in stream if line.strip())
        seen = set()
        for batch in iter_parquet_batches(cli.parquet, ('task_id', 'path', 'left_context'), cli.batch_size, cli.limit):
            for row in batch:
                ref = next(refs, None)
                if ref is None:
                    raise ValueError('Missing CCEVAL reference metadata')
                req = request(row, ref)
                if req['task_id'] in seen:
                    raise ValueError('Duplicate CCEVAL task ID')
                seen.add(req['task_id'])
                yield req
        if cli.limit is None and next(refs, None) is not None:
            raise ValueError('Reference contains additional CCEVAL tasks')

def signed(value):
    return dict(value, checksum=digest(value))

def checked(path):
    value = load_json(path)
    if value.get('checksum') != digest({k: v for k, v in value.items() if k != 'checksum'}):
        raise ValueError('GrepRAG recovery checksum mismatch: ' + str(path))
    return value

def compatible(previous, current):
    """Resume only the same experiment configuration and implementation."""
    if previous != current:
        raise ValueError('GrepRAG configuration changed; use a new output directory')

def configurations(output, current):
    result = {digest(current): current}
    location = output / 'resume_origin.json'
    if location.exists():
        origin = checked(location)
        previous = origin['configuration']
        compatible(previous, current)
        if origin['config_hash'] != digest(previous):
            raise ValueError('Recovery origin configuration mismatch')
        result[origin['config_hash']] = previous
    return result

def validate_record(record, config, task_id):
    if record.get('task_id') != task_id or record.get('checksum') != digest({k: v for k, v in record.items() if k != 'checksum'}):
        raise ValueError('GrepRAG prepared checksum/task mismatch: ' + task_id)
    for key in ('keyword_seconds', 'search_phase_seconds', 'retrieval_seconds'):
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid GrepRAG timing: ' + task_id)
    if record.get('retrieval_timing') != config.get('retrieval_timing', ONLINE_TIMING) or record['retrieval_seconds'] != record['keyword_seconds'] + record['search_phase_seconds'] or (not 0 <= record.get('prompt_tokens', -1) <= config['prompt_tokens']) or (not 0 <= record.get('crossfile_tokens', -1) <= config['crossfile_tokens']):
        raise ValueError('GrepRAG timing/budget mismatch: ' + task_id)
    if config.get('method') == BOUNDED_METHOD:
        from .greprag_keyword_input import validate_trace
        if record.get('method') != BOUNDED_METHOD:
            raise ValueError('Prepared keyword method mismatch')
        validate_trace(record.get('keyword_input'), config['keyword_input_tokens'])
        if 'keyword_command_policy' in config and record.get('keyword_command_policy') != config['keyword_command_policy']:
            raise ValueError('Prepared keyword command policy mismatch')

def audit(cli, output, current, previous):
    """Read-only: do not mutate a failed run until all reusable data is checked."""
    started = time.perf_counter()
    if cli.greprag_corpus != 'parquet':
        raise ValueError('Recovery requires immutable Parquet corpus, not a mutable physical repository')
    compatible(previous, current)
    configs = configurations(output, current)
    configs[digest(previous)] = previous
    for name in ('predictions.jsonl', 'predictions.jsonl.tmp', 'timings.jsonl', 'timings.jsonl.tmp'):
        if (output / name).exists():
            raise ValueError('Recovery is for interrupted retrieval, not partially/completely generated predictions: ' + name)
    init = load_json(output / 'model_initialization.json')
    expected_preprocessing = current.get('preprocessing') == 'source_corpora_and_model_initialization'
    if not init.get('excluded_from_retrieval') or init.get('repository_preprocessing') is not expected_preprocessing:
        raise ValueError('Missing original GrepRAG initialization identity')
    if current.get('method') == BOUNDED_METHOD:
        from .greprag_keyword_input import POLICY
        identity = init.get('identity', {})
        if identity.get('keyword_prompt_policy') != POLICY or identity.get('keyword_input_tokens') != current['keyword_input_tokens']:
            raise ValueError('Recovery keyword model input policy mismatch')
    kw = load_definitions(ROOT / 'third_party/GrepRAG/grep_train/qwen_generate_keywords.py', ('_extract_keywords', '_shell_single_quote', '_build_rg_commands'), dict(re=re))
    completed = {}
    uncommitted = []
    tasks = []
    keyword_total = 0.0
    with (output / 'keywords.jsonl').open(encoding='utf8') as stream, (output / 'keyword_timings.jsonl').open(encoding='utf8') as times:
        for index, req in enumerate(keyword_requests(cli)):
            raw = stream.readline()
            timing = times.readline()
            if not raw or not timing:
                raise ValueError('Keyword stage is incomplete; this recovery requires all keywords and timings')
            item = json.loads(raw)
            kt = json.loads(timing)
            if current.get('method') == BOUNDED_METHOD:
                from .greprag_keyword_input import validate_trace
                if item.get('method') != BOUNDED_METHOD:
                    raise ValueError('Recovery keyword method mismatch')
                validate_trace(item.get('keyword_input'), current['keyword_input_tokens'], req['prefix'])
            if not isinstance(item.get('raw_output'), str):
                raise ValueError('Missing original keyword model output: ' + req['task_id'])
            if item.get('request') != req or kt.get('task_id') != req['task_id']:
                raise ValueError('Recovery keyword task order/input mismatch: ' + req['task_id'])
            seconds = kt.get('seconds')
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Invalid original keyword timing: ' + req['task_id'])
            keys = kw['_extract_keywords'](item.get('raw_output'), 10)
            commands = kw['_build_rg_commands'](keys, '*.py', 10)
            if item.get('generated_keywords') != keys or item.get('generated_commands') != commands:
                raise ValueError('Recovery keywords/commands differ from official parser: ' + req['task_id'])
            keyword_total += seconds
            location = output / 'prepared' / f'{index:06d}.json'
            tasks.append(dict(task_id=req['task_id'], file=location.name))
            if not location.exists():
                continue
            if location.is_symlink():
                raise ValueError('Symlinked prepared record cannot be recovered')
            record = load_json(location)
            if 'checksum' not in record:
                if record.get('task_id') != req['task_id'] or record.get('config_hash') not in configs:
                    raise ValueError('Uncommitted record belongs to another experiment: ' + req['task_id'])
                uncommitted.append(index)
                continue
            if record.get('config_hash') not in configs:
                raise ValueError('Prepared record belongs to another experiment: ' + req['task_id'])
            validate_record(record, configs[record['config_hash']], req['task_id'])
            if record.get('path') != req['path'] or record.get('repository') != req['repository'] or record.get('keyword_seconds') != seconds or (record.get('raw_output') != item['raw_output']) or (record.get('generated_keywords') != keys) or (record.get('generated_commands') != commands):
                raise ValueError('Prepared record/keyword identity mismatch: ' + req['task_id'])
            if current.get('method') == BOUNDED_METHOD and record['keyword_input'] != item['keyword_input']:
                raise ValueError('Prepared keyword input trace mismatch')
            completed[str(index)] = dict(checksum=record['checksum'], config_hash=record['config_hash'])
        if stream.readline() or times.readline():
            raise ValueError('Additional keyword/timing records not present in dataset')
    if not tasks:
        raise ValueError('Empty recovery dataset')
    expected = {t['file'] for t in tasks}
    if any((p.name not in expected for p in (output / 'prepared').glob('*.json'))):
        raise ValueError('Unexpected prepared files in recovery directory')
    hashes = {name: file_hash(output / name) for name in ('keywords.jsonl', 'keyword_timings.jsonl', 'model_initialization.json')}
    origin_path = output / 'resume_origin.json'
    if origin_path.exists() and checked(origin_path)['artifact_hashes'] != hashes:
        raise ValueError('Original keyword/timing/initialization files changed after recovery was sealed')
    keyword_manifest = output / 'keyword_manifest.json'
    if keyword_manifest.exists():
        manifest = checked(keyword_manifest)
        if manifest.get('config_hash') not in configs or manifest['files'] != {k: v for k, v in hashes.items() if k != 'model_initialization.json'} or manifest['samples'] != len(tasks):
            raise ValueError('Keyword manifest mismatch')
    if (output / 'prepared_manifest.json').exists():
        manifest = checked(output / 'prepared_manifest.json')
        if manifest['config_hash'] not in configs or manifest['tasks'] != tasks or len(completed) != len(tasks):
            raise ValueError('Committed prepared manifest is incomplete or inconsistent')
    plan = dict(session_id=uuid.uuid4().hex, total=len(tasks), completed=completed, uncommitted=uncommitted, artifact_hashes=hashes, keyword_seconds=keyword_total, configs=configs, original_initialization=init, original_configuration=previous, audit_seconds=time.perf_counter() - started)
    return plan

def commit(output, plan):
    origin = output / 'resume_origin.json'
    if not origin.exists():
        atomic_json(origin, signed(dict(configuration=plan['original_configuration'], config_hash=digest(plan['original_configuration']), artifact_hashes=plan['artifact_hashes'])))
    history = output / 'resume_history.json'
    entries = checked(history)['sessions'] if history.exists() else []
    entries.append(dict(session_id=plan['session_id'], restored_keywords=plan['total'], restored_retrieval_samples=len(plan['completed']), rerun_uncommitted_samples=len(plan['uncommitted']), audit_seconds=plan['audit_seconds'], worker_recovery_seconds=0.0, note='Original completed timings retained; unrecorded work before failure cannot be reconstructed.'))
    atomic_json(history, signed(dict(sessions=entries)))

def verify_worker(plan, output):
    started = time.perf_counter()
    if any((file_hash(output / name) != sha for name, sha in plan['artifact_hashes'].items())):
        raise ValueError('Recovery artifacts changed between audit and worker start')
    previous = plan['original_initialization']['identity'].get('runtime_versions')
    if previous != runtime_versions():
        raise ValueError('GrepRAG worker dependencies changed; refusing mixed-runtime recovery')
    return time.perf_counter() - started

def finish(output, plan, seconds):
    history = checked(output / 'resume_history.json')
    entry = next((x for x in history['sessions'] if x['session_id'] == plan['session_id']))
    entry['worker_recovery_seconds'] = seconds
    atomic_json(output / 'resume_history.json', signed(dict(sessions=history['sessions'])))
