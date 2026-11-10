"""Generation-only baseline with independent, validated prepare/generation caches."""
import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import time
from infile_profiles import METHODS, validate_options
from benchmarking.common import AtomicJsonlWriter, iter_parquet_batches, normalize_api_base, stable_hash
from choruscoder import atomic_json, load_json
from benchmarking.retrieval_online import ONLINE_TIMING, online

def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--stage', choices=('prepare', 'generate', 'evaluate', 'all'), default='all')
    parser.add_argument('--parquet', nargs='+', required=True)
    parser.add_argument('--benchmark', default='cceval')
    parser.add_argument('--dataset-split')
    parser.add_argument('--language', default='python')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--infile-cache-dir', default='benchmark_cache/infile')
    parser.add_argument('--retrieval-timing', choices=('online',))
    parser.add_argument('--vllm-api-base', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--vllm-api-key', default='EMPTY')
    parser.add_argument('--vllm-request-timeout', type=float, default=7200)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--dataloader-num-workers', type=int, default=0)
    parser.add_argument('--dataloader-prefetch-factor', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-context-tokens', type=int, default=4096)
    parser.add_argument('--max-crossfile-tokens', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=50, help='Maximum newly generated completion tokens (default: 50)')
    return parser

def configuration(args):
    if not args.dataset_split:
        args.dataset_split = args.language
    profile = validate_options(args)
    paths = [Path(p).resolve() for p in args.parquet]
    if not paths or len({p.parent for p in paths}) != 1 or len(set(paths)) != len(paths):
        raise ValueError('In-File requires distinct shards from one dataset partition')
    versions = {'python': platform.python_version()}
    for name in ('torch', 'transformers', 'tree-sitter', 'pyarrow', 'openai'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parent.parent
    sources = ('infile_profiles.py', 'benchmarking/infile_runner.py', 'choruscoder.py', 'benchmarking/common.py', 'benchmarking/evaluation.py', 'benchmarking/metrics.py', 'utils/eval_utils.py')
    if args.benchmark == 'recceval':
        sources += ('benchmarking/recceval_metrics.py',)
    config = dict(method=args.method, profile=asdict(profile), model=args.model, benchmark=args.benchmark, language=args.language, dataset_split=args.dataset_split, dataset=[[str(p), file_hash(p)] for p in paths], limit=args.limit, seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers, dataloader_num_workers=args.dataloader_num_workers, dataloader_prefetch_factor=args.dataloader_prefetch_factor, prompt_tokens=profile.prompt_tokens, new_tokens=args.max_new_tokens, crossfile_tokens=0, recall_total=0, max_queries=0, max_support_segments=0, retrieval_mode='none', source_scope='visible_current_file_prefix', prefix_token_accounting='selected_ids_and_reencoded_text_v2', temperature=0, top_p=1, runtime_versions=versions, source_hashes={name: hashlib.sha256((root / name).read_text(encoding='utf-8').encode('utf-8')).hexdigest() for name in sources})
    if args.benchmark == 'recceval':
        from benchmarking.recceval_metrics import EVALUATION_VERSION
        config['evaluation_version'] = EVALUATION_VERSION
    if getattr(args, 'retrieval_timing', None) == 'online':
        config.update(retrieval_timing=ONLINE_TIMING, online_policy=dict(repository_preprocessing=False, task_cache='question_only', completed_tasks='current_output_only', end_to_end_includes_preparation=True))
    return config

def tokenizer_identity(tokenizer):
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else None
    return dict(type=type(tokenizer).__module__ + '.' + type(tokenizer).__name__, name=getattr(tokenizer, 'name_or_path', None), revision=getattr(tokenizer, 'init_kwargs', {}).get('_commit_hash'), vocabulary_hash=stable_hash(vocab), backend_hash=stable_hash(backend.to_str()) if backend is not None else None, special_tokens=getattr(tokenizer, 'all_special_ids', []), added_vocabulary_hash=stable_hash(tokenizer.get_added_vocab()) if hasattr(tokenizer, 'get_added_vocab') else None)

def resolved_configuration(args, config=None):
    config = dict(configuration(args) if config is None else config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config['generator_tokenizer'] = tokenizer_identity(tokenizer)
    return (config, tokenizer)

def safe_request(row):
    path = str(row.path).replace('\\', '/')
    parsed = PurePosixPath(path)
    if not path or parsed.is_absolute() or '..' in parsed.parts or (':' in path):
        raise ValueError('Unsafe in-file source path')
    return dict(task_id=str(row.task_id), path=str(parsed), prefix=str(row.left_context))

def construct_prompt(request, tokenizer, budget):
    """Match ChorusCoder's empty-evidence prompt and generator-token truncation."""
    header = f"\n\n# file path: {request['path']}\n\n"
    header_tokens = len(tokenizer.encode(header, add_special_tokens=False))
    if header_tokens > 128 or header_tokens > budget:
        raise ValueError('In-file path header exceeds its budget')
    ids = tokenizer.encode(request['prefix'], add_special_tokens=False)
    original = len(ids)
    available = budget - header_tokens
    ids = ids[-available:] if available else []
    text = tokenizer.decode(ids)
    while True:
        prompt = header + text
        count = len(tokenizer.encode(prompt, add_special_tokens=False))
        excess = count - budget
        if excess <= 0:
            return (prompt, dict(prefix_original_tokens=original, prefix_selected_tokens=len(ids), prefix_retained_tokens=len(tokenizer.encode(text, add_special_tokens=False)), prompt_tokens=count, crossfile_tokens=0, evidence_tokens=0))
        if not ids:
            raise ValueError('In-file prompt exceeds its budget')
        ids = ids[max(1, excess):]
        text = tokenizer.decode(ids)

def validate_record(record, config, tokenizer=None, request=None):
    required = {'record_hash', 'config_hash', 'request_hash', 'prompt_hash', 'prompt', 'task_id', 'path', 'prefix', 'queries', 'selected_segments', 'evidence', 'recall_quotas', 'eligible_supports', 'retrieval_seconds', 'preparation_seconds', 'context_usage'}
    usage_keys = {'prefix_original_tokens', 'prefix_selected_tokens', 'prefix_retained_tokens', 'prompt_tokens', 'crossfile_tokens', 'evidence_tokens'}
    if not isinstance(record, dict) or not required <= record.keys() or (not isinstance(record['context_usage'], dict)) or (not usage_keys <= record['context_usage'].keys()):
        raise ValueError('Incomplete in-file prepared cache record')
    body = {k: v for k, v in record.items() if k != 'record_hash'}
    source = {k: record[k] for k in ('task_id', 'path', 'prefix')}
    usage = record['context_usage']
    checks = dict(record_hash=record.get('record_hash') == stable_hash(body), config_hash=record['config_hash'] == stable_hash(config), request_hash=record['request_hash'] == stable_hash(source), request=request is None or source == request, prompt_hash=record['prompt_hash'] == stable_hash(record['prompt']), prompt_header=record['prompt'].startswith(f"\n\n# file path: {record['path']}\n\n"), no_retrieval=all((record[k] == [] for k in ('queries', 'selected_segments', 'evidence', 'recall_quotas'))) and record['retrieval_seconds'] == 0 and (record['eligible_supports'] == 0) and (usage['crossfile_tokens'] == 0) and (usage['evidence_tokens'] == 0), prompt_budget=type(usage['prompt_tokens']) is int and 0 <= usage['prompt_tokens'] <= config['prompt_tokens'], prefix_counts=all((type(usage[k]) is int and usage[k] >= 0 for k in ('prefix_original_tokens', 'prefix_selected_tokens', 'prefix_retained_tokens'))) and usage['prefix_selected_tokens'] <= usage['prefix_original_tokens'], preparation_seconds=valid_seconds(record['preparation_seconds']))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"In-file prepared cache identity/content mismatch: task_id={record['task_id']!r}; failed={','.join(failed)}")
    if tokenizer is not None and len(tokenizer.encode(record['prompt'], add_special_tokens=False)) != usage['prompt_tokens']:
        raise ValueError('In-file cached prompt token count mismatch')
    return record

def progress(output, stage, completed, total):
    atomic_json(output / 'progress.json', dict(stage=stage, completed=completed, total=total, percent=100 * completed / total if total else None, updated_unix_seconds=time.time()))

def valid_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and (value >= 0)

def validate_manifest(manifest, config):
    if not isinstance(manifest, dict) or not {'tasks', 'config_hash', 'manifest_hash'} <= manifest.keys() or (not isinstance(manifest['tasks'], list)):
        raise ValueError('Incomplete in-file manifest')
    tasks = manifest['tasks']
    if any((not isinstance(t, dict) or not {'task_id', 'key', 'file'} <= t.keys() or any((not isinstance(t[k], str) for k in ('task_id', 'key', 'file'))) for t in tasks)):
        raise ValueError('Invalid in-file manifest task')
    if manifest.get('manifest_hash') != stable_hash({k: v for k, v in manifest.items() if k != 'manifest_hash'}) or manifest['config_hash'] != stable_hash(config) or len({t['task_id'] for t in tasks}) != len(tasks) or (len({t['key'] for t in tasks}) != len(tasks)) or any((len(t['key']) != 64 or any((c not in '0123456789abcdef' for c in t['key'])) or t['file'] != 'prepared/' + t['key'] + '.json' for t in tasks)):
        raise ValueError('In-file manifest identity/content mismatch')
    return tasks

def prepare(args, config, tokenizer):
    output, cache = (Path(args.output_dir), Path(args.infile_cache_dir))
    started = time.perf_counter()
    fingerprint = stable_hash(config)
    tasks, seen, hits, misses = ([], set(), 0, 0)
    progress(output, 'preparation', 0, None)
    for rows in iter_parquet_batches(args.parquet, ('task_id', 'path', 'left_context'), args.batch_size, args.limit):
        for row in rows:
            source_started = time.perf_counter()
            request = safe_request(row)
            source_seconds = time.perf_counter() - source_started
            if request['task_id'] in seen:
                raise ValueError('Duplicate In-File task ID')
            seen.add(request['task_id'])
            key = stable_hash(request)
            local = output / 'prepared' / (key + '.json')
            shared = cache / 'prepared' / fingerprint / (key + '.json')
            existing = local if online(config) or local.is_file() else shared
            if existing.is_file():
                record = validate_record(load_json(existing), config, tokenizer, request)
                hits += 1
            else:
                timing = time.perf_counter()
                prompt, usage = construct_prompt(request, tokenizer, config['prompt_tokens'])
                record = dict(request, config_hash=fingerprint, request_hash=key, prompt=prompt, prompt_hash=stable_hash(prompt), context_usage=usage, queries=[], selected_segments=[], evidence=[], recall_quotas=[], eligible_supports=0, retrieval_seconds=0.0, preparation_seconds=time.perf_counter() - timing + (source_seconds if online(config) else 0.0))
                record['record_hash'] = stable_hash(record)
                validate_record(record, config, tokenizer, request)
                if not online(config):
                    atomic_json(shared, record)
                misses += 1
            if not local.is_file():
                atomic_json(local, record)
            tasks.append(dict(key=key, task_id=request['task_id'], file='prepared/' + key + '.json'))
        progress(output, 'preparation', len(tasks), None)
    manifest = dict(config_hash=fingerprint, tasks=tasks)
    manifest['manifest_hash'] = stable_hash(manifest)
    atomic_json(output / 'prepared_manifest.json', manifest)
    atomic_json(output / 'prepare_stats.json', dict(wall_seconds=time.perf_counter() - started, prepared_cache_hits=hits, prepared_cache_misses=misses, retrieval_seconds=0.0, retriever_model_loads=0, repository_index_builds=0))
    if online(config):
        atomic_json(output / 'preprocessing_stats.json', dict(version=ONLINE_TIMING, samples=len(tasks), seconds=0.0, repository_files=0, repository_encoded_texts=0, excluded_from_retrieval=True, not_applicable=True))
    progress(output, 'prepared', len(tasks), len(tasks))

def generation_key(record, config):
    return stable_hash([record['config_hash'], record['request_hash'], record['prompt_hash'], config['model'], config['generator_tokenizer'], config['seed'], config['new_tokens'], 0, 1])

def validate_generation(value, record, config):
    if not isinstance(value, dict) or not {'result_hash', 'generation_hash', 'task_id', 'prompt_hash', 'pred', 'context_usage', 'retrieval_seconds', 'preparation_seconds', 'generation_seconds', 'end_to_end_seconds'} <= value.keys():
        raise ValueError('Incomplete in-file generation cache record')
    if value.get('result_hash') != stable_hash({k: v for k, v in value.items() if k != 'result_hash'}) or value['generation_hash'] != generation_key(record, config) or value['task_id'] != record['task_id'] or (value['prompt_hash'] != record['prompt_hash']) or (value['retrieval_seconds'] != 0) or (not isinstance(value['pred'], str)) or (value['context_usage'] != record['context_usage']) or (value['preparation_seconds'] != record['preparation_seconds']) or any((not valid_seconds(value[k]) for k in ('preparation_seconds', 'generation_seconds', 'end_to_end_seconds'))) or (not math.isclose(value['end_to_end_seconds'], value['preparation_seconds'] + value['generation_seconds'])):
        raise ValueError('In-file generation cache identity/content mismatch')
    return value

def generate(args, config, tokenizer):
    output, cache = (Path(args.output_dir), Path(args.infile_cache_dir))
    manifest = load_json(output / 'prepared_manifest.json')
    tasks = validate_manifest(manifest, config)
    client, hits, misses, completed = (None, 0, 0, 0)
    progress(output, 'generation', 0, len(tasks))
    try:
        for start in range(0, len(tasks), args.batch_size):
            pending = []
            for task in tasks[start:start + args.batch_size]:
                record = validate_record(load_json(output / task['file']), config, tokenizer)
                if record['request_hash'] != task['key'] or record['task_id'] != task['task_id']:
                    raise ValueError('In-file manifest task identity mismatch')
                key = generation_key(record, config)
                local = output / 'generated' / (task['key'] + '.json')
                shared = cache / 'generated' / (key + '.json')
                existing = local if online(config) or local.is_file() else shared
                if existing.is_file():
                    value = validate_generation(load_json(existing), record, config)
                    if not local.is_file():
                        atomic_json(local, value)
                    hits += 1
                    completed += 1
                else:
                    pending.append((record, local, shared, key))
            if pending:
                if client is None:
                    from openai import OpenAI
                    client = OpenAI(base_url=normalize_api_base(args.vllm_api_base), api_key=args.vllm_api_key, timeout=args.vllm_request_timeout, max_retries=2)
                started = time.perf_counter()
                result = client.completions.create(model=args.model, prompt=[r['prompt'] for r, _, _, _ in pending], max_tokens=config['new_tokens'], temperature=0, top_p=1, seed=args.seed)
                elapsed = (time.perf_counter() - started) / len(pending)
                choices = sorted(result.choices, key=lambda item: item.index)
                if [c.index for c in choices] != list(range(len(pending))):
                    raise RuntimeError('In-file generator returned an incomplete batch')
                for (record, local, shared, key), choice in zip(pending, choices):
                    if not isinstance(choice.text, str):
                        raise RuntimeError('In-file generator returned non-text output')
                    value = dict(task_id=record['task_id'], pred=choice.text, generation_hash=key, prompt_hash=record['prompt_hash'], context_usage=record['context_usage'], retrieval_seconds=0.0, preparation_seconds=record['preparation_seconds'], generation_seconds=elapsed, end_to_end_seconds=record['preparation_seconds'] + elapsed)
                    value['result_hash'] = stable_hash(value)
                    if not online(config):
                        atomic_json(shared, value)
                    atomic_json(local, value)
                    misses += 1
                    completed += 1
            progress(output, 'generation', completed, len(tasks))
    finally:
        if client is not None:
            client.close()
    with AtomicJsonlWriter(output / 'predictions.jsonl') as predictions, AtomicJsonlWriter(output / 'timings.jsonl') as timings, AtomicJsonlWriter(output / 'retrieval.jsonl') as traces:
        for task in tasks:
            record = validate_record(load_json(output / task['file']), config)
            value = validate_generation(load_json(output / 'generated' / (task['key'] + '.json')), record, config)
            predictions.write(dict(task_id=value['task_id'], pred=value['pred']))
            timing = {k: value[k] for k in ('task_id', 'retrieval_seconds', 'preparation_seconds', 'generation_seconds', 'end_to_end_seconds')}
            if online(config):
                timing.update(retrieval_timing=ONLINE_TIMING, allocation='per_question_actual_compute')
            timings.write(timing)
            traces.write(dict(task_id=record['task_id'], request_hash=record['request_hash'], source_hash=stable_hash({k: record[k] for k in ('task_id', 'path', 'prefix')}), path=record['path'], queries=[], selected_segments=[], evidence=[], recall_quotas=[], context_usage=record['context_usage'], prompt_hash=record['prompt_hash'], generation_hash=value['generation_hash']))
    atomic_json(output / 'generation_stats.json', dict(generation_cache_hits=hits, generation_cache_misses=misses))
    progress(output, 'generated', len(tasks), len(tasks))

def evaluate(args, config):
    from benchmarking.evaluation import evaluate_predictions, aggregate_timings
    output = Path(args.output_dir)
    metrics = evaluate_predictions(output / 'predictions.jsonl', Path(args.parquet[0]).parent / 'test.jsonl', args.language, output, require_complete=args.limit is None, benchmark=args.benchmark)
    timings = [json.loads(line) for line in (output / 'timings.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    result = dict(method=args.method, model=Path(args.model).name, benchmark=args.benchmark, language=args.language, **metrics, **aggregate_timings(output / 'timings.jsonl'), preparation_seconds_per_sample=sum((r['preparation_seconds'] for r in timings)) / len(timings) if timings else 0.0, config_hash=stable_hash(config))
    if online(config):
        result['retrieval_timing'] = ONLINE_TIMING
    result['result_hash'] = stable_hash(result)
    atomic_json(output / 'result.json', result)
    return result

def completed_result(args):
    """Validate provenance even for explicitly skipped, already completed runs."""
    config, _ = resolved_configuration(args)
    output = Path(args.output_dir)
    wrapper = load_json(output / 'run_config.json')
    result = load_json(output / 'result.json')
    if wrapper.get('status') != 'complete' or wrapper['configuration'] != config or result.get('config_hash') != stable_hash(config) or (result.get('result_hash') != stable_hash({k: v for k, v in result.items() if k != 'result_hash'})) or any((result.get(k) != v for k, v in dict(method=args.method, model=Path(args.model).name, benchmark=args.benchmark, language=args.language).items())) or (False and result.get('subset') != args.dataset_split) or (type(result.get('total_samples')) is not int) or (result['total_samples'] < 0):
        raise ValueError('Completed In-File result differs from requested configuration; use a new output root')
    return result

def run(args):
    base = configuration(args)
    output = Path(args.output_dir)
    config_path = output / 'run_config.json'
    if args.stage == 'evaluate':
        saved = load_json(config_path)['configuration']
        config = dict(base, generator_tokenizer=saved['generator_tokenizer'])
        tokenizer = None
    else:
        config, tokenizer = resolved_configuration(args, base)
    if config_path.is_file() and load_json(config_path)['configuration'] != config:
        raise ValueError('In-file output directory contains a different configuration; use a new output root')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(config_path, dict(configuration=config, status='running', stage=args.stage))
    if args.stage in ('prepare', 'all'):
        prepare(args, config, tokenizer)
    if args.stage in ('generate', 'all'):
        generate(args, config, tokenizer)
    if args.stage in ('evaluate', 'all'):
        result = evaluate(args, config)
        progress(output, 'complete', result['total_samples'], result['total_samples'])
    status = 'complete' if args.stage in ('evaluate', 'all') else 'prepared' if args.stage == 'prepare' else 'generated'
    atomic_json(config_path, dict(configuration=config, status=status, stage=args.stage))
if __name__ == '__main__':
    run(build_parser().parse_args())
