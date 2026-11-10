"""CCEval/RECCEVAL runner for the pinned 6144-token GrepRAG method."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from greprag_profiles import METHOD, METHODS, VERSION, ONLINE_TIMING, defaults, validate_options, worker_python,timing_scope
from .common import model_slug, normalize_api_base, AtomicJsonlWriter
from .greprag_upstream import ROOT, file_hash, validate_source, validate_model
from choruscoder import atomic_json, digest, load_json
from . import greprag_resume as recovery


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, default=METHOD)
    p.add_argument('--stage', choices=('all', 'prepare', 'generate', 'evaluate'), default='all')
    p.add_argument('--parquet', nargs='+', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--benchmark', default='cceval')
    p.add_argument('--language', default='python')
    p.add_argument('--dataset-split', default='python')
    p.add_argument('--greprag-model', default='models/greprag-0.6b')
    p.add_argument('--greprag-python', default=worker_python())
    p.add_argument('--greprag-corpus', choices=('parquet',), default='parquet')
    p.add_argument('--greprag-repos-root', default='data_sources/crosscodeeval_repositories')
    p.add_argument('--greprag-resume', action='store_true')
    p.add_argument('--greprag-preprocess-cache-dir',default='benchmark_cache/greprag_corpus')
    p.add_argument('--vllm-api-base', default='http://127.0.0.1:8000/v1')
    p.add_argument('--vllm-api-key', default='EMPTY')
    p.add_argument('--vllm-request-timeout', type=float, default=7200)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--dataloader-num-workers', type=int, default=0)
    p.add_argument('--dataloader-prefetch-factor', type=int, default=1)
    p.add_argument('--limit', type=int)
    for name, value in defaults().items():
        p.add_argument('--' + name.replace('_', '-'), type=type(value), default=value)
    return p


def configuration(cli):
    profile = validate_options(cli)
    if not math.isfinite(cli.vllm_request_timeout) or cli.vllm_request_timeout <= 0:
        raise ValueError('Request timeout must be positive and finite')
    if not shutil.which('rg'):
        raise ValueError('Install ripgrep (rg) and add it to PATH before running GrepRAG')
    worker = Path(cli.greprag_python).absolute()
    model = Path(cli.greprag_model).resolve()
    if not worker.is_file() or not model.is_dir():
        raise ValueError('Missing GrepRAG worker Python or local distilled model; see README.md')
    for name in ('model.safetensors', 'config.json', 'tokenizer_config.json'):
        if not (model / name).is_file():
            raise ValueError('Missing local GrepRAG model file: ' + name)
    paths = [Path(p).resolve() for p in cli.parquet]
    if len(paths) != len(set(paths)) or len({p.parent for p in paths}) != 1:
        raise ValueError('Use distinct shards from one CCEVAL partition')
    sources = ('greprag_profiles.py', 'benchmarking/greprag_upstream.py', 'benchmarking/greprag_backend.py',
               'benchmarking/greprag_worker.py', 'benchmarking/greprag_runner.py',
               'benchmarking/common.py', 'benchmarking/evaluation.py', 'benchmarking/metrics.py',
               'requirements-greprag.txt','benchmarking/greprag_resume.py')
    model_pin = validate_model(model)
    sources+=('benchmarking/greprag_corpus_cache.py','benchmarking/source_corpus_cache.py')
    if cli.benchmark=='recceval':
        sources+=('benchmarking/recceval_metrics.py',)
    sources+=('benchmarking/greprag_keyword_input.py',)
    dataset=[];revisions=[]
    for path in paths:
        before=(path.stat().st_mtime_ns,path.stat().st_size)
        dataset.append([str(path),file_hash(path)])
        if before!=(path.stat().st_mtime_ns,path.stat().st_size):raise ValueError('Dataset changed during configuration')
        revisions.append(before)
    cli.corpus_dataset_revisions=revisions
    profile_data=asdict(profile)
    value=dict(version=VERSION, method=cli.method, profile=profile_data, benchmark=cli.benchmark,
        language=cli.language, dataset_split=cli.dataset_split, model=cli.model,
        distilled_model=dict(path=str(model), **model_pin),
        upstream=validate_source(), source_hashes={p: file_hash(ROOT / p) for p in sources},
        dataset=dataset,
        reference=[str(Path(cli.reference).resolve()), file_hash(cli.reference)],
        corpus_source=cli.greprag_corpus,repos_root=None,
        worker_python=str(worker),
        seed=cli.seed, limit=cli.limit, batch_size=cli.batch_size, workers=[1, 0, 1],
        prompt_tokens=cli.max_context_tokens, crossfile_tokens=cli.max_crossfile_tokens, new_tokens=cli.max_new_tokens,
        decoding=dict(temperature=0, top_p=1, seed=cli.seed), retrieval_timing=timing_scope(cli),
        preprocessing='source_corpora_and_model_initialization', retrieval_cache=False,
        command_error_policy='official_best_effort_no_regex_rewrite_or_retry',
        source_scope='benchmark_related_files_excluding_current',
        adaptations=['source_only_corpus_preparation_before_keywords_excluded_from_online_time',
                     'literal_glob_and_shell_escaping_for_current_file_exclusion',
                     'devnull_worker_stdin_so_rg_scans_repository_not_inherited_pipe',
                     'public_generator_4096_3072_50_budget_after_official_selection'],
        rg_version=subprocess.run(['rg', '--version'], capture_output=True, text=True, check=True).stdout,
        client_versions={p: importlib.metadata.version(p) for p in ('openai', 'pyarrow')})
    value['preprocess_cache_dir']=str(Path(cli.greprag_preprocess_cache_dir).resolve())
    if cli.benchmark=='recceval':
        from .recceval_metrics import EVALUATION_VERSION
        value['evaluation_version']=EVALUATION_VERSION
    from .greprag_keyword_input import POLICY
    value.update(keyword_input_policy=POLICY, keyword_input_tokens=profile.keyword_input_tokens,
            keyword_command_policy=recovery.KEYWORD_COMMAND_POLICY,
            online_measurement='actual_compute_excluding_outer_artifact_io_cuda_synchronized_v1')
    return value


def prepared_records(output, config):
    manifest = load_json(output / 'prepared_manifest.json')
    if (manifest['config_hash'] != digest(config) or not manifest['tasks'] or
            manifest.get('checksum') != digest({k:v for k,v in manifest.items() if k != 'checksum'})):
        raise ValueError('GrepRAG prepared manifest configuration mismatch')
    allowed=recovery.configurations(output,config)
    seen = set()
    for index, task in enumerate(manifest['tasks']):
        if task['file'] != f'{index:06d}.json' or task['task_id'] in seen:
            raise ValueError('Invalid GrepRAG prepared task identity')
        seen.add(task['task_id'])
        path = output / 'prepared' / task['file']
        record = load_json(path)
        if record['config_hash'] not in allowed or record['task_id'] != task['task_id']:
            raise ValueError('GrepRAG prepared task/configuration mismatch')
        if record.get('checksum') != digest({k:v for k,v in record.items() if k != 'checksum'}):
            raise ValueError('GrepRAG prepared checksum mismatch')
        for key in ('keyword_seconds', 'search_phase_seconds', 'retrieval_seconds'):
            if not isinstance(record[key], (int, float)) or not math.isfinite(record[key]) or record[key] < 0:
                raise ValueError('Invalid GrepRAG timing')
        if (record['retrieval_timing'] != config.get('retrieval_timing',ONLINE_TIMING) or
                record['retrieval_seconds'] != record['keyword_seconds'] + record['search_phase_seconds'] or
                not 0 <= record['prompt_tokens'] <= config['prompt_tokens'] or
                not 0 <= record['crossfile_tokens'] <= config['crossfile_tokens']):
            raise ValueError('GrepRAG timing/budget mismatch')
        recovery.validate_record(record,allowed[record['config_hash']],task['task_id'])
        yield task, record


def prepare(cli, config, output, resume_plan=None):
    if resume_plan is None and ((output / 'keywords.jsonl').exists() or (output / 'prepared_manifest.json').exists()):
        raise ValueError('GrepRAG retrieval is uncached; use a new output directory for a new measurement')
    if os.name != 'posix':
        raise ValueError('Official GrepRAG shell commands require Linux; use the documented Linux server')
    allowed = ('parquet', 'reference', 'output_dir', 'model', 'greprag_model', 'greprag_corpus', 'greprag_repos_root',
               'seed', 'batch_size', 'limit', 'max_context_tokens', 'max_crossfile_tokens')
    payload = dict(cli={k:getattr(cli,k) for k in allowed}, config_hash=digest(config))
    payload['cli'].update(method=cli.method,benchmark=cli.benchmark,greprag_preprocess_cache_dir=cli.greprag_preprocess_cache_dir,
        corpus_datasets=config['dataset'],corpus_dataset_revisions=getattr(cli,'corpus_dataset_revisions',None))
    if resume_plan is not None:
        payload['resume_plan']=resume_plan
    location = output / 'worker_request.json'
    atomic_json(location, payload)
    process = subprocess.Popen([str(Path(cli.greprag_python).absolute()), '-m', 'benchmarking.greprag_worker',
        '--request', str(location.resolve())], cwd=ROOT,
        env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONHASHSEED=str(cli.seed)),
        stdin=subprocess.DEVNULL)
    try:
        code = process.wait()
        if code:
            raise RuntimeError(f'GrepRAG worker exited with status {code}')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    # Materialize official outputs without altering the measured computations.
    with AtomicJsonlWriter(output / 'retrieval.jsonl') as traces:
        for _, record in prepared_records(output, config):
            traces.write(record)


def generate(cli, config, output, client_factory=None):
    import time
    if client_factory is None:
        from openai import OpenAI
        client_factory = OpenAI
    if (output / 'predictions.jsonl').exists():
        raise ValueError('Use a fresh generation stage; existing predictions are not silently reused')
    records = iter(prepared_records(output, config))
    client = client_factory(base_url=normalize_api_base(cli.vllm_api_base), api_key=cli.vllm_api_key,
                            timeout=cli.vllm_request_timeout, max_retries=2)
    try:
        with AtomicJsonlWriter(output / 'predictions.jsonl') as predictions, \
                AtomicJsonlWriter(output / 'timings.jsonl') as timings:
            while True:
                batch = []
                for _ in range(cli.batch_size):
                    row = next(records, None)
                    if row is None:
                        break
                    batch.append(row[1])
                if not batch:
                    break
                started = time.perf_counter()
                result = client.completions.create(model=cli.model, prompt=[r['prompt'] for r in batch],
                    max_tokens=cli.max_new_tokens, temperature=0, top_p=1, seed=cli.seed)
                seconds = (time.perf_counter() - started) / len(batch)
                choices = sorted(result.choices, key=lambda c:c.index)
                if [c.index for c in choices] != list(range(len(batch))):
                    raise RuntimeError('Incomplete GrepRAG generator response')
                for record, choice in zip(batch, choices):
                    predictions.write(dict(task_id=record['task_id'], pred=choice.text))
                    timings.write(dict(task_id=record['task_id'], retrieval_seconds=record['retrieval_seconds'],
                        generation_seconds=seconds, end_to_end_seconds=record['retrieval_seconds'] + seconds,
                        keyword_seconds=record['keyword_seconds'], repository_resolution_seconds=record['repository_resolution_seconds'],
                        corpus_materialization_seconds=record['corpus_materialization_seconds'],
                        corpus_cleanup_seconds=record['corpus_cleanup_seconds'],
                        official_search_seconds=record['official_search_seconds'],
                        prompt_adaptation_seconds=record['prompt_adaptation_seconds'], retrieval_timing=config.get('retrieval_timing',ONLINE_TIMING),
                        method=cli.method,keyword_input=record['keyword_input']))
    finally:
        client.close()


def evaluate(cli, config, output):
    from .evaluation import evaluate_predictions, aggregate_timings
    # Validate input artifacts even for an evaluate-only invocation.
    expected = [r['task_id'] for _, r in prepared_records(output, config)]
    predictions = [json.loads(s) for s in (output / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if s]
    times = [json.loads(s) for s in (output / 'timings.jsonl').read_text(encoding='utf-8').splitlines() if s]
    if [r['task_id'] for r in predictions] != expected or [r['task_id'] for r in times] != expected:
        raise ValueError('GrepRAG output task order/cardinality mismatch')
    for (_, record), timing in zip(prepared_records(output, config), times):
        if (timing['retrieval_timing'] != config.get('retrieval_timing',ONLINE_TIMING) or timing['retrieval_seconds'] != record['retrieval_seconds'] or
                not math.isfinite(timing['generation_seconds']) or timing['generation_seconds'] < 0 or
                timing['end_to_end_seconds'] != timing['retrieval_seconds'] + timing['generation_seconds']):
            raise ValueError('GrepRAG output timing mismatch')
    metrics = evaluate_predictions(output / 'predictions.jsonl', cli.reference, 'python', output,
                                   require_complete=cli.limit is None, benchmark=cli.benchmark)
    result = dict(method=cli.method, model=model_slug(cli.model), benchmark=cli.benchmark, language='python',
                  **metrics, **aggregate_timings(output / 'timings.jsonl'), retrieval_timing=config.get('retrieval_timing',ONLINE_TIMING))
    result.update(preprocessing_excluded=True,keyword_input_policy=config['keyword_input_policy'])
    if (output/'resume_history.json').exists():
        history=recovery.checked(output/'resume_history.json')['sessions']
        overhead=sum(s['audit_seconds']+s['worker_recovery_seconds'] for s in history)
        result.update(resumed=True,recovery_sessions=len(history),recovery_overhead_seconds=overhead,
            restored_keywords=history[0]['restored_keywords'],
            restored_retrieval_samples=history[0]['restored_retrieval_samples'],
            timing_note='Original valid task times retained; recovery overhead reported separately; unrecorded failed work is unknown.')
        result['retrieval_seconds_per_sample_with_recovery']=result['retrieval_seconds_per_sample']+overhead/len(expected)
        result['end_to_end_seconds_per_sample_with_recovery']=result['end_to_end_seconds_per_sample']+overhead/len(expected)
    atomic_json(output / 'result.json', result)
    return result


def run(cli):
    config = configuration(cli)
    output = Path(cli.output_dir)
    location = output / 'run_config.json'
    resume_plan=None
    if location.exists():
        previous = load_json(location)
        if getattr(cli,'greprag_resume',False):
            resume_plan=recovery.audit(cli,output,config,previous['configuration'])
            recovery.commit(output,resume_plan)
        elif previous['configuration'] != config:
            raise ValueError('GrepRAG configuration changed; use a new output directory')
        elif cli.stage in ('all', 'prepare'):
            raise ValueError('GrepRAG does not resume/reuse retrieval measurements; use a new output directory')
    elif getattr(cli,'greprag_resume',False):
        raise ValueError('--greprag-resume requires the original interrupted output directory')
    elif cli.stage in ('generate', 'evaluate'):
        raise ValueError('GrepRAG generate/evaluate requires an existing prepared run')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(location, dict(configuration=config, stage=cli.stage, status='running'))
    if cli.stage in ('all', 'prepare'):
        if resume_plan is None: prepare(cli, config, output)
        else: prepare(cli,config,output,resume_plan=resume_plan)
    if cli.stage in ('all', 'generate'):
        generate(cli, config, output)
    result = evaluate(cli, config, output) if cli.stage in ('all', 'evaluate') else None
    atomic_json(location, dict(configuration=config, stage=cli.stage, status='complete'))
    return result


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    run(build_parser().parse_args())
