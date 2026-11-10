"""Prepare, generate and evaluate registered ChorusCoder methods and partitions."""
from __future__ import annotations
from method_registry import CHORUSCODER_METHODS as METHODS
import argparse
from dataclasses import asdict, replace
import hashlib
from importlib import metadata
import platform
from pathlib import Path, PurePosixPath
import time
from benchmarking.retrieval_online import online
from benchmarking.common import iter_parquet_batches, normalize_crossfile_context, normalize_api_base, validate_generation_tokens
from choruscoder import VERSION, atomic_json, load_json, digest, window_view, select_segments, build_queries, round_robin_evidence, pack_evidence, construct_prompt, validate_coderag, source_query, render_evidence, query_evidence_coverage, method_recall_quotas, query_roles
from choruscoder_profiles import PROFILES, segment_limit, support_segment_limit, query_limit, context_config, recall_config, selection_cache_method
from choruscoder_benchmarks import experiment_target, ADAPTER_VERSION

class ProgressReporter:
    """Small live status file, separate from immutable result/cache identities."""

    def __init__(self, output):
        self.output = Path(output)
        self.started = time.perf_counter()
        self.last = -float('inf')

    def update(self, stage, completed, total=None, task_id=None, details=None, force=False):
        now = time.perf_counter()
        if not force and now - self.last < 10:
            return
        record = {'stage': stage, 'completed': completed, 'total': total, 'percent': 100 * completed / total if total else None, 'last_task_id': task_id, 'wall_seconds': now - self.started, 'updated_unix_seconds': time.time()}
        if details:
            record['last_task_diagnostics'] = details
        atomic_json(self.output / 'progress.json', record)
        self.last = now
        denominator = f'/{total}' if total is not None else ''
        print(f"[choruscoder-progress] {stage}: {completed}{denominator}; elapsed={record['wall_seconds']:.1f}s", flush=True)

def build_parser():

    class MethodParser(argparse.ArgumentParser):

        def parse_args(self, args=None, namespace=None):
            parsed = super().parse_args(args, namespace)
            if parsed.retrieval_timing is None:
                from choruscoder_profiles import supports_online_retrieval
                parsed.retrieval_timing = 'online' if supports_online_retrieval(parsed.method) else 'full'
            return parsed
    p = MethodParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--stage', choices=('prepare', 'generate', 'evaluate', 'all'), default='all')
    p.add_argument('--parquet', nargs='+', required=True)
    p.add_argument('--benchmark', default='cceval')
    p.add_argument('--dataset-split')
    p.add_argument('--language', default='python')
    p.add_argument('--model', default='deepseek-ai/deepseek-coder-1.3b-base')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--query-cache-dir', default='benchmark_cache/choruscoder')
    p.add_argument('--retrieval-timing', choices=('full', 'online'), default=None)
    p.add_argument('--query-max-segments', type=int)
    p.add_argument('--query-support-similarity-threshold', type=float, help='Support cosine threshold, or fused-score threshold for fused profiles; defaults come from the method')
    p.add_argument('--query-interface-similarity-threshold', type=float)
    p.add_argument('--query-interface-margin-threshold', type=float)
    p.add_argument('--query-probe-model', default='Salesforce/codet5p-220m')
    p.add_argument('--embedding-model', default=None, help="Defaults to the method's checkpoint (RLRetriever for RLCoder, UniXCoder otherwise)")
    p.add_argument('--coderag-root', default='third_party/CodeRAG')
    p.add_argument('--rlcoder-root', default='third_party/RLCoder')
    p.add_argument('--rlcoder-revision', default=None)
    p.add_argument('--vllm-api-base', default='http://127.0.0.1:8000/v1')
    p.add_argument('--vllm-api-key', default='EMPTY')
    p.add_argument('--vllm-request-timeout', type=float, default=7200)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--limit', type=int)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--dataloader-num-workers', type=int, default=0)
    p.add_argument('--dataloader-prefetch-factor', type=int, default=1)
    p.add_argument('--max-context-tokens', type=int)
    p.add_argument('--max-crossfile-tokens', type=int)
    p.add_argument('--max-new-tokens', type=int, default=50, help='Maximum newly generated completion tokens (default: 50)')
    return p

def configuration(cli):
    cli.query_max_segments = segment_limit(cli.method, cli.query_max_segments)
    profile = PROFILES[cli.method]
    from choruscoder_profiles import default_embedding_model, effective_api_support_config
    api_policy = effective_api_support_config(cli.method, getattr(cli, 'query_interface_similarity_threshold', None), getattr(cli, 'query_interface_margin_threshold', None))
    if profile.api_support_config is not None:
        from choruscoder_api_support import validate_runtime
        if profile.encoder_config is None:
            validate_runtime()
        else:
            validate_runtime(configurable_thresholds=True)
    if profile.callable_ablation_config is not None:
        from choruscoder_callable_ablations import validate_runtime
        validate_runtime()
    if cli.embedding_model is None:
        cli.embedding_model = default_embedding_model(cli.method)
    threshold = getattr(cli, 'query_support_similarity_threshold', None)
    if threshold is not None and profile.semantic_support_config is None:
        raise ValueError('Support similarity threshold requires a semantic-support method')
    if profile.support_ablation_config is not None and threshold is not None and (threshold != 0.4):
        raise ValueError('GaussRank support ablations require fused threshold 0.4')
    if profile.callable_ablation_config is not None and threshold not in (None, 0.4):
        raise ValueError('Callable support ablations require fused threshold 0.4')
    context = context_config(cli.method)
    expected_budgets = (context.prompt_tokens, context.crossfile_tokens)
    for name, value in zip(('max_context_tokens', 'max_crossfile_tokens'), expected_budgets):
        if getattr(cli, name) is None:
            setattr(cli, name, value)
    split = cli.dataset_split or (cli.language if cli.benchmark == 'cceval' else Path(cli.parquet[0]).parent.name)
    target = experiment_target(cli.method, cli.benchmark, cli.language, split)
    cli.dataset_split = target.data_key
    if (cli.max_context_tokens, cli.max_crossfile_tokens) != expected_budgets:
        raise ValueError(f"{cli.method} requires the common {'/'.join(map(str, expected_budgets))} input budgets")
    validate_generation_tokens(cli.max_new_tokens)
    if cli.batch_size < 1 or (cli.limit is not None and cli.limit < 1):
        raise ValueError('Batch and limit must be positive')
    if cli.num_workers != 1 or cli.dataloader_num_workers != 0:
        raise ValueError('ChorusCoder v1 requires worker=1 and DataLoader workers=0')
    if profile.coderag_query_config is not None:
        validate_coderag(cli.coderag_root)
    paths = [Path(p).resolve() for p in cli.parquet]
    if len({path.parent for path in paths}) != 1:
        raise ValueError('ChorusCoder Parquet shards must belong to a single benchmark partition')
    code = [Path(__file__), Path(__file__).parent / 'choruscoder_models.py', Path(__file__).parent.parent / 'choruscoder.py']
    config = {'version': VERSION, 'method': cli.method, 'model': cli.model, 'benchmark': cli.benchmark, 'language': cli.language, 'dataset_split': target.data_key, 'embedding_model': cli.embedding_model, 'probe_model': cli.query_probe_model, 'max_segments': cli.query_max_segments, 'seed': cli.seed, 'window_tokens': 256, 'chunk_tokens': context.chunk_tokens, 'overlap_tokens': context.overlap_tokens, 'runtime_versions': runtime_versions(), 'prompt_tokens': context.prompt_tokens, 'crossfile_tokens': context.crossfile_tokens, 'infile_tokens': context.infile_tokens, 'new_tokens': cli.max_new_tokens, 'probe_tokens': 5, 'probe_score': 'sum_max_softmax_probability', 'recall_total': recall_config(cli.method).total, 'evidence_count': None, 'source_scope': 'benchmark_related_files_excluding_current', 'dataset': [[str(p), file_digest(p)] for p in paths], 'limit': cli.limit, 'batch_size': cli.batch_size, 'num_workers': cli.num_workers, 'dataloader_num_workers': cli.dataloader_num_workers, 'source_hashes': {p.name: digest(p.read_text(encoding='utf-8')) for p in code}, 'coderag_query_mode': None, 'coderag_commit': None}
    if profile.encoder_config is not None:
        from benchmarking.choruscoder_encoders import local_checkpoint_identity
        from benchmarking.choruscoder_encoder_registry import implementation_sources
        sources = implementation_sources(profile.encoder_config)
        config['encoder_config'] = asdict(profile.encoder_config)
        for source in sources:
            config['source_hashes'][source.name] = digest(source.read_text(encoding='utf-8'))
        config['encoder_implementation'] = config['source_hashes'][sources[0].name]
        local = local_checkpoint_identity(cli.embedding_model)
        if local is not None:
            config['encoder_local_checkpoint'] = local
    timing = getattr(cli, 'retrieval_timing', None)
    if profile.callable_support_config is not None:
        timing = timing or 'full'
    if timing == 'online':
        from benchmarking.retrieval_online import ONLINE_TIMING, validate_runtime
        validate_runtime()
        from choruscoder_profiles import supports_online_retrieval
        if not supports_online_retrieval(cli.method):
            raise ValueError('Online timing requires a supported ChorusCoder method, including Prefix100 and CodeRAG-Select1')
        config['retrieval_timing'] = ONLINE_TIMING
        config['online_policy'] = dict(repository_preprocessing=True, task_cache='question_only', completed_tasks='current_output_only')
        source = Path(__file__).with_name('retrieval_online.py')
        config['source_hashes'][source.name] = digest(source.read_text(encoding='utf-8'))
        timing_source = Path(__file__).with_name('choruscoder_timing.py')
        config['source_hashes'][timing_source.name] = digest(timing_source.read_text(encoding='utf-8'))
        config['retrieval_timing_implementation'] = digest([config['source_hashes'][source.name], config['source_hashes'][timing_source.name]])
    if timing == 'full':
        if profile.rlcoder_config is not None:
            raise ValueError('Full timing requires an independent ChorusCoder retrieval run')
        from benchmarking.choruscoder_timing import FULL_RETRIEVAL_TIMING
        config['retrieval_timing'] = FULL_RETRIEVAL_TIMING
        source = Path(__file__).with_name('choruscoder_timing.py')
        config['source_hashes'][source.name] = digest(source.read_text(encoding='utf-8'))
        config['retrieval_timing_implementation'] = config['source_hashes'][source.name]
    from choruscoder_benchmarks import uses_cast_adapter
    if uses_cast_adapter(cli.method, cli.benchmark):
        from choruscoder_languages import grammar_identities
        config['grammar_identities'] = grammar_identities(cli.language)
        adapter = Path(__file__).parent.parent / 'choruscoder_languages.py'
        config['source_hashes'][adapter.name] = digest(adapter.read_text(encoding='utf-8'))
    from benchmarking.choruscoder_runtime_cache import runtime_cache_config
    runtime_config = runtime_cache_config(cli.method, cli.benchmark)
    if runtime_config is not None:
        config['repository_runtime_cache'] = runtime_config
        source = Path(__file__).with_name('choruscoder_runtime_cache.py')
        config['source_hashes'][source.name] = digest(source.read_text(encoding='utf-8'))
    config['recall_config'] = {'total': profile.recall_config.total, 'quotas_by_query_count': [list(row) for row in profile.recall_config.quotas_by_query_count]}
    config.update(context_profile=asdict(context), evidence_body_tokens=context.evidence_body_tokens, infile_mode='remaining' if context.infile_tokens is None else 'fixed', retrieval_mode='fresh')
    context_source = Path(__file__).parent / 'choruscoder_context.py'
    config['source_hashes'][context_source.name] = digest(context_source.read_text(encoding='utf-8'))
    config['source_hashes']['choruscoder_benchmarks.py'] = digest((Path(__file__).parent.parent / 'choruscoder_benchmarks.py').read_text(encoding='utf-8'))
    if cli.benchmark == 'recceval':
        from benchmarking.recceval_metrics import EVALUATION_VERSION
        config['evaluation_version'] = EVALUATION_VERSION
        source = Path(__file__).with_name('recceval_metrics.py')
        config['source_hashes'][source.name] = digest(source.read_text(encoding='utf-8'))
    if (cli.benchmark, cli.language) != ('cceval', 'python'):
        config.update(benchmark_adapter_version=ADAPTER_VERSION, reference_mode='standalone')
    profile = PROFILES[cli.method]
    if profile.scoring_mode == 'source_recency':
        config.update(scoring_mode=profile.scoring_mode, selection_mode=profile.selection_mode, max_support_segments=support_segment_limit(cli.method, cli.query_max_segments), max_queries=query_limit(cli.method, cli.query_max_segments))
        if profile.scoring_mode == 'source_recency':
            config['probe_score'] = 'source_end_byte_recency'
    if profile.scoring_mode == 'none':
        config.update(scoring_mode='none', selection_mode=profile.selection_mode, probe_score='none', max_support_segments=0, max_queries=1)
    if profile.coderag_query_config is not None:
        config.update(coderag_query_config=asdict(profile.coderag_query_config), window_tokens=None, query_source=profile.coderag_query_config.source, scoring_mode=profile.scoring_mode, selection_mode=profile.selection_mode, max_support_segments=1, max_queries=1, coderag_query_mode=profile.coderag_query_config.version, coderag_commit=validate_coderag(cli.coderag_root)['commit'])
        for name in ('choruscoder_coderag_query.py', 'choruscoder_prefix_query.py', 'choruscoder_chunk_controls.py'):
            source = Path(__file__).parent.parent / name
            config['source_hashes'][name] = digest(source.read_text(encoding='utf-8'))
    if profile.prefix_query_tokens is not None:
        config.update(prefix_query_tokens=profile.prefix_query_tokens, window_tokens=profile.prefix_query_tokens, query_source='visible_prefix_suffix', prefix_query_version='prefix_query_controls_v1')
        for name in ('choruscoder_prefix_query.py', 'choruscoder_chunk_controls.py'):
            source = Path(__file__).parent.parent / name
            config['source_hashes'][name] = digest(source.read_text(encoding='utf-8'))
    if profile.support_ablation_config is not None:
        config['support_ablation_config'] = asdict(profile.support_ablation_config)
        config['max_queries'] = query_limit(cli.method, cli.query_max_segments)
        for name in ('choruscoder_support_ablations.py', 'choruscoder_repository_lines.py'):
            source = Path(__file__).parent.parent / name
            config['source_hashes'][name] = digest(source.read_text(encoding='utf-8'))
    if profile.repository_chunk_config is not None:
        from benchmarking.choruscoder_cast import implementation_identity
        repository_identity = implementation_identity(cli.language, profile.repository_chunk_config)
        repository_sources = (Path(__file__).with_name('choruscoder_cast.py'),)
        config.update(repository_chunk_config=asdict(profile.repository_chunk_config), repository_chunk_identity=repository_identity)
        for repository_source in repository_sources:
            config['source_hashes'][repository_source.name] = digest(repository_source.read_text(encoding='utf-8'))
    if profile.semantic_support_config is not None:
        semantic_config = replace(profile.semantic_support_config, similarity_threshold=threshold) if threshold is not None else profile.semantic_support_config
        config.update(semantic_support_config=asdict(semantic_config), support_source_scope='visible_current_file_prefix', semantic_score='cosine(candidate_only,target_only)')
        for name in ('choruscoder_semantic_support.py', 'choruscoder_far.py'):
            source = Path(__file__).parent.parent / name
            config['source_hashes'][name] = digest(source.read_text(encoding='utf-8'))
    if profile.api_support_config is not None:
        config['api_support_config'] = asdict(api_policy)
        for name in ('choruscoder_api_support.py', 'choruscoder_api_support_candidates.py', 'choruscoder_api_support_combinations.py', 'choruscoder_api_support_interfaces.py', 'choruscoder_compact_support.py', 'choruscoder_compact_bindings.py', 'choruscoder_compact_apis.py', 'choruscoder_dependency_support.py', 'choruscoder_scoring.py'):
            config['source_hashes'][name] = digest((Path(__file__).parent.parent / name).read_text(encoding='utf-8'))
    if profile.callable_support_config is not None:
        config['callable_support_config'] = asdict(profile.callable_support_config)
        name = 'choruscoder_callable_support.py'
        config['source_hashes'][name] = digest((Path(__file__).parent.parent / name).read_text(encoding='utf-8'))
    if profile.callable_ablation_config is not None:
        config['callable_ablation_config'] = asdict(profile.callable_ablation_config)
        for name in ('choruscoder_callable_ablations.py', 'choruscoder_support_ablations.py', 'choruscoder_repository_lines.py'):
            config['source_hashes'][name] = digest((Path(__file__).parent.parent / name).read_text(encoding='utf-8'))
    if profile.chunk_control_config is not None:
        config['chunk_control_config'] = asdict(profile.chunk_control_config)
        config['window_tokens'] = None
        config['block_source_scope'] = 'visible_current_file_prefix'
        source = Path(__file__).parent.parent / 'choruscoder_chunk_controls.py'
        config['source_hashes'][source.name] = digest(source.read_text(encoding='utf-8'))
        source = Path(__file__).parent.parent / 'choruscoder_prefix_query.py'
        config['source_hashes'][source.name] = digest(source.read_text(encoding='utf-8'))
    if profile.fused_support_config is not None:
        config['support_threshold_kind'] = 'fused' if profile.fused_support_config is not None else 'cosine'
        if profile.fused_support_config is not None:
            config['fused_support_config'] = asdict(profile.fused_support_config)
        for name in ('choruscoder_selection_controls.py', 'choruscoder_languages.py'):
            source = Path(__file__).parent.parent / name
            config['source_hashes'][name] = digest(source.read_text(encoding='utf-8'))
    if profile.support_unit_config is not None:
        config['support_unit_config'] = asdict(profile.support_unit_config)
        if getattr(profile.support_unit_config, 'oversize_policy', 'exclude') == 'recursive':
            split_source = Path(__file__).parent / 'choruscoder_cast.py'
            config['source_hashes'][split_source.name] = digest(split_source.read_text(encoding='utf-8'))
    if profile.support_unit_config is not None:
        unit_source = Path(__file__).parent.parent / 'choruscoder_support_units.py'
        grammar = Path(__file__).parent.parent / 'utils/build/python-lang-parser.so'
        try:
            grammar_version = metadata.version('tree-sitter-python')
        except metadata.PackageNotFoundError:
            grammar_version = None
        config['support_unit_parser'] = {'package_version': grammar_version, 'library_sha256': file_digest(grammar) if grammar.exists() else None}
        config['source_hashes'][unit_source.name] = digest(unit_source.read_text(encoding='utf-8'))
    profile = PROFILES[cli.method]
    config.update(ablation_version=profile.strategy_version, candidate_mode=profile.candidate_mode, target_mode=profile.target_mode, query_mode=profile.query_mode)
    if profile.candidate_lines is not None:
        config['candidate_lines'] = profile.candidate_lines
    additional = [Path(__file__).parent.parent / 'choruscoder_profiles.py']
    config['source_hashes'].update({p.name: digest(p.read_text(encoding='utf-8')) for p in additional})
    if profile.rlcoder_config is not None:
        from benchmarking.rlcoder_backend import configure
        configure(cli, config, profile.rlcoder_config)
    if profile.rlcoder_config is not None:
        from benchmarking.choruscoder_inputs import INPUT_VERSION
        config.update(input_budget_version=INPUT_VERSION, dataloader_prefetch_factor=cli.dataloader_prefetch_factor, decoding={'temperature': 0, 'top_p': 1})
        source = Path(__file__).with_name('choruscoder_inputs.py')
        config['source_hashes'][source.name] = digest(source.read_text(encoding='utf-8'))
    if profile.api_support_config is not None:
        config['api_support_implementation'] = digest({name: config['source_hashes'][name] for name in config['source_hashes'] if name.startswith('choruscoder_api_support') or name in ('choruscoder_callable_support.py', 'choruscoder_callable_ablations.py', 'choruscoder_support_ablations.py', 'choruscoder_repository_lines.py', 'choruscoder_timing.py', 'choruscoder_compact_support.py', 'choruscoder_compact_bindings.py', 'choruscoder_compact_apis.py', 'choruscoder_dependency_support.py', 'choruscoder_scoring.py', 'choruscoder_far.py', 'choruscoder_chunk_controls.py', 'choruscoder_selection_controls.py', 'choruscoder_support_units.py', 'choruscoder_semantic_support.py', 'choruscoder_languages.py', 'choruscoder.py') or (profile.callable_ablation_config is not None and name in ('choruscoder_runner.py', 'choruscoder_profiles.py'))})
    return config

def runtime_versions():
    versions = {'python': platform.python_version()}
    for name in ('torch', 'transformers', 'tree-sitter', 'pyarrow', 'openai'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions

def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def safe_path(value):
    value = str(value).replace('\\', '/')
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or (':' in value):
        raise ValueError(f'Unsafe benchmark source path: {value!r}')
    return str(path)

def safe_request(row):
    """Explicit whitelist: even if the caller has references, they cannot leave here."""
    current = safe_path(row.path)
    files = {}
    for item in normalize_crossfile_context(row.crossfile_context, row.task_id):
        if not item.get('path'):
            continue
        path = safe_path(item['path'])
        if path == current:
            continue
        source = str(item.get('text', ''))
        if path in files and files[path] != source:
            raise ValueError(f'Conflicting source copies: {path}')
        files[path] = source
    return {'task_id': str(row.task_id), 'path': current, 'prefix': str(row.left_context), 'files': [{'path': p, 'text': text} for p, text in sorted(files.items())]}

def identity(request):
    return digest(request)

def canonical_method(method):
    return method

def selection_record(request, config, cache, tokenizer, probe_factory):
    selection_started = time.perf_counter()
    profile = PROFILES[config['method']]
    from benchmarking.retrieval_online import online, clear_support_memos
    if online(config):
        clear_support_memos()
    if profile.coderag_query_config is not None:
        from choruscoder_coderag_query import selection_record as coderag_selection_record
        return coderag_selection_record(request, config, cache, tokenizer, probe_factory)
    prefix_window_seconds = 0.0
    window = window_view(request['prefix'], tokenizer)
    strategy = {'candidate': profile.candidate_mode, 'target': profile.target_mode, 'version': config.get('ablation_version')}
    if config.get('grammar_identities'):
        from choruscoder_languages import grammar_variant
        strategy['grammar_variant'] = grammar_variant(config.get('language', 'python'), request['path'])
    strategy['recall_config'] = config['recall_config']
    strategy['method'] = selection_cache_method(config['method'])
    if profile.support_unit_config is not None:
        strategy.update(method=selection_cache_method(config['method']), support_units=config.get('support_unit_config'), parser=config['support_unit_parser'])
    if profile.semantic_support_config is not None:
        strategy.update(prefix_hash=digest(request['prefix']), semantic_support=config['semantic_support_config'])
    strategy['chunk_control_config'] = config['chunk_control_config']
    for name in ('fused_support_config', 'api_support_config', 'callable_support_config', 'callable_ablation_config', 'retrieval_timing'):
        if name in config:
            strategy[name] = config[name]
    if profile.support_ablation_config is not None:
        strategy['support_ablation_config'] = config['support_ablation_config']
        strategy['prefix_hash'] = digest(request['prefix'])
    if (config.get('benchmark', 'cceval'), config.get('language', 'python')) != ('cceval', 'python'):
        strategy['benchmark_target'] = [config['benchmark'], config['language'], config['dataset_split'], config['benchmark_adapter_version']]
    selection_key = digest({'window': window, 'version': VERSION, 'seed': config['seed'], 'probe_model': config['probe_model'], 'source': config['source_hashes'], 'runtime': config.get('runtime_versions'), 'e': False, 'coderag_commit': None, 'strategy': strategy})
    location = cache / 'selection' / f'{selection_key}.json'
    if location.exists() and (not online(config)):
        result = load_json(location)
        if profile.support_ablation_config is not None:
            if result.get('ablation_selection_hash') != digest({k: result[k] for k in ('window', 'structure', 'ablation_pool', 'chunk_features')}):
                raise ValueError('Support ablation selection cache checksum mismatch')
        elif profile.chunk_control_config is not None and result.get('chunk_selection_hash') != digest({k: result[k] for k in ('window', 'structure', 'semantic_pool', 'chunk_features')}):
            raise ValueError('Chunk selection cache checksum mismatch')
        return result
    parsing = time.perf_counter()
    profile = PROFILES[config['method']]
    if profile.callable_ablation_config is not None:
        from choruscoder_callable_ablations import build_selection
        from choruscoder_profiles import SemanticSupportConfig
        window, structure, chunk_pool, chunk_features = build_selection(request['prefix'], tokenizer, profile, SemanticSupportConfig(**config['semantic_support_config']), config.get('language', 'python'), request['path'])
    elif profile.support_ablation_config is not None:
        from choruscoder_support_ablations import build_support_selection
        window, structure, chunk_pool, chunk_features = build_support_selection(request['prefix'], tokenizer, profile, config.get('language', 'python'), request['path'])
    else:
        from choruscoder_chunk_controls import build_chunk_selection
        from choruscoder_profiles import SemanticSupportConfig
        window, structure, chunk_pool, chunk_features = build_chunk_selection(request['prefix'], tokenizer, profile.support_unit_config, SemanticSupportConfig(**config['semantic_support_config']), profile.chunk_control_config, language=config.get('language', 'python'), path=request['path'], reserve_near=profile.fused_support_config is None)
    prompts = [source_query(window['text'], c['context_spans'] + structure['target_context_spans']) for c in structure['candidates']]
    parse_seconds = time.perf_counter() - parsing + prefix_window_seconds
    started = time.perf_counter()
    scores = [float(max((end for _, end in c['spans']))) for c in structure['candidates']]
    parse_seconds += time.perf_counter() - started
    result = {'window': window, 'structure': structure, 'scores': scores, 'eligible_supports': len(prompts), 'probe_seconds': 0.0, 'parse_seconds': parse_seconds}
    if profile.support_ablation_config is not None:
        result.update(ablation_pool=chunk_pool, chunk_features=chunk_features)
        result['ablation_selection_hash'] = digest({k: result[k] for k in ('window', 'structure', 'ablation_pool', 'chunk_features')})
    if profile.semantic_support_config is not None:
        from choruscoder_profiles import SemanticSupportConfig
        semantic_started = time.perf_counter()
        result['semantic_pool'] = chunk_pool
        result['chunk_features'] = chunk_features
        result['chunk_selection_hash'] = digest({k: result[k] for k in ('window', 'structure', 'semantic_pool', 'chunk_features')})
        result['semantic_pool_seconds'] = time.perf_counter() - semantic_started
        result['parse_seconds'] += result['semantic_pool_seconds']
    if config.get('retrieval_timing'):
        result['selection_seconds'] = time.perf_counter() - selection_started
    if not online(config):
        atomic_json(location, result)
    return result

def create_retriever(cli, config, cache, tokenizer, enabled=True):
    if not enabled:
        return (None, 0.0)
    from benchmarking.choruscoder_cast import CastDenseIndex
    profile = PROFILES[cli.method]
    loading = time.perf_counter()
    encoder_options = {'encoder_config': profile.encoder_config} if profile.encoder_config is not None else {}
    backend = CastDenseIndex(cache, tokenizer, cli.embedding_model, repository_config=profile.repository_chunk_config, language=cli.language, chunk_tokens=config['chunk_tokens'], overlap_tokens=config['overlap_tokens'], **encoder_options)
    if cli.benchmark == 'recceval' or profile.api_support_config is not None:
        backend.record_repository_timings = True
    backend.cache_query_vectors = bool(profile.fused_support_config is not None or profile.support_ablation_config is not None or cli.benchmark == 'recceval')
    if config.get('repository_runtime_cache') is not None:
        from benchmarking.choruscoder_runtime_cache import enable_runtime_cache
        enable_runtime_cache(backend, config['repository_runtime_cache'])
    return (backend, time.perf_counter() - loading)

def prepare(cli, config, output, cache):
    if PROFILES[cli.method].rlcoder_config is not None:
        from benchmarking.rlcoder_backend import prepare as prepare_rlcoder
        return prepare_rlcoder(cli, config, output, cache)
    from transformers import AutoTokenizer
    from benchmarking.choruscoder_encoder_registry import load_tokenizer
    from benchmarking.choruscoder_models import Probe
    cli.query_max_segments = segment_limit(cli.method, cli.query_max_segments)
    started_prepare = time.perf_counter()
    progress = ProgressReporter(output)
    progress.update('repository_preprocessing' if online(config) else 'candidate_selection', 0, force=True)
    load_stats = {'probe_model_seconds': 0.0, 'retriever_model_seconds': 0.0}
    generator_tokenizer = AutoTokenizer.from_pretrained(cli.model)
    run_profile = PROFILES[cli.method]
    tokenizer = load_tokenizer(cli.embedding_model, run_profile.encoder_config)
    if run_profile.encoder_config is not None and run_profile.api_support_config is not None:
        from choruscoder_profiles import ApiSupportConfig
        run_profile = replace(run_profile, api_support_config=ApiSupportConfig(**config['api_support_config']))
    config = dict(config, coderag_root=cli.coderag_root)
    requests = output / 'requests'
    requests.mkdir(parents=True, exist_ok=True)
    prepared_dir = output / 'prepared'
    prepared_dir.mkdir(parents=True, exist_ok=True)
    canonical_mode = canonical_method(cli.method)
    cache_config = dict(config, method=canonical_mode)
    fingerprint = digest(cache_config)
    jobs, probe = ([], None)
    backend = None
    from benchmarking.retrieval_online import local_preparation_complete
    if online(config) and (not local_preparation_complete(output, fingerprint)):
        from benchmarking.retrieval_online import OnlineDenseIndex, preprocess_dataset
        progress.update('repository_preprocessing', 0, force=True)
        raw_backend, load_stats['retriever_model_seconds'] = create_retriever(cli, config, cache, tokenizer)
        backend = OnlineDenseIndex(raw_backend, cache)
        try:
            preprocess_dataset(cli, backend, output)
        except BaseException:
            backend.close()
            raise
        progress.update('candidate_selection', 0, force=True)
    reused_probe_seconds, probe_cache_hits = (0.0, 0)

    def get_probe():
        nonlocal probe
        if probe is None:
            if online(config) and PROFILES[cli.method].coderag_query_config is not None:
                switching = time.perf_counter()
                backend.set_encoder_device('cpu')
                load_stats['retriever_park_seconds'] = time.perf_counter() - switching
            loading = time.perf_counter()
            probe = Probe(cli.query_probe_model, seed=cli.seed)
            load_stats['probe_model_seconds'] += time.perf_counter() - loading
        return probe
    try:
        columns = ('task_id', 'path', 'left_context', 'crossfile_context')
        for rows in iter_parquet_batches(cli.parquet, columns, cli.batch_size, cli.limit):
            for row in rows:
                request = safe_request(row)
                key = identity(request)
                prepared_path = prepared_dir / f'{key}.json' if online(config) else cache / 'prepared' / fingerprint / f'{key}.json'
                if prepared_path.exists() and (not online(config) or load_json(prepared_path).get('config_hash') == fingerprint):
                    record = load_json(prepared_path)
                    from benchmarking.choruscoder_timing import validate_timing
                    validate_timing(record, config)
                    if record['request_hash'] != key:
                        raise ValueError('Prepared cache mismatch')
                    if record['evidence_hash'] != digest(record['evidence']):
                        raise ValueError('Prepared evidence cache checksum mismatch')
                    if PROFILES[cli.method].coderag_query_config is not None:
                        from choruscoder_coderag_query import validate_record
                        validate_record(record, config, tokenizer)
                    if PROFILES[cli.method].prefix_query_tokens is not None:
                        from choruscoder_prefix_query import validate_prefix_record
                        validate_prefix_record(record, config, tokenizer)
                    if PROFILES[cli.method].support_ablation_config is not None:
                        from choruscoder_support_ablations import validate_ablation_record
                        validate_ablation_record(record, config, tokenizer)
                    elif PROFILES[cli.method].chunk_control_config is not None:
                        from choruscoder_chunk_controls import validate_chunk_record
                        validate_chunk_record(record, config, tokenizer)
                    if PROFILES[cli.method].context_config is not None and record.get('context_profile') != config['context_profile']:
                        raise ValueError('Prepared context budget cache mismatch')
                    atomic_json(prepared_dir / f'{key}.json', record)
                    jobs.append({'key': key, 'prepared': str(prepared_path.resolve()), 'ready': True})
                    progress.update('candidate_selection', len(jobs), task_id=request['task_id'])
                    continue
                if PROFILES[cli.method].candidate_mode == 'window':
                    window_started = time.perf_counter()
                    profile = PROFILES[cli.method]
                    prefix_seconds = 0.0
                    if profile.prefix_query_tokens is not None:
                        from choruscoder_prefix_query import prefix_query_view
                        prefix_started = time.perf_counter()
                        window = prefix_query_view(request['prefix'], tokenizer, profile.prefix_query_tokens)
                        prefix_seconds = time.perf_counter() - prefix_started
                    else:
                        window = window_view(request['prefix'], tokenizer)
                    selected = {'window': window, 'structure': {'target': '', 'candidates': [], 'fallback': None}, 'scores': [], 'eligible_supports': 0, 'probe_seconds': 0.0, **({'selection_seconds': prefix_seconds} if profile.prefix_query_tokens is not None else {})}
                    if online(config):
                        selected['selection_seconds'] = time.perf_counter() - window_started
                else:
                    selected = selection_record(request, config, cache, tokenizer, get_probe)
                if selected.get('probe_cache_source'):
                    probe_cache_hits += 1
                    reused_probe_seconds += selected['probe_seconds']
                request['selection'] = selected
                request['config_hash'] = fingerprint
                atomic_json(requests / f'{key}.json', request)
                jobs.append({'key': key, 'request': str((requests / f'{key}.json').resolve()), 'prepared': str(prepared_path.resolve()), 'ready': False})
                progress.update('candidate_selection', len(jobs), task_id=request['task_id'])
    except BaseException:
        if backend is not None:
            backend.close()
        raise
    finally:
        if probe is not None:
            probe.close()
    manifest = output / 'jobs.json'
    atomic_json(manifest, {'config_hash': fingerprint, 'jobs': jobs})
    pending = [job for job in jobs if not job['ready']]
    completed_prepare = len(jobs) - len(pending)
    if online(config) and pending and (backend is None):
        raise ValueError('Incomplete local preparation manifest; use a new output directory')
    progress.update('retriever_loading' if pending else 'retrieval_preparation', completed_prepare, len(jobs), force=True)
    if backend is not None and getattr(backend, 'encoder_parked', False):
        switching = time.perf_counter()
        try:
            backend.set_encoder_device('cuda')
        except BaseException:
            backend.close()
            raise
        load_stats['retriever_restore_seconds'] = time.perf_counter() - switching
    if not online(config):
        backend, load_stats['retriever_model_seconds'] = create_retriever(cli, config, cache, tokenizer, bool(pending))
    progress.update('retrieval_preparation', completed_prepare, len(jobs), force=True)
    try:
        for job in pending:
            request = load_json(job['request'])
            selected = request['selection']
            profile = run_profile
            started = time.perf_counter()
            if online(config):
                backend.begin_question()
            semantic_trace = None
            semantic_scoring_seconds = 0.0
            semantic_encoding_stats = {}
            fused_trace = None
            target_query_trace = None
            ablation_selection_trace = None
            ablation_query_trace = None
            api_result = None
            api_trace = None
            if profile.api_support_config is not None:
                from choruscoder_api_support import retrieve_supports
                from choruscoder_profiles import SemanticSupportConfig
                api_result, api_trace = retrieve_supports(request, selected, backend, generator_tokenizer, profile, SemanticSupportConfig(**config['semantic_support_config']))
            if profile.coderag_query_config is not None:
                chosen, queries = (selected['selected_segments'], selected['queries'])
            elif api_result is not None:
                chosen, queries = (api_result['chosen'], api_result['queries'])
                semantic_trace, fused_trace = (api_result['semantic'], api_result['fused'])
                semantic_encoding_stats = api_result['encoding']
            else:
                profile = PROFILES[cli.method]
                if profile.support_ablation_config is not None and profile.support_ablation_config.selection == 'prefix_all':
                    from choruscoder_support_ablations import select_prefix_all
                    from choruscoder_selection_controls import finalize_fused_fallback
                    chosen, ablation_selection_trace = select_prefix_all(selected['ablation_pool'], request['prefix'], tokenizer)
                    if not chosen:
                        selected['structure']['fallback'] = 'no_prefix100_supports'
                    finalize_fused_fallback(selected['structure'], request['prefix'], tokenizer, chosen)
                else:
                    chosen = select_segments(selected['structure'], selected['scores'], support_segment_limit(cli.method, cli.query_max_segments))
                queries = build_queries(cli.method, selected['window'], selected['structure'], chosen, prefix=request['prefix'])
                if profile.support_ablation_config is not None:
                    from choruscoder_support_ablations import ablation_queries
                    queries, ablation_query_trace = ablation_queries(cli.method, request['prefix'], selected['structure'], chosen, tokenizer)
                elif profile.chunk_control_config is not None:
                    from choruscoder_chunk_controls import chunk_queries
                    queries = chunk_queries(cli.method, selected['window'], selected['structure'], chosen, prefix=request['prefix'])
                    if profile.fused_support_config is not None:
                        from choruscoder_selection_controls import control_queries
                        queries, target_query_trace = control_queries(cli.method, selected['window'], selected['structure'], chosen, prefix=request['prefix'])
                    if any((len(tokenizer.encode(q['text'], add_special_tokens=False)) > 508 for q in queries)):
                        raise ValueError('Chunk experiment query exceeds UniXcoder budget')
            quotas = method_recall_quotas(cli.method, len(queries), query_roles(queries))
            if api_result is not None:
                ranked = api_result['ranked']
            else:
                ranked = backend.retrieve(queries, request['files'], quotas)
            ordered = round_robin_evidence(queries, ranked)
            evidence = pack_evidence(ordered, generator_tokenizer, budget=config.get('evidence_body_tokens', 896), maximum=None, language=cli.language)
            index_seconds = getattr(backend, 'last_index_seconds', 0.0)
            retrieval_seconds = max(0.0, time.perf_counter() - started - index_seconds)
            profile = PROFILES[cli.method]
            raw = {'recall_quotas': quotas, 'ranked': ranked, 'index_build_seconds': index_seconds}
            if profile.coderag_query_config is not None:
                raw['coderag_selection'] = selected
            if profile.support_ablation_config is not None:
                raw['support_ablation'] = {'configuration': config['support_ablation_config'], 'pool': selected['ablation_pool'], 'selection': ablation_selection_trace, 'queries': ablation_query_trace}
            if profile.repository_chunk_config is not None:
                raw['repository_chunks'] = backend.last_repository_chunks
            if config.get('repository_runtime_cache') is not None:
                raw['repository_runtime_cache'] = dict(configuration=config['repository_runtime_cache'], validated_chunks=backend._validated_chunk_cache.statistics(), embedding_vectors=backend._repository_vector_cache.statistics() if getattr(backend, '_repository_vector_cache', None) is not None else None, timings=getattr(backend, 'last_repository_runtime_timings', {}))
            if semantic_trace is not None:
                raw.update(semantic_support=semantic_trace, semantic_pool_seconds=selected['semantic_pool_seconds'], semantic_scoring_seconds=semantic_scoring_seconds)
            if fused_trace is not None:
                raw['fused_support'] = fused_trace
            if api_trace is not None:
                raw['api_support'] = api_trace
            if fused_trace is not None:
                raw['semantic_encoding_cache'] = semantic_encoding_stats
            if getattr(backend, 'cache_query_vectors', False):
                raw['query_encoding_cache'] = getattr(backend, 'last_query_encoding_cache', {})
                raw['query_encoding_seconds'] = getattr(backend, 'last_query_encoding_seconds', 0.0)
            if target_query_trace is not None:
                raw['target_query'] = target_query_trace
            if profile.chunk_control_config is not None:
                raw['chunk_controls'] = selected['chunk_features']
            profile = PROFILES[cli.method]
            if profile.support_unit_config is not None:
                from choruscoder_support_units import unit_summary
                candidates = selected['structure']['candidates']
                if semantic_trace is not None:
                    candidates = candidates + semantic_trace['candidates']
                raw['support_units'] = unit_summary(candidates, chosen, queries, tokenizer)
                raw['support_units']['coverage'] = query_evidence_coverage(queries, ranked, evidence)
            structure = selected.get('structure', {})
            details = dict(structure.get('diagnostics', {}))
            if PROFILES[cli.method].candidate_lines is not None:
                details['candidate_lines'] = PROFILES[cli.method].candidate_lines
            if 'target' in structure:
                details.update(target=structure['target'], target_body_spans=structure.get('target_body_spans', [structure.get('target_span')]), target_context_spans=structure.get('target_context_spans', []), target_tokens=len(generator_tokenizer.encode(structure['target'], add_special_tokens=False)), original_target_tokens=len(generator_tokenizer.encode(details.get('original_target', structure['target']), add_special_tokens=False)), candidates=structure.get('candidates', []), scores=selected.get('scores', []))
                if profile.chunk_control_config is not None and 'fallback_query_text' in structure:
                    details['fallback_query_text'] = structure['fallback_query_text']
            if profile.semantic_support_config is not None:
                details['prefix_target_context_spans'] = structure.get('prefix_target_context_spans', [])
            raw['ablation'] = details
            if selected.get('probe_cache_source'):
                raw['probe_cache_source'] = selected['probe_cache_source']
            record = {'task_id': request['task_id'], 'path': request['path'], 'prefix': request['prefix'], 'window': selected['window'], 'queries': queries, 'selected_segments': chosen, 'eligible_supports': selected['eligible_supports'], 'fallback': selected.get('fallback', selected.get('structure', {}).get('fallback')), 'evidence': evidence, 'evidence_tokens': len(generator_tokenizer.encode(render_evidence(evidence, cli.language), add_special_tokens=False)), 'evidence_hash': digest(evidence), 'config_hash': fingerprint, 'request_hash': job['key'], 'probe_seconds': selected['probe_seconds'], 'retrieval_seconds': retrieval_seconds, 'selection_seconds': selected.get('selection_seconds', selected['probe_seconds'] + selected.get('parse_seconds', 0.0)), 'diagnostics': raw, 'official': False}
            if api_trace is not None:
                api_trace['full_retrieval_seconds'] = record['selection_seconds'] + record['retrieval_seconds'] + index_seconds
            if (cli.benchmark, cli.language) != ('cceval', 'python'):
                record.update(benchmark=cli.benchmark, language=cli.language, dataset_split=cli.dataset_split)
            record['context_profile'] = config['context_profile']
            usage = {}
            construct_prompt(record, generator_tokenizer, usage=usage)
            raw['context_usage'] = usage
            if config.get('retrieval_timing'):
                from benchmarking.choruscoder_timing import finish_retrieval
                finish_retrieval(record, started, config)
                if online(config):
                    raw['online_computation'] = dict(task_encoded_texts=backend.task_encoded_texts, task_encoding_seconds=backend.task_encoding_seconds, task_cache='question_only')
            if profile.support_ablation_config is not None:
                from choruscoder_support_ablations import ablation_checksum, validate_ablation_record
                raw['support_ablation_checksum'] = ablation_checksum(record)
                validation_started = time.perf_counter()
                validate_ablation_record(record, config)
                raw['preparation_validation_seconds'] = time.perf_counter() - validation_started
            elif profile.chunk_control_config is not None:
                from choruscoder_chunk_controls import chunk_checksum, validate_chunk_record
                raw['chunk_checksum'] = chunk_checksum(record)
                validation_started = time.perf_counter()
                validate_chunk_record(record, config)
                raw['preparation_validation_seconds'] = time.perf_counter() - validation_started
            if profile.prefix_query_tokens is not None:
                from choruscoder_prefix_query import validate_prefix_record
                validate_prefix_record(record, config)
            if profile.coderag_query_config is not None:
                from choruscoder_coderag_query import validate_record
                validate_record(record, config)
            atomic_json(job['prepared'], record)
            if Path(job['prepared']).resolve() != (prepared_dir / f"{job['key']}.json").resolve():
                atomic_json(prepared_dir / f"{job['key']}.json", record)
            if online(config):
                backend.end_question()
            completed_prepare += 1
            progress.update('retrieval_preparation', completed_prepare, len(jobs), request['task_id'], details={name: raw.get(name) for name in ('semantic_scoring_seconds', 'semantic_encoding_cache', 'query_encoding_seconds', 'query_encoding_cache', 'preparation_validation_seconds', 'index_build_seconds')} | {'repository_chunking_seconds': getattr(backend, 'last_chunk_seconds', 0.0)})
    finally:
        if backend is not None:
            backend.close()
    atomic_json(output / 'prepared_manifest.json', {'config_hash': fingerprint, 'tasks': [{'key': j['key'], 'file': str((prepared_dir / f"{j['key']}.json").resolve())} for j in jobs]})
    atomic_json(output / 'prepare_stats.json', {'wall_seconds': time.perf_counter() - started_prepare, 'prepared_cache_hits': len(jobs) - len(pending), 'prepared_cache_misses': len(pending), 'reused_logical_probe_seconds': reused_probe_seconds, 'probe_cache_hits': probe_cache_hits, 'model_load_seconds': load_stats, 'note': 'Generation timings reuse logical prepare cost; Cache hits are recorded separately from computation.'})
    progress.update('prepare_complete', len(jobs), len(jobs), force=True)

def generate(cli, output):
    from transformers import AutoTokenizer
    from openai import OpenAI
    manifest = load_json(output / 'prepared_manifest.json')
    expected = dict(load_json(output / 'run_config.json')['configuration'])
    if PROFILES[cli.method].rlcoder_config is None:
        expected['coderag_root'] = cli.coderag_root
    expected['method'] = canonical_method(cli.method)
    max_new_tokens = validate_generation_tokens(expected['new_tokens'])
    if cli.max_new_tokens != max_new_tokens:
        raise ValueError('Generation max-new-tokens differs from the prepared configuration')
    if manifest['config_hash'] != digest(expected):
        raise ValueError('Preparation configuration does not match generation')
    tokenizer = AutoTokenizer.from_pretrained(cli.model)
    actual_inputs = PROFILES[cli.method].rlcoder_config is not None
    definitions = None
    if PROFILES[cli.method].rlcoder_config is not None:
        from benchmarking.rlcoder_backend import upstream
        definitions = upstream(cli.rlcoder_root)[1]
    journal = output / 'generated'
    journal.mkdir(exist_ok=True)
    client = OpenAI(base_url=normalize_api_base(cli.vllm_api_base), api_key=cli.vllm_api_key, timeout=cli.vllm_request_timeout, max_retries=2)
    progress = ProgressReporter(output)
    completed_generation = 0
    progress.update('generation', 0, len(manifest['tasks']), force=True)
    service_verified = False
    try:
        tasks = manifest['tasks']
        for start in range(0, len(tasks), cli.batch_size):
            batch = []
            for task in tasks[start:start + cli.batch_size]:
                record = load_json(task['file'])
                from benchmarking.choruscoder_timing import validate_timing
                validate_timing(record, expected)
                if record['config_hash'] != manifest['config_hash'] or record['request_hash'] != task['key']:
                    raise ValueError('Prepared task identity mismatch')
                if record['evidence_hash'] != digest(record['evidence']):
                    raise ValueError('Prepared evidence checksum mismatch')
                if PROFILES[cli.method].rlcoder_config is not None:
                    from benchmarking.rlcoder_backend import validate_record
                    validate_record(record, expected, tokenizer, definitions)
                if PROFILES[cli.method].coderag_query_config is not None:
                    from choruscoder_coderag_query import validate_record
                    validate_record(record, expected)
                if PROFILES[cli.method].prefix_query_tokens is not None:
                    from choruscoder_prefix_query import validate_prefix_record
                    validate_prefix_record(record, expected)
                if PROFILES[cli.method].support_ablation_config is not None:
                    from choruscoder_support_ablations import validate_ablation_record
                    validate_ablation_record(record, expected)
                elif PROFILES[cli.method].chunk_control_config is not None:
                    from choruscoder_chunk_controls import validate_chunk_record
                    validate_chunk_record(record, expected)
                if expected.get('context_profile') is not None and record.get('context_profile') != expected['context_profile']:
                    raise ValueError('Prepared context budget does not match generation')
                if actual_inputs:
                    from benchmarking.choruscoder_inputs import validate_input
                    payload = validate_input(record, tokenizer, definitions)
                    prompt = payload['token_ids']
                else:
                    prompt = construct_prompt(record, tokenizer)
                location = journal / f"{task['key']}.json"
                request_hash = digest([prompt, cli.model, cli.seed, max_new_tokens, 0, 1])
                if location.exists():
                    cached_generation = load_json(location)
                    if actual_inputs:
                        from benchmarking.choruscoder_inputs import generation_checksum
                        if cached_generation.get('generation_checksum') != generation_checksum(cached_generation):
                            raise ValueError('Actual-input generation cache checksum mismatch')
                    if PROFILES[cli.method].rlcoder_config is not None:
                        from benchmarking.rlcoder_backend import record_checksum
                        if cached_generation.get('rlcoder_checksum') != record_checksum(cached_generation):
                            raise ValueError('RLCoder generation cache checksum mismatch')
                    if cached_generation['generation_hash'] != request_hash:
                        raise ValueError('Generation cache/config mismatch')
                    completed_generation += 1
                    continue
                batch.append((record, prompt, location, request_hash))
            if not batch:
                progress.update('generation', completed_generation, len(tasks))
                continue
            progress.update('generation', completed_generation, len(tasks))
            if actual_inputs and (not service_verified):
                from benchmarking.choruscoder_inputs import verify_service
                verify_service(client, cli.model, tokenizer, batch[0][0]['generation_input']['prompt_text'])
                service_verified = True
            started = time.perf_counter()
            result = client.completions.create(model=cli.model, prompt=[item[1] for item in batch], max_tokens=max_new_tokens, temperature=0, top_p=1, seed=cli.seed)
            if actual_inputs:
                from benchmarking.choruscoder_inputs import verify_usage
                verify_usage(result, [item[1] for item in batch])
            elapsed = (time.perf_counter() - started) / len(batch)
            choices = sorted(result.choices, key=lambda c: c.index)
            if [c.index for c in choices] != list(range(len(batch))):
                raise RuntimeError('Generator returned incomplete batch')
            for (record, prompt, location, request_hash), choice in zip(batch, choices):
                generated = {'task_id': record['task_id'], 'pred': choice.text, 'generation_hash': request_hash, 'prompt_hash': digest(prompt), 'max_new_tokens': max_new_tokens, 'finish_reason': getattr(choice, 'finish_reason', None), 'prompt_tokens': len(prompt) if actual_inputs else len(tokenizer.encode(prompt, add_special_tokens=False)), 'query_count': len(record['queries']), 'evidence_count': len(record['evidence']), 'eligible_supports': record['eligible_supports'], 'fallback': record['fallback'], 'evidence_hash': record['evidence_hash'], 'config_hash': record['config_hash'], 'retrieval_seconds': record.get('selection_seconds', record['probe_seconds']) + record['retrieval_seconds'], 'generation_seconds': elapsed, 'end_to_end_seconds': record.get('selection_seconds', record['probe_seconds']) + record['retrieval_seconds'] + elapsed}
                if actual_inputs:
                    from benchmarking.choruscoder_inputs import generation_checksum
                    generated.update(input_budget_version=expected['input_budget_version'], tokenizer_identity=record['generation_input']['tokenizer_identity'])
                    generated['generation_checksum'] = generation_checksum(generated)
                if PROFILES[cli.method].rlcoder_config is not None:
                    from benchmarking.rlcoder_backend import record_checksum
                    generated['rlcoder_checksum'] = record_checksum(generated)
                atomic_json(location, generated)
            completed_generation += len(batch)
            progress.update('generation', completed_generation, len(tasks), batch[-1][0]['task_id'])
    finally:
        client.close()
    from benchmarking.common import AtomicJsonlWriter
    with AtomicJsonlWriter(output / 'predictions.jsonl') as predictions, AtomicJsonlWriter(output / 'timings.jsonl') as timings, AtomicJsonlWriter(output / 'retrieval.jsonl') as traces:
        for task in manifest['tasks']:
            value = load_json(journal / f"{task['key']}.json")
            predictions.write({'task_id': value['task_id'], 'pred': value['pred']})
            timing = {k: value[k] for k in ('task_id', 'retrieval_seconds', 'generation_seconds', 'end_to_end_seconds')}
            if online(expected):
                timing.update(retrieval_timing=expected['retrieval_timing'], allocation='per_question_actual_compute')
            timings.write(timing)
            record = load_json(task['file'])
            record.pop('prefix', None)
            traces.write(dict(record, generation=value))
    progress.update('generation_complete', len(tasks), len(tasks), force=True)

def evaluate(cli, output):
    from benchmarking.evaluation import evaluate_predictions, aggregate_timings
    reference = Path(cli.parquet[0]).parent / 'test.jsonl'
    metrics = evaluate_predictions(output / 'predictions.jsonl', reference, cli.language, output, require_complete=cli.limit is None, benchmark=cli.benchmark)
    result = dict(method=cli.method, model=Path(cli.model).name, benchmark=cli.benchmark, language=cli.language, **metrics, **aggregate_timings(output / 'timings.jsonl'))
    if getattr(cli, 'retrieval_timing', None) == 'online':
        from benchmarking.retrieval_online import ONLINE_TIMING
        result['retrieval_timing'] = ONLINE_TIMING
    atomic_json(output / 'result.json', result)
    return result

def configuration_differences(recorded, current, prefix=''):
    """Describe incompatible fields without printing model/data contents."""
    changed = []
    for key in sorted(recorded.keys() | current.keys()):
        name = prefix + key
        if key not in recorded or key not in current:
            changed.append(name)
        elif isinstance(recorded[key], dict) and isinstance(current[key], dict):
            changed.extend(configuration_differences(recorded[key], current[key], name + '.'))
        elif recorded[key] != current[key]:
            changed.append(name)
    return changed

def run(cli):
    config = configuration(cli)
    from choruscoder_benchmarks import uses_cast_adapter
    if uses_cast_adapter(cli.method, cli.benchmark):
        for variant in config['grammar_identities']:
            from choruscoder import make_parser
            make_parser(variant)
    output = Path(cli.output_dir)
    cache = Path(cli.query_cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'run_config.json'
    if config_path.exists() and load_json(config_path)['configuration'] != config:
        changed = configuration_differences(load_json(config_path)['configuration'], config)
        detail = ', '.join(changed[:20]) + (' ...' if len(changed) > 20 else '')
        raise ValueError('Output directory contains a different experiment configuration; changed fields: ' + detail + '. Use a new output directory, or restore the original implementation to resume this run.')
    atomic_json(config_path, {'configuration': config, 'status': 'running', 'stage': cli.stage})
    if cli.stage in {'prepare', 'all'} and (not False):
        prepare(cli, config, output, cache)
    if cli.stage in {'generate', 'all'}:
        generate(cli, output)
    if cli.stage in {'evaluate', 'all'}:
        ProgressReporter(output).update('evaluation', 0, force=True)
        result = evaluate(cli, output)
        total = result['total_samples']
        ProgressReporter(output).update('complete', total, total, force=True)
    atomic_json(config_path, {'configuration': config, 'status': 'complete' if (output / 'result.json').exists() else 'prepared' if cli.stage == 'prepare' else 'generated', 'stage': cli.stage})
if __name__ == '__main__':
    run(build_parser().parse_args())
