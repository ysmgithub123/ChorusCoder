"""Run registered methods, individually or as one validated matrix."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from method_registry import METHODS, CHORUSCODER_METHODS, PREFIX100_ALL
from infile_profiles import METHODS as INFILE_METHODS, PROFILES as INFILE_PROFILES, validate_options as validate_infile_options, experiment_target as infile_target
from choruscoder_profiles import PROFILES, segment_limit, context_config
from choruscoder_benchmarks import experiment_target
from greprag_profiles import METHODS as GREPRAG_METHODS
from aircoder_profiles import METHODS as AIRCODER_BUDGET_METHODS
from benchmarking.common import benchmark_targets, ensure_dir, model_slug, normalize_api_base, resolve_dataset_shards, write_json, validate_generation_tokens

def explicit(argv, flag):
    return any((word == flag or word.startswith(flag + '=') for word in argv))

def without_options(argv, options):
    result = []
    index = 0
    while index < len(argv):
        word = argv[index]
        flag = word.split('=', 1)[0]
        if flag in options:
            index += 1 if '=' in word or flag == '--greprag-resume' else 2
        else:
            result.append(word)
            index += 1
    return result

def all_method_runs(args, argv, parser):
    if (args.benchmark, args.language, args.subset) != ('cceval', 'python', 'all'):
        parser.error('--method all requires --benchmark cceval --language python; select an individual supported method for ReccEval')
    for flag in ('--query-max-segments', '--query-support-similarity-threshold', '--query-interface-similarity-threshold', '--query-interface-margin-threshold'):
        if explicit(argv, flag):
            parser.error(flag + ' requires an individual method; all uses each fixed profile')
    runs = []
    for method in METHODS:
        ignored = {'--method', '--retrieval-timing', '--output-root'}
        if method in INFILE_METHODS:
            ignored.add('--max-crossfile-tokens')
        if method == 'RLCoder' or (PROFILES.get(method) is not None and PROFILES[method].encoder_config is not None):
            ignored.add('--embedding-model')
        if method not in AIRCODER_BUDGET_METHODS:
            ignored.update(('--dependency-model', '--dependency-vllm-api-base'))
        if method not in GREPRAG_METHODS:
            ignored.add('--greprag-resume')
        if method == PREFIX100_ALL:
            timing = 'full'
        elif method in CHORUSCODER_METHODS:
            timing = args.retrieval_timing or 'online'
        else:
            timing = 'online'
        values = without_options(argv, ignored)
        values += ['--method', method, '--output-root', args.output_root, '--retrieval-timing', timing]
        runs.append(parse_args(values))
    return tuple(runs)

def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=(*METHODS, 'all'), required=True)
    parser.add_argument('--model', required=True, help='Hugging Face ID or local generator model path')
    parser.add_argument('--dependency-model', default='same', help="AIRCoder dependency extractor; 'same' enforces the same model as --model")
    parser.add_argument('--allow-different-dependency-model', action='store_true')
    parser.add_argument('--benchmark', choices=('cceval', 'recceval'), default='cceval')
    parser.add_argument('--language', choices=['python'], required=True)
    parser.add_argument('--dataset-root', default='data')
    parser.add_argument('--output-root', default='benchmark_results')
    parser.add_argument('--infile-cache-dir', default='benchmark_cache/infile')
    parser.add_argument('--aircoder-root', default='third_party/AIRCoder')
    parser.add_argument('--aircoder-repos-root', default='data_sources/crosscodeeval_repositories')
    parser.add_argument('--choruscoder-python', required=True, help='Python executable for the main ChorusCoder environment')
    parser.add_argument('--aircoder-python', help='Required for AIRCoder-RRF or all: Python executable for its isolated environment')
    parser.add_argument('--vllm-api-base', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--dependency-vllm-api-base', help='Optional second endpoint for dependency-model ablations')
    parser.add_argument('--vllm-api-key', default='EMPTY')
    parser.add_argument('--vllm-request-timeout', type=float, default=7200)
    parser.add_argument('--embedding-model', default='microsoft/unixcoder-base')
    parser.add_argument('--greprag-model', default='models/greprag-0.6b')
    parser.add_argument('--greprag-python', help='Required for GrepRAG-0.6B or all: keyword-worker Python executable')
    parser.add_argument('--greprag-corpus', choices=('parquet',), default='parquet')
    parser.add_argument('--greprag-repos-root', default='data_sources/crosscodeeval_repositories')
    parser.add_argument('--greprag-resume', action='store_true', help='Verify and continue one interrupted GrepRAG retrieval run')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--dataloader-num-workers', type=int, default=0)
    parser.add_argument('--dataloader-prefetch-factor', type=int, default=1)
    parser.add_argument('--max-context-tokens', type=int, default=4096)
    parser.add_argument('--max-crossfile-tokens', type=int, default=3072)
    parser.add_argument('--max-new-tokens', type=int, default=50, help='Maximum newly generated completion tokens; overrides the method default')
    parser.add_argument('--aircoder-slice-size', type=int, default=800)
    parser.add_argument('--aircoder-num-rags', type=int, default=10)
    parser.add_argument('--dependency-max-input-tokens', type=int, default=2048)
    parser.add_argument('--dependency-max-new-tokens', type=int, default=100)
    parser.add_argument('--aircoder-metrics', nargs='+', default=['sparse', 'dense', 'file_es', 'file_dist', 'class_es', 'def_es', 'skelton', 'dataflow'])
    parser.add_argument('--aircoder-cache-dir', default='benchmark_cache/aircoder')
    parser.add_argument('--query-stage', choices=('prepare', 'generate', 'evaluate', 'all'), default='all')
    parser.add_argument('--query-max-segments', type=int)
    parser.add_argument('--query-support-similarity-threshold', type=float, default=None, help='Support cosine/fused-score threshold; default comes from the method')
    parser.add_argument('--query-interface-similarity-threshold', type=float, default=None)
    parser.add_argument('--query-interface-margin-threshold', type=float, default=None)
    parser.add_argument('--query-cache-dir', default='benchmark_cache/choruscoder')
    parser.add_argument('--query-probe-model', default='Salesforce/codet5p-220m')
    parser.add_argument('--retrieval-timing', choices=('full', 'online'), default=None, help='Retrieval accounting: full includes repository encoding; online prepares repositories first')
    parser.add_argument('--coderag-root', default='third_party/CodeRAG')
    parser.add_argument('--aircoder-recceval-root', default='data_sources/recceval')
    parser.add_argument('--greprag-preprocess-cache-dir', default='benchmark_cache/greprag_corpus')
    parser.add_argument('--rlcoder-root', default='third_party/RLCoder')
    parser.add_argument('--rlcoder-revision', default=None)
    parser.add_argument('--use-aircoder-dependency-cache', action='store_true')
    parser.add_argument('--skip-existing', action='store_true', help='Reuse a completed result.json instead of rerunning it')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.subset = 'all'
    if args.method in (*AIRCODER_BUDGET_METHODS, 'all') and (not args.aircoder_python):
        parser.error('--aircoder-python is required for AIRCoder-RRF or all')
    if args.method in (*GREPRAG_METHODS, 'all') and (not args.greprag_python):
        parser.error('--greprag-python is required for GrepRAG-0.6B or all')
    if args.retrieval_timing is None:
        args.retrieval_timing = 'full' if args.method == PREFIX100_ALL else 'online'
    if args.method == 'all':
        args.method_runs = all_method_runs(args, argv, parser)
        return args
    interface_values = (args.query_interface_similarity_threshold, args.query_interface_margin_threshold)
    if any((value is not None for value in interface_values)):
        from choruscoder_profiles import effective_api_support_config
        try:
            if args.method not in PROFILES:
                raise ValueError('Interface threshold overrides require a method with configurable API selection')
            effective_api_support_config(args.method, *interface_values)
        except ValueError as error:
            parser.error(str(error))
    if args.method in AIRCODER_BUDGET_METHODS:
        from aircoder_profiles import PROFILES as air_profiles
        policy = air_profiles[args.method]
        if not explicit(argv, '--dependency-model'):
            args.dependency_model = policy.dependency_model
        if args.dependency_model != policy.dependency_model:
            parser.error(f'{args.method} requires --dependency-model {policy.dependency_model}')
        for field, default in (('max_context_tokens', policy.context_tokens), ('max_crossfile_tokens', policy.crossfile_tokens), ('max_new_tokens', 50)):
            if not explicit(argv, '--' + field.replace('_', '-')):
                setattr(args, field, default)
        if args.retrieval_timing not in (None, 'online'):
            parser.error('AIRCoder requires --retrieval-timing online')
        if args.use_aircoder_dependency_cache:
            parser.error('AIRCoder online timing forbids cross-question dependency caching')
        for flag, expected in (('--aircoder-slice-size', policy.slice_size), ('--aircoder-num-rags', policy.num_rags), ('--dependency-max-input-tokens', policy.dependency_input_tokens), ('--dependency-max-new-tokens', policy.dependency_new_tokens), ('--aircoder-metrics', list(policy.metrics))):
            if explicit(argv, flag) and getattr(args, flag[2:].replace('-', '_')) != expected:
                parser.error(flag + ' differs from the pinned AIRCoder profile')
    if args.greprag_resume and args.method not in GREPRAG_METHODS:
        parser.error('--greprag-resume requires a GrepRAG method')
    if args.retrieval_timing == 'online':
        from choruscoder_profiles import supports_online_retrieval
        if not supports_online_retrieval(args.method) and args.method not in (*INFILE_METHODS, *GREPRAG_METHODS, *AIRCODER_BUDGET_METHODS):
            parser.error('Online timing is not supported by this method')
    if args.method in GREPRAG_METHODS:
        from greprag_profiles import defaults as grep_defaults, validate_options as validate_greprag
        for option, value in grep_defaults(args.method).items():
            if not explicit(argv, '--' + option.replace('_', '-')):
                setattr(args, option, value)
        try:
            validate_greprag(args)
        except ValueError as error:
            parser.error(str(error))
    elif args.method in CHORUSCODER_METHODS:
        from choruscoder_profiles import PREFIX100_CODET5P110M_METHOD, PREFIX100_CODESAGE_METHOD
        if args.method in (PREFIX100_CODET5P110M_METHOD, PREFIX100_CODESAGE_METHOD) and args.query_support_similarity_threshold is not None:
            parser.error(f'{args.method} has no support selection; omit --query-support-similarity-threshold')
        if not explicit(argv, '--embedding-model'):
            from choruscoder_profiles import default_embedding_model
            args.embedding_model = default_embedding_model(args.method)
        if PROFILES[args.method].rlcoder_config is not None:
            if not explicit(argv, '--embedding-model'):
                args.embedding_model = PROFILES[args.method].rlcoder_config.model
            if args.query_max_segments is not None:
                parser.error('RLCoder has no support segments; omit --query-max-segments')
        context = context_config(args.method)
        default_root = 'benchmark_results_choruscoder'
        for option, value in (('max_context_tokens', context.prompt_tokens), ('max_crossfile_tokens', context.crossfile_tokens), ('max_new_tokens', 50), ('output_root', default_root)):
            if not explicit(argv, '--' + option.replace('_', '-')):
                setattr(args, option, value)
        try:
            args.query_max_segments = segment_limit(args.method, args.query_max_segments)
        except ValueError as error:
            parser.error(str(error))
    elif args.method in INFILE_METHODS:
        profile = INFILE_PROFILES[args.method]
        for option, value in (('max_context_tokens', profile.prompt_tokens), ('max_crossfile_tokens', 0), ('max_new_tokens', profile.new_tokens), ('output_root', 'benchmark_results_infile_ctx4096_v1')):
            if not explicit(argv, '--' + option.replace('_', '-')):
                setattr(args, option, value)
        try:
            validate_infile_options(args)
            if args.query_max_segments is not None:
                raise ValueError('In-File has no support segments; omit --query-max-segments')
        except ValueError as error:
            parser.error(str(error))
    try:
        validate_generation_tokens(args.max_new_tokens)
    except ValueError as error:
        parser.error(str(error))
    return args

def _stop_child_process(process, grace_seconds=15):
    """Reap a runner and prevent it from surviving an interrupted CLI."""
    if process.poll() is not None:
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

def run_command(command, dry_run):
    print('\n$ ' + ' '.join((str(item) for item in command)), flush=True)
    if dry_run:
        return
    project_root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    existing_pythonpath = environment.get('PYTHONPATH', '')
    environment['PYTHONPATH'] = os.pathsep.join((value for value in (str(project_root), existing_pythonpath) if value))
    process = subprocess.Popen(command, cwd=project_root, env=environment)
    try:
        return_code = process.wait()
    except BaseException:
        _stop_child_process(process)
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def output_dir(args, method, target):
    data_key = getattr(target, 'data_key', target)
    return Path(args.output_root) / args.benchmark / data_key / method / model_slug(args.model) / f'seed_{args.seed}'

def greprag_command(args, target, parquets, run_dir, method):
    command = [args.choruscoder_python, '-m', 'benchmarking.greprag_runner', '--method', method, '--parquet', *(str(p) for p in parquets), '--reference', str(Path(args.dataset_root) / target.benchmark / target.data_key / 'test.jsonl'), '--dataset-split', target.data_key, '--output-dir', str(run_dir), '--stage', args.query_stage]
    for name in ('benchmark', 'language', 'model', 'greprag_model', 'greprag_python', 'greprag_corpus', 'greprag_repos_root', 'retrieval_timing', 'vllm_api_base', 'vllm_api_key', 'vllm_request_timeout', 'seed', 'batch_size', 'num_workers', 'dataloader_num_workers', 'dataloader_prefetch_factor', 'max_context_tokens', 'max_crossfile_tokens', 'max_new_tokens'):
        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    if args.limit is not None:
        command += ['--limit', str(args.limit)]
    if args.greprag_resume:
        command += ['--greprag-resume']
    from greprag_profiles import BOUNDED_METHOD
    if target.benchmark == 'recceval' or method == BOUNDED_METHOD:
        command += ['--greprag-preprocess-cache-dir', args.greprag_preprocess_cache_dir]
    return command

def aircoder_budget_command(args, target, parquets, run_dir, method):
    command = [args.aircoder_python, '-m', 'benchmarking.aircoder_budget_runner', '--method', method, '--stage', args.query_stage, '--parquet', *map(str, parquets), '--benchmark', target.benchmark, '--language', target.language, '--dataset-split', target.data_key, '--model', args.model, '--dependency-model', args.dependency_model, '--embedding-model', args.embedding_model, '--aircoder-root', args.aircoder_root, '--cache-dir', args.aircoder_cache_dir, '--repos-root', args.aircoder_repos_root, '--output-dir', str(run_dir), '--retrieval-timing', 'online', '--vllm-api-base', args.vllm_api_base, '--vllm-api-key', args.vllm_api_key, '--vllm-request-timeout', str(args.vllm_request_timeout), '--seed', str(args.seed), '--batch-size', str(args.batch_size), '--num-workers', str(args.num_workers), '--dataloader-num-workers', str(args.dataloader_num_workers), '--dataloader-prefetch-factor', str(args.dataloader_prefetch_factor), '--max-context-tokens', str(args.max_context_tokens), '--max-crossfile-tokens', str(args.max_crossfile_tokens), '--max-new-tokens', str(args.max_new_tokens)]
    if args.dependency_vllm_api_base:
        command += ['--dependency-vllm-api-base', args.dependency_vllm_api_base]
    if args.limit is not None:
        command += ['--limit', str(args.limit)]
    if target.benchmark == 'recceval':
        command += ['--recceval-root', args.aircoder_recceval_root]
    if args.skip_existing:
        command.append('--skip-existing')
    return command

def infile_command(args, target, parquets, run_dir, method):
    validate_infile_options(args)
    infile_target(target.benchmark, target.language, target.data_key)
    command = [args.choruscoder_python, '-m', 'benchmarking.infile_runner', '--method', method, '--stage', args.query_stage, '--parquet', *(str(p) for p in parquets), '--benchmark', target.benchmark, '--dataset-split', target.data_key, '--language', target.language, '--model', args.model, '--output-dir', str(run_dir)]
    for name in ('infile_cache_dir', 'vllm_api_base', 'vllm_api_key', 'vllm_request_timeout', 'seed', 'batch_size', 'num_workers', 'dataloader_num_workers', 'dataloader_prefetch_factor', 'max_context_tokens', 'max_crossfile_tokens', 'max_new_tokens'):
        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    if args.limit is not None:
        command += ['--limit', str(args.limit)]
    if getattr(args, 'retrieval_timing', None) is not None:
        command += ['--retrieval-timing', args.retrieval_timing]
    return command

def choruscoder_command(args, target, parquets, run_dir, method):
    experiment_target(method, target.benchmark, target.language, target.data_key)
    command = [args.choruscoder_python, '-m', 'benchmarking.choruscoder_runner', '--method', method, '--stage', args.query_stage, '--parquet', *(str(p) for p in parquets), '--benchmark', target.benchmark, '--dataset-split', target.data_key, '--language', target.language, '--model', args.model, '--output-dir', str(run_dir)]
    names = ('query_max_segments', 'query_cache_dir', 'query_probe_model', 'embedding_model', 'coderag_root', 'vllm_api_base', 'vllm_api_key', 'vllm_request_timeout', 'seed', 'batch_size', 'num_workers', 'dataloader_num_workers', 'dataloader_prefetch_factor', 'max_context_tokens', 'max_crossfile_tokens', 'max_new_tokens')
    for name in names:
        if PROFILES[method].rlcoder_config is not None and name in ('query_max_segments', 'query_probe_model', 'coderag_root'):
            continue
        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    if getattr(args, 'retrieval_timing', None) is not None:
        command += ['--retrieval-timing', args.retrieval_timing]
    if PROFILES[method].rlcoder_config is not None:
        command += ['--rlcoder-root', args.rlcoder_root]
        if args.rlcoder_revision is not None:
            command += ['--rlcoder-revision', args.rlcoder_revision]
    threshold = getattr(args, 'query_support_similarity_threshold', None)
    if threshold is not None:
        if PROFILES[method].semantic_support_config is None:
            raise ValueError('Support similarity threshold requires a semantic-support method')
        from dataclasses import replace
        replace(PROFILES[method].semantic_support_config, similarity_threshold=threshold)
        command += ['--query-support-similarity-threshold', str(threshold)]
    for name in ('query_interface_similarity_threshold', 'query_interface_margin_threshold'):
        value = getattr(args, name, None)
        if value is not None:
            command += ['--' + name.replace('_', '-'), str(value)]
    if args.limit is not None:
        command += ['--limit', str(args.limit)]
    return command

def completed_choruscoder_result(args, target, parquets, run_dir, method):
    """Reuse a completed result after configuration validation.



    Source hashes may differ because no model work is being resumed. All

    other recorded configuration fields must still match the requested run.

    The saved result/configuration retain their original provenance.

    """
    from benchmarking.choruscoder_runner import build_parser, configuration, configuration_differences
    config_path = run_dir / 'run_config.json'
    if not config_path.is_file():
        return None
    with config_path.open(encoding='utf-8') as handle:
        wrapper = json.load(handle)
    if wrapper.get('status') != 'complete':
        return None
    command = choruscoder_command(args, target, parquets, run_dir, method)
    current = configuration(build_parser().parse_args(command[3:]))
    recorded = wrapper['configuration']
    changed = configuration_differences({key: value for key, value in recorded.items() if key != 'source_hashes'}, {key: value for key, value in current.items() if key != 'source_hashes'})
    if changed:
        raise ValueError('--skip-existing cannot reuse a completed ChorusCoder result with different experiment parameters; changed fields: ' + ', '.join(changed[:20]) + '. Use a new output root.')
    with (run_dir / 'result.json').open(encoding='utf-8') as handle:
        result = json.load(handle)
    expected = {'method': method, 'model': Path(args.model).name, 'benchmark': target.benchmark, 'language': target.language}
    if any((result.get(key) != value for key, value in expected.items())):
        raise ValueError('Completed ChorusCoder result identity differs from its run configuration')
    count = result.get('total_samples')
    if type(count) is not int or count < 0:
        raise ValueError('Completed ChorusCoder result lacks a valid total_samples count')
    note = ' (recorded implementation; source hashes differ)' if recorded.get('source_hashes') != current.get('source_hashes') else ''
    print(f'[skip-existing] Reusing completed result: {run_dir}{note}', flush=True)
    return result

def print_table(rows):
    headers = ['Method', 'Model', 'Benchmark', 'Subset', 'Language', 'Samples', 'Code EM', 'Code ES', 'Identifier EM', 'Identifier F1', 'Retrieval s/sample', 'Generation s/sample', 'End-to-end s/sample']
    values = [[row['method'], row['model'], row['benchmark'], row.get('subset', ''), row['language'], row['total_samples'], row['code_em'], row['code_es'], row['identifier_em'], row['identifier_f1'], row['retrieval_seconds_per_sample'], row['generation_seconds_per_sample'], row['end_to_end_seconds_per_sample']] for row in rows]
    try:
        from prettytable import PrettyTable
        table = PrettyTable(headers)
        table.add_rows(values)
        print('\n' + str(table))
    except ImportError:
        print('\n' + '\t'.join(headers))
        for value in values:
            print('\t'.join((str(item) for item in value)))

def validate_run(args):
    args.vllm_api_base = normalize_api_base(args.vllm_api_base)
    if args.dependency_vllm_api_base:
        args.dependency_vllm_api_base = normalize_api_base(args.dependency_vllm_api_base)
    if args.num_workers < 0 or args.dataloader_num_workers < 0:
        raise ValueError('Worker counts must be non-negative')
    if args.dataloader_prefetch_factor < 1:
        raise ValueError('--dataloader-prefetch-factor must be at least 1')
    dependency_model = args.model if args.dependency_model == 'same' else args.dependency_model
    if args.method not in AIRCODER_BUDGET_METHODS and (not args.allow_different_dependency_model) and (dependency_model != args.model):
        raise ValueError('The dependency model must equal the generator unless explicitly allowed')
    targets = benchmark_targets(args.benchmark, args.language, args.subset)
    for target in targets:
        if args.method in AIRCODER_BUDGET_METHODS:
            from aircoder_profiles import validate_target
            validate_target(target.benchmark, target.language, target.data_key, args.method)
        elif args.method in INFILE_METHODS:
            infile_target(target.benchmark, target.language, target.data_key)
        elif args.method in GREPRAG_METHODS:
            if target.benchmark not in ('cceval', 'recceval') or (target.language, target.data_key) != ('python', 'python'):
                raise ValueError('GrepRAG requires CCEVAL or RECCEVAL Python')
        else:
            experiment_target(args.method, target.benchmark, target.language, target.data_key)
    return targets

def runner_command(args, target, parquets, run_dir):
    if args.method in AIRCODER_BUDGET_METHODS:
        build = aircoder_budget_command
    elif args.method in GREPRAG_METHODS:
        build = greprag_command
    elif args.method in INFILE_METHODS:
        build = infile_command
    else:
        build = choruscoder_command
    return build(args, target, parquets, run_dir, args.method)

def run_one(args, targets):
    from choruscoder_benchmarks import requires_repository_parser
    if requires_repository_parser(args.method, args.benchmark) and (not args.dry_run):
        from choruscoder_languages import grammar_identities
        from choruscoder import make_parser
        for target in targets:
            for variant in grammar_identities(target.language):
                make_parser(variant)
    rows = []
    method = args.method
    for target in targets:
        if args.dry_run:
            directory = Path(args.dataset_root) / args.benchmark / target.data_key
            monolithic = directory / 'test.parquet'
            shards = tuple(sorted(directory.glob('test_*.parquet')))
            parquets = (monolithic,) if monolithic.exists() else shards or (monolithic,)
        else:
            parquets, _ = resolve_dataset_shards(args.dataset_root, args.benchmark, target.data_key)
        run_dir = output_dir(args, method, target)
        if not args.dry_run:
            ensure_dir(run_dir)
        completed_result = run_dir / 'result.json'
        if args.skip_existing and completed_result.exists() and (not args.dry_run):
            if method in AIRCODER_BUDGET_METHODS:
                command = aircoder_budget_command(args, target, parquets, run_dir, method)
                run_command([*command, '--skip-existing'], False)
                result = json.loads(completed_result.read_text(encoding='utf-8'))
            elif method in CHORUSCODER_METHODS:
                result = completed_choruscoder_result(args, target, parquets, run_dir, method)
            elif method in INFILE_METHODS:
                from benchmarking.infile_runner import build_parser, completed_result as read_completed
                command = infile_command(args, target, parquets, run_dir, method)
                result = read_completed(build_parser().parse_args(command[3:]))
            else:
                raise ValueError('GrepRAG does not support --skip-existing')
            if result is not None:
                rows.append(result)
                print_table([result])
                continue
        command = runner_command(args, target, parquets, run_dir)
        run_command(command, args.dry_run)
        if args.dry_run or args.query_stage == 'prepare':
            continue
        if args.query_stage == 'generate' and method not in CHORUSCODER_METHODS:
            continue
        if completed_result.exists():
            result = json.loads(completed_result.read_text(encoding='utf-8'))
            rows.append(result)
            print_table([result])
    return rows

def main():
    args = parse_args()
    runs = args.method_runs if args.method == 'all' else (args,)
    planned = [(run, validate_run(run)) for run in runs]
    if args.method == 'all':
        for run, _ in planned:
            print(f'[all] {run.method}: context={run.max_context_tokens}, crossfile={run.max_crossfile_tokens}, retrieval-timing={run.retrieval_timing}', flush=True)
    rows = []
    for run, targets in planned:
        rows.extend(run_one(run, targets))
    if rows:
        print_table(rows)
        directory = ensure_dir(Path(args.output_root) / 'summaries')
        selection = args.language
        write_json(directory / f'{args.benchmark}_{selection}_{args.method}_{model_slug(args.model)}_seed_{args.seed}.json', rows)
if __name__ == '__main__':
    main()
