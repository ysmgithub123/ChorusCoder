"""Official AIRCoder-RRF with shared budgets and repository-preprocessed timing."""
import argparse
import importlib.metadata
from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import re
from pathlib import Path
import sys
import time
from copy import copy

from aircoder_profiles import QWEN_METHOD, METHODS, PROFILES, validate_target
from choruscoder import atomic_json, digest, load_json
from benchmarking.aircoder_official import (Corpus,
    dependency_definitions, dependency_prompt, qwen_dependency_prompt, file_hash, parse_dependencies, query, upstream)
from benchmarking.common import AtomicJsonlWriter, model_slug, normalize_api_base
from benchmarking.aircoder_repository import catalog

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,default=QWEN_METHOD)
    p.add_argument('--stage',choices=('prepare','generate','evaluate','all'),default='all')
    p.add_argument('--parquet',nargs='+',required=True)
    p.add_argument('--benchmark',default='cceval')
    p.add_argument('--language',default='python')
    p.add_argument('--dataset-split',default='python')
    p.add_argument('--model',required=True)
    p.add_argument('--dependency-model',default=None)
    p.add_argument('--dependency-vllm-api-base')
    p.add_argument('--embedding-model',default='microsoft/unixcoder-base')
    p.add_argument('--aircoder-root',default='third_party/AIRCoder')
    p.add_argument('--repos-root',default='data_sources/crosscodeeval_repositories')
    p.add_argument('--recceval-root',default='data_sources/recceval')
    p.add_argument('--metadata',help='Task-to-repository metadata; defaults to sibling test.jsonl')
    p.add_argument('--cache-dir',default='benchmark_cache/aircoder_official')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--check-only',action='store_true',help='Validate pinned sources, ABI, inputs and mappings without encoding or LLM requests')
    p.add_argument('--skip-existing',action='store_true',help='Strictly validate and reuse an already completed output')
    p.add_argument('--retrieval-timing',choices=('online',),default='online')
    p.add_argument('--vllm-api-base',default='http://127.0.0.1:8000/v1')
    p.add_argument('--vllm-api-key',default='EMPTY')
    p.add_argument('--vllm-request-timeout',type=float,default=7200)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--limit',type=int)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--num-workers',type=int,default=1)
    p.add_argument('--dataloader-num-workers',type=int,default=0)
    p.add_argument('--dataloader-prefetch-factor',type=int,default=1)
    p.add_argument('--max-context-tokens',type=int,default=4096)
    p.add_argument('--max-crossfile-tokens',type=int,default=3072)
    p.add_argument('--max-new-tokens',type=int,default=50)
    return p


def checkpoint_identity(name):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(name)
    local = Path(name)
    weights = {}
    if local.is_dir():
        for path in sorted(local.iterdir()):
            if path.is_file() and (path.suffix in ('.bin','.safetensors') or path.name == 'config.json'):
                weights[path.name] = file_hash(path)
    return dict(name=str(name),commit=getattr(config,'_commit_hash',None),local_weights=weights,
                hidden_size=getattr(config,'hidden_size',None),
                max_position_embeddings=getattr(config,'max_position_embeddings',None))


def load_tokenizer(args):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(args.model)
    t.padding_side = t.truncation_side = 'left'
    if t.pad_token_id is None: t.pad_token_id = t.eos_token_id
    return t


def tokenizer_identity(t):
    return dict(class_name=type(t).__module__+'.'+type(t).__qualname__,
        vocabulary=digest(t.get_vocab()),special=str(t.special_tokens_map),
        backend=t.backend_tokenizer.to_str() if hasattr(t,'backend_tokenizer') else None,
        bos=getattr(t,'bos_token_id',None),eos=t.eos_token_id,pad=t.pad_token_id,
        chat_template=getattr(t,'chat_template',None),
        probe=t.encode('变量 = helper("值")\n',add_special_tokens=True))


def runtime_versions():
    versions={}
    for name in ('torch','transformers','tokenizers','sentence-transformers','numpy',
                 'tree-sitter','tree-sitter-python','editdistance'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='missing'
    return versions


def separate_dependency(args):
    return getattr(PROFILES[args.method], 'dependency_model', None)


def resolved_dependency(args):
    requested = args.dependency_model
    if requested is None:
        return separate_dependency(args) or args.model
    return args.model if requested == 'same' else requested


def dependency_options(args):
    result = copy(args)
    result.model = resolved_dependency(args)
    result.vllm_api_base = getattr(args, 'dependency_vllm_api_base', None) or args.vllm_api_base
    return result


def load_dependency_tokenizer(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(resolved_dependency(args))
    tokenizer.padding_side = tokenizer.truncation_side = 'left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    if not tokenizer.chat_template: raise ValueError('Qwen dependency tokenizer has no chat template')
    return tokenizer


def dependency_identity(args):
    """Generation uses the recorded dependency identity, not the stopped service/model."""
    profile = PROFILES[args.method]
    output = Path(args.output_dir)
    frozen = args.stage in ('generate', 'evaluate') or (args.skip_existing and (output/'result.json').exists())
    if frozen:
        saved = load_json(output/'run_config.json')
        config = saved['configuration']
        if saved.get('configuration_checksum') != digest(config) or config.get('method') != args.method:
            raise ValueError('AIRCoder frozen configuration checksum/method mismatch')
        identity = config['dependency_identity']
        if (identity['checkpoint']['name'] != resolved_dependency(args) or
                identity['prompt_format'] != profile.dependency_prompt_format):
            raise ValueError('AIRCoder frozen dependency configuration mismatch')
        return identity
    tokenizer = load_dependency_tokenizer(args)
    args._dependency_tokenizer = tokenizer
    return dict(checkpoint=checkpoint_identity(resolved_dependency(args)), tokenizer=tokenizer_identity(tokenizer),
                prompt_format=profile.dependency_prompt_format, input_tokens=profile.dependency_input_tokens,
                new_tokens=profile.dependency_new_tokens, enable_thinking=False, truncation_side='left')


def validate_services(args):
    """Metadata checks only. A staged run never contacts the other model's service."""
    if not separate_dependency(args) or args.check_only: return
    options = []
    if args.stage in ('prepare', 'all'): options.append(dependency_options(args))
    if args.stage in ('generate', 'all'): options.append(args)
    for option in options:
        with client(option) as api:
            models = {item.id for item in api.models.list().data}
        if option.model not in models:
            raise ValueError(f'AIRCoder {args.stage}: {option.vllm_api_base} does not serve {option.model}; '
                             f'available={sorted(models)}. Use prepare, switch service, then generate/evaluate.')


def configuration(args, tokenizer):
    # One catalog per invocation; do not reuse stale file hashes if a caller
    # invokes the runner twice in the same Python process.
    if hasattr(args,'_repository_catalog'): del args._repository_catalog
    validate_target(args.benchmark,args.language,args.dataset_split,args.method)
    profile = PROFILES[args.method]
    dependency_model = resolved_dependency(args)
    if separate_dependency(args) and dependency_model != separate_dependency(args):
        raise ValueError(f'{args.method} requires dependency-model {separate_dependency(args)}')
    if not separate_dependency(args) and dependency_model != args.model:
        raise ValueError('AIRCoder fair comparison requires dependency-model == model')
    if not separate_dependency(args) and getattr(args, 'dependency_vllm_api_base', None) not in (None,args.vllm_api_base):
        raise ValueError('AIRCoder fair comparison requires the same generation/dependency endpoint')
    if args.embedding_model != 'microsoft/unixcoder-base':
        raise ValueError('This official profile requires microsoft/unixcoder-base')
    if (args.max_context_tokens,args.max_crossfile_tokens) != (profile.context_tokens,profile.crossfile_tokens):
        raise ValueError('AIRCoder-RRF requires 4096/3072 token limits')
    if args.num_workers != 1 or args.dataloader_num_workers != 0 or args.dataloader_prefetch_factor != 1:
        raise ValueError('AIRCoder fair comparison requires workers=1, DataLoader workers=0, prefetch=1')
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.limit is not None and args.limit < 1:
        raise ValueError('Batch size, generation length and limit must be positive')
    if not math.isfinite(args.vllm_request_timeout) or args.vllm_request_timeout <= 0:
        raise ValueError('AIRCoder request timeout must be positive and finite')
    if separate_dependency(args) and args.max_new_tokens != 50:
        raise ValueError('The Qwen3-8B comparison profile requires max-new-tokens 50')
    source = upstream(args.aircoder_root)
    root = Path(__file__).resolve().parents[1]
    names = ['aircoder_profiles.py','benchmarking/aircoder_budget_runner.py',
        'benchmarking/aircoder_repository.py',
        'benchmarking/aircoder_official.py','benchmarking/aircoder_upstream.json',
        'benchmarking/aircoder_runner.py','benchmarking/common.py','embedding_cache.py',
        'benchmarking/retrieval_online.py','benchmarking/choruscoder_runtime_cache.py',
        'choruscoder.py',
        'benchmarking/evaluation.py','benchmarking/metrics.py']
    if args.benchmark=='recceval':
        names+=['benchmarking/aircoder_recceval.py','benchmarking/source_corpus_cache.py',
            'benchmarking/prepare_recceval.py','benchmarking/recceval_metrics.py']
    value = dict(method=args.method,version=profile.version,profile=asdict(profile),
        benchmark=args.benchmark,language=args.language,dataset_split=args.dataset_split,
        generator=checkpoint_identity(args.model),encoder=checkpoint_identity(args.embedding_model),
        dependency_model=dependency_model,tokenizer=tokenizer_identity(tokenizer),upstream=source,
        runtime_versions=runtime_versions(),
        repository_catalog=catalog(args).identity(),
        dependency_prompt_format='official_instructions_plain_roles_prefix_suffix_v1',
        dataset=[[str(Path(p).resolve()),file_hash(p)] for p in args.parquet],limit=args.limit,
        reference=[catalog(args).reference,file_hash(catalog(args).reference)],
        seed=args.seed,batch_size=args.batch_size,num_workers=args.num_workers,
        dataloader_num_workers=args.dataloader_num_workers,dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        max_new_tokens=args.max_new_tokens,temperature=0,top_p=1,retrieval_timing='online',
        timing_scope='repository_preprocessed_online_v1',task_cache='question_only',
        source_hashes={n:file_hash(root/n) for n in names})
    if separate_dependency(args):
        value.update(dependency_identity=dependency_identity(args),
            dependency_prompt_format=profile.dependency_prompt_format,
            dependency_api_base=normalize_api_base(dependency_options(args).vllm_api_base),
            generation_api_base=normalize_api_base(args.vllm_api_base),
            request_timeout=args.vllm_request_timeout)
    if args.benchmark=='recceval':
        from benchmarking.recceval_metrics import EVALUATION_VERSION
        value.update(evaluation_version=EVALUATION_VERSION,recceval_root=str(Path(args.recceval_root).resolve()))
    return json.loads(json.dumps(value))


def rows(args):
    yield from catalog(args).rows(args)


def seal(record): return dict(record,checksum=digest(record))


def checked(path, fingerprint=None):
    record = load_json(path)
    if record.get('checksum') != digest({k:v for k,v in record.items() if k != 'checksum'}):
        raise ValueError('AIRCoder record checksum mismatch: '+str(path))
    if fingerprint is not None and record.get('config_hash') != fingerprint:
        raise ValueError('AIRCoder record configuration mismatch: '+str(path))
    return record


def record_path(output, request): return output/'prepared'/(digest(request)+'.json')


def check_prepared(record, request, fingerprint, tokenizer, profile):
    if record.get('config_hash') != fingerprint or record.get('request_hash') != digest(request):
        raise ValueError('AIRCoder prepared input mismatch')
    if record.get('task_id') != request['task_id'] or record.get('path') != request['path']:
        raise ValueError('AIRCoder prepared task/path mismatch')
    if (len(tokenizer.encode(record['prompt'],add_special_tokens=True)) != record['usage']['prompt_tokens'] or
            record['usage']['prompt_tokens'] > profile.context_tokens or
            record['usage']['crossfile_tokens'] > profile.crossfile_tokens or
            not math.isfinite(record['retrieval_seconds']) or record['retrieval_seconds'] < 0 or
            any(not isinstance(t,(int,float)) or not math.isfinite(t) or t < 0 for t in record['stage_seconds'].values()) or
            abs(sum(record['stage_seconds'].values())-record['retrieval_seconds']) > 1e-8):
        raise ValueError('AIRCoder prepared budget/timing mismatch')
    allowed = {f['path'] for f in request['files']} | {request['path']}
    if any(c['metadata']['file'] not in allowed for c in record['selected']):
        raise ValueError('AIRCoder prepared context is outside the allowed files')
    if len(record['selected']) > profile.num_rags or any(
            c['metadata']['file'] == request['path'] and
            c['metadata']['end_line_no'] > record['query_metadata']['start_line_no']
            for c in record['selected']):
        raise ValueError('AIRCoder prepared context violates official line filtering')


def complete_preparation(args, output, fingerprint, tokenizer, profile):
    tasks = []
    missing = False
    for batch in rows(args):
        for request in batch:
            path = record_path(output,request)
            if not path.exists():
                missing = True
                continue
            record = checked(path,fingerprint)
            check_prepared(record,request,fingerprint,tokenizer,profile)
            tasks.append(dict(file=str(path.resolve()),request_hash=digest(request),task_id=request['task_id']))
    if missing: return None
    if not tasks: raise ValueError('No AIRCoder samples')
    if len({t['task_id'] for t in tasks}) != len(tasks): raise ValueError('Duplicate task IDs')
    return tasks


def load_runtime(args, config, tokenizer):
    import torch
    from benchmarking.aircoder_runner import (import_aircoder,validate_tree_sitter_stack,
        install_safe_ast_text_access,build_llm_struct_retriever)
    validate_tree_sitter_stack()
    validate_evaluator()
    if not torch.cuda.is_available(): raise RuntimeError('AIRCoder requires a CUDA GPU')
    with upstream_imports():
        prompts,utils_repo,Builder,module,_,load_model = import_aircoder(args.aircoder_root)
        expected = Path(args.aircoder_root).resolve()/'src/retriever.py'
        if Path(module.__file__).resolve() != expected:
            raise ValueError('AIRCoder imported a different retriever module')
        install_safe_ast_text_access(Builder)
        model = load_model(args.embedding_model)
        positions = model[0].auto_model.config.max_position_embeddings
        actual_commit=getattr(model[0].auto_model.config,'_commit_hash',None)
        if config['encoder']['commit'] and actual_commit != config['encoder']['commit']:
            raise ValueError('AIRCoder encoder revision changed after startup validation')
        # Database.set_repo in the official code sets this before either embedding.
        model.max_seq_length = positions//2
        profile = PROFILES[args.method]
        chunker = Builder(tokenizer,max_chunk_size=profile.slice_size,language=args.language,metadata_template='default')
        retriever = build_llm_struct_retriever(module,tokenizer,model,list(profile.metrics),args.language)
    encoder_identity=dict(config['encoder'],runtime_versions=config['runtime_versions'],
                          tokenizer=tokenizer_identity(model[0].tokenizer),pooling='cls',normalize=False)
    identity = dict(upstream=config['upstream'],tokenizer=config['tokenizer'],encoder=encoder_identity,
                    encode_length=positions//2,language=args.language,slice_size=profile.slice_size,
                    corpus_format='full_snapshot_cls_v1')
    corpus = Corpus(args.cache_dir,identity,tokenizer,model,chunker,utils_repo,
                    profile.chunk_memory_bytes,profile.vector_memory_bytes)
    return corpus,retriever,prompts


@contextmanager
def upstream_imports():
    """Upstream's top-level `utils` must not shadow our evaluation package."""
    names={'utils','const','prompts','score','utils_transformer','utils_repo','retriever','astchunk','datasets'}
    # Our datasets.py is unrelated to the Hugging Face datasets package used by
    # recent SentenceTransformers. Keep it out of third-party import resolution.
    owned=lambda n:n.split('.')[0] in names
    saved={n:m for n,m in list(sys.modules.items()) if owned(n)}
    path=list(sys.path)
    project=Path(__file__).resolve().parents[1]
    sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=project]
    for name in saved: del sys.modules[name]
    try:
        yield
    finally:
        for name in list(sys.modules):
            if owned(name): del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:]=path


def client(args):
    from openai import OpenAI
    return OpenAI(base_url=normalize_api_base(args.vllm_api_base),api_key=args.vllm_api_key,
                  timeout=args.vllm_request_timeout,max_retries=0)


def validate_evaluator():
    from utils import eval_utils
    from benchmarking.evaluation import _parser
    _parser('python')
    return eval_utils


def completion(api,args,prompts,tokens):
    response = api.completions.create(model=args.model,prompt=prompts,temperature=0,top_p=1,
                                     max_tokens=tokens,seed=args.seed)
    choices = sorted(response.choices,key=lambda c:c.index)
    if [c.index for c in choices] != list(range(len(prompts))) or any(not isinstance(c.text,str) for c in choices):
        raise ValueError('Incomplete AIRCoder generation batch')
    return [c.text for c in choices]


def prepare(args, config, tokenizer):
    output = Path(args.output_dir); profile = PROFILES[args.method]; fingerprint = digest(config)
    tasks = complete_preparation(args,output,fingerprint,tokenizer,profile)
    if tasks is not None:
        atomic_json(output/'prepared_manifest.json',seal(dict(config_hash=fingerprint,tasks=tasks)))
        atomic_json(output/'preparation_resume.json',dict(reused_task_ids=[t['task_id'] for t in tasks]))
        dependency_statistics(output,tasks,fingerprint)
        return
    model_started = time.perf_counter()
    corpus,retriever,prompts = load_runtime(args,config,tokenizer)
    startup_seconds = time.perf_counter()-model_started
    try:
        started = time.perf_counter(); count = 0
        for name,repository in catalog(args).repositories.items():
            corpus.preprocess(repository['files'])
            count += 1
            print(f'[aircoder-preprocessing] repositories={count}; seconds={time.perf_counter()-started:.3f}; repository={name}',flush=True)
        corpus.flush()
        atomic_json(output/'preprocessing_stats.json',dict(repositories=count,samples=len(catalog(args).tasks),
            model_startup_seconds=startup_seconds,seconds=time.perf_counter()-started,
            catalog_scan_seconds=getattr(args,'_catalog_seconds',0),
            excluded_from_retrieval=True,source_cache=getattr(catalog(args),'source_cache_stats',None),**corpus.stats()))
        definitions = dependency_definitions(args.aircoder_root)
        tasks = []; reused = set()
        dep_args = dependency_options(args) if separate_dependency(args) else args
        dep_tokenizer = getattr(args,'_dependency_tokenizer',None) if separate_dependency(args) else tokenizer
        if separate_dependency(args) and dep_tokenizer is None:
            raise ValueError('Qwen dependency tokenizer was not initialized during preparation')
        prompt_builder = qwen_dependency_prompt if separate_dependency(args) else dependency_prompt
        with client(dep_args) as api:
            for batch in rows(args):
                pending = []
                for r in batch:
                    if record_path(output,r).exists(): reused.add(r['task_id'])
                    else: pending.append(r)
                if pending:
                    started = time.perf_counter()
                    dep_prompts = [prompt_builder(r['prefix'],args.language,dep_tokenizer,definitions,
                                                    profile.dependency_input_tokens) for r in pending]
                    answers = completion(api,dep_args,dep_prompts,profile.dependency_new_tokens)
                    deps = [parse_dependencies(a) for a in answers]
                    dependency_seconds = (time.perf_counter()-started)/len(pending)
                    for request,names,raw in zip(pending,deps,answers):
                        before_hits,before_disk = corpus.memory_hits,corpus.disk_keys
                        started = time.perf_counter()
                        value = query(request,names,corpus,retriever,prompts,profile)
                        actual = time.perf_counter()-started
                        stages = dict(dependency_extraction=dependency_seconds,**value['stage_seconds'])
                        # Include ordinary selection bookkeeping not covered by
                        # the sub-stage timers; exclude JSON write/read only.
                        stages['selection_overhead'] = max(0.0,actual-sum(value['stage_seconds'].values()))
                        selected = [dict(input_ids=c['input_ids'],metadata=c['metadata'],scores=c['scores'])
                                    for c in value.pop('selected')]
                        record = dict(value,selected=selected,task_id=request['task_id'],path=request['path'],
                            dependencies=names,raw_dependency_prediction=raw,config_hash=fingerprint,
                            snapshot_provenance=request['snapshot_provenance'],
                            dependency_answer_tag_present=re.search('<answer>(.*?)</answer>',raw,re.DOTALL) is not None,
                            request_hash=digest(request),stage_seconds=stages,retrieval_seconds=sum(stages.values()),
                            corpus_memory_hits=corpus.memory_hits-before_hits,corpus_disk_keys=corpus.disk_keys-before_disk)
                        if separate_dependency(args):
                            record['dependency_model'] = resolved_dependency(args)
                            record['dependency_prompt_format'] = profile.dependency_prompt_format
                        check_prepared(record,request,fingerprint,tokenizer,profile)
                        atomic_json(record_path(output,request),seal(record))
                for request in batch:
                    path = record_path(output,request)
                    record = checked(path,fingerprint)
                    check_prepared(record,request,fingerprint,tokenizer,profile)
                    tasks.append(dict(file=str(path.resolve()),request_hash=digest(request),task_id=request['task_id']))
                print(f'[aircoder-retrieval] samples={len(tasks)}',flush=True)
        if len({t['task_id'] for t in tasks}) != len(tasks): raise ValueError('Duplicate task IDs')
        atomic_json(output/'prepared_manifest.json',seal(dict(config_hash=fingerprint,tasks=tasks)))
        atomic_json(output/'preparation_resume.json',dict(reused_task_ids=sorted(reused)))
        dependency_statistics(output,tasks,fingerprint)
    finally: corpus.close()


def manifest(args, config):
    output = Path(args.output_dir); fingerprint = digest(config)
    m = checked(output/'prepared_manifest.json',fingerprint)
    for task in m['tasks']:
        if Path(task['file']).resolve().parent != (output/'prepared').resolve():
            raise ValueError('Prepared manifest references another output directory')
        if Path(task['file']).name != task['request_hash']+'.json':
            raise ValueError('Prepared manifest has a mismatched request filename')
    if [t['task_id'] for t in m['tasks']] != list(catalog(args).tasks):
        raise ValueError('Prepared manifest differs from input tasks/order')
    return m['tasks']


def dependency_statistics(output,tasks,fingerprint):
    stats=dict(samples=0,answer_tag_present=0,empty_dependencies=0,seconds=0.0,
               averaging='Actual dependency batch time / actual batch samples')
    for task in tasks:
        r=checked(task['file'],fingerprint)
        stats['samples']+=1
        stats['answer_tag_present']+=r['dependency_answer_tag_present']
        stats['empty_dependencies']+=not any(d.strip() for d in r['dependencies'])
        stats['seconds']+=r['stage_seconds']['dependency_extraction']
    atomic_json(output/'dependency_stats.json',stats)


def generated_path(output,task): return output/'generated'/(task['request_hash']+'.json')


def generation_record(args, config, task, prepared):
    if prepared['request_hash'] != task['request_hash'] or prepared['task_id'] != task['task_id']:
        raise ValueError('AIRCoder prepared manifest task mismatch')
    value = checked(generated_path(Path(args.output_dir),task),digest(config))
    if (value['request_hash'] != task['request_hash'] or value['prompt_hash'] != digest(prepared['prompt']) or
            value['task_id'] != task['task_id'] or
            value['retrieval_seconds'] != prepared['retrieval_seconds'] or
            not isinstance(value['pred'],str) or not math.isfinite(value['generation_seconds']) or value['generation_seconds'] < 0 or
            value['end_to_end_seconds'] != value['retrieval_seconds']+value['generation_seconds']):
        raise ValueError('AIRCoder generated result mismatch')
    return value


def export(args, config):
    output = Path(args.output_dir)
    with ExitStack() as stack:
        writers = {n:stack.enter_context(AtomicJsonlWriter(output/(n+'.jsonl')))
                   for n in ('predictions','timings','retrieval')}
        for task in manifest(args,config):
            prepared = checked(task['file'],digest(config))
            result = generation_record(args,config,task,prepared)
            writers['predictions'].write(dict(task_id=task['task_id'],pred=result['pred']))
            writers['timings'].write({k:result[k] for k in ('task_id','retrieval_seconds','generation_seconds','end_to_end_seconds')})
            writers['retrieval'].write(dict(prepared,prepared_reused=result['prepared_reused'],
                                            dependency_model=config['dependency_model']))


def generate(args, config):
    output = Path(args.output_dir); tasks = manifest(args,config)
    resumed = set(load_json(output/'preparation_resume.json').get('reused_task_ids',[])) if (output/'preparation_resume.json').exists() else set()
    with client(args) as api:
        for start in range(0,len(tasks),args.batch_size):
            pending = []
            for task in tasks[start:start+args.batch_size]:
                prepared = checked(task['file'],digest(config))
                if generated_path(output,task).exists(): generation_record(args,config,task,prepared)
                else: pending.append((task,prepared))
            if not pending: continue
            started = time.perf_counter()
            predictions = completion(api,args,[p['prompt'] for _,p in pending],args.max_new_tokens)
            elapsed = (time.perf_counter()-started)/len(pending)
            for (task,p),prediction in zip(pending,predictions):
                result = dict(task_id=task['task_id'],pred=prediction,request_hash=task['request_hash'],
                    config_hash=digest(config),prompt_hash=digest(p['prompt']),retrieval_seconds=p['retrieval_seconds'],
                    generation_seconds=elapsed,end_to_end_seconds=p['retrieval_seconds']+elapsed,
                    prepared_reused=args.stage == 'generate' or task['task_id'] in resumed)
                atomic_json(generated_path(output,task),seal(result))
    export(args,config)


def evaluate(args, config):
    from benchmarking.evaluation import evaluate_predictions, aggregate_timings
    output = Path(args.output_dir)
    export(args,config)
    metrics = evaluate_predictions(output/'predictions.jsonl',catalog(args).reference,
        args.language,output,require_complete=args.limit is None,benchmark=args.benchmark)
    result = dict(method=args.method,model=model_slug(args.model),benchmark=args.benchmark,language=args.language,
        **metrics,**aggregate_timings(output/'timings.jsonl'),config_hash=digest(config),
        retrieval_timing='online',preprocessing_excluded=True)
    atomic_json(output/'result.json',seal(result))
    return result


def completed_result(args, tokenizer=None, config=None):
    if tokenizer is None: tokenizer = load_tokenizer(args)
    if config is None: config = configuration(args,tokenizer)
    output = Path(args.output_dir)
    if load_json(output/'run_config.json').get('configuration') != config:
        raise ValueError('AIRCoder completed configuration differs; use a new output root')
    tasks = complete_preparation(args,output,digest(config),tokenizer,PROFILES[args.method])
    if tasks is None: raise ValueError('Missing AIRCoder prepared sample')
    stored = manifest(args,config)
    if tasks != stored: raise ValueError('AIRCoder prepared manifest differs from input')
    for task in tasks: generation_record(args,config,task,checked(task['file'],digest(config)))
    result = checked(output/'result.json',digest(config))
    if load_json(output/'run_config.json').get('status') != 'complete':
        raise ValueError('AIRCoder run has not completed evaluation')
    if result['total_samples'] != len(tasks): raise ValueError('AIRCoder sample count mismatch')
    return result


def run(args):
    validate_target(args.benchmark,args.language,args.dataset_split,args.method)
    tokenizer = load_tokenizer(args); config = configuration(args,tokenizer)
    if args.check_only:
        from benchmarking.aircoder_runner import validate_tree_sitter_stack, import_aircoder
        abi = validate_tree_sitter_stack()
        with upstream_imports(): import_aircoder(args.aircoder_root)
        validate_evaluator()
        print(json.dumps(dict(check='passed',samples=len(catalog(args).tasks),
            repositories=len(catalog(args).repositories),upstream=config['upstream']['commit'],
            tree_sitter=abi,dependency_model=config['dependency_model'],
            llm_requests=0,encoded_texts=0),ensure_ascii=False),flush=True)
        return
    if args.skip_existing and (Path(args.output_dir)/'result.json').exists():
        print(json.dumps(completed_result(args,tokenizer,config),ensure_ascii=False),flush=True)
        return
    output = Path(args.output_dir); path = output/'run_config.json'
    if path.exists() and load_json(path).get('configuration') != config:
        raise ValueError('AIRCoder configuration changed; use a new output root')
    validate_services(args)
    output.mkdir(parents=True,exist_ok=True)
    def save_status(status):
        value=dict(configuration=config,status=status,stage=args.stage)
        if separate_dependency(args): value['configuration_checksum']=digest(config)
        atomic_json(path,value)
    save_status('running')
    if args.stage in ('prepare','all'): prepare(args,config,tokenizer)
    if args.stage in ('generate','all'): generate(args,config)
    if args.stage in ('evaluate','all'): evaluate(args,config)
    save_status('complete' if args.stage in ('evaluate','all') else args.stage)


if __name__ == '__main__': run(build_parser().parse_args())
