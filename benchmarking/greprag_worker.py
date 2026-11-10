"""Isolated two-stage GrepRAG, with profile-controlled source preprocessing."""
import json
from pathlib import Path
from contextlib import nullcontext
import signal
import sys
import time

from .common import iter_parquet_batches
from .greprag_backend import KeywordModel, safe_path, retrieve, compose_prompt
from .greprag_upstream import upstream
from greprag_profiles import METHOD, PROFILES, timing_scope
from . import greprag_resume as recovery
from .greprag_resume import keyword_requests
from .greprag_upstream import file_hash


def write_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def generate_keywords(cli,output,keyword,config_hash):
    from choruscoder import atomic_json
    count=0
    with (output/'keywords.jsonl').open('w',encoding='utf8') as stream, \
            (output/'keyword_timings.jsonl').open('w',encoding='utf8') as times:
        requests=iter(keyword_requests(cli))
        while True:
            started=time.perf_counter()
            try: req=next(requests)
            except StopIteration: break
            started=time.perf_counter()  # Dataset/reference reads are outer I/O.
            value=dict(request=req,**keyword.generate(req['prefix']))
            value['method']=cli.method
            elapsed=time.perf_counter()-started
            write_line(stream,value)
            write_line(times,dict(task_id=req['task_id'],seconds=elapsed))
            count+=1
            print(f'[GrepRAG keywords] samples={count}',flush=True)
    if not count: raise ValueError('Empty GrepRAG dataset')
    atomic_json(output/'keyword_manifest.json',recovery.signed(dict(config_hash=config_hash,samples=count,
        files={name:file_hash(output/name) for name in ('keywords.jsonl','keyword_timings.jsonl')})))
    return count


def add_diagnostics(diagnostics,result):
    issues=result.get('command_issue_count',0)
    diagnostics['samples']+=1
    diagnostics['commands']+=len(result.get('commands',[]))
    diagnostics['command_issues']+=issues
    diagnostics['tasks_with_command_issues']+=int(issues>0)
    diagnostics['rerank_errors']+=int(bool(result.get('error')))
    diagnostics['empty_retrievals']+=int(not result['matches'])
    return issues


def run_retrieval(cli, output, config_hash, tokenizer_factory=None, keyword_factory=KeywordModel,resume_plan=None):
    from choruscoder import atomic_json
    if tokenizer_factory is None:
        from transformers import AutoTokenizer
        tokenizer_factory = AutoTokenizer.from_pretrained
    if getattr(cli,'method',METHOD)!=METHOD or cli.greprag_corpus!='parquet':
        raise ValueError('GrepRAG-0.6B requires its bounded Parquet profile')
    from .greprag_corpus_cache import preprocess,online_corpus
    corpus_cache,corpus_tasks=preprocess(cli,output)
    # Model startup and source-only corpus preparation are excluded separately.
    started = time.perf_counter()
    need_search=resume_plan is None or len(resume_plan['completed'])<resume_plan['total']
    kw_functions, functions = upstream() if need_search else (None,None)
    tokenizer = tokenizer_factory(cli.model) if need_search else None
    keyword_options = dict(input_limit=PROFILES[cli.method].keyword_input_tokens)
    keyword = keyword_factory(cli.greprag_model,kw_functions,cli.seed,**keyword_options) if resume_plan is None else None
    if resume_plan is None:
        atomic_json(output / 'model_initialization.json', dict(seconds=time.perf_counter() - started,
            excluded_from_retrieval=True, repository_preprocessing=True, identity=keyword.identity))
    else:
        atomic_json(output/f'resume_initialization_{resume_plan["session_id"]}.json',dict(
            seconds=time.perf_counter()-started,excluded_from_retrieval=True,
            keyword_model_loaded=False,tokenizer_loaded=need_search))
    keyword_time_path = output / 'keyword_timings.jsonl'
    prepared_dir = output / 'prepared'
    prepared_dir.mkdir(exist_ok=True)
    recovery_seconds=0.0
    try:
        # As in the two official CLIs: finish every keyword before any rg call.
        if resume_plan is None:
            count=generate_keywords(cli,output,keyword,config_hash)
        else:
            recovery_seconds=recovery.verify_worker(resume_plan,output)
            count=resume_plan['total']
            print(f'[GrepRAG resume] keywords={count} prepared={len(resume_plan["completed"])}'
                  f' pending={count-len(resume_plan["completed"])} keyword_model_loaded=False',flush=True)
        tasks = []
        diagnostics = dict(samples=0,commands=0,command_issues=0,tasks_with_command_issues=0,
                           rerank_errors=0,empty_retrievals=0,
                           error_policy='official_best_effort_no_regex_rewrite_or_retry')
        with (output / 'keywords.jsonl').open(encoding='utf-8') as stream, \
                keyword_time_path.open(encoding='utf-8') as times:
            columns=('task_id','path','left_context')
            rows = (row for batch in iter_parquet_batches(cli.parquet,
                columns,cli.batch_size,cli.limit)
                for row in batch)
            for index in range(count):
                started = time.perf_counter()
                item = json.loads(next(stream))
                kw_time = json.loads(next(times))
                req = item['request']
                row = next(rows)
                if (kw_time['task_id'] != req['task_id'] or
                        (str(row.task_id) != req['task_id'] or safe_path(row.path) != req['path'] or str(row.left_context) != req['prefix'])):
                    raise ValueError('GrepRAG stage input order/identity mismatch')
                location=prepared_dir/f'{index:06d}.json'
                if resume_plan is not None and str(index) in resume_plan['completed']:
                    record=json.loads(location.read_text(encoding='utf8'))
                    expected=resume_plan['completed'][str(index)]
                    if record.get('checksum')!=expected['checksum'] or record.get('config_hash')!=expected['config_hash']:
                        raise ValueError('Prepared record changed after recovery audit: '+req['task_id'])
                    recovery.validate_record(record,resume_plan['configs'][record['config_hash']],req['task_id'])
                    add_diagnostics(diagnostics,record['official_result'])
                    tasks.append(dict(task_id=req['task_id'],file=location.name))
                    recovery_seconds+=time.perf_counter()-started
                    continue
                audit_started = time.perf_counter()
                # Exclude saved keyword/timing reads; include corpus lookup and real query work.
                started=audit_started
                context = nullcontext(online_corpus(corpus_cache,corpus_tasks[index],req))
                try:
                    with context as corpus:
                        resolution_seconds = time.perf_counter() - audit_started
                        core_started = time.perf_counter()
                        result = retrieve(req, item['generated_commands'], corpus['repository_path'], functions,
                                          current_file=corpus['current_file'],
                                          protect_keyword_options=True)
                        core_seconds = time.perf_counter() - core_started
                except Exception:
                    print(f'[GrepRAG retrieval failed] task_id={req["task_id"]} sample={index+1}/{count}',
                          file=sys.stderr,flush=True)
                    raise
                cleanup_seconds = 0.0
                adapt_started = time.perf_counter()
                prompt = compose_prompt(result, req, tokenizer, cli.max_context_tokens, cli.max_crossfile_tokens)
                adaptation_seconds = time.perf_counter() - adapt_started
                online_seconds = time.perf_counter() - started
                issue_count=add_diagnostics(diagnostics,result)
                if issue_count or result.get('error'):
                    print(f'[GrepRAG upstream warning] task_id={req["task_id"]} command_issues={issue_count}'
                          f' rerank_error={result.get("error")!r}; see official_result in prepared record',
                          file=sys.stderr,flush=True)
                record = dict(task_id=req['task_id'], path=req['path'], repository=req['repository'],
                    config_hash=config_hash, generated_keywords=item['generated_keywords'],
                    generated_commands=item['generated_commands'], raw_output=item['raw_output'],
                    official_result=result, corpus=corpus, **prompt, keyword_seconds=kw_time['seconds'],
                    repository_resolution_seconds=resolution_seconds,
                    corpus_materialization_seconds=0.0,
                    corpus_cleanup_seconds=cleanup_seconds,official_search_seconds=core_seconds,
                    prompt_adaptation_seconds=adaptation_seconds, retrieval_timing=timing_scope(cli))
                record.update(method=cli.method, keyword_input=item['keyword_input'],
                              keyword_command_policy=recovery.KEYWORD_COMMAND_POLICY)
                location = prepared_dir / f'{index:06d}.json'
                # Persist records after measuring the actual online computation.
                record['search_phase_seconds'] = online_seconds
                record['retrieval_seconds'] = record['keyword_seconds'] + record['search_phase_seconds']
                from choruscoder import digest
                record['checksum'] = digest(record)
                atomic_json(location, record)
                tasks.append(dict(task_id=req['task_id'], file=location.name))
                print(f'[GrepRAG retrieval] samples={index + 1}/{count}', flush=True)
            if next(stream, None) is not None or next(times, None) is not None:
                raise ValueError('GrepRAG stage cardinality mismatch')
            if next(rows,None) is not None:
                raise ValueError('Additional Parquet corpus rows')
        from choruscoder import digest
        manifest = dict(config_hash=config_hash, tasks=tasks)
        atomic_json(output / 'prepared_manifest.json', dict(manifest, checksum=digest(manifest)))
        atomic_json(output / 'retrieval_diagnostics.json', diagnostics)
    finally:
        if keyword is not None: keyword.close()
        if resume_plan is not None: recovery.finish(output,resume_plan,recovery_seconds)


def main():
    import argparse
    from types import SimpleNamespace
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True)
    args = p.parse_args()
    payload = json.loads(Path(args.request).read_text(encoding='utf-8'))
    cli = SimpleNamespace(**payload['cli'])
    run_retrieval(cli,Path(cli.output_dir),payload['config_hash'],resume_plan=payload.get('resume_plan'))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    main()
