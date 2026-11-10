"""Prepare exact per-question Parquet corpora before keyword generation."""
from pathlib import Path
import time

from benchmarking.common import iter_parquet_batches
from benchmarking.greprag_backend import parquet_files,safe_path
from benchmarking.source_corpus_cache import SourceCorpusCache,sha,MissingCorpusError
from choruscoder import digest,atomic_json,load_json


def preprocess(cli,output):
    started=time.perf_counter()
    cache=SourceCorpusCache(cli.greprag_preprocess_cache_dir)
    datasets=getattr(cli,'corpus_datasets',None) or [[str(Path(p).resolve()),sha(p)] for p in cli.parquet]
    revisions=getattr(cli,'corpus_dataset_revisions',None)
    if [str(Path(p).resolve()) for p in cli.parquet]!=[p[0] for p in datasets]:raise ValueError('Prepared corpus dataset paths changed')
    if revisions is not None:
        if len(revisions)!=len(datasets):raise ValueError('Invalid dataset revision count')
        for path,expected,revision in zip(cli.parquet,[p[1] for p in datasets],revisions):
            current=(Path(path).stat().st_mtime_ns,Path(path).stat().st_size)
            if current!=tuple(revision) and sha(path)!=expected:raise ValueError('Dataset changed before corpus preprocessing')
    identity=dict(version='greprag_parquet_corpus_map_v1',datasets=datasets,limit=cli.limit)
    location=cache.root/'datasets'/(digest(identity)+'.json')
    mapping_hit=location.is_file()
    if mapping_hit:
        envelope=load_json(location)
        if envelope.get('identity')!=identity or envelope.get('checksum')!=digest(envelope.get('tasks')):
            raise ValueError('Corrupt GrepRAG corpus mapping')
        tasks=envelope['tasks']
        actual=[(str(row.task_id),safe_path(row.path)) for rows in
            iter_parquet_batches(cli.parquet,('task_id','path'),cli.batch_size,cli.limit) for row in rows]
        if actual!=[(t['task_id'],t['path']) for t in tasks] or len({t[0] for t in actual})!=len(actual):
            raise ValueError('Cached source mapping differs from dataset tasks')
        missing=set()
        for key in dict.fromkeys(t['corpus_key'] for t in tasks):
            try:cache.read(key);cache.hits+=1
            except MissingCorpusError:missing.add(key)
        if missing:
            expected={t['task_id']:t for t in tasks}
            for rows in iter_parquet_batches(cli.parquet,('task_id','path','crossfile_context'),cli.batch_size,cli.limit):
                for row in rows:
                    entry=expected[str(row.task_id)]
                    if entry['corpus_key'] in missing:
                        key,_=cache.prepare(parquet_files(row))
                        if key!=entry['corpus_key']:raise ValueError('Corpus differs from cached task mapping')
                        missing.remove(key)
            if missing:raise ValueError('Cannot restore missing corpus mapping')
    else:
        tasks=[];seen=set()
        for rows in iter_parquet_batches(cli.parquet,('task_id','path','crossfile_context'),cli.batch_size,cli.limit):
            for row in rows:
                task=str(row.task_id)
                if task in seen:raise ValueError('Duplicate corpus task')
                seen.add(task)
                files=parquet_files(row)
                key,_=cache.prepare(files)
                tasks.append(dict(task_id=task,path=safe_path(row.path),corpus_key=key))
        if not tasks:raise ValueError('No GrepRAG corpus tasks')
        atomic_json(location,dict(identity=identity,tasks=tasks,checksum=digest(tasks)))
    atomic_json(Path(output)/'preprocessing_stats.json',dict(seconds=time.perf_counter()-started,
        excluded_from_retrieval=True,samples=len(tasks),mapping_cache_hit=mapping_hit,
        unique_corpora=len({t['corpus_key'] for t in tasks}),**cache.stats()))
    return cache,tasks


def online_corpus(cache,task,req):
    if task['task_id']!=req['task_id'] or task['path']!=req['path']:raise ValueError('Prepared corpus task mismatch')
    value=cache.read(task['corpus_key'])
    entries=value['identity']['files']
    if req['path'] in {e['path'] for e in entries}:raise ValueError('Current file leaked into prepared corpus')
    return dict(repository_path=value['repository_path'],current_file=req['path'],file_count=len(entries),
        source_bytes=sum(e['size'] for e in entries),source_scope='benchmark_related_files_excluding_current',
        corpus_source='parquet.crossfile_context',persistent_corpus_cache=True,corpus_key=task['corpus_key'])
