"""Explicit official GrepRAG profile, independent of ChorusCoder retrieval."""
from dataclasses import dataclass
import os

METHOD = 'GrepRAG-0.6B'
BOUNDED_METHOD = METHOD
METHODS = (METHOD,)
OFFICIAL_COMMIT = '31a6875213540ae60c26ea8e4782d0707248b405'
MODEL_REVISION = '8fd8f3c0ecf6c687daaa69d702742506db74c87f'
VERSION = 'greprag_kw6144_online_v2'
ONLINE_TIMING = 'greprag_repository_preprocessed_online_v1'


def timing_scope(cli):
    return ONLINE_TIMING


@dataclass(frozen=True)
class GrepRAGProfile:
    prompt_tokens: int = 4096
    crossfile_tokens: int = 3072
    new_tokens: int = 50
    max_keywords: int = 10
    keyword_new_tokens: int = 64
    context_lines: int = 10
    official_budget: int = 4096
    final_topk: int = 10
    keyword_input_tokens: int = 6144


PROFILES = {METHOD: GrepRAGProfile()}


def worker_python():
    return '.venv-greprag/Scripts/python.exe' if os.name == 'nt' else '.venv-greprag/bin/python'


def defaults(method=METHOD):
    p = PROFILES[method]
    return dict(max_context_tokens=p.prompt_tokens, max_crossfile_tokens=p.crossfile_tokens,
                max_new_tokens=p.new_tokens, retrieval_timing='online')


def validate_options(args):
    if args.method not in METHODS or args.benchmark not in ('cceval','recceval') or args.language!='python':
        raise ValueError('GrepRAG supports CCEVAL and RECCEVAL Python only')
    if getattr(args, 'subset', 'all') not in ('all', None, '') or getattr(args, 'dataset_split', 'python') != 'python':
        raise ValueError('GrepRAG requires the CCEVAL Python partition')
    for name, expected in defaults(args.method).items():
        if getattr(args, name) != expected:
            raise ValueError(f'GrepRAG comparison requires --{name.replace("_", "-")} {expected}')
    if args.num_workers != 1 or args.dataloader_num_workers != 0 or args.dataloader_prefetch_factor != 1:
        raise ValueError('GrepRAG requires worker=1, DataLoader workers=0, prefetch=1')
    if args.batch_size < 1 or args.limit is not None and args.limit < 1:
        raise ValueError('Batch size and limit must be positive')
    if getattr(args, 'query_max_segments', None) is not None or getattr(args, 'query_reference_dir', None):
        raise ValueError('GrepRAG does not use ChorusCoder segments or reference runs')
    if getattr(args, 'query_support_similarity_threshold', None) is not None:
        raise ValueError('GrepRAG does not use a ChorusCoder support threshold')
    if getattr(args, 'skip_existing', False):
        raise ValueError('GrepRAG does not reuse completed retrievals; use a new output directory')
    if getattr(args,'greprag_corpus','parquet')!='parquet':
        raise ValueError('Preprocessed GrepRAG requires the exact Parquet corpus')
    if getattr(args,'greprag_resume',False):
        if getattr(args,'greprag_corpus','parquet')!='parquet':
            raise ValueError('GrepRAG recovery supports the Parquet corpus only')
        if getattr(args,'stage',getattr(args,'query_stage','all')) not in ('all','prepare'):
            raise ValueError('--greprag-resume is for interrupted retrieval: use stage all or prepare')
    return PROFILES[args.method]
