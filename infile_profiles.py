"""Explicit, immutable generation-only controls, separate from retrieval profiles."""
from dataclasses import dataclass
from types import MappingProxyType
from benchmarking.common import BenchmarkTarget, validate_generation_tokens

@dataclass(frozen=True)
class InFileProfile:
    prompt_tokens: int = 4096
    new_tokens: int = 50
    version: str = 'infile_no_retrieval_v1'

    def __post_init__(self):
        if any((type(n) is not int or n < 1 for n in (self.prompt_tokens, self.new_tokens))):
            raise ValueError('In-file prompt and generation budgets must be positive integers')
INFILE_METHOD = 'InFile-NoRetrieval'
PROFILES = MappingProxyType({INFILE_METHOD: InFileProfile()})
METHODS = tuple(PROFILES)

def experiment_target(benchmark, language, dataset_split=None):
    if language != 'python' or benchmark not in ('cceval', 'recceval') or dataset_split not in (None, 'python'):
        raise ValueError('In-File requires a paper Python benchmark')
    return BenchmarkTarget(benchmark, language)

def validate_options(args):
    profile = PROFILES[args.method]
    experiment_target(args.benchmark, args.language, getattr(args, 'dataset_split', None))
    if args.benchmark in ('cceval', 'recceval') and getattr(args, 'subset', 'all') not in (None, '', 'all'):
        raise ValueError(f'In-File {args.benchmark} requires the Python partition')
    if getattr(args, 'retrieval_timing', None) not in (None, 'online'):
        raise ValueError('In-File uses online timing with zero retrieval cost')
    if (args.max_context_tokens, args.max_crossfile_tokens) != (profile.prompt_tokens, 0):
        raise ValueError(f'{args.method} requires {profile.prompt_tokens}/0 input budgets')
    validate_generation_tokens(args.max_new_tokens)
    if args.batch_size < 1 or (getattr(args, 'limit', None) is not None and args.limit < 1):
        raise ValueError('Batch size and sample limit must be positive')
    if args.num_workers != 1 or args.dataloader_num_workers != 0 or args.dataloader_prefetch_factor != 1:
        raise ValueError('In-File requires worker=1, DataLoader workers=0 and prefetch=1')
    if getattr(args, 'query_support_similarity_threshold', None) is not None or getattr(args, 'query_reference_dir', None):
        raise ValueError('In-File does not accept support thresholds or retrieval reference directories')
    return profile
