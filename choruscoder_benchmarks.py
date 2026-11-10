
"""Python benchmark capabilities of the registered paper methods."""
from benchmarking.common import BenchmarkTarget
from choruscoder_profiles import PROFILES

RECCEVAL_CAST_METHODS = ('ChorusCoder', 'ChorusCoder-CodeT5p110M', 'cAST-Prefix100', 'CodeRAG-cAST')
RECCEVAL_METHODS = (*RECCEVAL_CAST_METHODS, 'RLCoder')
ADAPTER_VERSION = 'choruscoder_benchmarks_v1'

def uses_cast_adapter(method, benchmark):
    # Preserve the parser identity fields of the paper configurations.
    return (method in ('cAST-Prefix100', 'CodeRAG-cAST') or
            benchmark == 'cceval' and method in ('cAST-Prefix100-CodeT5p110M', 'cAST-Prefix100-CodeSageSmallV2', 'ChorusCoder-CodeSageSmallV2') or
            benchmark == 'recceval' and method in RECCEVAL_CAST_METHODS)

def requires_repository_parser(method, benchmark):
    return uses_cast_adapter(method, benchmark)

def experiment_target(method, benchmark, language, dataset_split=None):
    if language != 'python' or method not in PROFILES or benchmark not in ('cceval', 'recceval') or (benchmark == 'recceval' and method not in RECCEVAL_METHODS):
        raise ValueError('Unsupported paper method or Python benchmark')
    if dataset_split not in (None, '', 'python'):
        raise ValueError('The dataset split must be python')
    return BenchmarkTarget(benchmark, language)

