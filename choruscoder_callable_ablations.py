"""Source-only controls around callable selection; never encode or retrieve here."""
from choruscoder_chunk_controls import build_chunk_selection
from choruscoder_support_ablations import build_support_selection, ablation_queries


def validate_runtime():
    """Reject mixed updates before preparing candidates or loading weights."""
    import inspect
    import choruscoder_api_support as core
    import choruscoder_chunk_controls as chunks
    protocol = 'callable_support_ablations_v1'
    if (getattr(core, 'CALLABLE_ABLATION_PROTOCOL', None) != protocol or
            getattr(chunks, 'CALLABLE_ABLATION_PROTOCOL', None) != protocol or
            any('policy' not in inspect.signature(fn).parameters
                for fn in (build_support_selection, ablation_queries))):
        raise RuntimeError('Callable ablation files are incompatible. Synchronize choruscoder_profiles.py, '
            'choruscoder_callable_ablations.py, choruscoder_support_ablations.py, choruscoder_api_support.py, '
            'choruscoder_chunk_controls.py, benchmark_cli.py and benchmarking/choruscoder_runner.py '
            'from the same update. No candidate preparation or model loading has started.')


def build_selection(prefix, tokenizer, profile, semantic, language='python', path=None):
    policy = profile.callable_ablation_config
    if policy is not None and policy.support_partition == 'line3':
        return build_support_selection(prefix, tokenizer, profile, language, path, policy=policy)
    return build_chunk_selection(prefix, tokenizer, profile.support_unit_config, semantic,
                                 profile.chunk_control_config, language, path, reserve_near=False)


def final_queries(profile, prefix, structure, chosen, selection_queries, tokenizer=None, recorded=None):
    policy = profile.callable_ablation_config
    if policy is None or policy.query_composition != 'merged':
        return selection_queries, None
    return ablation_queries(None, prefix, structure, chosen, tokenizer, recorded, policy=policy)
