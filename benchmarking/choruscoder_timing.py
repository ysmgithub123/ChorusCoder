"""Explicit per-sample retrieval accounting; model startup and result IO separate."""
import math
import time

FULL_RETRIEVAL_TIMING = 'choruscoder_full_retrieval_v1'


def finish_retrieval(record, started, config=None):
    """The timer already includes all repository encodes; never add them twice."""
    elapsed = time.perf_counter() - started
    selection = record['selection_seconds']
    index = record['diagnostics']['index_build_seconds']
    if any(not math.isfinite(v) or v < 0 for v in (elapsed, selection, index)):
        raise ValueError('Invalid full retrieval timing')
    record['retrieval_seconds'] = elapsed
    total = selection + elapsed
    from benchmarking.retrieval_online import ONLINE_TIMING, online
    is_online = online(config or {})
    if is_online and index != 0:
        raise ValueError('Online retrieval performed repository encoding')
    record['diagnostics']['retrieval_timing'] = dict(
        version=ONLINE_TIMING if is_online else FULL_RETRIEVAL_TIMING, selection_seconds=selection,
        retrieval_and_packing_seconds=elapsed, repository_encoding_seconds=index,
        total_seconds=total, includes_repository_encoding=not is_online,
        model_startup_and_result_io_included=False)
    if is_online:
        record['diagnostics']['retrieval_timing'].update(preprocessing_excluded=True,
            post_bm25_candidate_encoding_included=bool((config or {}).get('rlcoder_config')))
    if 'api_support' in record['diagnostics']:
        record['diagnostics']['api_support']['full_retrieval_seconds'] = total


def validate_timing(record, config):
    from benchmarking.retrieval_online import ONLINE_TIMING, online
    is_online = online(config)
    if config.get('retrieval_timing') != FULL_RETRIEVAL_TIMING and not is_online:
        return
    detail = record['diagnostics'].get('retrieval_timing', {})
    if (detail.get('version') != (ONLINE_TIMING if is_online else FULL_RETRIEVAL_TIMING) or
            detail.get('includes_repository_encoding') is not (not is_online) or
            (is_online and detail.get('repository_encoding_seconds') != 0)):
        raise ValueError('Prepared retrieval timing policy mismatch')
    pairs = [(detail.get('selection_seconds'), record['selection_seconds']),
             (detail.get('retrieval_and_packing_seconds'), record['retrieval_seconds']),
             (detail.get('repository_encoding_seconds'), record['diagnostics']['index_build_seconds']),
             (detail.get('total_seconds'), record['selection_seconds'] + record['retrieval_seconds'])]
    if any(type(a) not in (int, float) or not math.isfinite(a) or a < 0 or
           not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in pairs):
        raise ValueError('Prepared full retrieval timing values mismatch')
