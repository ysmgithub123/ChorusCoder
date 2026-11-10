"""Fuse semantic similarity and source distance to select support segments."""
from dataclasses import asdict
import math
from choruscoder import select_segments, source_query
from choruscoder_chunk_controls import fully_covered, chunk_queries, suffix_span
from choruscoder_profiles import PROFILES

def support_distance_scores(candidates, target_start_line, policy):
    """Distance rules are profile policies; no method names or reserved slots."""
    ranks = None
    if policy.distance_curve == 'gaussian_rank_with_floor':
        order = sorted(range(len(candidates)), key=lambda i: (-max((b for a, b in candidates[i]['spans'])), candidates[i]['spans'][0][0]))
        ranks = {i: rank for rank, i in enumerate(order, 1)}
    rows = []
    for i, c in enumerate(candidates):
        distance = max(0, target_start_line - c['line_end'] - 1)
        if ranks is None:
            if policy.distance_curve != 'reciprocal_one_plus_distance':
                raise ValueError('Unknown support distance curve')
            row = {'distance_lines': distance, 'distance_score': 1 / (1 + distance)}
        else:
            rank = ranks[i]
            row = {'distance_lines': distance, 'distance_rank': rank, 'distance_score': policy.distance_floor + policy.distance_amplitude * math.exp(-policy.distance_decay * (rank - 1) ** 2)}
        rows.append(row)
    return rows

def rank_fused(pool, cosines, policy, threshold, structure, limit=4):
    candidates = pool['candidates']
    if len(candidates) != len(cosines):
        raise ValueError('One cosine is required per fused candidate')
    if any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not math.isfinite(v)) or (not -1 <= v <= 1) for v in cosines)):
        raise ValueError('Invalid fused support cosine')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid fused threshold')
    rows = []
    distances = support_distance_scores(candidates, pool.get('target_start_line', 0), policy)
    for c, cosine, distance in zip(candidates, cosines, distances):
        rows.append({'candidate_id': c['id'], 'spans': c['context_spans'], **distance, 'semantic_score': float(cosine), 'fused_score': policy.semantic_weight * cosine + policy.distance_weight * distance['distance_score']})
    ranking = sorted(range(len(rows)), key=lambda i: (-rows[i]['fused_score'], -rows[i]['semantic_score'], -max((b for a, b in candidates[i]['spans'])), candidates[i]['spans'][0][0]))
    occupied = list(pool['mapping']['target_spans'])
    chosen = []
    indices = []
    for i in ranking:
        row = rows[i]
        c = candidates[i]
        if row['fused_score'] < threshold:
            reason = 'below_fused_threshold'
        elif fully_covered(c['context_spans'], occupied):
            reason = 'covered_by_selected_union'
        elif len(chosen) >= limit:
            reason = 'support_limit'
        else:
            reason = None
            indices.append(i)
            chosen.append(dict(c, candidate_id=c['id'], id=f'S{len(chosen) + 1}', score=float(row['fused_score']), selection_role='fused', distance_lines=row['distance_lines'], distance_score=row['distance_score'], semantic_score=row['semantic_score'], **{'distance_rank': row['distance_rank']} if 'distance_rank' in row else {}))
            occupied += c['context_spans']
        row['exclusion_reason'] = reason
    near = select_segments(structure, [float(max((b for a, b in c['spans']))) for c in structure['candidates']], 2)
    reference = [c['context_spans'] for c in near]
    retained = [spans for spans in reference if any((c['context_spans'] == spans for c in chosen))]
    trace = {'version': policy.version, 'configuration': asdict(policy), 'threshold_kind': 'fused', 'threshold': threshold, 'fixed_support_count': 0, 'maximum_supports': limit, 'target_start_line': pool.get('target_start_line'), 'candidate_scores': rows, 'ranked_indices': ranking, 'selected_indices': indices, 'original_near_spans': reference, 'retained_original_near_spans': retained, 'threshold_exclusions': sum((r['exclusion_reason'] == 'below_fused_threshold' for r in rows)), 'coverage_exclusions': sum((r['exclusion_reason'] == 'covered_by_selected_union' for r in rows))}
    return (chosen, indices, trace)

def score_fused_supports(pool, structure, encode, semantic, policy):
    import numpy as np
    candidates = pool['candidates']
    scores = []
    if not pool['disabled_reason'] and candidates:
        texts = [structure['target']] + [c['text'] for c in candidates]
        values = encode(texts)
        values = values.numpy() if hasattr(values, 'numpy') else np.asarray(values, dtype=float)
        if values.ndim != 2 or len(values) != len(texts) or (not values.shape[1]) or (not np.isfinite(values).all()):
            raise ValueError('Invalid fused support encoder output')
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError('Zero/nonfinite fused encoder vectors')
        normalized = values / norms
        scores = np.clip(normalized[1:] @ normalized[0], -1, 1).tolist()
    evaluation = dict(pool, candidates=[] if pool['disabled_reason'] else candidates)
    chosen, indices, fused = rank_fused(evaluation, scores, policy, semantic.similarity_threshold, structure)
    trace = dict(pool, configuration=asdict(semantic), cosines=scores, selected_indices=indices, selected_extra_count=len(chosen), encoded_text_count=1 + len(candidates) if scores else 0, threshold_kind='fused')
    return (chosen, trace, fused)

def finalize_fused_fallback(structure, prefix, tokenizer, chosen):
    if chosen:
        structure['fallback'] = None
    else:
        structure['fallback'] = structure.get('fallback') or 'no_fused_support_passed_threshold'
        structure['fallback_query_text'] = source_query(prefix, [suffix_span(prefix, tokenizer, 508)])

def control_queries(method, window, structure, chosen, prefix=None):
    queries = [dict(q, role='support' if chosen else 'fallback') for q in chunk_queries(method, window, structure, chosen, prefix=prefix)]
    return (queries, None)

def validate_fused_selection(record, config):
    profile = PROFILES[config['method']]
    policy = profile.fused_support_config
    d = record['diagnostics']
    pool = d['semantic_support']
    s = d['ablation']
    if config.get('fused_support_config') != asdict(policy) or pool.get('threshold_kind') != 'fused' or pool.get('near_prefix_spans'):
        raise ValueError('Invalid fused policy/pool')
    scores = pool['cosines']
    if pool['disabled_reason']:
        if scores or pool['selected_indices']:
            raise ValueError('Disabled fused pool contains scores/selections')
        evaluation = dict(pool, candidates=[])
    else:
        evaluation = pool
    chosen, indices, expected = rank_fused(evaluation, scores, policy, config['semantic_support_config']['similarity_threshold'], s)
    if d.get('fused_support') != expected or pool['selected_indices'] != indices or pool['selected_extra_count'] != len(chosen) or (pool['encoded_text_count'] != (len(pool['candidates']) + 1 if scores else 0)):
        raise ValueError('Fused selection disagrees with recorded scores/coverage')
    return chosen
