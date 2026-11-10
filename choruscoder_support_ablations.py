"""Source-preserving, opt-in GaussRank support/query ablations."""
from dataclasses import asdict
from choruscoder import digest, make_parser, source_query, method_recall_quotas
from choruscoder_profiles import PROFILES, SemanticSupportConfig, LineRepositoryChunkConfig
from choruscoder_chunk_controls import build_chunk_selection, raw_partition, suffix_span, fully_covered, _candidate
from choruscoder_prefix_query import prefix_query_view, VERSION as PREFIX_VERSION

def build_support_selection(prefix, tokenizer, profile, language='python', path=None, policy=None):
    policy = policy or profile.support_ablation_config
    semantic = profile.semantic_support_config or SemanticSupportConfig()
    if policy.support_partition == 'syntax':
        return build_chunk_selection(prefix, tokenizer, profile.support_unit_config, semantic, profile.chunk_control_config, language, path, reserve_near=False)
    from choruscoder_repository_lines import line_partition
    from choruscoder_far import alignment
    view = {'text': prefix, 'line_offset': 0, 'start_token': 0, 'tokens': len(tokenizer.encode(prefix, add_special_tokens=False)), 'leading_truncated': False}
    unit, chunk = (profile.support_unit_config, profile.chunk_control_config)
    raw, details = raw_partition(view, unit, tokenizer, chunk, make_parser(language), language)
    end = len(prefix.encode('utf-8'))
    original_span = [raw[-1]['spans'][0][0], end] if raw else [0, end]
    original = source_query(prefix, [original_span])
    tail = suffix_span(original, tokenizer, chunk.target_tokens)
    used = [original_span[0] + tail[0], end]
    targets = [used] if original else []
    target_control = {'original_text': original, 'original_spans': [original_span] if original else [], 'original_tokens': len(tokenizer.encode(original, add_special_tokens=False)), 'used_tokens': len(tokenizer.encode(source_query(prefix, targets), add_special_tokens=False)), 'truncated_bytes': tail[0], 'used_spans': targets}
    parts = line_partition(prefix, tokenizer, LineRepositoryChunkConfig(max_tokens=unit.max_operation_tokens))
    candidates = []
    excluded = {'blank': 0, 'fully_covered': 0, 'query_oversize': 0}
    for part in parts:
        span = [part['start_byte'], part['end_byte']]
        c = _candidate(prefix, span, bool(part['degraded_reason']))
        c.update(support_tokens=part['token_count'], unit_kind='fixed_lines', dependency_analysis=['line3_no_syntax_split'], line_partition=part)
        if not c['text'].strip():
            excluded['blank'] += 1
        elif fully_covered([span], targets):
            excluded['fully_covered'] += 1
        elif len(tokenizer.encode(source_query(prefix, [span] + targets), add_special_tokens=False)) > 508:
            excluded['query_oversize'] += 1
        else:
            candidates.append(c)
    mapping = alignment(prefix, view, targets) if targets else {'status': 'empty_target', 'target_spans': []}
    structure = {'target': source_query(prefix, targets), 'target_span': used, 'target_body_spans': targets, 'target_context_spans': targets, 'prefix_target_context_spans': targets, 'candidates': candidates, 'fallback': None if candidates else 'no_line3_support_segments', 'diagnostics': {'target_policy': 'last_operation_unit', 'operation_partition': [{k: c[k] for k in ('spans', 'line_start', 'line_end', 'degraded')} for c in raw], 'parse_has_error': details.get('parse_has_error', False), 'target_control': target_control, 'chunk_policy': asdict(chunk), 'local_chunking': {'counts': excluded, 'split_records': []}}}
    pool = {'version': semantic.version, 'prefix_hash': digest(prefix), 'prefix_bytes': end, 'mapping': mapping, 'near_prefix_spans': [], 'disabled_reason': None if structure['target'].strip() else 'empty_target', 'chunk_policy': asdict(chunk), 'selection_pool_policy': 'all_supports_no_fixed_near', 'target_start_line': prefix.encode('utf-8')[:used[0]].count(b'\n') + 1, 'prefix_partition_diagnostics': {'counts': excluded, 'split_records': []}, 'prefix_partition_candidates': len(candidates), 'excluded': excluded, 'candidates': [dict(c, id='F' + digest(['prefix', c['spans'], c['text']])[:12], source_space='prefix', query_text=source_query(prefix, c['context_spans'] + targets)) for c in candidates]}
    features = {'configuration': asdict(chunk), 'local_chunking': structure['diagnostics']['local_chunking'], 'target_control': target_control, 'source_scope': 'visible_prefix', 'raw_partition': structure['diagnostics']['operation_partition'], 'line_partition': parts}
    return (view, structure, pool, features)

def select_prefix_all(pool, prefix, tokenizer=None, recorded_window=None):
    view = prefix_query_view(prefix, tokenizer, 100) if tokenizer is not None else recorded_window
    if view is None or view.get('prefix_hash') != digest(prefix) or view.get('prefix_query_tokens') != 100:
        raise ValueError('Missing/mismatched prefix-all source window')
    a, b = (view['source_start_byte'], view['source_end_byte'])
    data = prefix.encode('utf-8')
    if type(a) is not int or type(b) is not int or (not 0 <= a <= b == len(data)) or (data[a:b].decode('utf-8') != view['text']) or (type(view.get('tokens')) is not int) or (not 0 <= view['tokens'] <= 100) or (type(view.get('prefix_tokens')) is not int) or (view['prefix_tokens'] < view['tokens']) or (view.get('version') != PREFIX_VERSION) or (view.get('leading_truncated') != (a > 0)) or (view.get('line_offset') != prefix.count('\n') - view['text'].count('\n')):
        raise ValueError('Invalid prefix-all source window')
    candidates = [] if pool['disabled_reason'] else [c for c in pool['candidates'] if any((x < b and y > a for x, y in c['context_spans']))]
    candidates.sort(key=lambda c: (-max((y for x, y in c['spans'])), c['spans'][0][0]))
    chosen = [dict(c, candidate_id=c['id'], id=f'S{i + 1}', selection_role='prefix_all') for i, c in enumerate(candidates[:30])]
    trace = {'window': view, 'eligible_candidate_ids': [c['id'] for c in candidates], 'selected_candidate_ids': [c['candidate_id'] for c in chosen], 'limit_excluded_candidate_ids': [c['id'] for c in candidates[30:]], 'boundary_extended_candidate_ids': [c['candidate_id'] for c in chosen if any((x < a for x, y in c['context_spans']))]}
    return (chosen, trace)

def _merged_source(prefix, spans):
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    offset = 0
    mapping = []
    for a, b in merged:
        mapping.append({'query_start_byte': offset, 'query_end_byte': offset + b - a, 'source_span': [a, b]})
        offset += b - a + 1
    return (source_query(prefix, merged), mapping)

def ablation_queries(method, prefix, structure, chosen, tokenizer=None, recorded=None, policy=None):
    policy = policy or PROFILES[method].support_ablation_config
    merge = None
    if not chosen:
        queries = [{'id': 'Q0', 'text': structure['fallback_query_text'], 'segment_ids': [], 'role': 'fallback'}]
    elif policy.query_composition == 'merged':
        spans = [s for c in chosen for s in c['context_spans']] + structure['target_context_spans']
        full, mapping = _merged_source(prefix, spans)
        data = full.encode('utf-8')
        start = suffix_span(full, tokenizer, 508)[0] if tokenizer is not None else recorded['merge']['truncated_bytes']
        if type(start) is not int or not 0 <= start <= len(data):
            raise ValueError('Invalid merged query cut')
        text = data[start:].decode('utf-8')
        retained = [[a + max(0, start - m['query_start_byte']), b] for m in mapping for a, b in [m['source_span']] if m['query_end_byte'] > start]
        if not fully_covered(structure['target_context_spans'], retained):
            raise ValueError('Merged query encoder cut removed T')
        original_tokens = len(tokenizer.encode(full, add_special_tokens=False)) if tokenizer is not None else recorded['merge']['original_tokens']
        merge = {'full_text': full, 'original_tokens': original_tokens, 'truncated_bytes': start, 'source_mapping': mapping, 'retained_source_spans': retained, 'target_preserved': True}
        ordered = sorted(chosen, key=lambda c: c['spans'][0][0])
        queries = [{'id': 'Q0', 'text': text, 'segment_ids': [c['id'] for c in ordered], 'role': 'support'}]
    else:
        queries = [{'id': f'Q{i + 1}', 'text': source_query(prefix, c['context_spans'] + structure['target_context_spans']), 'segment_ids': [c['id']], 'role': 'support'} for i, c in enumerate(chosen)]
    tokens = [len(tokenizer.encode(q['text'], add_special_tokens=False)) for q in queries] if tokenizer is not None else recorded['query_tokens']
    if len(tokens) != len(queries) or any((type(n) is not int or not 0 <= n <= 508 for n in tokens)):
        raise ValueError('Support ablation query exceeds encoder budget')
    if merge is not None and (type(merge['original_tokens']) is not int or merge['original_tokens'] < tokens[0] or (merge['truncated_bytes'] and merge['original_tokens'] <= 508)):
        raise ValueError('Invalid merged query token accounting')
    return (queries, {'configuration': asdict(policy), 'query_tokens': tokens, 'merge': merge})

def ablation_checksum(record):
    d = record['diagnostics']
    return digest({'window': record['window'], 'ablation': d['ablation'], 'controls': d['support_ablation'], 'chunk_features': d['chunk_controls'], 'chosen': record['selected_segments'], 'queries': record['queries'], 'fallback': record['fallback'], 'eligible_supports': record['eligible_supports'], 'evidence': record['evidence'], 'ranked': d['ranked'], 'quotas': d['recall_quotas'], 'semantic': d.get('semantic_support'), 'fused': d.get('fused_support')})

def validate_ablation_record(record, config, tokenizer=None):
    from choruscoder_selection_controls import validate_fused_selection
    p = PROFILES[config['method']]
    policy = p.support_ablation_config
    d = record['diagnostics']
    s = d['ablation']
    control = d['support_ablation']
    prefix = record.get('prefix', record['window']['text'])
    if config.get('support_ablation_config') != asdict(policy) or control['configuration'] != asdict(policy) or d.get('support_ablation_checksum') != ablation_checksum(record):
        raise ValueError('Support ablation configuration/checksum mismatch')
    if record['window']['text'] != prefix or record['window']['leading_truncated'] or record['probe_seconds'] != 0 or record['official'] or (record.get('context_profile') != config['context_profile']):
        raise ValueError('Support ablation source/probe/context mismatch')
    if source_query(prefix, s['target_context_spans']) != s['target']:
        raise ValueError('Support ablation T differs from source')
    features = d['chunk_controls']
    pool = control['pool']
    if features['configuration'] != config['chunk_control_config'] or features['target_control']['used_tokens'] > 384 or pool['prefix_hash'] != digest(prefix) or pool['near_prefix_spans']:
        raise ValueError('Support ablation pool/target budget mismatch')
    for c in s['candidates'] + pool['candidates']:
        lines = c['line_partition']['parent_line_end'] - c['line_partition']['parent_line_start'] + 1 if policy.support_partition == 'line3' else len(c['text'].splitlines())
        if source_query(prefix, c['context_spans']) != c['text'] or not 0 <= c['support_tokens'] <= 96 or lines > 6:
            raise ValueError('Support ablation candidate source/budget mismatch')
    if s['scores'] != [float(max((b for a, b in c['spans']))) for c in s['candidates']] or record['eligible_supports'] != len(s['candidates']):
        raise ValueError('Support ablation structural scores/count mismatch')
    if policy.selection == 'prefix_all':
        if any((k in d for k in ('semantic_support', 'fused_support', 'semantic_encoding_cache'))):
            raise ValueError('Prefix-all record contains a removed scorer')
        chosen, selection = select_prefix_all(pool, prefix, tokenizer, control['selection']['window'])
        if selection != control['selection']:
            raise ValueError('Prefix-all window/selection mismatch')
    else:
        semantic = d['semantic_support']
        if semantic.get('configuration') != asdict(SemanticSupportConfig(**config['semantic_support_config'])):
            raise ValueError('Fused support encoder policy mismatch')
        for k, v in pool.items():
            if semantic.get(k) != v:
                raise ValueError('Fused pool differs from ablation source pool')
        chosen = validate_fused_selection(record, config)
        if control['selection'] is not None:
            raise ValueError('Scored ablation contains prefix-all selection')
    if chosen != record['selected_segments'] or bool(record['fallback']) != bool(not chosen):
        raise ValueError('Support ablation chosen/fallback mismatch')
    queries, qtrace = ablation_queries(config['method'], prefix, s, chosen, tokenizer, control['queries'])
    if queries != record['queries'] or qtrace != control['queries'] or d['recall_quotas'] != method_recall_quotas(config['method'], len(queries)):
        raise ValueError('Support ablation queries/quotas mismatch')
    if tokenizer is not None:
        view, expected, expected_pool, expected_features = build_support_selection(prefix, tokenizer, p, config.get('language', 'python'), record['path'])
        if view != record['window'] or expected_pool != pool or expected_features != features or (expected['target'] != s['target']) or (expected['target_context_spans'] != s['target_context_spans']) or (expected['candidates'] != s['candidates']):
            raise ValueError('Support ablation source/partition replay mismatch')
        if not chosen and s['fallback_query_text'] != source_query(prefix, [suffix_span(prefix, tokenizer, 508)]):
            raise ValueError('Support ablation fallback source mismatch')
    return control
