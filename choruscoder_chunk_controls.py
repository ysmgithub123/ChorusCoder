"""Syntax-aware support segmentation and bounded query construction."""
from bisect import bisect_left
from dataclasses import asdict
from functools import lru_cache
from copy import deepcopy
from choruscoder import build_queries, digest, make_parser, select_segments, source_query, window_view
from choruscoder_far import alignment
from choruscoder_semantic_support import overlaps
from choruscoder_support_units import SIMPLE, COMPOUND, _scope, operation_segments
CALLABLE_ABLATION_PROTOCOL = 'callable_support_ablations_v1'

def _boundaries(data):
    result = [0]
    for char in data.decode('utf-8'):
        result.append(result[-1] + len(char.encode('utf-8')))
    return result

@lru_cache(maxsize=4)
def _vocabulary_bytes(tokenizer):
    if not hasattr(tokenizer, 'byte_decoder') or not hasattr(tokenizer, 'get_vocab'):
        return ()
    pieces = set()
    for token in tokenizer.get_vocab():
        try:
            piece = bytes((tokenizer.byte_decoder[c] for c in token))
        except (KeyError, TypeError):
            continue
        if piece:
            pieces.add(piece)
    return tuple(sorted(pieces, key=len, reverse=True))

def suffix_span(text, tokenizer, budget):
    """Longest source suffix that fits, using exact byte-level BPE offsets.

    Decode is never used to reconstruct source. Exotic tokenizers without byte
    offsets use an exhaustive source-suffix check, avoiding monotonicity claims.
    """
    data = text.encode('utf-8')
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return [0, len(data)]
    from choruscoder_prefix_query import _stable_byte_bpe, _suffix_start
    try:
        stable = _stable_byte_bpe(tokenizer) and (not any((token and token in text for token in tokenizer.get_added_vocab())))
    except (AttributeError, TypeError, ValueError):
        stable = False
    if stable:
        return [_suffix_start(text, tokenizer, budget, len(ids)), len(data)]
    from benchmarking.choruscoder_cast import token_byte_offsets
    _, offsets = token_byte_offsets(text, tokenizer)
    boundaries = _boundaries(data)
    valid = set(boundaries)
    starts = [a for a in offsets if a in valid] if offsets else boundaries
    position = max(0, len(ids) - budget)
    hint = offsets[position] if offsets else 0
    start = next((a for a in starts if a >= hint and len(tokenizer.encode(data[a:].decode('utf-8'), add_special_tokens=False)) <= budget), len(data))
    lower = 0
    if offsets:
        try:
            vocabulary = _vocabulary_bytes(tokenizer)
        except TypeError:
            vocabulary = ()
        maximum = next((len(piece) for piece in vocabulary if piece in data), None) if getattr(tokenizer, 'add_prefix_space', None) is False else len(vocabulary[0]) if vocabulary else None
        if maximum:
            lower = max(0, len(data) - budget * maximum)
    for a in boundaries[bisect_left(boundaries, lower):]:
        if a >= start:
            break
        if len(tokenizer.encode(data[a:].decode('utf-8'), add_special_tokens=False)) <= budget:
            start = a
            break
    return [start, len(data)]

class SupportSplitter:
    containers = {'module', 'block', 'argument_list', 'parameters', 'list', 'tuple', 'set', 'dictionary', 'expression_list', 'pattern_list', 'boolean_operator'}

    def __init__(self, text, tokenizer, unit, root=None, language='python'):
        self.text = text
        self.data = text.encode('utf-8')
        self.tokenizer = tokenizer
        self.unit = unit
        self.parser = make_parser(language)
        self.root = root or self.parser.parse(self.data).root_node
        self.boundaries = _boundaries(self.data)
        self.newlines = [i for i, byte in enumerate(self.data) if byte == 10]
        self._measure_cache = {}
        self._split_cache = {}
        self.stats = {'syntax_splits': 0, 'line_bisections': 0, 'token_bisections': 0, 'degraded_fragments': 0}
        cuts = set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(node.named_children)
            if node.type in self.containers:
                for child in node.named_children[1:]:
                    cuts.add(child.start_byte)
        self.syntax_cuts = sorted(cuts)

    def measure(self, a, b):
        key = (a, b)
        if key not in self._measure_cache:
            text = self.data[a:b].decode('utf-8')
            self._measure_cache[key] = (len(text.splitlines()), len(self.tokenizer.encode(text, add_special_tokens=False)))
        return self._measure_cache[key]

    def fits(self, a, b):
        lines, tokens = self.measure(a, b)
        return lines <= self.unit.max_operation_lines and tokens <= self.unit.max_operation_tokens

    def _middle(self, a, b, lines, tokens):
        if lines > self.unit.max_operation_lines:
            breaks = [a + i + 1 for i, byte in enumerate(self.data[a:b]) if byte == 10 and a + i + 1 < b]
            if breaks:
                return (breaks[(len(breaks) - 1) // 2], 'line_bisection')
        text = self.data[a:b].decode('utf-8')
        from benchmarking.choruscoder_cast import token_byte_offsets
        ids, offsets = token_byte_offsets(text, self.tokenizer)
        legal = self.boundaries[bisect_left(self.boundaries, a + 1):bisect_left(self.boundaries, b)]
        if not legal:
            raise ValueError('A single Unicode character exceeds the support budget')
        if offsets and len(ids) > 1:
            mid = a + offsets[len(ids) // 2]
        else:
            lo, hi = (0, len(legal) - 1)
            goal = tokens // 2
            while lo < hi:
                position = (lo + hi) // 2
                count = len(self.tokenizer.encode(self.data[a:legal[position]].decode('utf-8'), add_special_tokens=False))
                if count < goal:
                    lo = position + 1
                else:
                    hi = position
            options = legal[max(0, lo - 1):min(len(legal), lo + 2)]
            mid = min(options, key=lambda x: (abs(len(self.tokenizer.encode(self.data[a:x].decode('utf-8'), add_special_tokens=False)) - goal), x))
        return (min(legal, key=lambda x: (abs(x - mid), x)), 'token_bisection')

    def split(self, span):
        key = tuple(span)
        if key in self._split_cache:
            pieces, counts = self._split_cache[key]
            for name, count in counts.items():
                self.stats[name] += count
            return deepcopy(pieces)
        initial = dict(self.stats)
        pieces = []
        pending = [(*span, False, [])]
        parent = span
        while pending:
            a, b, degraded, reasons = pending.pop()
            if self.fits(a, b):
                pieces.append({'span': [a, b], 'degraded': degraded, 'reasons': reasons})
                continue
            lines, tokens = self.measure(a, b)
            mid, fallback = self._middle(a, b, lines, tokens)
            cuts = self.syntax_cuts[bisect_left(self.syntax_cuts, a + 1):bisect_left(self.syntax_cuts, b)]
            if cuts:
                cut = min(cuts, key=lambda x: (abs(x - mid), x))
                reason = 'syntax_boundary'
                self.stats['syntax_splits'] += 1
            else:
                cut = mid
                reason = fallback
                degraded = True
                self.stats['line_bisections' if reason == 'line_bisection' else 'token_bisections'] += 1
            if not a < cut < b:
                raise ValueError('Recursive support split failed to make progress')
            pending.extend([(cut, b, degraded, reasons + [reason]), (a, cut, degraded, reasons + [reason])])
        cursor = parent[0]
        for piece in pieces:
            a, b = piece['span']
            if a != cursor or not a < b <= parent[1] or (not self.fits(a, b)):
                raise ValueError('Support split lost source or budget')
            if self.parser.parse(self.data[a:b]).root_node.has_error:
                piece['degraded'] = True
                piece['reasons'].append('incomplete_syntax_fragment')
            cursor = b
            self.stats['degraded_fragments'] += piece['degraded']
        if cursor != parent[1]:
            raise ValueError('Support split lost trailing source')
        self._split_cache[key] = (deepcopy(pieces), {name: value - initial[name] for name, value in self.stats.items()})
        return pieces

def _candidate(text, span, degraded=False, headers=None, *, data=None, newlines=None):
    raw = text.encode('utf-8') if data is None else data
    a, b = span
    if newlines is None:
        first = raw[:a].count(b'\n') + 1
        last = raw[:b].rstrip(b'\n').count(b'\n') + 1
    else:
        end = b
        while end and raw[end - 1] == 10:
            end -= 1
        first = bisect_left(newlines, a) + 1
        last = bisect_left(newlines, end) + 1
    return {'id': 'S' + digest([span])[:12], 'spans': [span], 'context_spans': [span], 'text': raw[a:b].decode('utf-8'), 'line_start': first, 'line_end': last, 'dependency_analysis': ['isolated_parse_fragment' if degraded else 'short_operation_unit'], 'unit_kind': 'operation', 'degraded': degraded, **({'header_spans': headers} if headers else {})}

def _header_partition(view, unit, tokenizer, parser):
    text = view['text']
    data = text.encode('utf-8')
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode('utf-8')))
    root = parser.parse(data).root_node

    def atom(node, last_row=None, degraded=False):
        first = node.start_point[0]
        last = node.end_point[0] - int(node.end_point[1] == 0) if last_row is None else last_row
        if last < first:
            return None
        a, b = (offsets[first], offsets[min(last + 1, len(lines))])
        if not data[a:b].strip():
            return None
        scope = _scope(node) + ((node.start_byte,) if node.type in {'class_definition', 'function_definition'} else ())
        return {'span': [a, b], 'scope': scope, 'block': node.parent.start_byte if node.parent else -1, 'degraded': degraded, 'headers': []}

    def visit(node):
        if node.type in SIMPLE:
            a = atom(node)
            return [a] if a else []
        if node.type == 'ERROR' and (not any((c.type in SIMPLE | COMPOUND for c in node.named_children))):
            a = atom(node, degraded=True)
            return [a] if a else []
        body = node.child_by_field_name('body') or next((c for c in node.named_children if c.type == 'block'), None)
        if node.type in COMPOUND and body and (body.start_point[0] == node.start_point[0]):
            a = atom(node, degraded=True)
            return [a] if a else []
        children = [a for child in node.named_children for a in visit(child)]
        if node.type in COMPOUND and body:
            head = atom(node, body.start_point[0] - 1)
            first = next((i for i, a in enumerate(children) if a['span'][1] > body.start_byte and a['span'][0] >= offsets[body.start_point[0]]), None)
            if head and first is not None:
                child = dict(children[first])
                child['span'] = [head['span'][0], child['span'][1]]
                child['headers'] = [head['span']] + child['headers']
                children[first] = child
            elif head:
                head['degraded'] = True
                head['headers'] = [head['span']]
                children.insert(0, head)
        return children
    atoms = visit(root)
    end = max((a['span'][1] for a in atoms), default=0)
    if data[end:].strip():
        atoms.append({'span': [end, len(data)], 'scope': (), 'block': ('tail', end), 'degraded': True, 'headers': []})
    result = []
    pending = None

    def output():
        nonlocal pending
        if pending:
            result.append(_candidate(text, pending['span'], pending['degraded'], pending['headers']))
        pending = None
    for a in sorted(atoms, key=lambda x: tuple(x['span'])):
        if pending and a['span'][0] < pending['span'][1]:
            if a['span'][1] <= pending['span'][1]:
                continue
            output()
        if pending:
            start, end = (pending['span'][0], a['span'][1])
            length = len(data[start:end].decode('utf-8').splitlines())
            merge = pending['scope'] == a['scope'] and pending['block'] == a['block'] and (not pending['degraded']) and (not a['degraded']) and (not data[pending['span'][1]:a['span'][0]].strip()) and (length <= unit.max_operation_lines) and (len(tokenizer.encode(data[start:end].decode('utf-8'), add_special_tokens=False)) <= unit.max_operation_tokens)
            if not merge:
                output()
        if pending:
            pending['span'][1] = a['span'][1]
            pending['headers'] += a['headers']
        else:
            pending = dict(a, span=list(a['span']), headers=list(a['headers']))
        if pending['degraded'] or len(data[pending['span'][0]:pending['span'][1]].decode('utf-8').splitlines()) >= unit.operation_lines:
            output()
    output()
    return (result, {'parse_has_error': root.has_error, 'head_body': True})

def raw_partition(view, unit, tokenizer, policy, parser, language='python'):
    if policy.head_body:
        return _header_partition(view, unit, tokenizer, parser)
    partitioned = operation_segments(view, unit, tokenizer, parser=parser, target_mode='last_operation_unit', language=language)
    data = view['text'].encode('utf-8')
    newlines = [i for i, byte in enumerate(data) if byte == 10]
    result = [_candidate(view['text'], u['spans'][0], u['degraded'], data=data, newlines=newlines) for u in partitioned['diagnostics'].get('operation_partition', [])]
    return (result, partitioned['diagnostics'])

def eligible_blocks(view, raw, unit, tokenizer, policy, exclude_cut=True, language='python', splitter=None):
    splitter = splitter or SupportSplitter(view['text'], tokenizer, unit, language=language)
    splitter.stats = {name: 0 for name in splitter.stats}
    result = []
    records = []
    counts = {'excluded_cut_blocks': 0, 'excluded_oversize_operations': 0, 'recovered_operations': 0, 'degraded_head_fragments': 0}
    cut = sum((len(l.encode('utf-8')) for l in view['text'].splitlines(keepends=True)[:3])) if exclude_cut and view.get('leading_truncated') else 0
    for c in raw:
        span = c['spans'][0]
        if span[0] < cut:
            counts['excluded_cut_blocks'] += 1
            continue
        if splitter.fits(*span):
            result.append(dict(c, support_tokens=splitter.measure(*span)[1]))
            continue
        if not policy.recursive_split:
            counts['excluded_oversize_operations'] += 1
            continue
        pieces = splitter.split(span)
        counts['recovered_operations'] += 1
        records.append({'parent_span': span, 'pieces': pieces, 'headers': c.get('header_spans', [])})
        for piece in pieces:
            a, b = piece['span']
            if not splitter.data[a:b].strip():
                continue
            head_only = any((h[0] <= a and b <= h[1] for h in c.get('header_spans', [])))
            child = _candidate(view['text'], [a, b], c['degraded'] or piece['degraded'] or head_only, c.get('header_spans'), data=splitter.data, newlines=splitter.newlines)
            child.update(source_operation_span=span, split_reasons=piece['reasons'], head_only_fragment=head_only)
            child['support_tokens'] = splitter.measure(a, b)[1]
            counts['degraded_head_fragments'] += head_only
            result.append(child)
    counts.update(splitter.stats)
    return (result, {'counts': counts, 'split_records': records})

def fully_covered(spans, occupied):
    merged = []
    for a, b in sorted(occupied):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return all((any((c <= a and b <= d for c, d in merged)) for a, b in spans))

def build_chunk_selection(prefix, tokenizer, unit, semantic, policy, language='python', path=None, reserve_near=True):
    from choruscoder_languages import grammar_variant
    language = grammar_variant(language, path)
    parser = make_parser(language)
    if policy.full_prefix:
        view = {'text': prefix, 'line_offset': 0, 'start_token': 0, 'tokens': len(tokenizer.encode(prefix, add_special_tokens=False)), 'leading_truncated': False}
    else:
        view = window_view(prefix, tokenizer)
    raw, details = raw_partition(view, unit, tokenizer, policy, parser, language=language)
    original_span = [raw[-1]['spans'][0][0], len(view['text'].encode('utf-8'))] if raw else [0, len(view['text'].encode('utf-8'))]
    original = source_query(view['text'], [original_span])
    tail = suffix_span(original, tokenizer, policy.target_tokens)
    used_span = [original_span[0] + tail[0], original_span[1]]
    target_spans = [used_span] if original else []
    local_splitter = SupportSplitter(view['text'], tokenizer, unit, language=language)
    candidates, local_diagnostics = eligible_blocks(view, raw[:-1], unit, tokenizer, policy, language=language, splitter=local_splitter)
    target_control = {'original_text': original, 'original_spans': [original_span] if original else [], 'original_tokens': len(tokenizer.encode(original, add_special_tokens=False)), 'used_tokens': len(tokenizer.encode(source_query(view['text'], target_spans), add_special_tokens=False)), 'truncated_bytes': tail[0], 'used_spans': target_spans}
    structure = {'target': source_query(view['text'], target_spans), 'target_span': used_span, 'target_body_spans': target_spans, 'target_context_spans': target_spans, 'candidates': candidates, 'fallback': None if candidates else 'no_operation_support_units', 'diagnostics': {'target_policy': 'last_operation_unit', 'operation_partition': [{k: c[k] for k in ('spans', 'line_start', 'line_end', 'degraded')} for c in raw], 'parse_has_error': details.get('parse_has_error', False), 'target_control': target_control, 'chunk_policy': asdict(policy), 'local_chunking': local_diagnostics}}
    scores = [float(max((b for a, b in c['spans']))) for c in candidates]
    near = select_segments(structure, scores, semantic.near_segments) if reserve_near else []
    mapping = alignment(prefix, view, target_spans) if target_spans else {'status': 'empty_target', 'target_spans': []}
    pool = {'version': semantic.version, 'prefix_hash': digest(prefix), 'prefix_bytes': len(prefix.encode('utf-8')), 'mapping': mapping, 'candidates': [], 'near_prefix_spans': [], 'disabled_reason': None, 'chunk_policy': asdict(policy)}
    if not reserve_near:
        pool['selection_pool_policy'] = 'all_supports_no_fixed_near'
        pool['target_start_line'] = prefix.encode('utf-8')[:used_span[0]].count(b'\n') + 1
    if mapping['status'] not in {'exact', 'exact_target_suffix'}:
        pool['disabled_reason'] = mapping['status']
    else:
        shift = mapping['window_shift']
        structure['prefix_target_context_spans'] = mapping['target_spans']
        pool['near_prefix_spans'] = [[[a + shift, b + shift] for a, b in c['context_spans']] for c in near]
        whole = {'text': prefix, 'leading_truncated': False}
        if policy.full_prefix:
            all_raw = raw
            whole_splitter = local_splitter
        else:
            all_raw, _ = raw_partition(whole, unit, tokenizer, policy, parser, language=language)
            whole_splitter = None
        full, diagnostics = eligible_blocks(whole, all_raw, unit, tokenizer, policy, exclude_cut=False, language=language, splitter=whole_splitter)
        pool['prefix_partition_diagnostics'] = diagnostics
        pool['prefix_partition_candidates'] = len(full)
        occupied = mapping['target_spans'] + [s for support in pool['near_prefix_spans'] for s in support]
        excluded = {'fully_covered': 0, 'overlap_target': 0, 'overlap_near': 0, 'query_oversize': 0, 'duplicate_span': 0, 'partial_overlap_retained': 0}
        seen = set()
        for c in full:
            spans = c['context_spans']
            signature = tuple(map(tuple, spans))
            target_overlap = overlaps(spans, mapping['target_spans'])
            near_overlap = any((overlaps(spans, s) for s in pool['near_prefix_spans']))
            if policy.partial_overlap:
                if fully_covered(spans, occupied):
                    excluded['fully_covered'] += 1
                    continue
            else:
                if max((b for a, b in spans)) > min((a for a, b in mapping['target_spans'])):
                    excluded['overlap_target'] += 1
                    continue
                if near_overlap:
                    excluded['overlap_near'] += 1
                    continue
            if signature in seen:
                excluded['duplicate_span'] += 1
                continue
            seen.add(signature)
            query = source_query(prefix, spans + mapping['target_spans'])
            if len(tokenizer.encode(query, add_special_tokens=False)) > policy.query_tokens:
                excluded['query_oversize'] += 1
                continue
            excluded['partial_overlap_retained'] += policy.partial_overlap and (target_overlap or near_overlap)
            pool['candidates'].append(dict(c, id='F' + digest(['prefix', spans, c['text']])[:12], source_space='prefix', query_text=query))
        pool['excluded'] = excluded
    if not structure['target'].strip():
        pool['disabled_reason'] = 'empty_target'
    if policy.full_prefix and (not candidates):
        fallback_span = suffix_span(prefix, tokenizer, policy.query_tokens)
        structure['fallback_query_text'] = source_query(prefix, [fallback_span])
    features = {'configuration': asdict(policy), 'local_chunking': local_diagnostics, 'target_control': target_control, 'source_scope': 'visible_prefix' if policy.full_prefix else 'choruscoder_window', 'raw_partition': structure['diagnostics']['operation_partition']}
    return (view, structure, pool, features)

def chunk_queries(method, view, structure, chosen, prefix=None):
    if not chosen and 'fallback_query_text' in structure:
        return [{'id': 'Q0', 'text': structure['fallback_query_text'], 'segment_ids': []}]
    return build_queries(method, view, dict(structure, fallback=None if chosen else structure.get('fallback')), chosen, prefix=prefix)

def chunk_checksum(record):
    d = record['diagnostics']
    value = {'window': record['window'], 'structure': d['ablation'], 'features': d['chunk_controls'], 'semantic': d['semantic_support'], 'chosen': record['selected_segments'], 'queries': record['queries'], 'fallback': record['fallback'], 'evidence': record['evidence'], 'ranked': d['ranked'], 'quotas': d['recall_quotas']}
    for name in ('fused_support', 'api_support'):
        if name in d:
            value[name] = d[name]
    return digest(value)

def validate_chunk_record(record, config, tokenizer=None):
    from choruscoder_profiles import PROFILES, SemanticSupportConfig
    from choruscoder import method_recall_quotas, query_roles
    p = PROFILES[config['method']]
    policy = p.chunk_control_config
    d = record['diagnostics']
    s = d['ablation']
    features = d['chunk_controls']
    pool = d['semantic_support']
    if d.get('chunk_checksum') != chunk_checksum(record):
        raise ValueError('Chunk control cache checksum mismatch')
    if config.get('chunk_control_config') != asdict(policy) or features['configuration'] != asdict(policy) or pool['chunk_policy'] != asdict(policy):
        raise ValueError('Chunk policy mismatch')
    semantic = SemanticSupportConfig(**config['semantic_support_config'])
    if pool['configuration'] != asdict(semantic):
        raise ValueError('Chunk semantic configuration mismatch')
    if source_query(record['window']['text'], s['target_context_spans']) != s['target']:
        raise ValueError('Chunk T source mismatch')
    if features['target_control']['used_tokens'] > 384:
        raise ValueError('Chunk T encoder budget exceeded')
    for c in s['candidates']:
        if source_query(record['window']['text'], c['context_spans']) != c['text']:
            raise ValueError('Chunk local support source mismatch')
    for c in s['candidates'] + pool['candidates']:
        if len(c['text'].splitlines()) > 6 or not 0 <= c.get('support_tokens', -1) <= 96:
            raise ValueError('Chunk support budget invalid')
    for diagnostic in (features['local_chunking'], pool.get('prefix_partition_diagnostics', {})):
        for splitting in diagnostic.get('split_records', []):
            cursor = splitting['parent_span'][0]
            for piece in splitting['pieces']:
                a, b = piece['span']
                if a != cursor or not a < b <= splitting['parent_span'][1]:
                    raise ValueError('Chunk split coverage invalid')
                cursor = b
            if cursor != splitting['parent_span'][1]:
                raise ValueError('Chunk split lost source')
    if policy.partial_overlap and (not pool['disabled_reason']):
        occupied = pool['mapping']['target_spans'] + [span for support in pool['near_prefix_spans'] for span in support]
        if any((fully_covered(c['context_spans'], occupied) for c in pool['candidates'])):
            raise ValueError('Fully covered candidate entered partial-overlap pool')
    from choruscoder_api_support import validate_selection
    chosen = validate_selection(record, config, tokenizer)
    if record['selected_segments'] != chosen:
        raise ValueError('Chunk selected supports mismatch')
    if s['scores'] != [float(max((b for a, b in c['spans']))) for c in s['candidates']]:
        raise ValueError('Chunk recency scores mismatch')
    queries = record['queries']
    if d['recall_quotas'] != method_recall_quotas(config['method'], len(queries), query_roles(queries)):
        raise ValueError('Chunk queries/quotas mismatch')
    if bool(record['fallback']) != (not record['selected_segments']):
        raise ValueError('Chunk fallback mismatch')
    if record.get('prefix') is not None and tokenizer is not None:
        if p.callable_ablation_config is not None:
            from choruscoder_callable_ablations import build_selection
            view, expected, expected_pool, expected_features = build_selection(record['prefix'], tokenizer, p, semantic, language=config.get('language', 'python'), path=record['path'])
        else:
            view, expected, expected_pool, expected_features = build_chunk_selection(record['prefix'], tokenizer, p.support_unit_config, semantic, policy, language=config.get('language', 'python'), path=record['path'], reserve_near=p.fused_support_config is None)
        if view != record['window'] or expected_features != features:
            raise ValueError('Chunk source/partition replay mismatch')
        if expected['target'] != s['target'] or expected['candidates'] != s['candidates']:
            raise ValueError('Chunk source candidates/T mismatch')
        for key, value in expected_pool.items():
            if pool.get(key) != value:
                raise ValueError('Chunk prefix candidate source mismatch: ' + key)
        if p.fused_support_config is not None and (not chosen):
            expected_text = source_query(record['prefix'], [suffix_span(record['prefix'], tokenizer, policy.query_tokens)])
            if s.get('fallback_query_text') != expected_text:
                raise ValueError('Fused fallback suffix mismatch')
        if any((len(tokenizer.encode(q['text'], add_special_tokens=False)) > policy.query_tokens for q in queries)):
            raise ValueError('Chunk query encoder budget exceeded')
    return features
