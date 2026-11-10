"""Official top-1 confidence selection with lossless, token-bounded source chunks.

Upstream's splitlines/newline serialization and stable confidence ranking are
preserved. Only oversized three-line groups are subdivided. Source coverage is
recorded separately from that upstream newline normalization.
"""
from dataclasses import asdict
from functools import lru_cache
import math
import time

from choruscoder import atomic_json, digest, load_json
from choruscoder_prefix_query import prefix_query_view


@lru_cache(maxsize=4)
def tokenizer_fingerprint(tokenizer):
    from benchmarking.choruscoder_cast import tokenizer_identity
    return digest(tokenizer_identity(tokenizer))


def full_prefix_view(prefix, tokenizer):
    return {'text':prefix, 'line_offset':0, 'start_token':0,
        'tokens':len(tokenizer.encode(prefix, add_special_tokens=False)),
        'leading_truncated':False, 'prefix_hash':digest(prefix)}


def source_groups(prefix, lines):
    groups, offset = [], 0
    raw_lines = prefix.splitlines(keepends=True)
    for start in range(0, len(raw_lines), lines):
        raw = ''.join(raw_lines[start:start+lines])
        end = offset + len(raw.encode('utf-8'))
        groups.append({'text':'\n'.join(raw.splitlines()), 'source_start_byte':offset,
            'source_end_byte':end, 'line_start':start+1,
            'line_end':min(start+lines, len(raw_lines))})
        offset = end
    return groups


def _parts(text, tokenizer, policy, target=None):
    """Split only when needed, from the tail, without assuming BPE monotonicity."""
    def fits(piece):
        return (len(tokenizer.encode(piece, add_special_tokens=False)) <= policy.chunk_tokens and
            (target is None or len(tokenizer.encode(piece+'\n'+target,
                add_special_tokens=False)) <= policy.query_tokens))
    if fits(text):
        return [(0, len(text))]
    parts, end = [], len(text)
    while end:
        remaining = text[:end]
        piece = prefix_query_view(remaining, tokenizer, policy.chunk_tokens)['text']
        if not piece:
            raise ValueError('A Unicode character cannot fit the CodeRAG chunk budget')
        # Joining can alter BPE boundaries: charge the exact S+T, not a sum.
        if not fits(piece):
            piece = next((piece[i:] for i in range(1, len(piece)) if fits(piece[i:])), None)
            if piece is None:
                raise ValueError('A Unicode character plus T cannot fit the CodeRAG query budget')
        start = end-len(piece)
        parts.append((start,end))
        end = start
    return list(reversed(parts))


def build_layout(prefix, tokenizer, policy):
    groups = source_groups(prefix, policy.lines_per_chunk)
    chunks = []
    for group_id, group in enumerate(groups):
        for start, end in _parts(group['text'], tokenizer, policy):
            chunks.append({'group':group_id, 'start_char':start, 'end_char':end,
                'text':group['text'][start:end]})
    if chunks:
        target = chunks[-1]
        supports = []
        for chunk in chunks[:-1]:
            for start, end in _parts(chunk['text'], tokenizer, policy, target['text']):
                supports.append(dict(chunk, start_char=chunk['start_char']+start,
                    end_char=chunk['start_char']+end, text=chunk['text'][start:end]))
        chunks = supports + [target]
    for i, chunk in enumerate(chunks):
        chunk['id'] = 'T' if i == len(chunks)-1 else f'S{i+1}'
        chunk['tokens'] = len(tokenizer.encode(chunk['text'], add_special_tokens=False))
    return groups, chunks


def select(prefix, tokenizer, policy, probe_factory, *, exclude_probe_loading=False):
    started = time.perf_counter()
    window = full_prefix_view(prefix, tokenizer)
    groups, chunks = build_layout(prefix, tokenizer, policy)
    fallback = len(chunks) <= 2 and window['tokens'] <= policy.query_tokens
    target = chunks[-1] if chunks else None
    prompts = [] if fallback else [c['text']+'\n'+target['text'] for c in chunks[:-1]]
    counts = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    if any(n > policy.query_tokens for n in counts):
        raise ValueError('CodeRAG candidate exceeds its query token budget')
    parsing_seconds = time.perf_counter()-started
    setup_seconds = 0.0
    probe = None
    if prompts and exclude_probe_loading:
        setup_started = time.perf_counter()
        probe = probe_factory()
        setup_seconds = time.perf_counter()-setup_started
    started = time.perf_counter()
    scores = list((probe if probe is not None else probe_factory())(prompts)) if prompts else []
    if len(scores) != len(prompts) or any(isinstance(s,bool) or not isinstance(s,(int,float)) or
            not math.isfinite(s) for s in scores):
        raise ValueError('Invalid CodeRAG confidence scores')
    scores = [float(s) for s in scores]
    probe_seconds = time.perf_counter()-started if prompts else 0.0
    chosen = [] if fallback else [chunks[max(range(len(scores)), key=scores.__getitem__)]]
    query = prefix if fallback else chosen[0]['text']+'\n'+target['text']
    query_tokens = len(tokenizer.encode(query, add_special_tokens=False))
    if query_tokens > policy.query_tokens:
        raise ValueError('CodeRAG final query exceeds its query token budget')
    result = {'window':window, 'policy':asdict(policy), 'groups':groups, 'chunks':chunks,
        'scores':scores, 'candidate_query_tokens':counts, 'selected_segments':chosen,
        'queries':[{'id':'Q0','text':query,'segment_ids':[] if fallback else [chosen[0]['id'],'T']}],
        'query_tokens':[query_tokens], 'fallback':'official_short_prefix' if fallback else None,
        'eligible_supports':max(0,len(chunks)-1), 'probe_seconds':probe_seconds,
        'parse_seconds':parsing_seconds}
    if exclude_probe_loading:
        result['probe_model_setup_seconds'] = setup_seconds
    result['checksum'] = selection_checksum(result)
    return result


def selection_checksum(result):
    return digest({k:v for k,v in result.items() if k not in ('checksum','probe_cache_source')})


def validate_selection(result, prefix, policy, tokenizer=None):
    if (result.get('checksum') != selection_checksum(result) or result.get('policy') != asdict(policy) or
            result['window']['text'] != prefix or result['window'].get('prefix_hash') != digest(prefix) or
            result['window']['line_offset'] != 0 or result['window']['start_token'] != 0 or
            result['window']['leading_truncated'] is not False):
        raise ValueError('CodeRAG selection source/configuration/checksum mismatch')
    groups = source_groups(prefix,policy.lines_per_chunk)
    if groups != result['groups']:
        raise ValueError('CodeRAG groups differ from visible source')
    chunks = result['chunks']
    positions = [0]*len(groups)
    previous = -1
    seen_groups = set()
    for i,c in enumerate(chunks):
        g = c['group']
        if (type(g) is not int or not 0 <= g < len(groups) or g < previous or
                type(c['start_char']) is not int or type(c['end_char']) is not int or
                c['start_char'] != positions[g] or not c['start_char'] <= c['end_char'] <= len(groups[g]['text']) or
                c['text'] != groups[g]['text'][c['start_char']:c['end_char']] or
                c['id'] != ('T' if i == len(chunks)-1 else f'S{i+1}') or
                type(c['tokens']) is not int or not 0 <= c['tokens'] <= policy.chunk_tokens):
            raise ValueError('CodeRAG chunk source coverage or token count mismatch')
        positions[g] = c['end_char']; previous = g; seen_groups.add(g)
        if tokenizer is not None and c['tokens'] != len(tokenizer.encode(c['text'],add_special_tokens=False)):
            raise ValueError('CodeRAG chunk token count mismatch')
    if positions != [len(g['text']) for g in groups] or seen_groups != set(range(len(groups))):
        raise ValueError('CodeRAG chunks omit source content')
    tokens = result['window']['tokens']
    if type(tokens) is not int or tokens < 0 or tokenizer is not None and tokens != len(tokenizer.encode(prefix,add_special_tokens=False)):
        raise ValueError('CodeRAG prefix token count mismatch')
    fallback = len(chunks) <= 2 and tokens <= policy.query_tokens
    scores = result['scores']
    if (len(scores) != (0 if fallback else len(chunks)-1) or
            any(isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) for s in scores)):
        raise ValueError('CodeRAG scores differ from the candidate pool')
    chosen = [] if fallback else [chunks[max(range(len(scores)),key=scores.__getitem__)]]
    query = prefix if fallback else chosen[0]['text']+'\n'+chunks[-1]['text']
    expected = [{'id':'Q0','text':query,'segment_ids':[] if fallback else [chosen[0]['id'],'T']}]
    if (result['selected_segments'] != chosen or result['queries'] != expected or
            result['fallback'] != ('official_short_prefix' if fallback else None) or
            result['eligible_supports'] != max(0,len(chunks)-1) or
            len(result['candidate_query_tokens']) != len(scores) or
            len(result['query_tokens']) != 1 or
            any(type(n) is not int or not 0 <= n <= policy.query_tokens for n in
                result['candidate_query_tokens']+result['query_tokens'])):
        raise ValueError('CodeRAG selected query or token budget mismatch')
    for name in ('probe_seconds','parse_seconds'):
        value = result[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid CodeRAG timing')
    if 'probe_model_setup_seconds' in result:
        for name in ('probe_model_setup_seconds', 'selection_seconds'):
            if name not in result:
                continue  # select() is also usable before the record timer ends.
            value = result[name]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid online CodeRAG timing')
    if fallback and result['probe_seconds'] != 0:
        raise ValueError('Short CodeRAG prefixes must not invoke the probe')
    if fallback and result.get('probe_model_setup_seconds', 0) != 0:
        raise ValueError('Short CodeRAG prefixes must not load the probe')
    if tokenizer is not None:
        counts = [] if fallback else [len(tokenizer.encode(c['text']+'\n'+chunks[-1]['text'],
            add_special_tokens=False)) for c in chunks[:-1]]
        if (counts != result['candidate_query_tokens'] or
                result['query_tokens'] != [len(tokenizer.encode(query,add_special_tokens=False))]):
            raise ValueError('CodeRAG query token count mismatch')


def selection_record(request, config, cache, tokenizer, probe_factory):
    from choruscoder_profiles import PROFILES
    from benchmarking.retrieval_online import online
    started = time.perf_counter()
    is_online = online(config)
    policy = PROFILES[config['method']].coderag_query_config
    identity = {'prefix':digest(request['prefix']), 'policy':asdict(policy),
        'tokenizer':tokenizer_fingerprint(tokenizer), 'probe_model':config['probe_model'],
        'probe_tokens':config['probe_tokens'], 'probe_score':config['probe_score'],
        'seed':config['seed'], 'upstream':config['coderag_commit'],
        'runtime':config.get('runtime_versions'), 'sources':config['source_hashes']}
    key = digest(identity)
    location = cache/'coderag_selection'/f'{key}.json'
    if not is_online and location.exists():
        result = load_json(location)
        if result.get('cache_identity') != identity:
            raise ValueError('CodeRAG selection cache identity mismatch')
        validate_selection(result,request['prefix'],policy,tokenizer)
        result['probe_cache_source'] = str(location)
        return result
    result = select(request['prefix'],tokenizer,policy,probe_factory,
                    exclude_probe_loading=is_online)
    result['cache_identity'] = identity
    result['checksum'] = selection_checksum(result)
    validate_selection(result,request['prefix'],policy)
    if is_online:
        result['selection_seconds'] = max(0.0, time.perf_counter()-started-result['probe_model_setup_seconds'])
        result['checksum'] = selection_checksum(result)
    else:
        atomic_json(location,result)
    return result


def validate_record(record, config, tokenizer=None):
    from choruscoder_profiles import PROFILES
    policy = PROFILES[config['method']].coderag_query_config
    if policy is None or config.get('coderag_query_config') != asdict(policy):
        raise ValueError('CodeRAG run configuration mismatch')
    selection = record['diagnostics']['coderag_selection']
    identity = selection.get('cache_identity',{})
    expected = {'prefix':digest(record['prefix']), 'policy':asdict(policy),
        'probe_model':config['probe_model'], 'probe_tokens':config['probe_tokens'],
        'probe_score':config['probe_score'], 'seed':config['seed'], 'upstream':config['coderag_commit'],
        'runtime':config.get('runtime_versions'), 'sources':config['source_hashes']}
    if (any(identity.get(k) != v for k,v in expected.items()) or not identity.get('tokenizer') or
            tokenizer is not None and identity['tokenizer'] != tokenizer_fingerprint(tokenizer)):
        raise ValueError('Prepared CodeRAG selection identity mismatch')
    validate_selection(selection,record['prefix'],policy,tokenizer)
    for field in ('window','queries','selected_segments','eligible_supports','fallback','probe_seconds'):
        if record[field] != selection[field]:
            raise ValueError('Prepared CodeRAG record differs from its selection')

