"""ChorusCoder query construction, evidence fusion and prompt assembly."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from choruscoder_profiles import PROFILES, QueryContextConfig, recall_config
VERSION = 'choruscoder_v1'
CODERAG_COMMIT = 'f50251c86839bc50dbf8b928ae4f3eeacd75e4c5'
STATEMENTS = {'expression_statement', 'return_statement', 'raise_statement', 'assert_statement', 'import_statement', 'import_from_statement', 'delete_statement', 'pass_statement', 'break_statement', 'continue_statement', 'global_statement', 'nonlocal_statement', 'if_statement', 'for_statement', 'while_statement', 'try_statement', 'with_statement', 'match_statement', 'function_definition', 'class_definition', 'decorated_definition'}
SCOPES = {'function_definition', 'class_definition'}

def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def make_parser(language_name='python'):
    from choruscoder_languages import make_parser as shared_parser
    return shared_parser(language_name)

def window_view(prefix, tokenizer, size=256):
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    text = tokenizer.decode(ids[-size:])
    return {'text': text, 'line_offset': prefix.count('\n') - text.count('\n'), 'start_token': max(0, len(ids) - size), 'tokens': min(size, len(ids)), 'leading_truncated': len(ids) > size}

def source_query(text, spans):
    """Serialize a union of visible source spans; never duplicate scope headers."""
    data = text.encode('utf-8')
    merged = []
    for start, end in sorted(spans):
        if not 0 <= start <= end <= len(data):
            raise ValueError('Query source span escaped the window')
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return '\n'.join((data[start:end].decode('utf-8') for start, end in merged))

def select_segments(structure, scores, maximum):
    if len(scores) != len(structure['candidates']):
        raise ValueError('One probe score is required for each candidate')
    ranked = sorted((dict(candidate, score=float(score)) for candidate, score in zip(structure['candidates'], scores)), key=lambda c: (-c['score'], c['spans'][0][0]))
    return [dict(c, candidate_id=c['id'], id=f'S{i + 1}') for i, c in enumerate(ranked[:maximum])]

def build_queries(method, window, structure, selected, prefix=None):
    mode = PROFILES[method].query_mode
    if mode == 'single' or not selected or structure.get('fallback'):
        return [{'id': 'Q0', 'text': window['text'], 'segment_ids': []}]
    queries = []
    for i, c in enumerate(selected):
        if c.get('source_space') == 'prefix':
            spans = c['context_spans'] + structure['prefix_target_context_spans']
            text = source_query(prefix, spans) if prefix is not None else c['query_text']
            if text != c.get('query_text', c['text'] + '\n' + structure['target']):
                raise ValueError('Query source parts disagree')
        else:
            text = source_query(window['text'], c['context_spans'] + structure['target_context_spans'])
        queries.append({'id': f'Q{i + 1}', 'text': text, 'segment_ids': [c['id']]})
    return queries

def recall_quotas(count, total=20):
    if not 1 <= count <= min(8, total):
        raise ValueError('Query count must be between one and eight with a positive quota each')
    return [total // count + int(i < total % count) for i in range(count)]

def query_roles(queries):
    return [q.get('role', 'support' if q['segment_ids'] else 'fallback') for q in queries]

def method_recall_quotas(method, count, roles=None):
    """Allocate the paper recall budget across actual queries."""
    config = recall_config(method)
    policy = PROFILES[method].support_ablation_config
    if policy is not None and policy.selection == 'prefix_all':
        if type(count) is not int or not 1 <= count <= PROFILES[method].max_segments:
            raise ValueError('Invalid prefix-all query count')
        return [config.total // count + int(i < config.total % count) for i in range(count)]
    if type(count) is not int or not 1 <= count <= min(8, config.total):
        raise ValueError('Invalid actual query count for recall policy')
    if config.quotas_by_query_count:
        if count > len(config.quotas_by_query_count):
            raise ValueError('Actual query count exceeds the recall quota table')
        return list(config.quotas_by_query_count[count - 1])
    return recall_quotas(count, config.total)

def round_robin_evidence(queries, ranked, maximum=4):
    """Deduplicate by identity and preserve all recall associations, even later ones."""
    associations = {}
    for query, results in zip(queries, ranked):
        for block in results:
            associations.setdefault(block['id'], [])
            if query['id'] not in associations[block['id']]:
                associations[block['id']].append(query['id'])
    ordered, seen = ([], set())
    for rank in range(max((len(items) for items in ranked), default=0)):
        for items in ranked:
            if rank < len(items) and items[rank]['id'] not in seen:
                item = items[rank]
                seen.add(item['id'])
                ordered.append(dict(item, query_ids=associations[item['id']]))
    return ordered

def query_evidence_coverage(queries, ranked, evidence):
    """Count coverage after deduplication and packing; a block may cover many queries."""
    if len(queries) != len(ranked):
        raise ValueError('One retrieval result list is required per query')
    retained = {block['id'] for block in evidence}
    coverage = []
    for query, results in zip(queries, ranked):
        ids = list(dict.fromkeys((block['id'] for block in results)))
        kept = [key for key in ids if key in retained]
        coverage.append({'query_id': query['id'], 'segment_ids': query['segment_ids'], 'retrieved_count': len(ids), 'retained_count': len(kept), 'retained_ids': kept, 'covered': bool(kept)})
    return coverage

def evidence_text(block, language='python'):
    if block.get('official'):
        return block['text']
    comment = '#' if language == 'python' else '//'
    location = f"{comment} evidence {block.get('display_id', block['id'])} source: {block['path']} tokens: {block.get('start_token', '')}-{block.get('end_token', '')}"
    return location + '\n' + '\n'.join((comment + ' ' + line for line in block['text'].splitlines()))

def render_evidence(blocks, language='python'):
    return '\n\n'.join((evidence_text(dict(block, display_id=f'E{i + 1}'), language) for i, block in enumerate(blocks)))

def pack_evidence(blocks, tokenizer, budget=896, maximum=4, language='python'):
    chosen = []
    for block in blocks:
        if maximum is not None and len(chosen) >= maximum:
            break
        proposed = render_evidence(chosen + [block], language)
        if len(tokenizer.encode(proposed, add_special_tokens=False)) <= budget:
            chosen.append(block)
    return chosen

def construct_prompt(prepared, tokenizer, usage=None):
    if prepared.get('rlcoder_config') is not None:
        from benchmarking.choruscoder_inputs import build_input
        payload = build_input(prepared, tokenizer)
        if usage is not None:
            usage.update(payload['usage'])
        return payload['prompt_text']
    language = prepared.get('language', 'python')
    evidence = render_evidence(prepared['evidence'], language)
    if prepared.get('official') and evidence:
        evidence = "'''\n" + evidence + "\n'''"
    comment = '#' if language == 'python' else '//'
    header = f"\n\n{comment} file path: {prepared['path']}\n\n"
    if len(tokenizer.encode(header, add_special_tokens=False)) > 128:
        raise ValueError('Path header exceeds 128 tokens')
    if prepared.get('context_profile') is not None:
        context = QueryContextConfig(**prepared['context_profile'])
        cross = evidence
        cross_tokens = len(tokenizer.encode(cross, add_special_tokens=False))
        evidence_tokens = len(tokenizer.encode(evidence, add_special_tokens=False))
        if cross_tokens > context.crossfile_tokens or evidence_tokens > context.evidence_body_tokens:
            raise ValueError('Prepared evidence exceeds its configured context budget')
        leading = cross + header
        available = context.prompt_tokens - len(tokenizer.encode(leading, add_special_tokens=False))
        if available < 0:
            raise ValueError('Evidence and path header exceed the total prompt budget')
        if context.infile_tokens is not None:
            available = min(available, context.infile_tokens)
        prefix_ids = tokenizer.encode(prepared['prefix'], add_special_tokens=False)
        original_prefix_tokens = len(prefix_ids)
        prefix_ids = prefix_ids[-available:] if available else []
        prefix_text = tokenizer.decode(prefix_ids)
        prompt = leading + prefix_text
        while True:
            excess = len(tokenizer.encode(prompt, add_special_tokens=False)) - context.prompt_tokens
            if context.infile_tokens is not None:
                excess = max(excess, len(tokenizer.encode(prefix_text, add_special_tokens=False)) - context.infile_tokens)
            if excess <= 0:
                break
            if not prefix_ids:
                raise ValueError('Prepared prompt exceeds its configured context budget')
            prefix_ids = prefix_ids[max(1, excess):]
            prefix_text = tokenizer.decode(prefix_ids)
            prompt = leading + prefix_text
        if usage is not None:
            usage.update(evidence_tokens=evidence_tokens, crossfile_tokens=cross_tokens, prefix_original_tokens=original_prefix_tokens, prefix_retained_tokens=len(tokenizer.encode(prefix_text, add_special_tokens=False)), prompt_tokens=len(tokenizer.encode(prompt, add_special_tokens=False)))
        return prompt
    prefix = tokenizer.decode(tokenizer.encode(prepared['prefix'], add_special_tokens=False)[-2944:])
    prompt = evidence + header + prefix
    cross_tokens = len(tokenizer.encode(evidence, add_special_tokens=False))
    if cross_tokens > 1024 or len(tokenizer.encode(prompt, add_special_tokens=False)) > 4096:
        raise ValueError('Prepared prompt exceeds experiment budget')
    return prompt

def validate_coderag(root):
    root = Path(root)
    manifest = load_json(root / 'SOURCE_MANIFEST.json')
    if manifest['commit'] != CODERAG_COMMIT:
        raise ValueError('Wrong pinned CodeRAG revision')
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Modified CodeRAG source: {name}')
    return manifest
