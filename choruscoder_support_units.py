"""Visible-source operation units with configurable target construction."""
from __future__ import annotations
from choruscoder import digest, make_parser, source_query
SIMPLE = {'expression_statement', 'return_statement', 'raise_statement', 'assert_statement', 'import_statement', 'import_from_statement', 'delete_statement', 'pass_statement', 'break_statement', 'continue_statement', 'global_statement', 'nonlocal_statement', 'comment', 'decorator'}
COMPOUND = {'function_definition', 'class_definition', 'if_statement', 'for_statement', 'while_statement', 'try_statement', 'with_statement', 'match_statement', 'elif_clause', 'else_clause', 'except_clause', 'finally_clause', 'case_clause'}
SCOPES = {'function_definition', 'class_definition'}

def _scope(node):
    scope = []
    parent = node.parent
    while parent is not None:
        if parent.type in SCOPES:
            scope.append(parent.start_byte)
        parent = parent.parent
    return tuple(reversed(scope))

def operation_segments(window, config, tokenizer, parser=None, target_mode='last_operation_unit', language='python'):
    """Pack short adjacent operations, never splitting a multiline statement.

    Three lines is a soft packing goal, six lines/96 tokens are default hard
    bounds for supports. An oversize cursor operation remains T rather than
    moving the target to an older statement. No statement is split.
    ERROR leaves may remain as explicitly degraded, isolated source units.
    """
    if target_mode != 'last_operation_unit':
        raise ValueError(f'Unsupported operation target policy: {target_mode}')
    result = {'target_span': [0, len(window['text'].encode('utf-8'))], 'diagnostics': {}}
    if not window['text']:
        return {'target': '', 'candidates': [], 'fallback': 'empty_window', 'diagnostics': {}}
    if 'target_span' not in result:
        return result
    text, target = (window['text'], result['target_span'])
    data = text.encode('utf-8')
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode('utf-8')))
    cut_end = offsets[min(3, len(lines))] if window.get('leading_truncated') else 0
    root = (parser or make_parser(language)).parse(data).root_node
    atoms = []
    source_nodes = {}
    counts = {'raw_operation_units': 0, 'excluded_cut_blocks': 0, 'excluded_target_blocks': 0, 'excluded_blank_blocks': 0, 'excluded_oversize_operations': 0, 'degraded_operation_units': 0, 'parse_has_error': root.has_error}

    def emit(node, last_row=None, isolated=False):
        first = node.start_point[0]
        last = node.end_point[0] - int(node.end_point[1] == 0) if last_row is None else last_row
        if last < first:
            return
        a, b = (offsets[first], offsets[min(last + 1, len(lines))])
        source_nodes[a, b] = node
        counts['raw_operation_units'] += 1

        def is_oversize():
            return last - first + 1 > config.max_operation_lines or len(tokenizer.encode(data[a:b].decode('utf-8'), add_special_tokens=False)) > config.max_operation_tokens
        oversize = True and is_oversize()
        if True and data[a:b].strip():
            parent = node.parent
            scope = _scope(node) + ((node.start_byte,) if node.type in SCOPES else ())
            atoms.append({'span': [a, b], 'scope': scope, 'block': ('isolated', node.start_byte) if isolated or oversize else parent.start_byte if parent else -1, 'line_start': first + 1, 'line_end': last + 1, 'degraded': isolated, 'oversize': oversize})
        elif a < cut_end:
            counts['excluded_cut_blocks'] += 1
        elif a < target[1] and b > target[0]:
            counts['excluded_target_blocks'] += 1
        elif not data[a:b].strip():
            counts['excluded_blank_blocks'] += 1
        elif is_oversize():
            counts['excluded_oversize_operations'] += 1
        else:
            parent = node.parent
            block = parent.start_byte if parent is not None else -1
            scope = _scope(node)
            if node.type in SCOPES:
                scope += (node.start_byte,)
            if isolated:
                block = ('isolated', node.start_byte)
            atoms.append({'span': [a, b], 'scope': scope, 'block': block, 'line_start': first + 1, 'line_end': last + 1, 'degraded': isolated})

    def visit(node):
        if node.type in SIMPLE:
            emit(node)
            return
        if node.type in COMPOUND:
            body = node.child_by_field_name('body') or next((c for c in node.named_children if c.type == 'block'), None)
            if body is not None:
                if body.start_point[0] == node.start_point[0]:
                    emit(node, isolated=True)
                    return
                emit(node, body.start_point[0] - 1)
        if node.type == 'ERROR' and (not any((c.type in SIMPLE | COMPOUND for c in node.named_children))):
            emit(node, isolated=True)
            return
        for child in node.named_children:
            visit(child)
    visit(root)
    end = max((a['span'][1] for a in atoms), default=0)
    if data[end:].strip():
        first = next((i for i, offset in enumerate(offsets[1:]) if offset > end))
        atoms.append({'span': [end, len(data)], 'scope': (), 'block': ('tail', end), 'line_start': first + 1, 'line_end': len(lines), 'degraded': True, 'oversize': len(lines) - first > config.max_operation_lines or len(tokenizer.encode(data[end:].decode('utf-8'), add_special_tokens=False)) > config.max_operation_tokens})
    candidates, pending = ([], None)

    def flush():
        nonlocal pending
        if pending is None:
            return
        spans = [pending['span']]
        candidates.append({'id': 'S' + digest(spans)[:12], 'spans': spans, 'context_spans': spans, 'text': source_query(text, spans), 'line_start': pending['line_start'], 'line_end': pending['line_end'], 'dependency_analysis': ['isolated_parse_fragment' if pending['degraded'] else 'short_operation_unit'], 'unit_kind': 'operation', 'degraded': pending['degraded']})
        counts['degraded_operation_units'] += int(pending['degraded'])
        pending = None
    for atom in sorted(atoms, key=lambda x: (x['span'][0], x['span'][1])):
        if pending and atom['span'][0] < pending['span'][1]:
            if atom['span'][1] <= pending['span'][1]:
                continue
            flush()
        if pending:
            start, end = (pending['span'][0], atom['span'][1])
            can_merge = pending['scope'] == atom['scope'] and pending['block'] == atom['block'] and (not pending['degraded']) and (not atom['degraded']) and (not data[pending['span'][1]:atom['span'][0]].strip()) and (atom['line_end'] - pending['line_start'] + 1 <= config.max_operation_lines) and (len(tokenizer.encode(data[start:end].decode('utf-8'), add_special_tokens=False)) <= config.max_operation_tokens)
            if not can_merge:
                flush()
        if pending is None:
            pending = dict(atom)
        else:
            pending['span'][1] = atom['span'][1]
            pending['line_end'] = atom['line_end']
        if pending['degraded'] or pending['line_end'] - pending['line_start'] + 1 >= config.operation_lines:
            flush()
    flush()
    partition = [{key: c[key] for key in ('spans', 'line_start', 'line_end', 'degraded')} for c in candidates]
    if candidates:
        last = candidates.pop()
        span = [last['spans'][0][0], len(data)]
        target_degraded = last['degraded']
    else:
        span, target_degraded = ([0, len(data)], bool(data.strip()))
    result = dict(result, target=source_query(text, [span]), target_span=span, target_body_spans=[span], target_context_spans=[span])
    supports = []
    for candidate in candidates:
        a, b = candidate['spans'][0]
        if a < cut_end:
            counts['excluded_cut_blocks'] += 1
        elif candidate['line_end'] - candidate['line_start'] + 1 > config.max_operation_lines or len(tokenizer.encode(data[a:b].decode('utf-8'), add_special_tokens=False)) > config.max_operation_tokens:
            if getattr(config, 'oversize_policy', 'exclude') == 'recursive':
                from benchmarking.choruscoder_cast import chunk_source
                from choruscoder_profiles import RepositoryChunkConfig
                pieces, detail = chunk_source(text, tokenizer, RepositoryChunkConfig(max_tokens=config.max_operation_tokens), root_node=source_nodes.get((a, b), root), byte_span=(a, b), max_lines=config.max_operation_lines)
                counts['split_oversize_operations'] = counts.get('split_oversize_operations', 0) + 1
                counts['support_oversize_leaf_count'] = counts.get('support_oversize_leaf_count', 0) + detail['oversize_leaf_count']
                for first, last, reason in pieces:
                    if not data[first:last].strip():
                        continue
                    spans = [[first, last]]
                    begin_line = next((i for i, offset in enumerate(offsets[1:]) if offset > first))
                    end_line = next((i for i, offset in enumerate(offsets[1:]) if offset >= last))
                    supports.append(dict(candidate, id='S' + digest(spans)[:12], spans=spans, context_spans=spans, text=source_query(text, spans), line_start=begin_line + 1, line_end=end_line + 1, source_operation_span=[a, b], degraded=bool(reason) or candidate['degraded'], degraded_reason=reason, dependency_analysis=['recursive_operation_fragment']))
            else:
                counts['excluded_oversize_operations'] += 1
        else:
            supports.append(candidate)
    candidates = supports
    target_first = next((i for i, offset in enumerate(offsets[1:]) if offset > span[0]))
    counts.update(excluded_target_blocks=int(bool(partition)), operation_partition=partition, target_line_start=target_first + 1, target_line_end=len(lines), target_degraded=target_degraded, target_oversize=len(lines) - target_first > config.max_operation_lines or len(tokenizer.encode(data[span[0]:].decode('utf-8'), add_special_tokens=False)) > config.max_operation_tokens, target_policy=target_mode)
    if getattr(config, 'oversize_policy', 'exclude') == 'recursive':
        counts.update(support_oversize_policy='recursive', split_oversize_operations=counts.get('split_oversize_operations', 0), support_oversize_leaf_count=counts.get('support_oversize_leaf_count', 0))
    return dict(result, candidates=candidates, diagnostics=dict(result['diagnostics'], **counts, operation_candidate_count=len(candidates)), fallback=None if candidates else 'no_operation_support_units')

def unit_summary(candidates, selected, queries, tokenizer):
    return {'single_candidates': sum((c.get('unit_kind', 'single') == 'single' for c in candidates)), 'pair_candidates': sum((c.get('unit_kind') == 'pair' for c in candidates)), 'operation_candidates': sum((c.get('unit_kind') == 'operation' for c in candidates)), 'selected_pairs': sum((c.get('unit_kind') == 'pair' for c in selected)), 'query_tokens': [len(tokenizer.encode(q['text'], add_special_tokens=False)) for q in queries], 'support_units': [{'id': c['id'], 'candidate_id': c.get('candidate_id', c['id']), 'kind': c.get('unit_kind', 'single'), 'members': c.get('member_ids', [c.get('candidate_id', c['id'])])} for c in selected]}
