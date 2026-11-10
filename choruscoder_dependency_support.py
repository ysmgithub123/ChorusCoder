"""Cursor statements and receiver bindings extracted from visible Python source."""
import io
import itertools
import tokenize
from choruscoder import make_parser
from choruscoder_far import PrefixFacts, path
from choruscoder_scoring import _parents, _scope, CONDITIONAL

def cursor_statement(prefix):
    """Tokenize incomplete Python without completing or adding source tokens."""
    lines = prefix.splitlines(keepends=True)
    if not lines or prefix.endswith(('\n', '\r')) or (not lines[-1].strip()) or lines[-1].lstrip().startswith('#'):
        return {'text': '', 'spans': [], 'status': 'empty_cursor_line'}
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode('utf-8')))

    def byte(position):
        row, col = position
        return offsets[min(row - 1, len(lines))] + len((lines[row - 1][:col] if row <= len(lines) else '').encode('utf-8'))
    tokens = []
    start = None
    depth = 0
    partial_error = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(prefix).readline):
            if tok.type in {tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}:
                continue
            if tok.type == tokenize.NEWLINE or (tok.type == tokenize.OP and tok.string == ';' and (not depth)):
                start = None
                tokens = []
                continue
            if tok.type in {tokenize.NL, tokenize.COMMENT}:
                continue
            if tok.type == tokenize.ERRORTOKEN and tok.string.isspace():
                continue
            if start is None:
                start = byte(tok.start)
            tokens.append({'type': tok.type, 'text': tok.string, 'start': byte(tok.start), 'end': byte(tok.end)})
            if tok.type == tokenize.OP:
                depth += int(tok.string in '([{') - int(tok.string in ')]}')
    except (tokenize.TokenError, IndentationError, SyntaxError) as error:
        partial_error = str(error)
    raw = prefix.encode('utf-8')
    if start is None and partial_error is None:
        try:
            all_tokens = list(tokenize.generate_tokens(io.StringIO(prefix).readline))
            relevant = [t for t in all_tokens if t.type not in {tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.ENDMARKER, tokenize.ENCODING}]
            if relevant:
                last = relevant[-1]
                boundary = 0
                for t in all_tokens:
                    if t.start >= last.start:
                        break
                    if t.type == tokenize.NEWLINE or (t.type == tokenize.OP and t.string == ';'):
                        boundary = byte(t.end)
                chosen = [t for t in relevant if byte(t.start) >= boundary]
                if chosen:
                    start = byte(chosen[0].start)
                    tokens = [{'type': t.type, 'text': t.string, 'start': byte(t.start), 'end': byte(t.end)} for t in chosen]
        except (tokenize.TokenError, IndentationError, SyntaxError):
            pass
    if start is None:
        first = offsets[-2]
        text = raw[first:].decode('utf-8')
        return {'text': text if not text.lstrip().startswith('#') else '', 'spans': [[first, len(raw)]], 'status': 'physical_line_fallback', 'error': partial_error}
    if any((t['type'] == tokenize.ERRORTOKEN or (t['type'] == tokenize.OP and t['text'] not in tokenize.EXACT_TOKEN_TYPES) for t in tokens)) or depth < 0:
        first = offsets[-2]
        return {'text': raw[first:].decode('utf-8'), 'spans': [[first, len(raw)]], 'status': 'physical_line_fallback', 'tokens': [], 'error': partial_error}
    return {'text': raw[start:].decode('utf-8'), 'spans': [[start, len(raw)]], 'status': 'logical_statement', 'tokens': tokens, 'error': partial_error}

def _statement(index, entry):
    n = index.root.descendant_for_byte_range(entry['start'], entry['start'] + 1)
    if entry.get('parameter'):
        owner = next((p for p in itertools.chain([n], _parents(n)) if p.type == 'function_definition'), None)
        if owner is None:
            return None
        body = owner.child_by_field_name('body')
        end = body.start_byte - body.start_point[1] if body and body.start_point[0] > owner.start_point[0] else None
        return [owner.start_byte, end] if end else None
    return entry['span']

class ReceiverBindings:
    """Small, conservative extension for static attribute writes and aliases."""

    def __init__(self, prefix, parser=None):
        self.index = PrefixFacts(prefix, parser or make_parser('python'))
        for n in self.index.nodes:
            if n.type != 'parameters':
                continue
            for c in n.named_children:
                if c.type not in {'typed_parameter', 'typed_default_parameter'}:
                    continue
                ident = c.child_by_field_name('name') or next((k for k in c.named_children if k.type == 'identifier'), None)
                if ident is None:
                    continue
                name = self.index.text(ident)
                if any((e['start'] == c.start_byte for e in self.index.writes.get((_scope(c), name), []))):
                    continue
                annotation = path(c.child_by_field_name('type'), self.index.raw)
                self.index.add(name, c, c.has_error, origin=['annotation', annotation] if annotation else None, annotation=annotation, available=n.end_byte, parameter=True)
        for entries in self.index.writes.values():
            entries.sort(key=lambda e: e['available'])
        self.attributes = []
        for n in self.index.nodes:
            if n.type not in {'assignment', 'augmented_assignment', 'delete_statement'}:
                continue
            left = n.child_by_field_name('left')
            if n.type == 'delete_statement':
                left = next((c for c in n.named_children if c.type == 'attribute'), None)
            name = path(left, self.index.raw)
            owner = next((p for p in _parents(n) if p.type == 'class_definition'), None)
            method = next((p for p in _parents(n) if p.type == 'function_definition'), None)
            if name and '.' not in name and owner and (method is None) and n.child_by_field_name('type'):
                name = 'self.' + name
            if not name or '.' not in name:
                continue
            head = name.split('.')[0]
            root_node = next((c for c in self.index.nodes if c.type == 'identifier' and left.start_byte <= c.start_byte < c.end_byte <= left.end_byte and (self.index.text(c) == head)), None)
            bound = self.index.binding(head, root_node) if root_node else None
            scope = _scope(n)
            rhs = n.child_by_field_name('right')
            annotation = path(n.child_by_field_name('type'), self.index.raw)
            producer = path(rhs.child_by_field_name('function'), self.index.raw) if rhs and rhs.type == 'call' and (not rhs.has_error) else None
            self.attributes.append({'name': name, 'scope': scope, 'start': n.start_byte, 'available': n.end_byte, 'span': [n.start_byte, n.end_byte], 'parameter': False, 'annotation': annotation, 'origin': ['producer', producer] if producer else ['annotation', annotation] if annotation else None, 'unsafe': n.has_error or n.type != 'assignment' or any((p.type in CONDITIONAL | {'ERROR'} for p in _parents(n))), 'root_key': self.key(bound), 'class': owner.start_byte if owner else None, 'method': self.index.text(method.child_by_field_name('name')) if method else None})

    @staticmethod
    def key(bound):
        return (tuple(bound['scope']), bound['name'], bound['start']) if bound else None

    def binding(self, name, node):
        if '.' not in name:
            return self.index.binding(name, node)
        head = name.split('.')[0]
        root = self.index.binding(head, node)
        if not root:
            return None
        scope = _scope(node)
        owner = next((p for p in itertools.chain([node], _parents(node)) if p.type == 'class_definition'), None)
        current = [a for a in self.attributes if a['name'] == name and a['scope'] == scope and (a['root_key'] == self.key(root)) and (a['available'] <= node.start_byte)]
        if current:
            entry = max(current, key=lambda a: a['available'])
            return None if entry['unsafe'] else entry
        if head not in {'self', 'cls'} or not root.get('parameter') or owner is None:
            return None
        earlier = [a for a in self.attributes if a['name'] == name and a['class'] == owner.start_byte and (a['available'] <= node.start_byte)]
        all_writes = [a for a in self.attributes if a['name'] == name and a['class'] == owner.start_byte]
        init = [a for a in earlier if a['method'] in {'__init__', None} and (not a['unsafe'])]
        if len(init) != 1 or any((a['method'] not in {'__init__', None} or a['unsafe'] for a in all_writes)):
            return None
        return init[0]

    def rhs_name(self, entry):
        n = self.index.root.descendant_for_byte_range(entry['start'], entry['start'] + 1)
        n = next((p for p in itertools.chain([n], _parents(n)) if p.type == 'assignment'), None)
        rhs = n.child_by_field_name('right') if n else None
        return path(rhs, self.index.raw)

    def resolve(self, receiver):
        index = self.index
        name = receiver['name']
        node = index.root.descendant_for_byte_range(receiver['span'][0], receiver['span'][0] + 1)
        chain = []
        origin = None
        reason = 'unresolved_binding'
        mode = 'object'
        seen = set()
        for hop in range(3):
            entry = self.binding(name, node)
            suffix = []
            if entry is None and '.' in name:
                head, *suffix = name.split('.')
                imported = index.binding(head, node)
                if imported and imported['origin'] and (imported['origin'][0] == 'import'):
                    entry = imported
                else:
                    break
            if entry is None:
                break
            key = self.key(entry)
            if key in seen:
                reason = 'dependency_cycle'
                break
            seen.add(key)
            span = _statement(index, entry)
            if span:
                chain.append({'name': name, 'hop': hop, 'span': span, 'origin': entry['origin']})
            value = entry['origin']
            if value and value[0] == 'import':
                _, level, module, symbol = value
                if module is None:
                    imported_tail = symbol.split('.')[1:]
                    if entry['name'] != symbol.split('.')[0]:
                        module = symbol
                        entity = suffix
                    elif suffix[:len(imported_tail)] == imported_tail:
                        module = symbol
                        entity = suffix[len(imported_tail):]
                    else:
                        module = entry['name']
                        entity = suffix
                else:
                    entity = [symbol] + suffix
                origin = {'level': level, 'module': module, 'entity': entity, 'mode': mode}
                reason = 'resolved_import'
                break
            successor = None
            if entry.get('annotation'):
                successor = entry['annotation']
                mode = 'type'
            elif value and value[0] == 'producer':
                successor = value[1]
                mode = 'constructor'
            else:
                successor = self.rhs_name(entry)
            if not successor:
                reason = 'definition_without_static_origin'
                break
            if hop == 2:
                reason = 'hop_limit'
                break
            name = successor
            node = index.root.descendant_for_byte_range(entry['start'], entry['start'] + 1)
        return {'receiver': receiver, 'chain': chain, 'api_origin': origin, 'resolution': reason}
