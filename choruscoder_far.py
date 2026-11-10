"""Visible-prefix syntax facts, source alignment and cached text encoding."""
from bisect import bisect_right
import ast
from choruscoder_scoring import _walk, _parents, _scope, CONDITIONAL
STOP_WORDS = frozenset('self cls get set run value data args kwargs obj item result return def class test init true false none producer import annotation definition'.split())

def path(node, raw):
    if node is None or node.is_missing:
        return None
    if node.type == 'identifier':
        return raw[node.start_byte:node.end_byte].decode('utf-8')
    if node.type == 'attribute':
        base = path(node.child_by_field_name('object'), raw)
        member = node.child_by_field_name('attribute')
        return base + '.' + path(member, raw) if base and member and path(member, raw) else None
    if node.type == 'type' and len(node.named_children) == 1:
        return path(node.named_children[0], raw)
    return None

def alignment(prefix, window, target_spans):
    """Use only byte-exact common suffixes; never guess a normalized target."""
    raw, view = (prefix.encode('utf-8'), window['text'].encode('utf-8'))
    common = 0
    while common < min(len(raw), len(view)) and raw[-common - 1] == view[-common - 1]:
        common += 1
    shift = len(raw) - len(view)
    if shift < 0 or any((a < len(view) - common for a, b in target_spans)):
        return {'status': 'unaligned_target', 'boundary': 0, 'target_spans': []}
    boundary = max(0, min(shift, len(raw) - common))
    if common != len(view):
        first_line = len(view.split(b'\n', 1)[0]) + int(b'\n' in view)
        if not window.get('leading_truncated') or len(view) - common > first_line:
            return {'status': 'unaligned_window', 'boundary': 0, 'target_spans': []}
        boundary = min(boundary, raw.rfind(b'\n', 0, len(raw) - common) + 1)
    while boundary > 0 and boundary < len(raw) and (raw[boundary] & 192 == 128):
        boundary -= 1
    return {'status': 'exact' if common == len(view) else 'exact_target_suffix', 'boundary': boundary, 'window_shift': shift, 'target_spans': [[a + shift, b + shift] for a, b in target_spans]}

class PrefixFacts:
    """Conservative lexical reaching bindings; conditional/unknown writes stop lookup."""

    def __init__(self, text, parser):
        self.raw = text.encode('utf-8')
        self.root = parser.parse(self.raw).root_node
        self.nodes = list(_walk(self.root))
        self.scope_types = {n.start_byte: n.type for n in self.nodes if n.type in {'function_definition', 'class_definition'}}
        self.writes = {}
        self.declared = set()
        for n in self.nodes:
            if n.type in {'global_statement', 'nonlocal_statement'}:
                self.declared.update(((_scope(n), self.text(c)) for c in n.named_children if c.type == 'identifier'))
        for n in self.nodes:
            unsafe = n.has_error or any((p.type in CONDITIONAL | {'ERROR'} for p in _parents(n)))
            if n.type in {'import_statement', 'import_from_statement'} and (not n.has_error):
                try:
                    imp = ast.parse(self.text(n)).body[0]
                except (SyntaxError, ValueError):
                    continue
                for alias in imp.names:
                    if alias.name == '*':
                        continue
                    name = alias.asname or (alias.name.split('.')[0] if isinstance(imp, ast.Import) else alias.name)
                    self.add(name, n, unsafe, origin=['import', getattr(imp, 'level', 0), getattr(imp, 'module', None), alias.name])
            elif n.type in {'assignment', 'augmented_assignment', 'named_expression', 'for_statement', 'for_in_clause'}:
                left = n.child_by_field_name('left') or n.child_by_field_name('name')
                if left is None:
                    continue
                for ident in _walk(left):
                    if ident.type != 'identifier' or any((p.type in {'attribute', 'subscript'} for p in _parents(ident.parent) if p.start_byte >= left.start_byte and p.end_byte <= left.end_byte)):
                        continue
                    rhs = n.child_by_field_name('right')
                    annotation = path(n.child_by_field_name('type'), self.raw)
                    producer = path(rhs.child_by_field_name('function'), self.raw) if rhs and rhs.type == 'call' and (not rhs.has_error) else None
                    self.add(self.text(ident), n, unsafe or n.type != 'assignment', origin=['producer', producer] if producer else ['annotation', annotation] if annotation else None, annotation=annotation, available=left.end_byte if n.type in {'for_statement', 'for_in_clause'} else None)
            elif n.type in {'delete_statement', 'as_pattern_target'}:
                for ident in _walk(n):
                    if ident.type == 'identifier':
                        self.add(self.text(ident), n, True)
            elif n.type == 'parameters':
                for c in n.named_children:
                    name = c if c.type == 'identifier' else c.child_by_field_name('name')
                    if name is not None and name.type == 'identifier':
                        annotation = path(c.child_by_field_name('type'), self.raw)
                        self.add(self.text(name), c, c.has_error, origin=['annotation', annotation] if annotation else None, annotation=annotation, available=n.end_byte, parameter=True)
            elif n.type in {'function_definition', 'class_definition'}:
                name = n.child_by_field_name('name')
                if name is not None:
                    self.add(self.text(name), n, unsafe, origin=['definition', n.type, self.text(name)])
        for entries in self.writes.values():
            entries.sort(key=lambda x: x['available'])

    def text(self, n):
        return self.raw[n.start_byte:n.end_byte].decode('utf-8')

    def add(self, name, n, unsafe, origin=None, annotation=None, available=None, parameter=False):
        scope = _scope(n)
        source_end = n.end_byte
        if n.type in {'function_definition', 'class_definition'}:
            body = n.child_by_field_name('body')
            if body is not None and body.start_point[0] > n.start_point[0]:
                source_end = body.start_byte - body.start_point[1]
        entry = {'name': name, 'scope': scope, 'start': n.start_byte, 'available': n.end_byte if available is None else available, 'unsafe': unsafe, 'origin': origin, 'annotation': annotation, 'span': [n.start_byte, source_end], 'parameter': parameter}
        self.writes.setdefault((scope, name), []).append(entry)

    def binding(self, name, n):
        scope = _scope(n)
        if (scope, name) in self.declared:
            return None
        for size in range(len(scope), -1, -1):
            here = scope[:size]
            if size < len(scope) and size and (self.scope_type(here[-1]) == 'class_definition'):
                continue
            entries = self.writes.get((here, name), [])
            pos = bisect_right([e['available'] for e in entries], n.start_byte) - 1
            if pos >= 0:
                return None if entries[pos]['unsafe'] else entries[pos]
            if entries and size == len(scope):
                return None
        return None

    def scope_type(self, start):
        return self.scope_types.get(start)

    def identity(self, name, n):
        parts = name.split('.')
        if len(parts) > 1:
            return None
        if parts[0] in {'self', 'cls'}:
            bound = self.binding(parts[0], n)
            if bound is None or not bound['parameter']:
                return None
            owner = next((p for p in _parents(n) if p.type == 'class_definition'), None)
            if owner is None:
                return None
            label = owner.child_by_field_name('name')
            if label is None:
                return None
            if len(parts) > 1:
                return None
            return {'key': ['class', owner.start_byte], 'type': ['class', owner.start_byte], 'origin': ['class', self.text(label)], 'span': [owner.start_byte, label.end_byte]}
        entry = self.binding(parts[0], n)
        if entry is None:
            return None
        return {'key': ['binding', list(entry['scope']), entry['name'], entry['start'], parts[1:]], 'type': self.qualified_annotation(entry, n) if len(parts) == 1 else None, 'origin': entry['origin'] if len(parts) == 1 else None, 'span': entry['span']}

    def qualified_annotation(self, entry, n):
        annotation = entry['annotation']
        if not annotation:
            return None
        head, *tail = annotation.split('.')
        declared = self.root.descendant_for_byte_range(entry['start'], min(entry['start'] + 1, len(self.raw)))
        binding = self.binding(head, declared)
        if binding and binding['origin']:
            return [binding['origin'], tail]
        return [list(entry['scope']), annotation]

    @staticmethod
    def match(a, b):
        if not a or not b:
            return 0
        if a['key'] == b['key']:
            return 1
        return 0.5 if a['type'] is not None and a['type'] == b['type'] else 0

    def calls(self):
        out = []
        for n in self.nodes:
            if n.type != 'call' or n.has_error:
                continue
            fn = n.child_by_field_name('function')
            if fn is None or fn.type != 'attribute':
                continue
            receiver = path(fn.child_by_field_name('object'), self.raw)
            member = path(fn.child_by_field_name('attribute'), self.raw)
            if not receiver or not member:
                continue
            identity = self.identity(receiver, n)
            if not identity:
                continue
            args = n.child_by_field_name('arguments')
            keywords = [self.text(c.child_by_field_name('name')) for c in args.named_children if c.type == 'keyword_argument' and c.child_by_field_name('name')] if args else []
            parent = n.parent
            while parent and parent.type in {'await', 'parenthesized_expression'}:
                parent = parent.parent
            left = parent.child_by_field_name('left') if parent and parent.type == 'assignment' else None
            arity = len(left.named_children) if left and left.type in {'pattern_list', 'tuple_pattern', 'tuple', 'list_pattern'} else 0
            out.append({'span': [n.start_byte, n.end_byte], 'identity': identity, 'member': member, 'keywords': keywords, 'arity': arity, 'arguments': self.text(args) if args else ''})
        return sorted(out, key=lambda x: x['span'])

    def operations(self, calls):
        out = list(calls)
        for n in self.nodes:
            if n.type not in {'assignment', 'augmented_assignment'} or n.has_error:
                continue
            left = n.child_by_field_name('left')
            right = n.child_by_field_name('right')
            if left is None or right is None or left.type != 'attribute':
                continue
            name = path(left, self.raw)
            if not name or '.' not in name:
                continue
            head, tail = name.split('.', 1)
            identity = self.identity(head, n)
            if identity is None:
                continue
            operator = next((self.text(c) for c in n.children if c.type in {'=', '+=', '-=', '*=', '/='}), '=')
            value = self.text(right).strip()
            out.append({'span': [n.start_byte, n.end_byte], 'identity': identity, 'member': 'state:' + tail + operator + value, 'keywords': [], 'arity': 0, 'arguments': value})
        return sorted(out, key=lambda x: x['span'])
TASK_VECTOR_PROTOCOL = 'question_only_v1'

def cached_encode(index, texts, statistics=None):
    """Reuse only the existing encoder namespace; all values are normalized vectors."""
    import numpy as np
    if getattr(index, '_task_vectors', None) is not None:
        from benchmarking.retrieval_online import task_encode
        return task_encode(index, texts, statistics)
    if not hasattr(index, 'cache'):
        if statistics is not None:
            statistics.update(texts=len(texts), unique_texts=len(set(texts)), cache_hits=0, encoded_texts=len(texts))
        values = index.encode(texts)
        return values.numpy() if hasattr(values, 'numpy') else np.asarray(values)
    keys = [index.cache.key(text) for text in texts]
    cached = index.cache.get_many(keys)
    missing = {key: text for key, text in zip(keys, texts) if key not in cached}
    if statistics is not None:
        statistics.update(texts=len(texts), unique_texts=len(set(keys)), cache_hits=len(set(keys)) - len(missing), encoded_texts=len(missing))
    if missing:
        values = index.encode(list(missing.values()))
        values = values.numpy() if hasattr(values, 'numpy') else np.asarray(values)
        added = dict(zip(missing, values))
        index.cache.put_many(added)
        cached.update(added)
    return np.stack([cached[key] for key in keys])
