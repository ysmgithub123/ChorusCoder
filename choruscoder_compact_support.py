"""Visible-prefix features, retrieval memoization and evidence packing for API selection."""
import json
import math
import re
from functools import lru_cache
from choruscoder import make_parser, evidence_text, pack_evidence, render_evidence
from choruscoder_compact_bindings import extract, VisibleNames, current_statement, in_spans
from choruscoder_compact_apis import active_header, words, resolve_provider, all_apis, hypothesis_scores

def prefix_score(f):
    return 4 * f['descriptor_named'] + 2.8 * f['meaningful_header'] + 1.8 * f['is_source'] + 0.6 * f['receiver_call'] * f['current_scope'] + 0.45 * math.log1p(f['mentions']) + 0.3 * f['near'] + 0.2 * f['current_scope'] + 0.15 * f['assertions'] - 0.0003 * f['length']

def prefix_features(prefix, target_T, candidates, parser=None):
    parser = parser or make_parser('python')
    dependency = extract(prefix, candidates, [], parser)
    core = [b for b in dependency['bindings'] if b['confidence'] == 'core' and b['mapped_candidates']]
    span = core[0]['span'] if core else None
    visible = VisibleNames(prefix, parser)
    variables = visible.active_variables(current_statement(prefix))
    names = {v['name'] for v in variables}
    occurrences = visible.occurrences(variables)
    start, name = active_header(prefix, target_T)
    meaningful = bool(words(name))
    fs = []
    for i, c in enumerate(candidates):
        text = c['text']
        point = min((a for a, b in c['spans']))
        current = start is not None and point >= start
        header = bool(start is not None and any((a <= start < b for a, b in c['spans'])))
        descriptor = bool(re.match('\\s*(?:[rubf]*[\\"\\\']{3}|#)', text, re.I))
        named = re.findall('\\b[a-zA-Z]\\w*_[a-zA-Z]\\w*\\b|\\b[A-Z][a-z]+[A-Z]\\w*\\b', text)
        mentions = sum((in_spans(c['spans'], *o['span']) for o in occurrences))
        distance_rank = len(candidates) - i
        fs.append(dict(index=i, label=f'C{i + 1}', descriptor_named=bool(current and descriptor and named), meaningful_header=bool(header and meaningful), is_source=bool(span and in_spans(c['spans'], *span)), receiver_call=any((re.search('\\b' + re.escape(n) + '\\.\\w+\\s*\\(', text) for n in names)), current_scope=current, mentions=mentions, near=1 / (1 + math.log1p(distance_rank)), assertions=bool(re.search('\\bassert\\b|\\.assert\\w*\\(', text)), length=len(text)))
    return dict(features=fs, bindings=core, source_span=span, variables=variables, dependency_status=dependency['status'])

class FastPacker:

    def __init__(self, tokenizer):
        cfg = json.loads(tokenizer.backend_tokenizer.to_str())
        steps = cfg['pre_tokenizer']['pretokenizers']
        if cfg['normalizer'] != {'type': 'Sequence', 'normalizers': []} or not steps or steps[0] != {'type': 'Split', 'pattern': {'Regex': '[\r\n]'}, 'behavior': 'Isolated', 'invert': False} or (not any((s == {'type': 'Digits', 'individual_digits': True} for s in steps))) or (not all((len(tokenizer.encode(str(i), add_special_tokens=False)) == 1 for i in range(10)))):
            raise ValueError('Tokenizer does not support exact cached packing')
        self.tokenizer = tokenizer
        self.costs = {}
        self.separator = len(tokenizer.encode('\n\n', add_special_tokens=False))

    def pack(self, blocks, budget=3072):
        chosen = []
        cost = 0
        for b in blocks:
            if len(chosen) >= 30:
                break
            if b['id'] not in self.costs:
                self.costs[b['id']] = len(self.tokenizer.encode(evidence_text(dict(b, display_id='E1'), 'python'), add_special_tokens=False))
            proposed = cost + self.costs[b['id']] + (len(str(len(chosen) + 1)) - 1 if not b.get('official') else 0) + (self.separator if chosen else 0)
            if proposed <= budget:
                chosen.append(b)
                cost = proposed
        return (chosen, cost)

class ContextPacker:
    """Use the verified fast path only for its exact tokenizer contract."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        try:
            self.fast = FastPacker(tokenizer)
        except (AttributeError, KeyError, TypeError, ValueError):
            self.fast = None

    def pack(self, blocks, budget=3072):
        if self.fast is not None:
            return self.fast.pack(blocks, budget)
        chosen = pack_evidence(blocks, self.tokenizer, budget=budget, maximum=30, language='python')
        return (chosen, len(self.tokenizer.encode(render_evidence(chosen, 'python'), add_special_tokens=False)))

@lru_cache(maxsize=128)
def module_apis(path, text, class_name):
    return all_apis([dict(paths=[path], class_name=class_name)], [dict(path=path, text=text)])

def prepare_supports(prefix, target, candidates, path, files, parser=None):
    detail = prefix_features(prefix, target, candidates, parser)
    providers = resolve_provider(detail['bindings'], path, files)
    mapping = {f['path']: f['text'] for f in files}
    apis = []
    for provider in providers:
        for name in provider['paths']:
            apis.extend(module_apis(name, mapping[name], provider.get('class_name')))
    hypotheses = {h['name']: h for h in hypothesis_scores(prefix, target, apis)}
    apis = [dict(a, hypothesis=hypotheses[a['name']]) for a in apis]
    reason = None
    if not candidates:
        reason = 'no_support_candidates'
    elif not detail['variables'] and (not detail['bindings']):
        reason = 'no_cursor_variables'
    elif not any((f['mentions'] or f['is_source'] for f in detail['features'])):
        reason = 'no_candidate_mentions_cursor_variables'
    elif not providers:
        reason = 'variable_source_unresolved'
    elif not apis:
        reason = 'no_scorable_interfaces'
    return dict(features=[{}] + detail['features'], apis=apis, provider_count=len(providers), dependency_status=detail['dependency_status'], fallback_reason=reason)
