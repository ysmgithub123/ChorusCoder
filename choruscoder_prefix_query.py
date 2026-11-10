"""Source-preserving prefix windows and parser-free fixed-line segment queries."""
from functools import lru_cache
import json
from choruscoder import digest
VERSION = 'prefix_query_controls_v1'

@lru_cache(maxsize=4)
def _stable_byte_bpe(tokenizer):
    """Recognize deterministic GPT-2/Roberta byte BPE without normalization.

    A cut can change its first whitespace/word unit, but cannot change the
    units after it. Other tokenizer configurations keep the exhaustive path.
    """
    if getattr(tokenizer, 'add_prefix_space', False) or not hasattr(tokenizer, 'get_added_vocab'):
        return False
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is not None:
        pre = json.loads(backend.pre_tokenizer.__getstate__())
        model = json.loads(backend.model.__getstate__())
        return backend.normalizer is None and backend.truncation is None and (backend.padding is None) and (pre.get('type') == 'ByteLevel') and (pre.get('add_prefix_space') is False) and (pre.get('use_regex') is True) and (model.get('type') == 'BPE') and (model.get('dropout') is None) and (not model.get('continuing_subword_prefix')) and (not model.get('end_of_word_suffix'))
    pattern = getattr(getattr(tokenizer, 'pat', None), 'pattern', None)
    return hasattr(tokenizer, 'byte_decoder') and pattern == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

@lru_cache(maxsize=4)
def _source_vocabulary_bytes(tokenizer):
    """Byte BPE pieces, only when every source byte has a vocabulary token.

    Full byte coverage rules out unknown-token substitution, so a suffix of
    at most B tokens cannot contain more than B * M bytes. Added tokens in the
    source are excluded by the caller before this bound is used.
    """
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    decoder = getattr(tokenizer, 'byte_decoder', None)
    if decoder is None and backend is not None:
        values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        characters = values[:]
        extra = 0
        for value in range(256):
            if value not in values:
                values.append(value)
                characters.append(256 + extra)
                extra += 1
        decoder = dict(zip(map(chr, characters), values))
    if decoder is None or set(decoder.values()) != set(range(256)):
        return ()
    getter = getattr(tokenizer, 'get_vocab', None) or getattr(backend, 'get_vocab', None)
    if getter is None:
        return ()
    vocab = getter()
    if not all((character in vocab for character in decoder)):
        return ()
    pieces = set()
    for token in vocab:
        try:
            piece = bytes((decoder[character] for character in token))
        except (KeyError, TypeError, ValueError):
            continue
        if piece:
            pieces.add(piece)
    return tuple(sorted(pieces, key=lambda piece: (-len(piece), piece)))

def _suffix_start(prefix, tokenizer, budget, prefix_tokens):
    if prefix_tokens <= budget:
        return 0
    try:
        stable = _stable_byte_bpe(tokenizer)
    except (AttributeError, TypeError, ValueError):
        stable = False
    if stable and any((token and token in prefix for token in tokenizer.get_added_vocab())):
        stable = False
    if not stable:
        from choruscoder_chunk_controls import suffix_span
        return suffix_span(prefix, tokenizer, budget)[0]
    import regex
    boundaries = sorted({0, len(prefix), *(match.start() for match in regex.finditer('\\s+', prefix))})
    counts = {0: prefix_tokens, len(prefix): 0}

    def count(start):
        if start not in counts:
            counts[start] = len(tokenizer.encode(prefix[start:], add_special_tokens=False))
        return counts[start]
    lo, hi = (0, len(boundaries) - 1)
    while hi - lo > 1:
        middle = (lo + hi) // 2
        if count(boundaries[middle]) <= budget:
            hi = middle
        else:
            lo = middle
    first = boundaries[lo] + 1
    if boundaries[hi] - boundaries[lo] > budget:
        try:
            vocabulary = _source_vocabulary_bytes(tokenizer)
        except (AttributeError, TypeError, ValueError):
            vocabulary = ()
        if vocabulary:
            raw = prefix.encode('utf-8')
            maximum = next((len(piece) for piece in vocabulary if piece in raw))
            lower = max(0, len(raw) - budget * maximum)
            while lower < len(raw) and raw[lower] & 192 == 128:
                lower += 1
            first = max(first, len(raw[:lower].decode('utf-8')))
    for start in range(first, boundaries[hi] + 1):
        if count(start) <= budget:
            return len(prefix[:start].encode('utf-8'))
    raise ValueError('No fitting prefix suffix at a verified byte-BPE boundary')

def prefix_query_view(prefix, tokenizer, budget):
    if type(budget) is not int or not 1 <= budget <= 508:
        raise ValueError('Prefix query budget must be an integer in 1..508')
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    start = _suffix_start(prefix, tokenizer, budget, prefix_tokens)
    raw = prefix.encode('utf-8')
    end = len(raw)
    text = raw[start:end].decode('utf-8')
    count = len(tokenizer.encode(text, add_special_tokens=False))
    if count > budget or end != len(raw):
        raise ValueError('Prefix suffix exceeds its token budget or misses the cursor')
    return {'text': text, 'line_offset': prefix.count('\n') - text.count('\n'), 'start_token': len(tokenizer.encode(raw[:start].decode('utf-8'), add_special_tokens=False)), 'tokens': count, 'leading_truncated': start > 0, 'source_start_byte': start, 'source_end_byte': end, 'prefix_hash': digest(prefix), 'prefix_tokens': prefix_tokens, 'prefix_query_tokens': budget, 'version': VERSION}

def validate_prefix_record(record, config, tokenizer=None):
    """Validate Prefix100 records and cached source boundaries."""
    budget = config.get('prefix_query_tokens')
    if type(budget) is not int or not 1 <= budget <= 508:
        raise ValueError('Invalid recorded prefix query budget')
    prefix, view = (record['prefix'], record['window'])
    raw = prefix.encode('utf-8')
    start, end = (view.get('source_start_byte'), view.get('source_end_byte'))
    if type(start) is not int or type(end) is not int or (not 0 <= start <= end == len(raw)) or (view.get('prefix_hash') != digest(prefix)) or (view.get('version') != VERSION) or (view.get('prefix_query_tokens') != budget):
        raise ValueError('Prefix query source span/hash/version mismatch')
    try:
        text = raw[start:end].decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError('Prefix query starts inside a UTF-8 character') from exc
    if view['text'] != text or type(view.get('tokens')) is not int or (not 0 <= view['tokens'] <= budget) or (type(view.get('prefix_tokens')) is not int) or (view['prefix_tokens'] < view['tokens']) or (view.get('leading_truncated') != (start > 0)) or (view.get('line_offset') != prefix.count('\n') - text.count('\n')):
        raise ValueError('Prefix query text/token metadata differs from visible source')
    if tokenizer is not None and view != prefix_query_view(prefix, tokenizer, budget):
        raise ValueError('Prefix query is not the tokenizer-counted source suffix')
    expected = [{'id': 'Q0', 'text': text, 'segment_ids': []}]
    details = record.get('diagnostics', {}).get('ablation', {})
    if record['queries'] != expected or record['selected_segments'] or record['eligible_supports'] != 0 or (record['probe_seconds'] != 0) or (record['fallback'] is not None) or record['official'] or (details.get('target') != '') or (details.get('target_context_spans') != []) or (details.get('candidates') != []) or (details.get('scores') != []) or any((key in record.get('diagnostics', {}) for key in ('semantic_support', 'support_control', 'support_units', 'chunk_controls', 'far_selection', 'value_gate'))):
        raise ValueError('Prefix query record contains supports, a target or selection logic')
    return view
