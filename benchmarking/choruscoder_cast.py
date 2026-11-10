"""cAST token-budget adaptation; see third_party/astchunk_official/NOTICE.md.

The split/merge skeleton follows pinned ASTChunk 0.1.0. Size is measured on
actual source spans, not summed non-whitespace counts. Trivia and oversize
leaves are retained rather than lost. No upstream dependency is imported.
"""
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import hashlib
import time
from bisect import bisect_left, bisect_right
from copy import deepcopy
from choruscoder import atomic_json, digest, load_json, make_parser
from choruscoder_profiles import RepositoryChunkConfig
from benchmarking.choruscoder_context import BudgetDenseIndex
UPSTREAM = Path(__file__).resolve().parents[1] / 'third_party/astchunk_official'

@lru_cache(maxsize=1)
def upstream_identity():
    manifest = load_json(UPSTREAM / 'upstream.json')
    if manifest['commit'] != RepositoryChunkConfig().upstream_commit or manifest['version'] != '0.1.0':
        raise ValueError('cAST upstream revision differs from the pinned source')
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'cAST upstream source checksum mismatch: {name}')
    return manifest

def parser_identity(language='python'):
    from choruscoder_languages import parser_identity as shared_identity
    return shared_identity(language)

def implementation_identity(language='python', config=None):
    result = {'upstream': upstream_identity(), 'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'parser': parser_identity(language)}
    return result

def tokenizer_identity(tokenizer):
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else None
    return {'class': type(tokenizer).__module__ + '.' + type(tokenizer).__qualname__, 'name': str(getattr(tokenizer, 'name_or_path', '')), 'vocab_hash': digest(vocab), 'special_tokens': str(getattr(tokenizer, 'special_tokens_map', {})), 'init_kwargs': str(sorted(getattr(tokenizer, 'init_kwargs', {}).items()))}

def chunk_source(text, tokenizer, config, parser=None, *, root_node=None, byte_span=None, max_lines=None):
    """Return a contiguous, lossless partition, including whitespace-only spans.

    Nodes are packed in source order and recursively split when too large,
    following assign_nodes_to_windows/merge_adjacent_windows upstream. A child's
    span includes its preceding trivia; the final child also receives trailing
    trivia. Source bytes are always copied, never reconstructed from AST text.
    """
    data = text.encode('utf-8')
    stats = {'parse_has_error': False, 'parser_fallback': False, 'fallback_reason': None, 'oversize_leaf_count': 0}

    @lru_cache(maxsize=None)
    def size(a, b):
        return len(tokenizer.encode(data[a:b].decode('utf-8'), add_special_tokens=False))

    def fits(a, b):
        return size(a, b) <= config.max_tokens and (max_lines is None or len(data[a:b].decode('utf-8').splitlines()) <= max_lines)

    def raw_split(a, b, reason):
        offsets = [a]
        for char in data[a:b].decode('utf-8'):
            offsets.append(offsets[-1] + len(char.encode('utf-8')))
        position = 0
        while position < len(offsets) - 1:
            start = offsets[position]
            hi = min(len(offsets) - 1, position + config.max_tokens)
            fit = position
            while fits(start, offsets[hi]):
                fit = hi
                if hi == len(offsets) - 1:
                    break
                hi = min(len(offsets) - 1, position + 2 * (hi - position))
            lo = fit + 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if fits(start, offsets[mid]):
                    fit = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if fit == position:
                raise ValueError('A single UTF-8 character cannot fit the repository token budget')
            end = offsets[fit]
            newline = data.rfind(b'\n', start, end)
            if newline >= start and size(start, newline + 1) <= config.max_tokens:
                end = newline + 1
            yield (start, end, reason)
            position = bisect_left(offsets, end)

    def pack(spans):
        pending = None
        for a, b, reason in spans:
            if pending and pending[2] == reason and fits(pending[0], b):
                pending = (pending[0], b, reason)
            else:
                if pending:
                    yield pending
                pending = (a, b, reason)
        if pending:
            yield pending

    def split(node, a, b, depth=0):
        if fits(a, b):
            yield (a, b, None)
            return
        children = [c for c in node.children if c.end_byte > c.start_byte and c.end_byte > a and (c.start_byte < b)]
        if not children or depth >= 200:
            stats['oversize_leaf_count'] += 1
            yield from raw_split(a, b, 'oversize_leaf' if not children else 'depth_limit')
            return
        pending = None
        start = a
        for index, child in enumerate(children):
            end = b if index == len(children) - 1 else min(b, child.end_byte)
            if not start < end <= b:
                raise ValueError('Parser returned non-monotone AST source ranges')
            if not fits(start, end):
                if pending:
                    yield pending
                    pending = None
                yield from pack(split(child, start, end, depth + 1))
            elif pending and fits(pending[0], end):
                pending = (pending[0], end, None)
            else:
                if pending:
                    yield pending
                pending = (start, end, None)
            start = end
        if pending:
            yield pending
    if not data:
        return ([], stats)
    first, last = byte_span or (0, len(data))
    if not 0 <= first < last <= len(data):
        raise ValueError('Invalid AST source subrange')
    try:
        root = root_node if root_node is not None else (parser or make_parser('python')).parse(data).root_node
    except Exception as error:
        stats.update(parser_fallback=True, fallback_reason=type(error).__name__)
        spans = list(raw_split(first, last, 'parser_exception'))
    else:
        stats['parse_has_error'] = bool(root.has_error)
        spans = list(split(root, first, last))
    cursor = first
    for a, b, reason in spans:
        if a != cursor or not a < b <= last or (not fits(a, b)):
            raise ValueError('cAST partition failed source coverage or token-budget validation')
        cursor = b
    if cursor != last:
        raise ValueError('cAST partition lost source bytes')
    return (spans, stats)

def token_byte_offsets(text, tokenizer):
    """Map window spans to original bytes, without relying on IDs
    to compare different chunkers. Byte-level BPE permits Unicode-split tokens.
    For other tokenizers, only exact reversible prefix mappings are accepted.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    offsets = [0]
    if hasattr(tokenizer, 'byte_decoder') and hasattr(tokenizer, 'convert_ids_to_tokens'):
        try:
            pieces = [bytes((tokenizer.byte_decoder[c] for c in token)) for token in tokenizer.convert_ids_to_tokens(ids)]
        except (KeyError, TypeError):
            pieces = None
        if pieces is not None and b''.join(pieces) == text.encode('utf-8'):
            for piece in pieces:
                offsets.append(offsets[-1] + len(piece))
        else:
            offsets = []
    else:
        pieces = [tokenizer.decode([token]).encode('utf-8') for token in ids]
        if b''.join(pieces) == text.encode('utf-8'):
            for piece in pieces:
                offsets.append(offsets[-1] + len(piece))
        else:
            for i in range(1, len(ids) + 1):
                prefix = tokenizer.decode(ids[:i])
                if not text.startswith(prefix):
                    offsets = []
                    break
                offsets.append(len(prefix.encode('utf-8')))
    return (ids, offsets)

def fixed_window_spans(path, text, tokenizer, limit, coordinates=None):
    ids, offsets = coordinates or token_byte_offsets(text, tokenizer)
    result = []
    for start in range(0, len(ids), limit):
        end = min(start + limit, len(ids))
        body = tokenizer.decode(ids[start:end])
        result.append({'id': 'E' + digest([path, start, end, body])[:12], 'start_byte': offsets[start] if offsets else None, 'end_byte': offsets[end] if offsets else None})
    return result

class CastDenseIndex(BudgetDenseIndex):
    """Only repository build_chunks changes; encoding/ranking stay inherited."""

    def __init__(self, cache_dir, tokenizer, model_name='microsoft/unixcoder-base', repository_config=None, language='python', **kwargs):
        self.repository_config = repository_config or RepositoryChunkConfig()
        from choruscoder_languages import grammar_variant
        grammar_variant(language)
        self.language = language
        self.chunk_cache = Path(cache_dir) / 'repository_chunks'
        self.chunk_identity = {'config': asdict(self.repository_config), 'implementation': implementation_identity(language, self.repository_config), 'tokenizer': tokenizer_identity(tokenizer)}
        self.parser = make_parser(language)
        super().__init__(cache_dir, tokenizer, model_name, **kwargs)

    def build_chunks(self, files):
        started = time.perf_counter()
        chunks, diagnostics = ([], [])
        memory = getattr(self, '_validated_chunk_cache', None)
        if memory is not None:
            from benchmarking.choruscoder_runtime_cache import file_revision
        live_config = asdict(self.repository_config)
        for file in files:
            text, path = (file['text'], file['path'])
            from choruscoder_languages import grammar_variant
            variant = grammar_variant(getattr(self, 'language', 'python'), path)
            parser = self.parser
            data = text.encode('utf-8')
            source_hash = digest(text)
            key = digest({'identity': self.chunk_identity, 'path': path, 'source_hash': source_hash, 'grammar_variant': variant})
            location = self.chunk_cache / (key + '.json')
            hit = location.is_file()
            revision = file_revision(location) if memory is not None else None
            entry = memory.get(key) if memory is not None else None
            memory_hit = entry is not None and entry[0] == revision and (revision is not None) and (entry[2] == live_config)
            if entry is not None and (not memory_hit):
                memory.discard(key)
            if memory_hit:
                record = entry[1]
            elif hit:
                record = load_json(location)
                if record.get('key') != key or digest(record['chunks']) != record['chunks_hash']:
                    raise ValueError('cAST repository chunk cache checksum mismatch')
            else:
                chunker = chunk_source
                spans, stats = chunker(text, self.tokenizer, self.repository_config, parser)
                coordinates = token_byte_offsets(text, self.tokenizer)
                offsets = coordinates[1]
                blocks = []
                for a, b, reason in spans:
                    body = data[a:b].decode('utf-8')
                    if not body.strip():
                        continue
                    blocks.append({'id': 'E' + digest([path, a, b, body])[:12], 'path': path, 'text': body, 'source_hash': source_hash, 'start_byte': a, 'end_byte': b, 'start_token': max(0, bisect_right(offsets, a) - 1) if offsets else '', 'end_token': bisect_left(offsets, b) if offsets else '', 'token_count': len(self.tokenizer.encode(body, add_special_tokens=False)), 'chunk_kind': self.repository_config.mode, 'degraded_reason': reason})
                mapping = fixed_window_spans(path, text, self.tokenizer, self.repository_config.max_tokens, coordinates)
                record = {'key': key, 'grammar_variant': variant, 'chunks': blocks, 'chunks_hash': digest(blocks), 'diagnostics': stats, 'fixed_window_spans': mapping, 'mapping_hash': digest(mapping)}
                atomic_json(location, record)
                if memory is not None:
                    revision = file_revision(location)
            if not memory_hit and digest(record['fixed_window_spans']) != record['mapping_hash']:
                raise ValueError('cAST baseline source mapping cache checksum mismatch')
            blocks = record['chunks']
            for block in [] if memory_hit else blocks:
                a, b = (block['start_byte'], block['end_byte'])
                if block['path'] != path or block['source_hash'] != source_hash or data[a:b].decode('utf-8') != block['text'] or (block.get('chunk_kind') != self.repository_config.mode) or (not 0 < block['token_count'] <= self.repository_config.max_tokens) or (len(self.tokenizer.encode(block['text'], add_special_tokens=False)) != block['token_count']):
                    raise ValueError('cAST cached chunk source identity or budget mismatch')
            if record.get('grammar_variant') != variant:
                raise ValueError('cAST repository grammar variant cache mismatch')
            if memory is not None and (not memory_hit):
                if file_revision(location) != revision:
                    raise ValueError('cAST repository cache changed during validation')
                memory.put(key, (revision, record, live_config))
            if memory is not None:
                chunks.extend((deepcopy(block) for block in blocks))
            else:
                chunks.extend(blocks)
            diagnostic = dict(record['diagnostics'], path=path, grammar_variant=variant, source_hash=source_hash, source_bytes=len(data), cache_hit=hit, chunk_count=len(blocks), degraded_chunks=sum((bool(b['degraded_reason']) for b in blocks)), fixed_window_spans=deepcopy(record['fixed_window_spans']) if memory is not None else record['fixed_window_spans'])
            if memory is not None:
                diagnostic['validation_memory_hit'] = memory_hit
            diagnostics.append(diagnostic)
        self.last_chunk_seconds = time.perf_counter() - started
        self.last_repository_chunks = {'configuration': asdict(self.repository_config), 'identity': self.chunk_identity, 'files': diagnostics, 'chunk_count': len(chunks), 'token_counts': [c['token_count'] for c in chunks], 'degraded': any((f['parser_fallback'] or f['degraded_chunks'] for f in diagnostics)), 'chunks_seconds': self.last_chunk_seconds}
        if memory is not None:
            self.last_repository_chunks['validation_runtime_cache'] = memory.statistics()
        return chunks
