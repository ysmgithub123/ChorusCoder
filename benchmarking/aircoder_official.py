"""Pinned AIRCoder-RRF primitives and repository-only preprocessing storage."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import time

import numpy as np

from choruscoder import digest, load_json
from benchmarking.choruscoder_runtime_cache import BoundedLRU, embedding_revision
from benchmarking.retrieval_online import RepositoryStore
from embedding_cache import SQLiteEmbeddingCache
from benchmarking.aircoder_repository import source_text

MANIFEST = Path(__file__).with_name('aircoder_upstream.json')


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def upstream(root):
    manifest = load_json(MANIFEST)
    root = Path(root)
    for name, expected in manifest['sha256'].items():
        if not (root/name).is_file() or file_hash(root/name) != expected:
            raise ValueError('Missing or changed pinned AIRCoder source: '+str(root/name))
    return manifest


def dependency_definitions(root):
    """Read the author's exact instruction literals without constructing VClient."""
    tree = ast.parse((Path(root)/'src/llm_dep_ext.py').read_text(encoding='utf-8'))
    client = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VClient')
    init = next(n for n in client.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    system = next(ast.literal_eval(n.value) for n in ast.walk(init) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Attribute) and t.attr == 'system_prompt' for t in n.targets))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    prompt = next(n.value for n in ast.walk(run) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in n.targets))
    return system, compile(ast.Expression(prompt), 'aircoder_dependency_instruction', 'eval')


def dependency_prompt(prefix, language, tokenizer, definitions, max_tokens=4000):
    system, instruction = definitions
    user = eval(instruction, {'__builtins__': {}}, {'sample': {'prompt': prefix}, 'lang': language})
    # Approved base-model adaptation: always plain text, including models that
    # happen to advertise a chat template. The author's instructions are intact.
    text = system+'\n\nUser: '+user+'\n\nAssistant:'
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) > max_tokens:
        # Keep the instruction and the cursor end rather than silently discarding
        # the system instruction. Truncate only the supplied visible code.
        code_ids = tokenizer.encode(prefix, add_special_tokens=False)
        excess = len(ids)-max_tokens
        if not code_ids or excess >= len(code_ids):
            raise ValueError('Dependency instruction cannot fit the input budget')
        return dependency_prompt(tokenizer.decode(code_ids[excess:]), language, tokenizer,
                                 definitions, max_tokens)
    return text


def parse_dependencies(text):
    """The official llm_dep_ext.run parser, including its strict answer tags."""
    answer = re.search('<answer>(.*?)</answer>', text, re.DOTALL)
    if answer is None:
        return []
    predictions = answer.groups()[0].split(',')
    if len(predictions) < 2:
        predictions = answer.groups()[0].split('and')
    return [name.strip() for prediction in predictions for name in prediction.split('.')]


def qwen_dependency_prompt(prefix, language, tokenizer, definitions, max_tokens=4000):
    """Exact VClient.completion chat serialization; the whole prompt truncates left."""
    system, instruction = definitions
    user = eval(instruction, {'__builtins__': {}}, {'sample': {'prompt': prefix}, 'lang': language})
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       enable_thinking=False, truncation=True, max_length=max_tokens)
    return tokenizer.decode(ids)


def structure_text(chunk):
    m = chunk['metadata']
    return str(dict(file=m['file'], ancestors=m['ancestors'], local_definitions=m['struct_context']))


class QuestionEncoder:
    """Query-only memo, discarded after this question; never reads corpus vectors."""
    def __init__(self, model):
        self.model, self.memo = model, {}
        self.seconds = 0.0
        self.encoded_texts = 0

    def encode(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.memo))
        if missing:
            tick = time.perf_counter()
            values = self.model.encode(missing)
            self.seconds += time.perf_counter()-tick
            self.encoded_texts += len(missing)
            self.memo.update(zip(missing, values))
        return np.stack([self.memo[t] for t in texts])

    def similarity(self, *args, **kwargs):
        return self.model.similarity(*args, **kwargs)


class Corpus:
    """Only source files and corpus vectors cross question/run boundaries."""
    def __init__(self, root, identity, tokenizer, embedding_model, chunker, utils_repo,
                 chunk_memory=64*1024**2, vector_memory=256*1024**2):
        self.tokenizer, self.model = tokenizer, embedding_model
        self.chunker, self.utils_repo = chunker, utils_repo
        self.identity = dict(identity,artifact_version='prepared_vector_keys_v1')
        self.store = RepositoryStore(Path(root)/'repository_chunks', self.identity, chunk_memory)
        namespace = digest(dict(version='aircoder_official_cls_v1', encoder=identity['encoder'],
                                encode_length=identity['encode_length']))
        self.vectors = SQLiteEmbeddingCache(Path(root)/('vectors_'+namespace[:20]+'.sqlite3'), namespace)
        self.memory = BoundedLRU(vector_memory)
        self.revision = embedding_revision(self.vectors)
        self.seen, self.pending = set(), {}
        self.encoded = self.chunked = self.disk_keys = self.memory_hits = 0
        self.chunk_seconds = self.lookup_seconds = self.encoding_seconds = 0.0
        self.dimension = None

    @staticmethod
    def key(file): return dict(path=file['path'], source_hash=file['source_hash'])

    def texts(self, chunks):
        size = self.identity['encode_length']
        return ([self.tokenizer.decode(c['input_ids'][:size]) for c in chunks] +
                [structure_text(c) for c in chunks])

    def encode(self, texts):
        values = np.asarray(self.model.encode(texts, batch_size=10, convert_to_numpy=True),dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(texts) or not np.isfinite(values).all():
            raise ValueError('Invalid AIRCoder encoder output')
        if self.dimension is not None and values.shape[1] != self.dimension:
            raise ValueError('AIRCoder encoder dimension changed')
        self.validate_vectors(values)
        self.dimension = values.shape[1]
        return values

    def flush(self):
        if self.pending:
            started = time.perf_counter()
            keys = list(self.pending)
            values = self.encode(list(self.pending.values()))
            self.vectors.put_many(dict(zip(keys,values)))
            self.encoded += len(keys)
            self.encoding_seconds += time.perf_counter()-started
            self.pending.clear()

    def preprocess(self, files):
        for file in files:
            key = self.key(file); token = digest(key)
            if token in self.seen: continue
            def build():
                self.chunked += 1
                chunks = self.utils_repo.slice_code_ast(source_text(file),file['path'],self.chunker)
                size = self.identity['encode_length']
                for c in chunks:
                    c['_aircoder_code_key'] = self.vectors.key(self.tokenizer.decode(c['input_ids'][:size]))
                    c['_aircoder_structure_key'] = self.vectors.key(structure_text(c))
                return chunks
            started = time.perf_counter()
            chunks = self.store.get(key,build)
            self.chunk_seconds += time.perf_counter()-started
            keys = list(dict.fromkeys(c[k] for c in chunks for k in ('_aircoder_code_key','_aircoder_structure_key')))
            started = time.perf_counter()
            cached = self.vectors.get_many(keys)
            self.validate_vectors(cached.values())
            self.lookup_seconds += time.perf_counter()-started
            available = set(cached) | set(self.pending)
            for c in chunks:
                for field in ('_aircoder_code_key','_aircoder_structure_key'):
                    k = c[field]
                    if k not in available:
                        available.add(k)
                        self.pending[k] = (self.tokenizer.decode(c['input_ids'][:self.identity['encode_length']])
                                           if field == '_aircoder_code_key' else structure_text(c))
                        # Keep full official encoder batches of ten across files.
                        if len(self.pending) == 160: self.flush()
            self.seen.add(token)

    def candidates(self, files):
        """No builders and no encoder: missing/corrupt artifacts are errors."""
        if self.pending: raise ValueError('Repository preprocessing has not finished')
        chunks = [c for f in files for c in self.store.get(self.key(f))]
        keys = list(dict.fromkeys(c[k] for c in chunks for k in ('_aircoder_code_key','_aircoder_structure_key')))
        revision = embedding_revision(self.vectors)
        if revision != self.revision: self.memory.clear()
        self.revision = revision
        values = {}
        for key in keys:
            v = self.memory.get(key)
            if v is not None: values[key] = v
        self.memory_hits += len(values)
        needed = [k for k in keys if k not in values]
        loaded = self.vectors.get_many(needed) if needed else {}
        self.disk_keys += len(needed)
        if any(k not in loaded for k in needed):
            raise ValueError('Missing prepared AIRCoder vectors; rerun preprocessing')
        self.validate_vectors(loaded.values())
        for k,v in loaded.items():
            v.flags.writeable = False
            self.memory.put(k,v)
        values.update(loaded)
        for i,c in enumerate(chunks):
            c.update(id=i,emb=values[c['_aircoder_code_key']],
                     struct_emb=values[c['_aircoder_structure_key']], scores={})
        return chunks

    def validate_vectors(self, vectors):
        expected = self.identity['encoder'].get('hidden_size') or self.dimension
        for v in vectors:
            if v.ndim != 1 or not len(v) or not np.isfinite(v).all() or expected is not None and len(v) != expected:
                raise ValueError('Invalid prepared AIRCoder vector dimension/content')
            if expected is None: expected = len(v)
        if expected is not None: self.dimension = expected

    def stats(self):
        return dict(repository_files=len(self.seen), repository_chunked_files=self.chunked,
            repository_encoded_texts=self.encoded, artifact_hits=self.store.hits,
            repository_chunking_seconds=self.chunk_seconds, vector_lookup_seconds=self.lookup_seconds,
            repository_encoding_and_write_seconds=self.encoding_seconds)

    def close(self): self.vectors.close()


def budget_prompt(sample, ranked, tokenizer, prompts, profile):
    """Official top-prefix/reverse presentation; only budgets are adapted."""
    prompt_code = tokenizer.decode(sample['prompt_input_ids'])
    current = '# '+sample['metadata']['file']+'\n'+prompt_code
    for n in range(min(profile.num_rags,len(ranked)),0,-1):
        chosen = ranked[:n]
        codes = ''.join('# '+s['metadata'].get('file',sample['metadata']['file'])+'\n'+
                        tokenizer.decode(s['input_ids'])+'\n\n' for s in chosen[::-1]).strip()
        prompt = prompts.RAG_PROMPT.format(rag_codes=codes,code=current).strip()
        # Upstream strips the whole prompt, including cursor indentation. Count
        # the actual context prefix rather than slicing by the unstripped code.
        context = prompt[:-len(current.rstrip())]
        if (len(tokenizer.encode(context,add_special_tokens=False)) <= profile.crossfile_tokens and
                len(tokenizer.encode(prompt,add_special_tokens=True)) <= profile.context_tokens):
            return prompt, chosen, dict(prompt_tokens=len(tokenizer.encode(prompt,add_special_tokens=True)),
                crossfile_tokens=len(tokenizer.encode(context,add_special_tokens=False)))
    # Official no-context sentinel returns the query code. Oversized AST leaves
    # need an explicit suffix adaptation to enforce the shared generation limit.
    ids = tokenizer.encode(prompt_code,add_special_tokens=False)
    while len(tokenizer.encode(prompt_code,add_special_tokens=True)) > profile.context_tokens:
        excess = len(tokenizer.encode(prompt_code,add_special_tokens=True))-profile.context_tokens
        ids = ids[max(1,excess):]; prompt_code = tokenizer.decode(ids)
    return prompt_code, [], dict(prompt_tokens=len(tokenizer.encode(prompt_code,add_special_tokens=True)),crossfile_tokens=0)


def query(request, dependencies, corpus, retriever, prompts, profile):
    """Official prefix query and full-snapshot candidate line filtering."""
    started = time.perf_counter()
    left = corpus.utils_repo.slice_code_ast(request['prefix'],request['path'],corpus.chunker)
    if not left:
        raise ValueError('AIRCoder cannot construct a query chunk: '+request['task_id'])
    last = left[-1]
    sample = dict(metadata=dict(last['metadata'],task_id=request['task_id'],file=request['path'],
                  pred_df=[['llm',d] for d in dependencies]),prompt_input_ids=last['input_ids'])
    candidates = corpus.candidates(request['files'])
    cross = [c for c in candidates if c['metadata']['file'] != request['path']]
    if len(cross) == len(candidates):
        raise ValueError('Current file has no prepared chunks: '+request['task_id'])
    visible = [c for c in candidates if c['metadata']['file'] == request['path']
               and c['metadata']['end_line_no'] <= last['metadata']['start_line_no']]
    # Database.get_filtered_slices_by_sample puts cross-file chunks first.
    candidates = cross + visible
    preparation = time.perf_counter()-started
    started = time.perf_counter()
    local_encoder = QuestionEncoder(corpus.model)
    original = (retriever.dense.model, retriever.skelton.model)
    retriever.dense.model = retriever.skelton.model = local_encoder
    try:
        ranked = retriever.query(candidates,sample) if candidates else []
    finally:
        retriever.dense.model, retriever.skelton.model = original
    ranking = time.perf_counter()-started
    started = time.perf_counter()
    prompt, selected, usage = budget_prompt(sample,ranked,corpus.tokenizer,prompts,profile)
    return dict(prompt=prompt,selected=selected,usage=usage,query_metadata=sample['metadata'],
        query_input_ids=sample['prompt_input_ids'],query_texts=list(local_encoder.memo),
        candidate_count=len(candidates),
        ranking=[dict(file=c['metadata']['file'],start_line_no=c['metadata']['start_line_no'],
                      end_line_no=c['metadata']['end_line_no'],scores=deepcopy(c['scores'])) for c in ranked],
        stage_seconds=dict(query_and_candidate_preparation=preparation,
                           query_encoding=local_encoder.seconds,ranking=ranking-local_encoder.seconds,
                           packing=time.perf_counter()-started),visible_candidate_count=len(visible),
        query_encoded_texts=local_encoder.encoded_texts,repository_online_encoded_texts=0,
        repository_online_chunked_files=0)
