"""Configurable repository windows with optional embedding adapters.

UniXcoder encoding and content-addressed repository caches.
"""
from choruscoder import digest
from benchmarking.choruscoder_models import DenseIndex


class BudgetDenseIndex(DenseIndex):
    def __init__(self, cache_dir, tokenizer, model_name="microsoft/unixcoder-base",
                 chunk_tokens=192, overlap_tokens=0, encoder_config=None):
        if type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 508:
            raise ValueError("Chunk size must be in 1..508")
        if type(overlap_tokens) is not int or not 0 <= overlap_tokens < chunk_tokens:
            raise ValueError("Overlap must be non-negative and smaller than chunk size")
        self.chunk_tokens, self.overlap_tokens = chunk_tokens, overlap_tokens
        if encoder_config is None:
            super().__init__(cache_dir,tokenizer,model_name)
        else:
            super().__init__(cache_dir,tokenizer,model_name,encoder_config=encoder_config)

    def build_chunks(self, files):
        chunks = []
        for file in files:
            ids = self.tokenizer.encode(file["text"],add_special_tokens=False)
            for start in range(0,len(ids),self.chunk_tokens-self.overlap_tokens):
                end = min(start+self.chunk_tokens,len(ids))
                text = self.tokenizer.decode(ids[start:end])
                chunks.append({"id":"E"+digest([file["path"],start,end,text])[:12],
                    "path":file["path"],"text":text,"start_token":start,"end_token":end})
                if end == len(ids):
                    break
        return chunks

    def retrieve(self, queries, files, quotas):
        import numpy as np
        import time
        if len(queries) != len(quotas):
            raise ValueError("One recall quota is required per query")
        self.last_index_seconds = 0.0
        self.last_query_encoding_cache={}
        self.last_query_encoding_seconds=0.0
        self.last_repository_runtime_timings={}
        chunks = self.build_chunks(files)
        if not chunks:
            return [[] for _ in queries]
        keys = [self.cache.key(c["text"]) for c in chunks]
        unique = dict(zip(keys,(c["text"] for c in chunks)))
        vector_started=time.perf_counter()
        memory=getattr(self,'_repository_vector_cache',None)
        disk_seconds=0.0
        if memory is None:
            disk_started=time.perf_counter()
            cached=self.cache.get_many(unique)
            disk_seconds=time.perf_counter()-disk_started
        else:
            from benchmarking.choruscoder_runtime_cache import embedding_revision
            revision=embedding_revision(self.cache)
            if getattr(self,'_repository_vector_revision',revision)!=revision:
                memory.clear()
            self._repository_vector_revision=revision
            cached={}
            for key in unique:
                vector=memory.get(key)
                if vector is not None:
                    cached[key]=vector
            needed=[key for key in unique if key not in cached]
            disk_started=time.perf_counter()
            loaded=self.cache.get_many(needed) if needed else {}
            disk_seconds=time.perf_counter()-disk_started
            cached.update(loaded)
            for key,vector in loaded.items():
                stored=np.array(vector,dtype=np.float32,copy=True)
                stored.flags.writeable=False
                memory.put(key,stored)
        missing = [key for key in unique if key not in cached]
        if missing:
            started = time.perf_counter()
            encoded = self.encode([unique[key] for key in missing]).numpy()
            values = dict(zip(missing,encoded))
            self.cache.put_many(values)
            cached.update(values)
            self.last_index_seconds = time.perf_counter()-started
            if memory is not None:
                for key,vector in values.items():
                    # Match SQLite's float32 representation on subsequent
                    # reads, while retaining original first-encode values.
                    stored=np.array(vector,dtype=np.float32,copy=True)
                    stored.flags.writeable=False
                    memory.put(key,stored)
        vector_seconds=max(0.0,time.perf_counter()-vector_started-self.last_index_seconds)
        matrix_started=time.perf_counter()
        matrix = np.stack([cached[key] for key in keys])
        matrix_seconds=time.perf_counter()-matrix_started
        query_started=time.perf_counter()
        self.last_query_encoding_cache={}
        if getattr(self,'cache_query_vectors',False):
            from choruscoder_far import cached_encode
            qvectors=cached_encode(self,[q['text'] for q in queries],statistics=self.last_query_encoding_cache)
        else:
            qvectors = self.encode([q["text"] for q in queries]).numpy()
        self.last_query_encoding_seconds=time.perf_counter()-query_started
        results = []
        ranking_started=time.perf_counter()
        for query,quota in zip(qvectors,quotas):
            scores = matrix @ query
            ranked = np.lexsort((np.arange(len(chunks)),-scores))[:quota]
            results.append([dict(chunks[int(i)],score=float(scores[i])) for i in ranked])
        if memory is not None or getattr(self,'record_repository_timings',False):
            self.last_repository_runtime_timings=dict(vector_lookup_seconds=vector_seconds,
                vector_disk_read_seconds=disk_seconds,matrix_assembly_seconds=matrix_seconds,
                query_encoding_seconds=self.last_query_encoding_seconds,
                scoring_sorting_seconds=time.perf_counter()-ranking_started)
            if getattr(self,'record_repository_timings',False):
                self.last_repository_runtime_timings.update(repository_unique_texts=len(unique),
                    repository_encoded_texts=len(missing),repository_cached_texts=len(unique)-len(missing),
                    repository_encoding_seconds=self.last_index_seconds)
        return results
