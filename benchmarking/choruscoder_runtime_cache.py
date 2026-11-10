"""Bounded, process-local reuse; persistent caches remain the source of truth."""
from collections import OrderedDict
import sys


def retained_bytes(value, seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size=sys.getsizeof(value)
    if isinstance(value,dict):
        size+=sum(retained_bytes(key,seen)+retained_bytes(item,seen) for key,item in value.items())
    elif isinstance(value,(tuple,list)):
        size+=sum(retained_bytes(item,seen) for item in value)
    return size


class BoundedLRU:
    def __init__(self,limit_bytes):
        if type(limit_bytes) is not int or limit_bytes<0:
            raise ValueError('Runtime cache byte limit must be a nonnegative integer')
        self.limit_bytes=limit_bytes
        self.items=OrderedDict()
        self.bytes=0
        self.hits=self.misses=self.evictions=self.rejected=0

    def get(self,key):
        item=self.items.get(key)
        if item is None:
            self.misses+=1
            return None
        self.hits+=1
        self.items.move_to_end(key)
        return item[0]

    def discard(self,key):
        item=self.items.pop(key,None)
        if item is not None:
            self.bytes-=item[1]

    def put(self,key,value):
        self.discard(key)
        cost=retained_bytes(key)+retained_bytes(value)+256
        if cost>self.limit_bytes:
            self.rejected+=1
            return
        while self.items and self.bytes+cost>self.limit_bytes:
            _,(_,size)=self.items.popitem(last=False)
            self.bytes-=size
            self.evictions+=1
        self.items[key]=(value,cost)
        self.bytes+=cost

    def clear(self):
        self.items.clear()
        self.bytes=0

    def statistics(self):
        return dict(limit_bytes=self.limit_bytes,retained_bytes=self.bytes,entries=len(self.items),
                    hits=self.hits,misses=self.misses,evictions=self.evictions,rejected=self.rejected)


def runtime_cache_config(method, benchmark):
    from choruscoder_benchmarks import RECCEVAL_CAST_METHODS
    from choruscoder_profiles import PROFILES
    if benchmark == 'recceval' and method in RECCEVAL_CAST_METHODS:
        return dict(version='recceval_repository_reuse_v1', validated_chunks_bytes=64*1024**2, embedding_vectors_bytes=256*1024**2)
    profile = PROFILES.get(method)
    if benchmark == 'cceval' and profile is not None and (profile.api_support_config is not None or profile.support_ablation_config is not None):
        return dict(version='gaussrank_validated_chunks_v1', validated_chunks_bytes=64*1024**2, embedding_vectors_bytes=0)
    return None


def enable_runtime_cache(backend,config):
    backend._validated_chunk_cache=BoundedLRU(config['validated_chunks_bytes'])
    if config['embedding_vectors_bytes']:
        backend._repository_vector_cache=BoundedLRU(config['embedding_vectors_bytes'])


def file_revision(path):
    try:
        stat=path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def embedding_revision(cache):
    connection=getattr(cache,'connection',None)
    if connection is None:
        return None
    # SQLite increments data_version for changes committed by other
    # connections; our own inserts do not invalidate our validated vectors.
    return connection.execute('PRAGMA data_version').fetchone()[0]
