"""Immutable source-only trees: prepared once, read without online builders."""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from choruscoder import atomic_json, digest, load_json

VERSION = 'source_corpus_v1'


class MissingCorpusError(ValueError):
    pass


def safe_path(value):
    p=PurePosixPath(value)
    if not value or not p.parts or p.is_absolute() or '..' in p.parts or any(c in value for c in ('\\',':','\x00')):
        raise ValueError('Unsafe source cache path: '+repr(value))
    return p.as_posix()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SourceCorpusCache:
    def __init__(self,root,max_entries=128,max_bytes=32*1024**2):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.max_entries=max_entries;self.memo=OrderedDict();self.verified=OrderedDict();self.blob_memo=OrderedDict()
        self.max_bytes=max_bytes;self.memo_bytes=0;self.memo_costs={}
        self.hits=self.builds=self.blobs_written=self.bytes_written=self.repaired_files=0

    def prepare(self,files):
        entries=[];contents={}
        for file in files:
            name=safe_path(file['path']);body=file['text'].encode('utf8')
            token=hashlib.sha256(body).hexdigest()
            entries.append(dict(path=name,sha256=token,size=len(body)));contents[token]=body
        if len({e['path'] for e in entries})!=len(entries): raise ValueError('Duplicate source cache path')
        identity=dict(version=VERSION,files=entries);key=digest(identity)
        target=self.root/'corpora'/key
        if target.exists():
            try:value=self.read(key)
            except MissingCorpusError:
                envelope=load_json(target/'manifest.json')
                if envelope!=dict(identity=identity,checksum=digest(identity)):
                    raise ValueError('Corrupt incomplete source corpus identity')
                source=target/'source';source.mkdir(exist_ok=True)
                actual={p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
                if not actual<={e['path'] for e in entries}:raise ValueError('Unexpected source corpus file')
                for e in entries:
                    path=source/e['path']
                    if path.is_file():
                        if path.is_symlink() or sha(path)!=e['sha256']:raise ValueError('Corrupt prepared source')
                    elif not path.exists():
                        if not path.resolve().is_relative_to(source.resolve()):raise ValueError('Escaping prepared source path')
                        path.parent.mkdir(parents=True,exist_ok=True)
                        with TemporaryDirectory(prefix='.repair_',dir=target) as repair:
                            temporary=Path(repair)/'source'
                            blob=self.root/'blobs'/e['sha256']
                            if blob.is_file() and sha(blob)==e['sha256']:
                                try:os.link(blob,temporary)
                                except OSError:temporary.write_bytes(contents[e['sha256']])
                            else:temporary.write_bytes(contents[e['sha256']])
                            temporary.replace(path)
                        self.repaired_files+=1
                    else:raise ValueError('Prepared source path is not a file')
                self.memo.pop(key,None);self.memo_bytes-=self.memo_costs.pop(key,0)
                self.verified.pop(key,None);value=self.read(key)
            if value['identity']!=identity: raise ValueError('Source corpus cache identity mismatch')
            self.hits+=1;return key,value
        target.parent.mkdir(parents=True,exist_ok=True)
        blobs=self.root/'blobs';blobs.mkdir(exist_ok=True)
        for token,body in contents.items():
            blob=blobs/token
            if blob.exists():
                signature=(blob.stat().st_mtime_ns,blob.stat().st_size)
                if self.blob_memo.get(token)!=signature and sha(blob)!=token:
                    raise ValueError('Corrupt source blob: '+str(blob))
            else:
                temporary=blob.with_suffix('.tmp');temporary.write_bytes(body);temporary.replace(blob)
                self.blobs_written+=1;self.bytes_written+=len(body)
            self.blob_memo[token]=(blob.stat().st_mtime_ns,blob.stat().st_size)
            self.blob_memo.move_to_end(token)
            while len(self.blob_memo)>16384:self.blob_memo.popitem(last=False)
        with TemporaryDirectory(prefix='.build_',dir=target.parent) as temporary:
            staging=Path(temporary);source=staging/'source';source.mkdir()
            for e in entries:
                path=source/e['path'];path.parent.mkdir(parents=True,exist_ok=True)
                try: os.link(blobs/e['sha256'],path)
                except OSError: path.write_bytes(contents[e['sha256']])
            envelope=dict(identity=identity,checksum=digest(identity))
            atomic_json(staging/'manifest.json',envelope)
            # Rename only the owned source tree; manifests never enter rg's corpus.
            ready=staging/'ready';ready.mkdir()
            source.rename(ready/'source');(staging/'manifest.json').rename(ready/'manifest.json')
            ready.rename(target)
        self.builds+=1
        return key,self.read(key)

    def read(self,key):
        if len(key)!=64 or any(c not in '0123456789abcdef' for c in key): raise ValueError('Invalid corpus key')
        directory=self.root/'corpora'/key;manifest=directory/'manifest.json';source=directory/'source'
        if not directory.exists():raise MissingCorpusError('Missing prepared source corpus: '+key)
        if directory.is_symlink() or not directory.resolve().is_relative_to(self.root):raise ValueError('Escaping source corpus')
        if not manifest.is_file():raise ValueError('Corrupt prepared corpus: missing manifest')
        if source.is_symlink():raise ValueError('Symlinked source corpus')
        if not source.is_dir():raise MissingCorpusError('Missing prepared source directory: '+key)
        cached=self.memo.get(key)
        if cached is not None:
            identity=cached['identity']
        else:
            raw=load_json(manifest);identity=raw['identity']
            if raw.get('checksum')!=digest(identity) or digest(identity)!=key or identity['version']!=VERSION:
                raise ValueError('Corrupt source corpus manifest')
        paths=[source/safe_path(e['path']) for e in identity['files']]
        for p in paths:
            if not p.exists():raise MissingCorpusError('Missing prepared source: '+str(p))
            if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(source.resolve()):
                raise ValueError('Invalid or escaping prepared source: '+str(p))
        directories={source}
        for p in paths:
            directories.update(parent for parent in p.parents if parent==source or source in parent.parents)
        signature=digest((manifest.stat().st_mtime_ns,manifest.stat().st_size,
            tuple((p.stat().st_mtime_ns,p.stat().st_size) for p in paths),
            [(str(p),p.stat().st_mtime_ns) for p in sorted(directories)]))
        if self.verified.get(key)!=signature:
            raw=load_json(manifest)
            if raw.get('identity')!=identity or raw.get('checksum')!=digest(identity): raise ValueError('Changed source corpus manifest')
            actual={p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
            if actual!={e['path'] for e in identity['files']}: raise ValueError('Unexpected source corpus file')
            for e,p in zip(identity['files'],paths):
                if p.stat().st_size!=e['size'] or sha(p)!=e['sha256']: raise ValueError('Corrupt prepared source: '+str(p))
        self.verified[key]=signature;self.verified.move_to_end(key)
        while len(self.verified)>16384:self.verified.popitem(last=False)
        value=dict(identity=identity,repository_path=str(source),_signature=signature)
        cost=self.memo_costs.get(key) or (1024+sum(1024+len(e['path'])*4 for e in identity['files']))
        self.memo_bytes-=self.memo_costs.get(key,0)
        self.memo_costs[key]=cost;self.memo_bytes+=cost
        self.memo[key]=value;self.memo.move_to_end(key)
        while self.memo and (len(self.memo)>self.max_entries or self.memo_bytes>self.max_bytes):
            removed,_=self.memo.popitem(last=False);self.memo_bytes-=self.memo_costs.pop(removed)
        return value

    def stats(self):
        return dict(cache_hits=self.hits,corpora_built=self.builds,blobs_written=self.blobs_written,
            bytes_written=self.bytes_written,repaired_files=self.repaired_files,memory_entries=len(self.memo),
            estimated_memory_bytes=self.memo_bytes)
