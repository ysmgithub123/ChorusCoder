"""RECCEVAL full snapshots from the existing archive/source SQLite cache."""
from contextlib import closing
import json
import sqlite3
from pathlib import Path

from choruscoder import digest
from benchmarking.aircoder_repository import Catalog
from benchmarking.prepare_recceval import open_sources,ArchiveSources,json_hash,file_hash
from benchmarking.source_corpus_cache import SourceCorpusCache


class CachedArchiveSources(ArchiveSources):
    """Read an already converted archive cache even if only SQLite is retained."""
    def __init__(self,root,archive_sha):
        location=root/'.recceval_source_cache'/(archive_sha+'.sqlite3')
        sidecar=json.loads(location.with_suffix('.json').read_text(encoding='utf8'))
        if sidecar!=dict(archive_sha256=archive_sha,cache_sha256=file_hash(location)):
            raise ValueError('Corrupt existing RECCEVAL source SQLite cache')
        self.connection=sqlite3.connect(location.resolve().as_uri()+'?mode=ro',uri=True)
        if dict(self.connection.execute('SELECT key,value FROM metadata'))!={'version':'recceval_archive_sources_v1','archive_sha256':archive_sha}:
            self.connection.close();raise ValueError('RECCEVAL SQLite source identity mismatch')
        self.archive_sha256=archive_sha


class ReccevalCatalog(Catalog):
    def __init__(self,args):
        self.source_cache=SourceCorpusCache(Path(args.cache_dir)/'recceval_sources')
        root=Path(getattr(args,'recceval_root','data_sources/recceval'))
        location=Path(args.parquet[0]).with_name('manifest.json')
        manifest=json.loads(location.read_text(encoding='utf8')) if location.is_file() else None
        archive_sha=manifest['identity'].get('source_archive_sha256') if manifest else None
        archive=root/'Source_Code.tar.gz'
        existing=root/'.recceval_source_cache'/(str(archive_sha)+'.sqlite3')
        if archive_sha and existing.is_file():
            if archive.is_file() and file_hash(archive)!=archive_sha:
                raise ValueError('RECCEVAL sources differ from converted dataset; reconvert before retrieval')
            provider=CachedArchiveSources(root,archive_sha)
        elif archive.is_file() or (root/'Source_Code').is_dir():
            provider=open_sources(root)
        elif manifest and manifest['identity'].get('source_archive_sha256'):
            provider=CachedArchiveSources(root,manifest['identity']['source_archive_sha256'])
        else:raise ValueError('Missing RECCEVAL source archive, directory and validated SQLite cache')
        with closing(provider) as source:
            self.raw_source=source
            inventory=source.inventory()
            self.snapshot=json_hash(inventory)
            if manifest and manifest['identity']['source_snapshot_sha256']!=self.snapshot:
                raise ValueError('RECCEVAL sources differ from converted dataset; reconvert before retrieval')
            super().__init__(args)
        self.raw_source=None
        self.source_cache_stats=self.source_cache.stats()

    def scan(self,root,repo):
        values,skipped=self.raw_source.repository(repo)
        files=[];texts=[]
        for value in values:
            path=value['path'];text=value['text']
            if not text.strip() or Path(path).name=='config_batch_dict.py':continue
            # The chunker sees logical paths; storage aliases also work for
            # template filenames that are not representable on Windows.
            physical='f_'+digest(path)+'.py'
            texts.append(dict(path=physical,text=text));files.append((path,physical))
        if not files:raise ValueError('Empty RECCEVAL AIRCoder repository: '+repo)
        key,cached=self.source_cache.prepare(texts)
        hashes={e['path']:e['sha256'] for e in cached['identity']['files']}
        records=[dict(path=path,physical_path=str(Path(cached['repository_path'])/physical),
            source_hash=hashes[physical]) for path,physical in files]
        snapshot=dict(kind='recceval_official_source',inventory_sha256=self.snapshot,corpus_key=key)
        return dict(files=records,skipped=skipped,snapshot=snapshot,provenance='recceval_official_source',
            fingerprint=digest(dict(repository=repo,snapshot=snapshot,
                files=[dict(path=f['path'],source_hash=f['source_hash']) for f in records])))
