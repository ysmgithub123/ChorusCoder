"""Ordered full-snapshot catalog. No completion labels enter retrieval requests."""
import hashlib
import json
import os
import time
from pathlib import Path, PurePosixPath

from benchmarking.common import iter_parquet_batches
from choruscoder import digest

COLUMNS = ('task_id', 'path', 'left_context')


def logical_path(value):
    value = str(value).replace('\\', '/')
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or ':' in path.parts[0]:
        raise ValueError('Invalid repository-relative path: '+value)
    return path.as_posix()


def source_text(file):
    data = Path(file['physical_path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != file['source_hash']:
        raise ValueError('Repository changed during preprocessing: '+file['physical_path'])
    # Match upstream utils.read_file's UTF-8 text-mode newline conversion.
    return data.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')


class Catalog:
    def __init__(self, args):
        self.repositories, self.tasks = {}, {}
        reference = Path(args.metadata) if getattr(args, 'metadata', None) else Path(args.parquet[0]).with_name('test.jsonl')
        metadata = {}
        with reference.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                m = row.get('metadata', row)
                task = str(m.get('task_id', row.get('task_id', '')))
                if not task or task in metadata:
                    raise ValueError('Missing or duplicate task metadata: '+task)
                metadata[task] = (m.get('repository'), m.get('file'))
        self.reference = str(reference.resolve())
        root = Path(args.repos_root).resolve()
        for batch in iter_parquet_batches(args.parquet, COLUMNS, args.batch_size, args.limit):
            for row in batch:
                task, path = str(row.task_id), logical_path(row.path)
                if task in self.tasks or task not in metadata:
                    raise ValueError('Missing mapping or duplicate input task: '+task)
                repo, mapped = metadata[task]
                if not repo or logical_path(mapped) != path or Path(str(repo)).name != str(repo):
                    raise ValueError('Repository/file metadata mismatch: '+task)
                if repo not in self.repositories:
                    self.repositories[repo] = self.scan(root, repo)
                repository = self.repositories[repo]
                if path not in {f['path'] for f in repository['files']}:
                    raise ValueError('Current file missing or excluded from full snapshot: '+task+' '+path)
                self.tasks[task] = dict(repository=repo, path=path)
        if not self.tasks:
            raise ValueError('No AIRCoder samples')

    @staticmethod
    def scan(root, repo):
        directory = (root/repo).resolve()
        if directory.parent != root or not directory.is_dir():
            raise FileNotFoundError('Repository snapshot missing: '+str(directory))
        snapshot_path = directory/'_snapshot.json'
        snapshot = json.loads(snapshot_path.read_text(encoding='utf-8')) if snapshot_path.exists() else {}
        mapping = snapshot.get('portable_path_map', {})
        files, skipped, seen = [], [], set()
        # Deliberately unsorted: upstream get_repo_files follows os.walk order.
        for current, _, names in os.walk(directory):
            for name in names:
                physical = Path(current)/name
                if physical.is_symlink() or not name.endswith('.py') or name == 'config_batch_dict.py':
                    continue
                if not physical.resolve().is_relative_to(directory):
                    raise ValueError('Source path escapes repository: '+str(physical))
                relative = physical.relative_to(directory).as_posix()
                logical = logical_path(mapping.get(relative, relative))
                if logical in seen:
                    raise ValueError('Duplicate logical snapshot file: '+logical)
                seen.add(logical)
                data = physical.read_bytes()
                try:
                    text = data.decode('utf-8')
                except UnicodeDecodeError:
                    skipped.append(dict(path=logical, reason='upstream_utf8_read_failure'))
                    continue
                if not text.strip():
                    skipped.append(dict(path=logical, reason='upstream_empty_file'))
                    continue
                files.append(dict(path=logical, physical_path=str(physical.resolve()),
                                  source_hash=hashlib.sha256(data).hexdigest()))
        if not files:
            raise ValueError('Empty repository snapshot: '+str(directory))
        provenance = snapshot.get('snapshot_kind', 'exact_github_revision' if snapshot.get('revision') else 'unmarked_local_snapshot')
        identity = dict(repository=repo, snapshot=snapshot,
                        ordered_files=[{k:f[k] for k in ('path','source_hash')} for f in files], skipped=skipped)
        return dict(files=files, provenance=provenance, snapshot=snapshot,
                    fingerprint=digest(identity), skipped=skipped)

    def identity(self):
        return dict(tasks=self.tasks, repositories={name:dict(fingerprint=r['fingerprint'],
            provenance=r['provenance'], files=len(r['files']), skipped=r['skipped'])
            for name,r in self.repositories.items()})

    def rows(self, args):
        for batch in iter_parquet_batches(args.parquet, COLUMNS, args.batch_size, args.limit):
            result = []
            for row in batch:
                task = str(row.task_id)
                mapped = self.tasks[task]
                if logical_path(row.path) != mapped['path'] or not isinstance(row.left_context, str):
                    raise ValueError('Invalid AIRCoder input: '+task)
                repo = self.repositories[mapped['repository']]
                result.append(dict(task_id=task, path=mapped['path'], prefix=row.left_context,
                    repository=mapped['repository'], repository_fingerprint=repo['fingerprint'],
                    snapshot_provenance=repo['provenance'], files=repo['files']))
            yield result


def catalog(args):
    if not hasattr(args, '_repository_catalog'):
        started = time.perf_counter()
        if getattr(args,'benchmark','cceval')=='recceval':
            from benchmarking.aircoder_recceval import ReccevalCatalog
            args._repository_catalog=ReccevalCatalog(args)
        else: args._repository_catalog = Catalog(args)
        args._catalog_seconds = time.perf_counter()-started
    return args._repository_catalog
