"""Normalize official ReccEval sources once for the shared benchmark runner."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from tempfile import TemporaryDirectory


VERSION = 'recceval_normalized_v1'


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def relative_path(value):
    if not isinstance(value, str) or '\\' in value or ':' in value:
        raise ValueError(f'Invalid ReccEval relative path: {value!r}')
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError(f'Unsafe ReccEval relative path: {value!r}')
    return path


class ArchiveSources:
    """Read official archive bytes without creating executable source files."""
    def __init__(self, root):
        import sqlite3
        self.archive = root / 'Source_Code.tar.gz'
        archive_hash = file_hash(self.archive)
        cache = root / '.recceval_source_cache'
        cache.mkdir(exist_ok=True)
        location = cache / (archive_hash + '.sqlite3')
        cache_manifest = location.with_suffix('.json')
        self.connection = None
        try:
            cache_valid = (location.is_file() and cache_manifest.is_file() and
                json.loads(cache_manifest.read_text(encoding='utf-8')) ==
                dict(archive_sha256=archive_hash, cache_sha256=file_hash(location)))
        except (OSError, ValueError):
            cache_valid = False
        if cache_valid:
            connection = None
            try:
                connection = sqlite3.connect(location.as_uri() + '?mode=ro', uri=True)
                meta = dict(connection.execute('SELECT key, value FROM metadata'))
                if meta == {'version': 'recceval_archive_sources_v1', 'archive_sha256': archive_hash}:
                    self.connection = connection
                else:
                    connection.close()
            except sqlite3.DatabaseError:
                if connection is not None:
                    connection.close()
        if self.connection is None:
            with TemporaryDirectory(prefix='.import_', dir=cache) as directory:
                temporary = Path(directory) / 'sources.sqlite3'
                connection = sqlite3.connect(temporary)
                try:
                    connection.execute('CREATE TABLE files (path TEXT PRIMARY KEY, pkg TEXT NOT NULL, body BLOB NOT NULL, sha256 TEXT NOT NULL)')
                    connection.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
                    with tarfile.open(self.archive, 'r:gz') as source:
                        for member in source:
                            path = relative_path(member.name)
                            if path.parts[0] != 'Source_Code' or not (member.isdir() or member.isfile()):
                                raise ValueError(f'Unsupported ReccEval archive member: {member.name}')
                            if member.isdir():
                                continue
                            if len(path.parts) < 3:
                                raise ValueError(f'Source must belong to a package: {path}')
                            with source.extractfile(member) as incoming:
                                body = incoming.read()
                            logical = PurePosixPath(*path.parts[1:]).as_posix()
                            try:
                                connection.execute('INSERT INTO files VALUES (?, ?, ?, ?)',
                                    (logical, path.parts[1], body, hashlib.sha256(body).hexdigest()))
                            except sqlite3.IntegrityError as error:
                                raise ValueError(f'Duplicate archive source: {logical}') from error
                    connection.execute('CREATE INDEX files_pkg ON files(pkg)')
                    connection.executemany('INSERT INTO metadata VALUES (?, ?)',
                        [('version', 'recceval_archive_sources_v1'), ('archive_sha256', archive_hash)])
                    connection.commit()
                finally:
                    connection.close()
                os.replace(temporary, location)
                sidecar = Path(directory) / 'sources.json'
                sidecar.write_text(json.dumps(dict(archive_sha256=archive_hash, cache_sha256=file_hash(location))), encoding='utf-8')
                os.replace(sidecar, cache_manifest)
            self.connection = sqlite3.connect(location.as_uri() + '?mode=ro', uri=True)
        self.archive_sha256 = archive_hash

    def inventory(self):
        result = [dict(path=path, sha256=sha) for path, sha in
                  self.connection.execute("SELECT path, sha256 FROM files WHERE path LIKE '%.py' ORDER BY path")]
        if not result:
            raise ValueError('Archive has no Python sources')
        for item in result:
            relative_path(item['path'])
        return result

    def repository(self, pkg):
        files, skipped = [], []
        for logical, body, expected in self.connection.execute('SELECT path, body, sha256 FROM files WHERE pkg=? ORDER BY path', (pkg,)):
            if not logical.endswith('.py'):
                continue
            if hashlib.sha256(body).hexdigest() != expected:
                raise ValueError(f'Corrupt archived source cache: {logical}')
            try:
                text = body.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
            except UnicodeDecodeError:
                skipped.append(dict(path=logical, reason='invalid_utf8'))
                continue
            files.append(dict(path=PurePosixPath(*relative_path(logical).parts[1:]).as_posix(), text=text))
        return files, skipped

    def close(self):
        self.connection.close()


class DirectorySources:
    def __init__(self, root):
        self.root = root
        self.groups = defaultdict(list)

    def inventory(self):
        result = []
        for folder, dirs, names in os.walk(self.root, followlinks=False):
            base = Path(folder)
            if any((base / name).is_symlink() for name in dirs + names):
                raise ValueError(f'Symlink in ReccEval source tree: {base}')
            for name in names:
                if not name.endswith('.py'):
                    continue
                path = base / name
                logical = path.relative_to(self.root).as_posix()
                parts = relative_path(logical).parts
                if len(parts) < 2:
                    raise ValueError(f'Source must belong to a package: {logical}')
                sha = file_hash(path)
                result.append(dict(path=logical, sha256=sha))
                self.groups[parts[0]].append((path, PurePosixPath(*parts[1:]).as_posix(), sha))
        return sorted(result, key=lambda item: item['path'])

    def repository(self, pkg):
        files, skipped = [], []
        for path, relative, expected in sorted(self.groups.get(pkg, ()), key=lambda item: item[1]):
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != expected:
                raise ValueError(f'Source changed during preparation: {path}')
            try:
                text = body.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
            except UnicodeDecodeError:
                skipped.append(dict(path=f'{pkg}/{relative}', reason='invalid_utf8'))
                continue
            files.append(dict(path=relative, text=text))
        return files, skipped

    def close(self):
        pass


def open_sources(root, sources_dir=None):
    root = Path(root).resolve()
    if sources_dir is None and (root / 'Source_Code.tar.gz').is_file():
        return ArchiveSources(root)
    directory = Path(sources_dir) if sources_dir is not None else root / 'Source_Code'
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f'Expected regular source directory: {directory}')
    return DirectorySources(directory)


def prepare(recceval_root, output_root='data', sources_dir=None):
    from contextlib import closing
    root = Path(recceval_root)
    if not (root / 'metadata.jsonl').is_file():
        raise FileNotFoundError(root / 'metadata.jsonl')
    with closing(open_sources(root, sources_dir)) as source:
        return _prepare(source, root / 'metadata.jsonl', output_root)


def _prepare(source, metadata, output_root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    snapshot = source.inventory()
    source_identity = dict(version=VERSION, implementation_sha256=file_hash(__file__), metadata_sha256=file_hash(metadata),
                           source_snapshot_sha256=json_hash(snapshot), source_archive_sha256=getattr(source, 'archive_sha256', None))
    output = Path(output_root) / 'recceval' / 'python'
    manifest_path = output / 'manifest.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (previous.get('identity') == source_identity and
                all((output / name).is_file() and file_hash(output / name) == expected
                    for name, expected in previous.get('outputs', {}).items()) and
                set(previous.get('outputs', {})) == {'test.parquet', 'test.jsonl'}):
            return dict(previous, reused=True)

    packages = defaultdict(list)
    count = 0
    with metadata.open(encoding='utf-8') as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            pkg = str(item['pkg'])
            if relative_path(pkg).parts != (pkg,):
                raise ValueError(f'Invalid package: {pkg}')
            path = relative_path(item['fpath'])
            if len(path.parts) < 2 or path.parts[0] != pkg:
                raise ValueError(f'Package/path mismatch: {pkg}/{path}')
            prefix, target = item['input'], item['gt']
            if not isinstance(prefix, str) or not isinstance(target, str) or '\n' in target or '\r' in target:
                raise ValueError(f'Expected current-line text at metadata row {source_index}')
            packages[pkg].append(dict(task_id=f'{pkg}/{len(packages[pkg])}',
                path=PurePosixPath(*path.parts[1:]).as_posix(), prefix=prefix, target=target,
                source_index=source_index))
            count += 1
    if not count:
        raise ValueError('ReccEval metadata is empty')

    schema = pa.schema([('task_id', pa.string()), ('path', pa.string()),
                        ('left_context', pa.string()), ('right_context', pa.string()),
                        ('crossfile_context', pa.list_(pa.struct([('path', pa.string()), ('text', pa.string())]))),
                        ('groundtruth', pa.string())])
    output.mkdir(parents=True, exist_ok=True)
    skipped = []
    # Temporary files are on the output filesystem for atomic replacement.
    with TemporaryDirectory(prefix='.prepare_', dir=output) as directory:
        staging = Path(directory)
        with pq.ParquetWriter(staging / 'test.parquet', schema, compression='zstd') as parquet, \
                (staging / 'test.jsonl').open('w', encoding='utf-8', newline='\n') as reference:
            for pkg, samples in packages.items():
                files, omitted = source.repository(pkg)
                skipped.extend(omitted)
                contents = {file['path']: file['text'] for file in files}
                batch = []
                for item in samples:
                    code = contents.get(item['path'])
                    if code is None or not code.startswith(item['prefix'] + item['target']):
                        raise ValueError(f"Source does not match input/gt: {item['task_id']} {item['path']}")
                    # The full target file, including both future and past code,
                    # is excluded; only the official prefix becomes query input.
                    related = [file for file in files if file['path'] != item['path']]
                    row = dict(task_id=item['task_id'], path=item['path'], left_context=item['prefix'],
                               right_context='', crossfile_context=related, groundtruth=item['target'])
                    batch.append(row)
                    if len(batch) == 8:
                        parquet.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                    ref = dict(prompt=item['prefix'], groundtruth=item['target'], right_context='',
                               metadata=dict(task_id=item['task_id'], repository=pkg, file=item['path'],
                                             source_index=item['source_index']))
                    reference.write(json.dumps(ref, ensure_ascii=False) + '\n')
                if batch:
                    parquet.write_table(pa.Table.from_pylist(batch, schema=schema))
        manifest = dict(identity=source_identity, total_samples=count, repository_count=len(packages),
                        source_files=snapshot, skipped_files=skipped,
                        outputs={name: file_hash(staging / name) for name in ('test.parquet', 'test.jsonl')})
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        for name in ('test.parquet', 'test.jsonl', 'manifest.json'):
            os.replace(staging / name, output / name)
    return dict(manifest, reused=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recceval-root', default='data_sources/recceval')
    parser.add_argument('--output-root', default='data')
    parser.add_argument('--sources-dir', help='Explicit extracted source directory; otherwise prefer the official archive')
    args = parser.parse_args()
    result = prepare(args.recceval_root, args.output_root, args.sources_dir)
    print(json.dumps({key: result[key] for key in ('total_samples', 'repository_count', 'skipped_files', 'reused')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
