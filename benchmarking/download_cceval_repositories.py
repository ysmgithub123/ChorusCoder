import argparse
import concurrent.futures
import json
import os
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from benchmarking.common import ensure_dir, write_json, write_jsonl
EXTENSIONS = {'python': ('.py',)}

def parse_args():
    parser = argparse.ArgumentParser(description='Download the exact GitHub source snapshots referenced by CrossCodeEval')
    parser.add_argument('--official-root', default='data_sources/crosscodeeval_official/extracted', help='Extracted official CrossCodeEval directory')
    parser.add_argument('--repos-root', default='data_sources/crosscodeeval_repositories', help='Destination for language-filtered repository snapshots')
    parser.add_argument('--download-dir', default='data_sources/crosscodeeval_downloads', help='Temporary archive directory')
    parser.add_argument('--languages', nargs='+', choices=sorted(EXTENSIONS), default=list(EXTENSIONS))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--retries', type=int, default=4)
    parser.add_argument('--repository-id', action='append', help='Download only the named dataset repository id; repeat as needed')
    return parser.parse_args()

def load_jsonl(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def load_repository_map(official_root):
    path = Path(official_root) / 'LICENSES' / 'project_license_map.txt'
    mapping = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            full_name = line.split(' ', 1)[0].strip()
            if not full_name or '/' not in full_name:
                continue
            flattened = full_name.replace('/', '-')
            if flattened in mapping and mapping[flattened] != full_name:
                raise ValueError(f'Ambiguous flattened repository name: {flattened}')
            mapping[flattened] = full_name
    return mapping

def collect_repositories(official_root, languages):
    official_root = Path(official_root)
    repository_map = load_repository_map(official_root)
    repositories = {}
    for language in languages:
        source = official_root / language / 'line_completion.jsonl'
        for row in load_jsonl(source):
            repository_id = str(row['metadata']['repository'])
            match = re.fullmatch('(.+)-([0-9a-fA-F]{7,40})', repository_id)
            if not match:
                raise ValueError(f'Unexpected repository identifier: {repository_id}')
            flattened, revision = match.groups()
            full_name = repository_map.get(flattened)
            if not full_name:
                raise KeyError(f'No GitHub mapping for {repository_id}')
            current = repositories.get(repository_id)
            value = {'repository_id': repository_id, 'github_repository': full_name, 'revision': revision, 'language': language, 'extensions': EXTENSIONS[language]}
            if current and current != value:
                raise ValueError(f'Conflicting metadata for {repository_id}')
            repositories[repository_id] = value
    return [repositories[key] for key in sorted(repositories)]

def completed_snapshot(target, item):
    metadata_path = target / '_snapshot.json'
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open('r', encoding='utf-8') as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get('github_repository') != item['github_repository'] or metadata.get('revision') != item['revision'] or metadata.get('language') != item['language']:
        return None
    return metadata

def download_archive(item, destination, timeout, retries):
    url = f"https://codeload.github.com/{item['github_repository']}/tar.gz/{item['revision']}"
    request = urllib.request.Request(url, headers={'User-Agent': 'CrossCodeEval-research-snapshot-downloader/1.0', 'Accept': 'application/octet-stream'})
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if getattr(response, 'status', 200) != 200:
                    raise RuntimeError(f'HTTP {response.status}')
                with destination.open('wb') as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            return url
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            if destination.exists():
                destination.unlink()
            if attempt < retries:
                time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f'Download failed after {retries} attempts: {last_error}')

def portable_parts(relative):
    path_map = {}
    output = []
    invalid = re.compile('[<>:"\\\\|?*]')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{number}' for number in range(1, 10)), *(f'LPT{number}' for number in range(1, 10))}
    for part in relative.parts:
        portable = invalid.sub(lambda match: f'~{ord(match.group(0)):02X}~', part)
        while portable.endswith((' ', '.')):
            portable = portable[:-1] + f'~{ord(portable[-1]):02X}~'
        if portable.split('.', 1)[0].upper() in reserved:
            portable = '~RESERVED~' + portable
        output.append(portable)
    portable_relative = PurePosixPath(*output)
    if portable_relative != relative:
        path_map[portable_relative.as_posix()] = relative.as_posix()
    return (output, path_map)

def extract_source_files(archive_path, partial_dir, extensions):
    file_count = 0
    source_bytes = 0
    path_map = {}
    with tarfile.open(archive_path, mode='r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            archive_path_value = PurePosixPath(member.name)
            parts = archive_path_value.parts
            if len(parts) < 2:
                continue
            relative = PurePosixPath(*parts[1:])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unsafe archive member: {member.name}')
            if not str(relative).lower().endswith(tuple((ext.lower() for ext in extensions))):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            destination_parts, member_path_map = portable_parts(relative)
            path_map.update(member_path_map)
            destination = partial_dir.joinpath(*destination_parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open('wb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            file_count += 1
            source_bytes += member.size
    if file_count == 0:
        raise ValueError(f'Archive contains no source files matching {extensions}')
    return (file_count, source_bytes, path_map)

def safe_remove_partial(path, allowed_root):
    path = path.resolve()
    allowed_root = allowed_root.resolve()
    if path.parent != allowed_root or not path.name.endswith('.partial'):
        raise ValueError(f'Refusing to remove unexpected path: {path}')
    if path.exists():
        shutil.rmtree(path)

def process_repository(item, repos_root, download_dir, timeout, retries):
    target = repos_root / item['repository_id']
    existing = completed_snapshot(target, item)
    if existing:
        return {**item, **existing, 'status': 'existing'}
    archive_path = download_dir / f"{item['repository_id']}.tar.gz"
    partial_dir = repos_root / f"{item['repository_id']}.partial"
    safe_remove_partial(partial_dir, repos_root)
    partial_dir.mkdir(parents=True)
    try:
        url = download_archive(item, archive_path, timeout, retries)
        archive_bytes = archive_path.stat().st_size
        file_count, source_bytes, path_map = extract_source_files(archive_path, partial_dir, item['extensions'])
        metadata = {'repository_id': item['repository_id'], 'github_repository': item['github_repository'], 'revision': item['revision'], 'language': item['language'], 'source_extensions': list(item['extensions']), 'source_file_count': file_count, 'source_bytes': source_bytes, 'download_bytes': archive_bytes, 'source_url': url, 'portable_path_map': path_map, 'snapshot_kind': 'exact_github_revision'}
        write_json(partial_dir / '_snapshot.json', metadata)
        if target.exists():
            raise FileExistsError(f'Target exists without matching snapshot metadata: {target}')
        os.replace(partial_dir, target)
        return {**item, **metadata, 'status': 'downloaded'}
    finally:
        if archive_path.exists():
            archive_path.unlink()
        if partial_dir.exists():
            safe_remove_partial(partial_dir, repos_root)

def main():
    args = parse_args()
    official_root = Path(args.official_root).resolve()
    repos_root = ensure_dir(Path(args.repos_root).resolve())
    download_dir = ensure_dir(Path(args.download_dir).resolve())
    repositories = collect_repositories(official_root, args.languages)
    if args.repository_id:
        selected = set(args.repository_id)
        repositories = [item for item in repositories if item['repository_id'] in selected]
        missing = selected - {item['repository_id'] for item in repositories}
        if missing:
            raise ValueError(f'Unknown repository ids: {sorted(missing)}')
    print(f'Preparing {len(repositories)} exact repository snapshots', flush=True)
    results = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_repository, item, repos_root, download_dir, args.timeout, args.retries): item for item in repositories}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                print(f"[{completed}/{len(repositories)}] {result['status']}: {item['repository_id']} ({result['source_file_count']} files)", flush=True)
            except Exception as error:
                failure = {**item, 'status': 'failed', 'error': str(error)}
                failures.append(failure)
                print(f"[{completed}/{len(repositories)}] FAILED: {item['repository_id']}: {error}", flush=True)
    current_failures = {row['repository_id']: row for row in failures}
    manifest = []
    successful_rows = []
    failed_rows = []
    for item in repositories:
        metadata = completed_snapshot(repos_root / item['repository_id'], item)
        if metadata:
            row = {**item, **metadata, 'status': 'available'}
            manifest.append(row)
            successful_rows.append(row)
        else:
            row = current_failures.get(item['repository_id'], {**item, 'status': 'failed', 'error': 'Exact GitHub revision is no longer publicly available'})
            manifest.append(row)
            failed_rows.append(row)
    manifest.sort(key=lambda row: row['repository_id'])
    write_jsonl(repos_root / 'snapshot_manifest.jsonl', manifest)
    summary = {'requested': len(repositories), 'successful': len(successful_rows), 'failed': len(failed_rows), 'source_files': sum((row.get('source_file_count', 0) for row in successful_rows)), 'source_bytes': sum((row.get('source_bytes', 0) for row in successful_rows)), 'failures': failed_rows}
    write_json(repos_root / 'snapshot_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failed_rows:
        raise SystemExit(2)
if __name__ == '__main__':
    main()
