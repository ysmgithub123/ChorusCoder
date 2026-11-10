import argparse
import json
import os
from pathlib import Path
import pandas as pd
from benchmarking.common import BENCHMARKS, LANGUAGES, ensure_dir, normalize_crossfile_context, task_id_from_reference, write_jsonl
EXTENSIONS = {'python': ('.py',)}

def parse_args():
    parser = argparse.ArgumentParser(description='Normalize benchmark data for all method runners')
    parser.add_argument('--input', required=True, help='Canonical benchmark JSONL')
    parser.add_argument('--repos-root', help='Repository snapshots; needed when JSONL has no crossfile_context')
    parser.add_argument('--output-root', default='data')
    parser.add_argument('--benchmark', choices=BENCHMARKS, default='cceval')
    parser.add_argument('--language', required=True, choices=LANGUAGES)
    return parser.parse_args()

def repository_files(repos_root, repository, language):
    repo_dir = Path(repos_root) / repository
    if not repo_dir.is_dir():
        raise FileNotFoundError(f'Repository snapshot not found: {repo_dir}')
    extensions = EXTENSIONS[language]
    portable_path_map = {}
    snapshot_metadata = repo_dir / '_snapshot.json'
    if snapshot_metadata.exists():
        with snapshot_metadata.open('r', encoding='utf-8') as handle:
            portable_path_map = json.load(handle).get('portable_path_map', {})
    files = []
    for root, dirs, names in os.walk(repo_dir):
        dirs[:] = [name for name in dirs if name not in {'.git', 'node_modules', 'vendor', 'dist', 'build'}]
        for name in names:
            if name.endswith(extensions):
                absolute = Path(root) / name
                stored_relative = absolute.relative_to(repo_dir).as_posix()
                relative = portable_path_map.get(stored_relative, stored_relative)
                try:
                    text = absolute.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    text = absolute.read_text(encoding='utf-8', errors='ignore')
                files.append({'path': relative, 'text': text})
    if not files:
        raise ValueError(f'No {language} files found in {repo_dir}')
    return files

def main():
    args = parse_args()
    if args.benchmark == 'recceval' and args.language != 'python':
        raise ValueError('ReccEval is Python-only')
    data_key = args.language
    with Path(args.input).open('r', encoding='utf-8') as handle:
        source = [json.loads(line) for line in handle if line.strip()]
    output_dir = Path(args.output_root) / args.benchmark / data_key
    excluded_task_ids = set()
    repo_cache = {}
    normalized = []
    references = []
    for item in source:
        metadata = dict(item.get('metadata') or {})
        task_id = task_id_from_reference(item)
        if not task_id:
            raise ValueError('Every sample must contain metadata.task_id or task_id')
        if task_id in excluded_task_ids:
            continue
        path = metadata.get('file', item.get('path'))
        if not path:
            raise ValueError(f'Sample {task_id} does not contain metadata.file or path')
        crossfiles = item.get('crossfile_context')
        if crossfiles is None:
            repository = metadata.get('repository')
            if not repository or not args.repos_root:
                raise ValueError(f'Sample {task_id} has no crossfile_context. Both metadata.repository and --repos-root are required.')
            cache_key = (repository, args.language)
            if cache_key not in repo_cache:
                repo_cache[cache_key] = repository_files(args.repos_root, repository, args.language)
            crossfiles = [row for row in repo_cache[cache_key] if row['path'] != str(path).replace('\\', '/')]
        crossfiles = normalize_crossfile_context(crossfiles, task_id)
        normalized.append({'task_id': task_id, 'path': str(path), 'left_context': str(item.get('prompt', item.get('left_context', ''))), 'right_context': str(item.get('right_context', '')), 'crossfile_context': list(crossfiles), 'groundtruth': str(item['groundtruth'])})
        reference = dict(item)
        reference['prompt'] = normalized[-1]['left_context']
        reference['groundtruth'] = normalized[-1]['groundtruth']
        reference['metadata'] = metadata
        reference['metadata']['task_id'] = task_id
        reference['metadata'].setdefault('file', str(path))
        references.append(reference)
    output_dir = ensure_dir(output_dir)
    pd.DataFrame(normalized).to_parquet(output_dir / 'test.parquet', index=False)
    write_jsonl(output_dir / 'test.jsonl', references)
    print(f'Prepared {len(normalized)} samples in {output_dir}')
if __name__ == '__main__':
    main()
