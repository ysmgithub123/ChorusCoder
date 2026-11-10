import hashlib
import json
import os
import re
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
LANGUAGES = ('python',)
BENCHMARKS = ('cceval', 'recceval')
MARKDOWN_URL = re.compile('^\\[(https?://[^\\]]+)\\]\\((https?://[^)]+)\\)$')

@dataclass(frozen=True)
class BenchmarkTarget:
    """One independently runnable benchmark partition."""
    benchmark: str
    language: str
    subset: str = ''

    @property
    def data_key(self):
        return self.subset or self.language

def validate_generation_tokens(value):
    """Validate a resolved generation limit, independent of retrieval profiles."""
    if type(value) is not int or value < 1:
        raise ValueError('max-new-tokens must be a positive integer')
    return value

def benchmark_targets(benchmark, language, subset='all'):
    """Resolve the two Python benchmarks used by the paper."""
    if benchmark not in BENCHMARKS or language != 'python' or subset not in (None, '', 'all'):
        raise ValueError('Select cceval/python or recceval/python')
    return (BenchmarkTarget(benchmark, 'python'),)

def model_slug(model_name):
    value = model_name.rstrip('/\\').split('/')[-1].split('\\')[-1]
    return re.sub('[^A-Za-z0-9_.-]+', '_', value)

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def normalize_api_base(value):
    """Return a plain HTTP(S) API base, including when copied from rich text."""
    value = str(value).strip()
    markdown_match = MARKDOWN_URL.fullmatch(value)
    if markdown_match:
        label, target = markdown_match.groups()
        if label != target:
            raise ValueError(f'The vLLM API URL label and target differ: {value}')
        value = target
    if not re.fullmatch('https?://[^\\s]+', value):
        raise ValueError(f'Invalid vLLM API base URL: {value!r}')
    return value.rstrip('/')

def normalize_crossfile_context(value, task_id=''):
    """Normalize Parquet/JSON cross-file values to a list of dictionaries."""
    if value is None:
        return []
    if hasattr(value, 'as_py'):
        value = value.as_py()
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f'Invalid crossfile_context JSON for task {task_id!r}') from error
    if hasattr(value, 'tolist') and (not isinstance(value, (dict, list, tuple))):
        value = value.tolist()
    if isinstance(value, dict):
        value = [value]
    else:
        try:
            value = list(value)
        except TypeError as error:
            raise ValueError(f'crossfile_context for task {task_id!r} must be a sequence, got {type(value).__name__}') from error
    normalized = []
    for index, item in enumerate(value):
        if hasattr(item, 'as_py'):
            item = item.as_py()
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid cross-file item JSON at index {index} for task {task_id!r}') from error
        if not isinstance(item, dict):
            try:
                item = dict(item)
            except (TypeError, ValueError) as error:
                raise ValueError(f'Cross-file item {index} for task {task_id!r} must be a mapping, got {type(item).__name__}') from error
        normalized.append(item)
    return normalized

def iter_parquet_batches(paths, columns, batch_size, limit=None):
    """Yield rows across one or more Parquet shards without materializing them."""
    if batch_size < 1:
        raise ValueError('batch_size must be at least 1')
    if limit is not None and limit < 0:
        raise ValueError('limit must be non-negative')
    import pyarrow.parquet as pq
    if isinstance(paths, (str, os.PathLike)):
        paths = (paths,)
    else:
        paths = tuple(paths)
    if not paths:
        raise ValueError('At least one Parquet path is required')
    columns = tuple(columns)
    row_type = namedtuple('ParquetRow', columns)
    remaining = limit
    for path in paths:
        if remaining is not None and remaining <= 0:
            break
        parquet_file = pq.ParquetFile(str(path))
        available = set(parquet_file.schema_arrow.names)
        missing = set(columns) - available
        if missing:
            raise ValueError(f'Dataset shard {path} is missing required columns: {sorted(missing)}')
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(columns)):
            if remaining is not None and remaining <= 0:
                break
            values_by_column = record_batch.to_pydict()
            row_count = record_batch.num_rows
            if remaining is not None:
                row_count = min(row_count, remaining)
            rows = [row_type(*(values_by_column[column][index] for column in columns)) for index in range(row_count)]
            del values_by_column, record_batch
            if rows:
                yield rows
            if remaining is not None:
                remaining -= row_count

def write_json(path, value):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)

def write_jsonl(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

class AtomicJsonlWriter:
    """Stream JSONL rows without exposing a partial final output file."""

    def __init__(self, path):
        self.path = Path(path)
        self.temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        self.handle = None

    def __enter__(self):
        ensure_dir(self.path.parent)
        self.handle = self.temporary.open('w', encoding='utf-8')
        return self

    def write(self, row):
        if self.handle is None:
            raise RuntimeError('AtomicJsonlWriter is not open')
        self.handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if exc_type is None:
            os.replace(self.temporary, self.path)
        else:
            self.temporary.unlink(missing_ok=True)
        return False

def read_jsonl(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def stable_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def resolve_dataset_shards(dataset_root, benchmark, data_key):
    root = Path(dataset_root) / benchmark / data_key
    parquet = root / 'test.parquet'
    reference = root / 'test.jsonl'
    if parquet.exists():
        parquets = (parquet,)
    else:

        def shard_key(path):
            match = re.fullmatch('test_(\\d+)\\.parquet', path.name)
            return (0, int(match.group(1))) if match else (1, path.name)
        parquets = tuple(sorted(root.glob('test_*.parquet'), key=shard_key))
    if not parquets:
        raise FileNotFoundError(f'Dataset parquet not found: expected {parquet} or test_*.parquet shards in {root}')
    if not reference.exists():
        raise FileNotFoundError(f'Evaluation reference not found: {reference}')
    return (parquets, reference)

def task_id_from_reference(row):
    metadata = row.get('metadata') or {}
    return str(metadata.get('task_id', row.get('task_id', '')))
