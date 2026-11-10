"""Load only unmodified, SHA256-pinned upstream function definitions.

The upstream CLI modules have API clients, logging and heavyweight imports at
module scope. Selecting AST definitions avoids those unrelated side effects;
the selected function bodies, prompt constants and globals are unchanged.
"""
import ast
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from greprag_profiles import OFFICIAL_COMMIT, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_source(root=None):
    root = Path(root or ROOT / 'third_party/GrepRAG')
    manifest = json.loads((ROOT / 'benchmarking/greprag_upstream.json').read_text(encoding='utf-8'))
    if manifest['commit'] != OFFICIAL_COMMIT:
        raise ValueError('GrepRAG upstream commit mismatch')
    for name, expected in manifest['files'].items():
        if not (root / name).is_file() or file_hash(root / name) != expected:
            raise ValueError('Missing or changed official GrepRAG source: ' + str(root / name))
    return manifest


def validate_model(root):
    root = Path(root)
    pin = json.loads((ROOT / 'benchmarking/greprag_model.json').read_text(encoding='utf-8'))
    if pin['revision'] != MODEL_REVISION:
        raise ValueError('GrepRAG model revision mismatch')
    for name, expected in pin['files'].items():
        if not (root / name).is_file() or file_hash(root / name) != expected:
            raise ValueError('Missing or changed official GrepRAG distilled model file: ' + name)
    return pin


def load_definitions(path, names, namespace=None):
    namespace = {} if namespace is None else namespace
    tree = ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))
    selected = []
    found = set()
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            selected.append(node)
            found.add(name)
    if found != set(names):
        raise ValueError('Missing official definitions: ' + str(set(names) - found))
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def upstream(root=None):
    from rank_bm25 import BM25Okapi
    import tiktoken
    root = Path(root or ROOT / 'third_party/GrepRAG')
    validate_source(root)
    keyword = load_definitions(root / 'grep_train/qwen_generate_keywords.py',
        ('INSTRUCTION', '_build_chat_prompt', '_extract_keywords', '_shell_single_quote', '_build_rg_commands'),
        dict(re=re))
    retrieval = load_definitions(root / 'grep_dedup/deepseek_generate_grep.py',
        ('run_command_in_repo', 'run_commands_in_repo', 'parse_rg_json', 'count_tokens',
         'select_matches_by_token_budget', 'merge_overlapping_segments', 'rerank_matches'),
        dict(os=os, json=json, re=re, subprocess=subprocess, tempfile=tempfile,
             concurrent=concurrent, time=time, GREP_TIMEOUT=30,
             BM25Okapi=BM25Okapi, ENCODER=tiktoken.get_encoding('cl100k_base')))
    # Reuse the official CFC assembly statements too, including its literal
    # string escapes. Do not silently "repair" the published prompt format.
    tree = ast.parse((root / 'grep_dedup/deepseek_generate_grep.py').read_text(encoding='utf-8'))
    process = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_dataset')
    loop = process.body[2].body[0]  # with input/output streams -> for each sample
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'retrieved_list' for t in n.targets))
    end = next(i for i, n in enumerate(loop.body[start:], start) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                       and t.slice.value == 'crossfile_context' for t in n.targets))
    body = loop.body[start:end + 1]
    fn = ast.FunctionDef(name='build_context', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg='data'), ast.arg(arg='repo_path')], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[*body, ast.Return(value=ast.Subscript(value=ast.Name(id='data', ctx=ast.Load()),
            slice=ast.Constant(value='crossfile_context'), ctx=ast.Load()))], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(root / 'grep_dedup/deepseek_generate_grep.py'), 'exec'), retrieval)
    return keyword, retrieval
