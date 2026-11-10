"""Thin adaptation around pinned GrepRAG functions; no retrieval optimizations."""
from pathlib import Path, PurePosixPath
import json

from .common import normalize_crossfile_context


def safe_path(value):
    value = str(value).replace('\\', '/')
    p = PurePosixPath(value)
    if (not value or str(p) == '.' or p.is_absolute() or '..' in p.parts or
            ':' in value or '\x00' in value or '\n' in value or '\r' in value):
        raise ValueError('Unsafe repository-relative path: ' + repr(value))
    return str(p)


def request(row, reference):
    meta = reference.get('metadata') or {}
    current = safe_path(row.path)
    if (str(meta.get('task_id', reference.get('task_id', ''))) != str(row.task_id) or
            safe_path(meta.get('file', '')) != current or reference.get('prompt') != row.left_context):
        raise ValueError('CCEVAL metadata/Parquet task order, path or prefix mismatch')
    repo = safe_path(meta.get('repository', ''))
    if '/' in repo:
        raise ValueError('Repository ID must be a single directory name')
    # Neither groundtruth nor right_context is forwarded to an upstream function.
    return dict(task_id=str(row.task_id), path=current, repository=repo, prefix=str(row.left_context))




def parquet_files(row):
    """Same source normalization as the ChorusCoder runner's safe_request."""
    current = safe_path(row.path)
    files = {}
    for item in normalize_crossfile_context(row.crossfile_context, row.task_id):
        if not item.get('path'):
            continue
        path = safe_path(item['path'])
        if path == current:
            continue
        text = str(item.get('text', ''))
        if path in files and files[path] != text:
            raise ValueError('Conflicting source copies: ' + path)
        files[path] = text
    paths = set(files)
    for path in paths:
        if any(str(parent) in paths for parent in PurePosixPath(path).parents if str(parent) != '.'):
            raise ValueError('Parquet file/directory path conflict: ' + path)
    return [dict(path=path,text=text) for path,text in sorted(files.items())]




class KeywordModel:
    def __init__(self, name, functions, seed=123, *, input_limit=6144):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        if not torch.cuda.is_available():
            raise ValueError('This benchmark requires a CUDA GPU; refusing an unnoticed CPU run')
        torch.cuda.set_device(0)
        self.device = 'cuda:0'
        self.functions = functions
        if input_limit != 6144:
            raise ValueError("GrepRAG-0.6B requires the 6144-token keyword input policy")
        self.input_limit = input_limit
        functions['torch'] = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        # Identical to the upstream CUDA loading branch. No use_cache override,
        # prompt repair, thinking-mode switch, batching or generation-config edits.
        attention = 'flash_attention_2'
        try:
            import flash_attn  # noqa: F401
        except Exception:
            attention = 'sdpa'
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
            trust_remote_code=True, attn_implementation=attention).to(self.device).eval()
        self.identity = dict(attention=attention, dtype='float16', device=self.device,
            model_config=self.model.config.to_dict(), generation_config=self.model.generation_config.to_dict(),
            max_new_tokens=64)
        import importlib.metadata
        self.identity['runtime_versions'] = {p:importlib.metadata.version(p)
            for p in ('torch','transformers','tokenizers','numpy','rank-bm25','tiktoken','pyarrow')}
        self.identity['gpu'] = torch.cuda.get_device_name(0)
        from .greprag_keyword_input import POLICY
        self.identity.update(keyword_prompt_policy=POLICY, keyword_input_tokens=input_limit)

    def generate(self, prefix):
        return self._generate_bounded(prefix)

    def _generate_bounded(self, prefix):
        import time
        import torch
        from .greprag_keyword_input import prepare_input
        f = self.functions
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        inputs, trace = prepare_input(self.tokenizer, f, prefix, self.input_limit)
        trace['input_preparation_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=64, do_sample=False,
                temperature=None, pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize(self.device)
        trace['inference_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        text = self.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        keywords = f['_extract_keywords'](text, 10)
        commands = f['_build_rg_commands'](keywords, '*.py', 10)
        trace['output_processing_seconds'] = time.perf_counter() - started
        return dict(raw_output=text, generated_keywords=keywords,
                    generated_commands=commands, keyword_input=trace)

    def close(self):
        import torch
        del self.model
        torch.cuda.empty_cache()


def retrieve(req, commands, repo, functions, current_file=None, *, protect_keyword_options=True):
    """Execute upstream concurrency, parser, two BM25 calls and budget policy."""
    # Input-only adaptation: the official function wraps !current_file in shell
    # single quotes and treats it as an rg glob. Preserve the exact current-file
    # exclusion even for template names containing quotes or glob characters.
    path = safe_path(current_file or req['path'])
    glob_literal = ''.join('\\' + c if c in '\\*?[]{}' else c for c in path)
    quoted_path = glob_literal.replace("'", "'\"'\"'")
    cmd_objs = [c for c in commands[:10] if isinstance(c, dict)]
    if protect_keyword_options is not True:
        raise ValueError('GrepRAG-0.6B always protects dash-prefixed keyword patterns')
    originals = {}
    # Official keyword builder always emits rg '<literal pattern>' ... .
    # Shell quotes do not stop rg interpreting '--version' as an option.
    # -e preserves the regex itself, flags, concurrency and completion order.
    safe_commands = []
    for item in cmd_objs:
        command = item.get('command', '')
        if command.startswith("rg '-"):
            executed = 'rg -e ' + command[3:]
            originals[executed] = command
            item = dict(item, command=executed)
        safe_commands.append(item)
    cmd_objs = safe_commands
    results = functions['run_commands_in_repo'](quoted_path, cmd_objs, str(repo))
    matches, diagnostics = [], []
    try:
        for item in results:  # Preserve upstream as_completed order.
            result = item.get('run_res') or {}
            cmd_obj = item.get('cmd_obj') or {}
            filename = result.get('json_file')
            available = bool(filename and Path(filename).is_file())
            # Match official process_dataset: return codes do not decide which
            # JSON outputs are parsed. Invalid regexes normally leave an empty
            # file; timed-out searches may leave usable matches. No retry,
            # prevalidation, regex repair or fixed-string fallback is added.
            try:
                found = functions['parse_rg_json'](filename, cmd_obj.get('command')) if available else []
            except (json.JSONDecodeError, UnicodeDecodeError) as failure:
                with Path(filename).open('rb') as stream:
                    preview = stream.read(240).decode('utf8', errors='replace')
                raise ValueError(f'Invalid rg JSON: task={req.get("task_id")} '
                    f'command={cmd_obj.get("command")!r} returncode={result.get("returncode")} '
                    f'preview={preview!r}') from failure
            matches.extend(found)
            diagnostics.append(dict(command=cmd_obj.get('command'), match_count=len(found),
                returncode=result.get('returncode'), stderr=result.get('stderr') or '',
                json_output_available=available))
            if cmd_obj.get('command') in originals:
                diagnostics[-1].update(original_command=originals[cmd_obj['command']],
                                       keyword_option_protected=True)
        total = len(matches)
        first, merged, ranked, selected, error = [], [], [], matches, None
        if matches:
            try:
                first = functions['rerank_matches'](matches, req['prefix'], max(100, int(total * 0.5)))
                merged = functions['merge_overlapping_segments'](first)
                ranked = functions['rerank_matches'](merged, req['prefix'], top_k=10)
                selected = functions['select_matches_by_token_budget'](ranked, budget=4096)
            except Exception as failure:
                # Official process_dataset retains all_matches on rerank errors.
                # Preserve that behavior and report it rather than aborting or
                # introducing an alternative ranking algorithm.
                error = '; rerank failed: ' + str(failure)
        context = functions['build_context']({'matches': selected}, str(repo))
        return dict(matches=selected, crossfile_text=context['text'], total_match_count=total,
                    first_count=len(first), merged_count=len(merged), commands=diagnostics, error=error,
                    command_issue_count=sum(c['returncode'] not in (0, 1) or not c['json_output_available']
                                            for c in diagnostics))
    finally:
        # Official code leaves these files behind. Ownership cleanup changes no
        # retrieval function and is included in the surrounding online timer.
        for item in results:
            name = (item.get('run_res') or {}).get('json_file')
            if name:
                Path(name).unlink(missing_ok=True)


def compose_prompt(result, req, tokenizer, prompt_budget=4096, crossfile_budget=3072):
    """Common prefix-only generator budget, applied AFTER official selection."""
    cross_ids = tokenizer.encode(result['crossfile_text'], add_special_tokens=False)[:crossfile_budget]
    cross = tokenizer.decode(cross_ids)
    while len(tokenizer.encode(cross, add_special_tokens=False)) > crossfile_budget:
        cross_ids = cross_ids[:-1]
        cross = tokenizer.decode(cross_ids)
    leading = cross + f"\n\n# file path: {req['path']}\n\n"
    available = prompt_budget - len(tokenizer.encode(leading, add_special_tokens=False))
    if available < 0:
        raise ValueError('Crossfile text and file header exceed the common prompt budget')
    original = tokenizer.encode(req['prefix'], add_special_tokens=False)
    prefix = original[-available:] if available else []
    while True:
        prompt = leading + tokenizer.decode(prefix)
        count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if count <= prompt_budget:
            break
        if not prefix:
            raise ValueError('Cannot fit the common prompt budget')
        prefix = prefix[max(1, count - prompt_budget):]
    return dict(prompt=prompt, prompt_tokens=count, crossfile_text=cross,
                crossfile_tokens=len(tokenizer.encode(cross, add_special_tokens=False)),
                prefix_original_tokens=len(original), prefix_retained_tokens=len(prefix))
