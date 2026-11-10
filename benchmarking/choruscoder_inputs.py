"""Tokenizer-exact RLCoder inputs and generation validation."""
from __future__ import annotations
from types import SimpleNamespace
from urllib.parse import urljoin
from choruscoder import digest
INPUT_VERSION = 'actual_input_ids_v2'

def generation_checksum(value):
    return digest({key: item for key, item in value.items() if key not in ('generation_checksum', 'rlcoder_checksum')})

def tokenizer_identity(tokenizer):
    cached = getattr(tokenizer, '_choruscoder_input_identity', None)
    if cached is None:
        cached = digest(dict(vocab=tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else None, special=getattr(tokenizer, 'special_tokens_map', {}), bos=getattr(tokenizer, 'bos_token_id', None), eos=getattr(tokenizer, 'eos_token_id', None), pad=getattr(tokenizer, 'pad_token_id', None), add_bos=getattr(tokenizer, 'add_bos_token', None), add_eos=getattr(tokenizer, 'add_eos_token', None), probe=tokenizer.encode("变量 = helper('λ')\n", add_special_tokens=True)))
        tokenizer._choruscoder_input_identity = cached
    return cached

def official_input(record, tokenizer, definitions):
    """Execute the pinned constructor; duplicate only __getitem__'s encoding step."""
    context = record['context_profile']
    blocks = [definitions['CodeBlock'](b['path'], b['description'], b['text'], 'python', b['type']) for b in record['evidence']]
    args = SimpleNamespace(generator_max_context_length=context['prompt_tokens'], generator_max_crossfile_length=context['crossfile_tokens'])
    instance = SimpleNamespace(args=args, tokenizer=tokenizer)
    example = SimpleNamespace(file_path=record['path'], left_context=record['prefix'])
    text = definitions['construct_prompts'](instance, example, blocks)
    raw_cross = '\n\n'.join(map(str, blocks))[:context['crossfile_tokens'] * 10]
    cross_ids = tokenizer.encode(raw_cross, add_special_tokens=False)[:context['crossfile_tokens']]
    ids = tokenizer.encode(text, add_special_tokens=True)[-context['prompt_tokens']:]
    return dict(prompt_text=text, token_ids=ids, crossfile_token_ids=cross_ids, crossfile_text=tokenizer.decode(cross_ids), prompt_format='rlcoder_official', usage=dict(prompt_tokens=len(ids), crossfile_tokens=len(cross_ids), evidence_tokens=len(cross_ids)))

def build_input(record, tokenizer, definitions=None):
    if record.get('input_budget_version') != INPUT_VERSION:
        raise ValueError('Missing or incompatible actual-input budget version')
    if not record.get('rlcoder_config'):
        raise ValueError('Token-ID prompt construction is exclusive to RLCoder')
    if definitions is None:
        from .rlcoder_backend import upstream
        definitions = upstream('third_party/RLCoder')[1]
    payload = official_input(record, tokenizer, definitions)
    if len(payload['token_ids']) > record['context_profile']['prompt_tokens'] or len(payload['crossfile_token_ids']) > record['context_profile']['crossfile_tokens']:
        raise ValueError('Actual input exceeds its token budget')
    payload.update(version=INPUT_VERSION, tokenizer_identity=tokenizer_identity(tokenizer))
    payload['checksum'] = digest(payload)
    return payload

def validate_input(record, tokenizer=None, definitions=None):
    payload = record.get('generation_input', {})
    if payload.get('version') != INPUT_VERSION or record.get('input_budget_version') != INPUT_VERSION or payload.get('checksum') != digest({k: v for k, v in payload.items() if k != 'checksum'}) or (len(payload['token_ids']) > record['context_profile']['prompt_tokens']) or (len(payload['crossfile_token_ids']) > record['context_profile']['crossfile_tokens']) or (payload['usage']['prompt_tokens'] != len(payload['token_ids'])):
        raise ValueError('Actual generation input checksum/version/budget mismatch')
    if tokenizer is not None and payload != build_input(record, tokenizer, definitions):
        raise ValueError('Actual generation input/tokenizer mismatch')
    return payload

def verify_service(client, model, tokenizer, text):
    """Check text->IDs once before submitting a nonempty batch of cached IDs."""
    import httpx
    from openai import APIConnectionError, APIStatusError
    url = urljoin(str(client.base_url), '../tokenize')
    for probe in ("变量 = helper('λ')\n", text):
        expected = tokenizer.encode(probe, add_special_tokens=True)
        try:
            response = client.post(url, body=dict(model=model, prompt=probe, add_special_tokens=True), cast_to=httpx.Response)
        except APIStatusError as exc:
            raise RuntimeError(f'Generation service /tokenize request failed (HTTP {exc.status_code})') from exc
        except APIConnectionError as exc:
            raise RuntimeError('Cannot connect to generation service /tokenize endpoint') from exc
        try:
            response.raise_for_status()
            value = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f'Generation service /tokenize request failed (HTTP {exc.response.status_code})') from exc
        except ValueError as exc:
            raise ValueError('Generation service /tokenize returned invalid JSON') from exc
        if not isinstance(value, dict) or not isinstance(value.get('tokens'), list):
            raise ValueError('Generation service /tokenize response must contain a tokens list')
        if value['tokens'] != expected:
            raise ValueError('Generation service tokenizer/BOS does not match local generation inputs')

def verify_usage(response, prompts):
    usage = getattr(response, 'usage', None)
    if usage is None or getattr(usage, 'prompt_tokens', None) != sum(map(len, prompts)):
        raise ValueError('Generation service changed token-ID inputs or did not report actual prompt usage')
