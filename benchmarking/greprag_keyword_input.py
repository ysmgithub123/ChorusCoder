"""Bound keyword inputs without rewriting either the template or code suffix."""

POLICY = 'original_template_visible_code_suffix_total6144_v1'


def validate_trace(trace, limit, prefix=None):
    import math
    if not isinstance(trace, dict) or trace.get('policy') != POLICY or trace.get('input_limit') != limit:
        raise ValueError('Keyword input policy/limit mismatch')
    names = ('original_input_tokens', 'input_tokens', 'prefix_start_character',
             'original_prefix_characters', 'retained_prefix_characters', 'tokenization_passes')
    if any(type(trace.get(n)) is not int or trace[n] < 0 for n in names):
        raise ValueError('Invalid keyword input counts')
    if (not 0 < trace['input_tokens'] <= limit or trace['tokenization_passes'] < 1 or
            trace['original_input_tokens'] < trace['input_tokens'] or
            trace['prefix_start_character'] + trace['retained_prefix_characters'] != trace['original_prefix_characters'] or
            type(trace.get('truncated')) is not bool or
            trace['truncated'] != (trace['prefix_start_character'] > 0) or
            trace['truncated'] != (trace['original_input_tokens'] > limit)):
        raise ValueError('Inconsistent keyword input truncation')
    if prefix is not None and len(prefix) != trace['original_prefix_characters']:
        raise ValueError('Keyword input/source length mismatch')
    for name in ('input_preparation_seconds', 'inference_seconds', 'output_processing_seconds'):
        value = trace.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid keyword stage timing')


def prepare_input(tokenizer, functions, prefix, limit):
    """Return final CPU tensors once; offsets refer to original Unicode text.

    The estimate deliberately leaves eight tokens for boundary re-tokenization.
    The rebuilt prompt is always checked exactly, never assumed to fit. No
    decode/encode round trip can change the retained source characters.
    """
    build = functions['_build_chat_prompt']
    instruction = functions['INSTRUCTION']
    marker = '\x00GREPRAG_CODE_POSITION\x00'
    header, footer = build(instruction, marker).split(marker)
    original = build(instruction, prefix)
    if original != header + prefix + footer:
        raise ValueError('Unexpected keyword template: code is not a literal insertion')
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('Bounded keyword input requires the pinned fast tokenizer offsets')
    inputs = tokenizer(original, return_tensors='pt', return_offsets_mapping=True)
    original_tokens = inputs['input_ids'].shape[1]
    start = 0
    passes = 1
    if original_tokens > limit:
        empty_tokens = len(tokenizer.encode(header + footer, add_special_tokens=True))
        if empty_tokens > limit:
            raise ValueError('Keyword instruction and template exceed the input limit')
        while inputs['input_ids'].shape[1] > limit:
            offsets = inputs['offset_mapping'][0].tolist()
            excess = inputs['input_ids'].shape[1] - limit
            token_index = min(len(offsets)-1, excess + empty_tokens + 8)
            advance = offsets[token_index][0] - len(header)
            # Zero-width special tokens must not cause an unbounded retry.
            remaining = len(prefix) - start
            advance = min(remaining, max(1, advance))
            if not remaining:
                raise ValueError('Cannot fit keyword input despite empty code')
            start += advance
            inputs = tokenizer(build(instruction, prefix[start:]), return_tensors='pt',
                               return_offsets_mapping=True)
            passes += 1
    inputs.pop('offset_mapping')
    return inputs, dict(policy=POLICY, input_limit=limit, original_input_tokens=original_tokens,
        input_tokens=inputs['input_ids'].shape[1], truncated=start > 0,
        prefix_start_character=start, original_prefix_characters=len(prefix),
        retained_prefix_characters=len(prefix)-start, tokenization_passes=passes)
