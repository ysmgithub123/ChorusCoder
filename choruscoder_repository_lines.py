"""Lossless fixed physical-line repository partition, independent of caching."""

def line_partition(text, tokenizer, config):
    import re
    lines = re.findall('[^\\r\\n]*(?:\\r\\n|\\r|\\n)|[^\\r\\n]+$', text)
    parts = []
    cursor = 0

    def split(body, a, parent_start, parent_end, line_start, line_end):
        tokens = len(tokenizer.encode(body, add_special_tokens=False))
        b = a + len(body.encode('utf-8'))
        if tokens <= config.max_tokens:
            parts.append({'start_byte': a, 'end_byte': b, 'token_count': tokens, 'parent_start_byte': parent_start, 'parent_end_byte': parent_end, 'parent_line_start': line_start, 'parent_line_end': line_end, 'degraded_reason': 'encoder_limit_bisection' if (a, b) != (parent_start, parent_end) else None})
            return
        if len(body) < 2:
            raise ValueError('A source character exceeds the repository encoder budget')
        mid = len(body) // 2
        left, right = (body[:mid], body[mid:])
        split(left, a, parent_start, parent_end, line_start, line_end)
        split(right, a + len(left.encode('utf-8')), parent_start, parent_end, line_start, line_end)
    for i in range(0, len(lines), config.lines_per_chunk):
        body = ''.join(lines[i:i + config.lines_per_chunk])
        end = cursor + len(body.encode('utf-8'))
        split(body, cursor, cursor, end, i + 1, min(i + config.lines_per_chunk, len(lines)))
        cursor = end
    if cursor != len(text.encode('utf-8')):
        raise ValueError('Fixed-line partition does not cover the original source')
    return parts
