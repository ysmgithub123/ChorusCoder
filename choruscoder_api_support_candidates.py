"""Raw-prefix candidates for API-first support selection.

Byte ranges always refer to visible source; queries never synthesize code.
"""

import io
import re
import tokenize
from choruscoder_compact_apis import words, overlap
from choruscoder_compact_support import prefix_features
from choruscoder_compact_bindings import extract, in_spans, PLACEHOLDER
from choruscoder_far import PrefixFacts
from choruscoder import source_query
from choruscoder_chunk_controls import fully_covered


def purpose(prefix, candidates, target_spans, parser):
    raw = prefix.encode("utf-8")
    marker = b"__purpose_selection_cursor__"
    tree = parser.parse(raw + marker)
    node = tree.root_node.descendant_for_byte_range(
        len(raw), len(raw) + len(marker) - 1
    )
    while node is not None and node.type != "function_definition":
        node = node.parent
    if node is None:
        return dict(head=None, doc=None, reason="no_enclosing_function")
    name = node.child_by_field_name("name")
    body = node.child_by_field_name("body")
    if name is None or body is None:
        return dict(head=None, doc=None, reason="function_header_unavailable")
    fn_name = raw[name.start_byte : name.end_byte].decode()
    specific = words(fn_name) - {"main", "run", "f", "foo"}
    part = raw[node.start_byte : min(body.start_byte, len(raw))].decode()
    lines = part.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode()))
    depth = 0
    end = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(part).readline):
            if tok.type != tokenize.OP:
                continue
            if tok.string in "([{":
                depth += 1
            elif tok.string in ")]}":
                depth -= 1
            elif tok.string == ":" and depth == 0:
                end = (
                    node.start_byte
                    + offsets[tok.end[0] - 1]
                    + len(lines[tok.end[0] - 1][: tok.end[1]].encode())
                )
                break
    except (tokenize.TokenError, IndentationError):
        pass
    header = [node.start_byte, end] if end is not None else None
    eligible = [
        i
        for i, c in enumerate(candidates)
        if not fully_covered(c["context_spans"], target_spans)
    ]
    heads = [
        i
        for i in eligible
        if header
        and specific
        and any(
            a <= header[0] and header[1] <= b for a, b in candidates[i]["context_spans"]
        )
    ]
    head = min(heads, key=lambda i: (len(candidates[i]["text"]), i)) if heads else None
    doc_span = None
    for statement in body.named_children:
        if statement.type == "comment":
            continue
        if (
            statement.type == "expression_statement"
            and statement.named_children
            and statement.named_children[0].type in ("string", "concatenated_string")
        ):
            start = statement.start_byte
            stop = min(statement.end_byte, len(raw))
            text = raw[start:stop].decode()
            # The operation paragraph, before long parameter/return sections.
            first = re.split(
                r"\n\s*\n|\n\s*(?:Args|Arguments|Parameters|Returns)\s*[:\n]",
                text,
                maxsplit=1,
            )[0]
            if len(words(first)) >= 3:
                doc_span = [start, start + len(first.encode())]
        break
    docs = [
        i
        for i in eligible
        if doc_span and overlap(candidates[i]["context_spans"], [doc_span]) > 0
    ]
    doc = (
        max(
            docs,
            key=lambda i: (
                overlap(candidates[i]["context_spans"], [doc_span]),
                -len(candidates[i]["text"]),
                -i,
            ),
        )
        if docs
        else None
    )
    return dict(
        head=head,
        doc=doc,
        header_span=header,
        doc_span=doc_span,
        function_name=fn_name,
        reason=None,
    )


def analyze(bank, parser):
    d = purpose(
        bank["prefix"],
        bank["candidates"],
        bank["structure"]["prefix_target_context_spans"],
        parser,
    )
    f = prefix_features(bank["prefix"], bank["target_T"], bank["candidates"], parser)
    return dict(
        purpose=d,
        sources=[x["index"] for x in f["features"] if x["is_source"]],
        source_span=f["source_span"],
    )


def spare_source(bank, detail, version, tokenizer):
    old = list(bank["old_indices"])

    def result(indices, reason):
        return dict(indices=indices, groups=[[i] for i in indices], reason=reason)

    if not old or len(old) >= 4:
        return result(old, "no_spare_slot")
    sources = [
        i
        for i in detail["sources"]
        if i not in old
        and all(
            bank["candidates"][i]["query_text"] != bank["candidates"][j]["query_text"]
            for j in old
        )
    ]
    if not sources:
        return result(old, "no_missing_complete_source")
    source = max(
        sources,
        key=lambda i: (bank["old_cosines"][i], -len(bank["candidates"][i]["text"]), -i),
    )
    assert (
        len(
            tokenizer.encode(
                bank["candidates"][source]["query_text"], add_special_tokens=False
            )
        )
        <= 508
    )
    return result([*old, source], None)


def prepare_chain(source, parser, base):
    cs = source["candidates"]
    prefix = source["prefix"]
    facts = extract(prefix, cs, [], parser)
    index = PrefixFacts(prefix + PLACEHOLDER, parser)
    pairs = []
    for b in facts["bindings"]:
        provider = b.get("provider") or {}
        if provider.get("usage") != "constructor" or b["confidence"] not in {
            "core",
            "initializer_assumption",
        }:
            continue
        at = index.root.descendant_for_byte_range(b["start"], b["start"] + 1)
        imported = index.binding(provider["bound_name"], at)
        if imported is None or imported.get("origin") != provider["import_origin"]:
            continue
        origins = [
            i for i, c in enumerate(cs) if in_spans(c["context_spans"], *b["span"])
        ]
        imports = [
            i
            for i, c in enumerate(cs)
            if in_spans(c["context_spans"], *imported["span"])
        ]
        choices = [
            (a, d)
            for a in origins
            for d in imports
            if a != d and cs[a]["query_text"] != cs[d]["query_text"]
        ]
        if not choices:
            continue
        a, d = min(
            choices,
            key=lambda p: (
                p[0] not in base["indices"],
                len(cs[p[0]]["text"]) + len(cs[p[1]]["text"]),
                p,
            ),
        )
        pairs.append(
            dict(
                origin=a,
                import_index=d,
                confidence=b["confidence"],
                binding=b["code"],
                import_span=imported["span"],
            )
        )
    return dict(pairs=pairs, receiver=facts["receiver"])


def protected_chain(source, p, base, cosines, version, tokenizer):
    if version != "protected_members":
        raise ValueError("Unknown API-first source/import rule")
    old = base["indices"]
    cs = source["candidates"]
    if len(old) < 3:
        return None
    pairs = p["pairs"]
    for pair in pairs:
        joined = [pair["origin"], pair["import_index"]]
        if set(joined) & set(old[:2]):
            continue
        others = [
            i
            for i in old
            if i not in joined
            and cs[i]["query_text"] not in {cs[j]["query_text"] for j in joined}
        ]
        if len(others) < 2:
            continue
        others = old[:2]
        text = source_query(
            source["prefix"],
            [s for i in joined for s in cs[i]["context_spans"]]
            + source["structure"]["prefix_target_context_spans"],
        )
        if len(tokenizer.encode(text, add_special_tokens=False)) > 128:
            continue
        groups = [[i] for i in others] + [joined]
        assert len(groups) == 3 and len({i for g in groups for i in g}) == 4
        return dict(
            indices=[i for g in groups for i in g],
            groups=groups,
            reason=None,
            pair=pair,
            original_first_two_preserved=True,
        )
    return None
