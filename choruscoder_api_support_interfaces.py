"""Resolve visible owners, preserve their declarations, and score raw S+T."""

import re
from functools import lru_cache
from choruscoder import source_query, round_robin_evidence
from choruscoder_compact_bindings import extract, in_spans
from choruscoder_compact_apis import resolve_provider, all_apis, words, merged
from choruscoder_far import PrefixFacts, path as node_path
from choruscoder_compact_support import prefix_features
from choruscoder_chunk_controls import fully_covered
from choruscoder_api_support_candidates import purpose
from choruscoder_api_support_combinations import QUOTAS

VERSIONS = ("api_first_support", "api_two_supports", "api_visible_purpose")


def prepare_owner(source, files, parser):
    bindings = [
        b
        for b in extract(source["prefix"], source["candidates"], [], parser)["bindings"]
        if b["confidence"] in {"core", "initializer_assumption"}
    ]
    providers = resolve_provider(bindings, source["path"], files)
    return dict(apis=all_apis(providers, files), providers=providers)


def preserve_headers(prepared, old, new, parser, version):
    if version != "keep_heads":
        raise ValueError("Unknown API-first declaration preservation rule")
    if old["prompt_hash"] == new["prompt_hash"]:
        return False, dict(reason="identical_proposal")
    apis = prepared["apis"]
    if not apis:
        return True, dict(reason="unscorable_pass_through", reliable_owner=False)
    new_spans = {}
    for b in new["blocks"]:
        new_spans.setdefault(b["path"], []).append([b["start_byte"], b["end_byte"]])
    new_spans = {p: merged(v) for p, v in new_spans.items()}

    def covered(path, a, z):
        return in_spans(new_spans.get(path, []), a, z)

    required = set()
    headers = []
    for block in old["blocks"]:
        for api in apis:
            if api["path"] != block["path"]:
                continue
            for a, z in api["headers"]:
                if block["start_byte"] <= a and z <= block["end_byte"]:
                    required.add((api["path"], a, z))
                    headers.append(api["name"])
    lost = [x for x in sorted(required) if not covered(*x)]
    return not lost, dict(
        reason=(
            "lost_original_owner_information"
            if lost
            else "original_owner_information_preserved"
        ),
        required=len(required),
        lost=lost,
        old_heads=sorted(set(headers)),
        reliable_owner=True,
    )


@lru_cache(maxsize=512)
def raw_prefix(text, limit, tokenizer):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text
    tokens = tokenizer.convert_ids_to_tokens(ids[:limit])
    raw = bytes(tokenizer.byte_decoder[c] for token in tokens for c in token)
    prefix = raw.decode("utf-8", errors="ignore")
    if not text.startswith(prefix):
        raise ValueError("Tokenizer prefix does not preserve original source")
    while prefix and len(tokenizer.encode(prefix, add_special_tokens=False)) > limit:
        prefix = prefix[:-1]
    return prefix


def has_header(blocks, api):
    a, z = api["headers"][0]
    return any(
        b["path"] == api["path"] and b["start_byte"] <= a and z <= b["end_byte"]
        for b in blocks
    )


def candidate_pool(source, prepared, current, predictions, cosines):
    cs = source["candidates"]
    target = source["structure"]["prefix_target_context_spans"]
    raw = list(dict.fromkeys(current["indices"]))
    extra = []
    names = {p["api"]["name"] for p in predictions.values() if p}
    for i, c in enumerate(cs):
        if any(
            re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", c["text"]) for n in names
        ):
            extra.append(i)
    d = prepared["purpose"]
    fs = prepared["features"]
    extra.extend(i for i in (d["doc"], d["head"]) if i is not None)
    extra.extend(i for i, f in enumerate(fs) if f["is_source"])
    extra.extend(
        i for i, f in enumerate(fs) if f["receiver_call"] and f["current_scope"]
    )
    extra.extend(
        sorted(range(len(cs)), key=lambda i: (-cosines[i], -cs[i]["line_end"], i))
    )
    current_groups = current.get("groups") or [[i] for i in current["indices"]]
    joined = sum(len(g) > 1 for g in current_groups)
    limit = 8 - joined
    added = 0
    for i in extra:
        if len(raw) >= limit or added >= 4:
            break
        if (
            i in raw
            or fully_covered(cs[i]["context_spans"], target)
            or cs[i]["query_text"] in {cs[j]["query_text"] for j in raw}
        ):
            continue
        raw.append(i)
        added += 1
    assert len(raw) + joined <= 8
    return raw


def queries(source, groups):
    cs = source["candidates"]
    target = source["structure"]["prefix_target_context_spans"]
    return [
        dict(
            id=f"Q{j+1}",
            text=source_query(
                source["prefix"],
                [s for i in g for s in cs[i]["context_spans"]] + target,
            ),
            segment_ids=[f"C{i+1}" for i in g],
            role="support",
        )
        for j, g in enumerate(groups)
    ]


def prepare(source, files, parser, current, tokenizer):
    owned = prepare_owner(source, files, parser)
    apis = [a for a in owned["apis"] if not a["name"].startswith("__")]
    mapping = {f["path"]: f["text"].encode() for f in files}
    d = purpose(
        source["prefix"],
        source["candidates"],
        source["structure"]["prefix_target_context_spans"],
        parser,
    )
    hints = words((d.get("function_name") or "") + " " + source["prefix"][-1500:])
    apis = sorted(
        apis, key=lambda a: (-len(words(a["name"]) & hints), a["path"], a["name"])
    )[:32]
    texts = [
        raw_prefix(
            "\n".join(
                mapping[a["path"]][x:z].decode() for x, z in merged(a["regions"])
            ),
            192,
            tokenizer,
        )
        for a in apis
    ]
    groups = current.get("groups") or [[i] for i in current["indices"]]
    qs = queries(source, groups)
    purpose_index = d["doc"] if d["doc"] is not None else d["head"]
    purpose_query = (
        source["candidates"][purpose_index]["query_text"]
        if purpose_index is not None
        else None
    )
    return dict(
        apis=apis,
        texts=texts,
        owner=owned,
        purpose=d,
        qs=qs,
        purpose_query=purpose_query,
    )


def predict(p, scores, *, minimum_similarity=0.4, minimum_margin=0.015):
    if not scores:
        return None
    order = sorted(
        range(len(scores)),
        key=lambda i: (-scores[i], p["apis"][i]["path"], p["apis"][i]["name"]),
    )
    i = order[0]
    margin = scores[i] - scores[order[1]] if len(order) > 1 else None
    if scores[i] < minimum_similarity or margin is not None and margin < minimum_margin:
        return None
    return dict(
        api=p["apis"][i],
        score=float(scores[i]),
        margin=float(margin) if margin is not None else None,
    )


def pool(source, p, current, predictions):
    return candidate_pool(
        source,
        dict(
            p,
            features=prefix_features(
                source["prefix"], source["target_T"], source["candidates"]
            )["features"],
        ),
        current,
        predictions,
        source["cosines"],
    )


def quality(blocks, api, parser, call_cache=None):
    head = has_header(blocks, api)
    spans = merged(
        [
            [max(a, b["start_byte"]), min(z, b["end_byte"])]
            for b in blocks
            if b["path"] == api["path"]
            for a, z in api["regions"]
            if max(a, b["start_byte"]) < min(z, b["end_byte"])
        ]
    )
    calls = set()
    for b in blocks:
        key = (b['path'], b['start_byte'], b['text'], api['name'])
        cached = call_cache.get(key) if call_cache is not None else None
        if cached is None:
            found = set()
            facts = PrefixFacts(b["text"], parser)
            for n in facts.nodes:
                if n.type != "call" or n.has_error:
                    continue
                fn = n.child_by_field_name("function")
                name = node_path(fn, facts.raw) if fn is not None else None
                if name and name.split(".")[-1] == api["name"]:
                    found.add((b["path"], b["start_byte"] + n.start_byte))
            cached = frozenset(found)
            if call_cache is not None:
                call_cache[key] = cached
        calls.update(cached)
    return [int(head), len(calls), sum(z - a for a, z in spans)]


def replacement_candidates(source, current, indices):
    """The same admissibility rule used before retrieval and final comparison."""
    old_groups = current.get("groups") or [[i] for i in current["indices"]]
    if len(old_groups) < 3:
        return None, []
    cs = source["candidates"]
    drop = min(
        range(2, len(old_groups)),
        key=lambda j: (
            sum(source["cosines"][i] for i in old_groups[j]) / len(old_groups[j]),
            -j,
        ),
    )
    fixed = [i for j, g in enumerate(old_groups) if j != drop for i in g]
    occupied = source["structure"]["prefix_target_context_spans"] + [
        s for i in fixed for s in cs[i]["context_spans"]
    ]
    eligible = []
    for i in indices:
        if i in fixed or fully_covered(cs[i]["context_spans"], occupied):
            continue
        groups = [list(g) for g in old_groups]
        groups[drop] = [i]
        selected = [i for g in groups for i in g]
        if len(selected) > 4 or len(set(selected)) != len(selected):
            continue
        eligible.append(i)
    return drop, eligible


def select(source, p, current, prediction, indices, rankings, packer, parser):
    if prediction is None:
        return None
    drop, eligible = replacement_candidates(source, current, indices)
    if drop is None:
        return None
    old_groups = current.get("groups") or [[i] for i in current["indices"]]
    api = prediction["api"]
    call_cache = {}
    old_quality = quality(current["blocks"], api, parser, call_cache)
    options = []
    for i in eligible:
        groups = [list(g) for g in old_groups]
        groups[drop] = [i]
        selected = [i for g in groups for i in g]
        qs = queries(source, groups)
        blocks, tokens = packer.pack(
            round_robin_evidence(
                qs,
                [rankings[q["text"]][:quota] for q, quota in zip(qs, QUOTAS[len(qs)])],
            )
        )
        q = quality(blocks, api, parser, call_cache)
        if not q[0] or q <= old_quality:
            continue
        accepted, guard = preserve_headers(
            p["owner"],
            current,
            dict(prompt_hash="context_api_proposal", blocks=blocks),
            parser,
            "keep_heads",
        )
        if accepted:
            options.append(
                dict(
                    indices=selected,
                    groups=groups,
                    queries=qs,
                    blocks=blocks,
                    tokens=tokens,
                    prediction=prediction,
                    old_api_quality=old_quality,
                    api_quality=q,
                    guard=guard,
                    score=[*q, i in current["indices"], source["cosines"][i]],
                    drop_query_position=drop,
                )
            )
    return max(options, key=lambda s: s["score"]) if options else None
