"""Bounded source/interface combinations and packed-content comparisons."""

import ast
import re
import textwrap
from functools import lru_cache
from choruscoder import round_robin_evidence
from choruscoder_compact_support import prepare_supports, prefix_score
from choruscoder_compact_apis import active_header, words

QUOTAS = {1: [30], 2: [15, 15], 3: [12, 10, 8], 4: [10, 8, 6, 6]}
VERSIONS = ("headers",)


def queries(candidates, indices):
    return [
        dict(
            id=f"Q{k+1}",
            text=candidates[i]["query_text"],
            segment_ids=[f"S{k+1}"],
            role="support",
        )
        for k, i in enumerate(indices)
    ]


def groups(bank, indices):
    return [
        [
            dict(bank["chunks"][b["id"]], score=b["score"])
            for b in bank["rankings"][i + 1]
        ][:n]
        for i, n in zip(indices, QUOTAS[len(indices)])
    ]


def pack(bank, indices, packer, baseline_exact=True):
    if baseline_exact and list(indices) == bank["old_indices"]:
        blocks = bank["old_evidence"]
        from choruscoder import render_evidence

        return blocks, len(
            packer.tokenizer.encode(render_evidence(blocks), add_special_tokens=False)
        )
    return packer.pack(
        round_robin_evidence(
            queries(bank["candidates"], indices), groups(bank, indices)
        )
    )


@lru_cache(maxsize=512)
def parameters(header):
    if "def " not in header:
        return frozenset()
    try:
        node = ast.parse(textwrap.dedent(header) + "\n    pass").body[0]
        return frozenset(
            a.arg
            for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if a.arg not in ("self", "cls")
        )
    except (SyntaxError, AttributeError, ValueError):
        return frozenset()


def prepare(bank, files, parser=None):
    d = prepare_supports(
        bank["prefix"],
        bank["target_T"],
        bank["candidates"],
        bank["path"],
        files,
        parser,
    )
    features = d["features"][1:]
    old = bank["old_indices"]
    extra = []
    roles = {}
    start, _ = active_header(bank["prefix"], bank["target_T"])
    current = (
        bank["prefix"].encode()[start:].decode()
        if start is not None
        else bank["prefix"][-2000:]
    )
    current_parameters = parameters(current.partition(":\n")[0] + ":")
    param_mentions = [
        len(current_parameters & set(re.findall(r"\b[A-Za-z_]\w*\b", c["text"])))
        for c in bank["candidates"]
    ]
    ranked = sorted(
        range(len(features)),
        key=lambda i: (-prefix_score(features[i]), -features[i]["near"], i),
    )
    predicates = [
        ("source", lambda f: f["is_source"]),
        ("call", lambda f: f["receiver_call"] and f["mentions"]),
        ("parameters", lambda f: param_mentions[f["index"]] > 0),
        ("purpose", lambda f: f["descriptor_named"] or f["meaningful_header"]),
    ]
    for role, predicate in predicates:
        order = (
            sorted(ranked, key=lambda i: (-param_mentions[i], -features[i]["near"], i))
            if role == "parameters"
            else ranked
        )
        existing = next((i for i in order if predicate(features[i])), None)
        if existing is not None:
            roles[role] = existing
        added = next(
            (
                i
                for i in order
                if predicate(features[i]) and i not in old and i not in extra
            ),
            None,
        )
        if added is not None:
            extra.append(added)
    pool = list(dict.fromkeys([*old, *extra]))[:8]
    combinations = [tuple(old)] if old else []
    anchors = old[:2]
    if len(anchors) == 2:
        combinations.extend(tuple([*anchors, i]) for i in pool if i not in anchors)
    role_set = list(
        dict.fromkeys(roles[r] for r in ("source", "call", "purpose") if r in roles)
    )
    for i in sorted(pool, key=lambda i: (-prefix_score(features[i]), i)):
        if len(role_set) >= 3:
            break
        if i not in role_set:
            role_set.append(i)
    if len(role_set) == 3:
        combinations.append(tuple(role_set))
    combinations = list(dict.fromkeys(combinations))[:8]
    available = set(re.findall(r"\b[A-Za-z_]\w*\b", current))
    mapping = {f["path"]: f["text"].encode() for f in files}
    apis = []
    for api in d["apis"]:
        params = set()
        for a, b in api["headers"]:
            params.update(parameters(mapping[api["path"]][a:b].decode()))
        apis.append(
            dict(
                api,
                parameter_fit=len(params & available) / max(1, len(params)),
                prior=0.2 + min(6, api["hypothesis"]["score"]) / 2,
            )
        )
    return dict(
        pool=pool,
        combinations=combinations,
        features=features,
        roles=roles,
        apis=apis,
        available=available,
        fallback_reason=d["fallback_reason"],
    )


def annotate(prepared, blocks):
    """Precompute source spans and visible-argument calls once per returned block."""
    annotations = {}
    apis = prepared["apis"]
    available = prepared["available"]
    names = {a["name"] for a in apis}
    for b in blocks:
        heads = []
        calls = {}
        for i, a in enumerate(apis):
            if a["path"] == b["path"]:
                for x, y in a["headers"]:
                    if b["start_byte"] <= x and y <= b["end_byte"]:
                        heads.append(i)
                        break
        for line in b["text"].splitlines():
            if re.match(r"\s*(?:async\s+)?def\b", line):
                continue
            for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(([^)\n]*)", line):
                if m[1] not in names:
                    continue
                args = set(re.findall(r"\b[A-Za-z_]\w*\b", m[2]))
                calls[m[1]] = max(
                    calls.get(m[1], 0), 0.2 + len(args & available) / max(1, len(args))
                )
        annotations[b["id"]] = dict(heads=heads, calls=calls)
    return annotations


def quality(prepared, blocks, indices, annotations):
    heads = set()
    calls = {}
    for b in blocks:
        a = annotations[b["id"]]
        heads.update(a["heads"])
        for name, value in a["calls"].items():
            calls[name] = max(calls.get(name, 0), value)
    hs = [
        prepared["apis"][i]["prior"] * (1 + 0.5 * prepared["apis"][i]["parameter_fit"])
        for i in heads
    ]
    cs = [a["prior"] * calls[a["name"]] for a in prepared["apis"] if a["name"] in calls]
    header = max(hs, default=0) + 0.15 * sum(hs)
    call = max(cs, default=0) + 0.15 * sum(cs)
    fs = [prepared["features"][i] for i in indices]
    role = sum(
        [
            any(f["is_source"] for f in fs),
            any(f["receiver_call"] and f["current_scope"] for f in fs),
            any(f["mentions"] and f["current_scope"] for f in fs),
            any(f["meaningful_header"] or f["descriptor_named"] for f in fs),
        ]
    )
    return dict(
        header=header,
        call=call,
        roles=role,
        balanced=header + 0.75 * call + 0.25 * role,
    )


def select(bank, prepared, packer, versions=VERSIONS):
    if tuple(versions) != VERSIONS:
        raise ValueError("Unknown API-first interface comparison rule")
    if prepared["fallback_reason"] or not bank["old_indices"]:
        return {
            v: dict(
                indices=bank["old_indices"],
                blocks=bank["old_evidence"],
                fallback=prepared["fallback_reason"],
                pool=[],
                combinations=0,
                quality=None,
            )
            for v in versions
        }
    allblocks = {b["id"]: b for b in bank["old_evidence"]}
    for i in prepared["pool"]:
        for b in bank["rankings"][i + 1]:
            allblocks[b["id"]] = bank["chunks"][b["id"]]
    annotations = annotate(prepared, allblocks.values())
    options = []
    for indices in prepared["combinations"]:
        blocks, cost = pack(bank, indices, packer)
        q = quality(prepared, blocks, indices, annotations)
        options.append(
            dict(
                indices=list(indices),
                blocks=blocks,
                tokens=cost,
                quality=q,
                changed=len(set(indices) ^ set(bank["old_indices"])),
            )
        )
    out = {}
    for version in versions:
        best = max(
            options,
            key=lambda x: (
                x["quality"]["header"],
                x["quality"]["call"],
                x["quality"]["roles"],
                -x["changed"],
            ),
        )
        out[version] = dict(
            best, fallback=None, pool=prepared["pool"], combinations=len(options)
        )
    return out


def compare(prepared, original, proposed):
    blocks = {b["id"]: b for b in [*original["blocks"], *proposed["blocks"]]}
    annotations = annotate(prepared, blocks.values())
    before = quality(prepared, original["blocks"], original["indices"], annotations)
    after = quality(prepared, proposed["blocks"], proposed["indices"], annotations)
    strong = set()
    for i, api in enumerate(prepared["apis"]):
        h = api["hypothesis"]
        nw, hw = words(api["name"]), words(h["header"])
        matched = len(nw & hw)
        if h["exact"] or h["in_doc"] or (matched >= 2 and matched >= len(nw) / 2):
            strong.add(i)

    def heads(variant):
        return {i for b in variant["blocks"] for i in annotations[b["id"]]["heads"]}

    gained_strong = (heads(proposed) - heads(original)) & strong
    lost_strong = (heads(original) - heads(proposed)) & strong
    header_gain = after["header"] > before["header"] + 1e-9
    calls_preserved = after["call"] + 1e-9 >= before["call"]
    decisions = {
        "header_gain": header_gain,
        "call_preserving_gain": header_gain and calls_preserved,
        "specific_interface_gain": bool(gained_strong)
        and not lost_strong
        and calls_preserved,
    }
    # No role/source-count bonus. The actual input to generation must improve.
    return decisions, dict(
        before=before,
        after=after,
        gained_specific_interfaces=sorted(gained_strong),
        lost_specific_interfaces=sorted(lost_strong),
    )


def strong_interface(prepared):
    for api in prepared["apis"]:
        h = api["hypothesis"]
        nw, hw = words(api["name"]), words(h["header"])
        matched = len(nw & hw)
        if h["exact"] or h["in_doc"] or (matched >= 2 and matched >= len(nw) / 2):
            return True
    return False
