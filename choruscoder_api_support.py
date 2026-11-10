"""API-first support selection using the ordinary repository retrieval backend.

Only visible prefix/source files are inputs. Every stage shares a question-local
top-30 ranking memo and an eight-query budget; final packing never re-retrieves.
Repository chunks and vectors keep the backend's original cache identities.
"""

import time
import inspect
from dataclasses import asdict

import numpy as np

from choruscoder import digest, render_evidence, round_robin_evidence, source_query
from choruscoder_compact_support import ContextPacker
from choruscoder_compact_apis import merged
from choruscoder_far import cached_encode
from choruscoder_selection_controls import (
    control_queries,
    finalize_fused_fallback,
    score_fused_supports,
)
from choruscoder_api_support_candidates import (
    analyze,
    spare_source,
    prepare_chain,
    protected_chain,
)
from choruscoder_api_support_combinations import QUOTAS
import choruscoder_api_support_combinations as combinations
import choruscoder_api_support_interfaces as interfaces

CALLABLE_ABLATION_PROTOCOL = 'callable_support_ablations_v1'


def validate_runtime(configurable_thresholds=False):
    """Reject mixed deployments once, before candidate preparation/model work."""
    required = ('replacement_candidates', 'quality', 'prepare', 'pool', 'queries',
                'preserve_headers', 'select')
    missing = [name for name in required if not callable(getattr(interfaces, name, None))]
    quality = getattr(interfaces, 'quality', None)
    if callable(quality) and 'call_cache' not in inspect.signature(quality).parameters:
        missing.append('quality(..., call_cache=None)')
    if configurable_thresholds:
        from choruscoder_callable_support import select as callable_select
        for name, function in (('predict', getattr(interfaces, 'predict', None)),
                               ('callable_select', callable_select)):
            if not callable(function) or 'minimum_similarity' not in inspect.signature(function).parameters or \
                    'minimum_margin' not in inspect.signature(function).parameters:
                missing.append(name+'(..., minimum_similarity=..., minimum_margin=...)')
    if missing:
        location = getattr(interfaces, '__file__', '<unknown module path>')
        raise RuntimeError(
            'API support files are from incompatible versions. Missing: '
            + ', '.join(missing) + '. Loaded interface module: ' + str(location)
            + '. Synchronize choruscoder_api_support.py, choruscoder_api_support_interfaces.py, '
              'choruscoder_callable_support.py, choruscoder_profiles.py, benchmark_cli.py, '
              'benchmarking/choruscoder_runner.py and benchmarking/choruscoder_timing.py '
              'from the same update. No candidate selection or model inference has started.')


class QuerySession:
    """Memoize full rankings within one question, independent of query labels."""

    def __init__(self, backend, files, limit=8):
        self.backend, self.files, self.limit = backend, files, limit
        self.rankings = {}
        self.trials = []
        self.calls = []
        self.index_seconds = 0.0
        self.query_seconds = 0.0

    def fits(self, queries):
        return (
            len(self.rankings)
            + len({q["text"] for q in queries} - self.rankings.keys())
            <= self.limit
        )

    def retrieve(self, queries, quotas):
        if len(queries) != len(quotas) or any(
            type(n) is not int or not 0 <= n <= 30 for n in quotas
        ):
            raise ValueError("Invalid API-first retrieval quotas")
        missing = {q["text"]: q for q in queries if q["text"] not in self.rankings}
        if not self.fits(queries):
            raise ValueError("API-first complete pipeline exceeds eight trial queries")
        if missing:
            backend = self.backend
            if any(
                len(backend.tokenizer.encode(text, add_special_tokens=False)) > 508
                for text in missing
            ):
                raise ValueError("API-first query exceeds the original encoder budget")
            # Empty repositories must not inherit counters from a previous call.
            backend.last_index_seconds = 0.0
            backend.last_repository_runtime_timings = {}
            backend.last_repository_chunks = {}
            backend.last_query_encoding_cache = {}
            backend.last_query_encoding_seconds = 0.0
            started = time.perf_counter()
            hits = backend.retrieve(
                list(missing.values()), self.files, [30] * len(missing)
            )
            elapsed = time.perf_counter() - started
            if len(hits) != len(missing) or any(len(group) > 30 for group in hits):
                raise ValueError("Invalid API-first top-30 trial results")
            self.rankings.update(zip(missing, hits))
            self.trials.extend(
                dict(query=q, ranked=group) for q, group in zip(missing.values(), hits)
            )
            index_seconds = getattr(backend, "last_index_seconds", 0.0)
            self.index_seconds += index_seconds
            self.query_seconds += elapsed
            self.calls.append(
                dict(
                    seconds=elapsed,
                    index_seconds=index_seconds,
                    repository=dict(
                        getattr(backend, "last_repository_runtime_timings", {})
                    ),
                    chunks=dict(getattr(backend, "last_repository_chunks", {})),
                    query_encoding=dict(
                        getattr(backend, "last_query_encoding_cache", {})
                    ),
                    query_encoding_seconds=getattr(
                        backend, "last_query_encoding_seconds", 0.0
                    ),
                )
            )
        return [self.rankings[q["text"]][:n] for q, n in zip(queries, quotas)]

    def publish_timings(self):
        """Include every trial, including a rejected proposal's index work."""
        backend = self.backend
        backend.last_index_seconds = self.index_seconds
        repository = [call["repository"] for call in self.calls]
        keys = set().union(*(r.keys() for r in repository))
        backend.last_repository_runtime_timings = {
            k: (
                max(r.get(k, 0) for r in repository)
                if k == "repository_unique_texts"
                else sum(r.get(k, 0) for r in repository)
            )
            for k in keys
        }
        chunks = [call["chunks"] for call in self.calls if call["chunks"]]
        if chunks:
            backend.last_repository_chunks = dict(
                chunks[0],
                chunks_seconds=sum(c.get("chunks_seconds", 0) for c in chunks),
            )
        backend.last_query_encoding_seconds = sum(
            c["query_encoding_seconds"] for c in self.calls
        )
        keys = set().union(*(c["query_encoding"].keys() for c in self.calls))
        backend.last_query_encoding_cache = {
            k: sum(c["query_encoding"].get(k, 0) for c in self.calls) for k in keys
        }


def _packed(session, queries, packer):
    ranked = session.retrieve(queries, QUOTAS[len(queries)])
    return packer.pack(round_robin_evidence(queries, ranked))


def _source_first(source, backend, session, packer, original, log):
    """Fill a spare source slot, otherwise compare at most eight combinations."""
    old = source["old_indices"]
    plan = None
    if 0 < len(old) < 4:
        detail = analyze(source, backend.parser)
        plan = spare_source(source, detail, "spare_source", backend.tokenizer)
        log["source"] = dict(reason=plan["reason"], sources=detail["sources"])
        if plan["reason"] is not None:
            plan = None
    prepared = None
    if plan is None and old:
        prepared = combinations.prepare(source, session.files, backend.parser)
        log["interface_fallback_reason"] = prepared["fallback_reason"]
        if prepared["fallback_reason"] or not combinations.strong_interface(prepared):
            prepared = None
    indices = old
    branch = "original"
    if prepared is None:
        if plan is not None:
            indices, branch = plan["indices"], "source"
            queries = combinations.queries(source["candidates"], indices)
        else:
            queries = original
        blocks, tokens = _packed(session, queries, packer)
    else:
        pool = prepared["pool"]
        queries = combinations.queries(source["candidates"], pool)
        groups = session.retrieve(queries, [30] * len(pool))
        by_index = dict(zip(pool, groups))
        old_ranked = [by_index[i][:n] for i, n in zip(old, QUOTAS[len(old)])]
        base, tokens = packer.pack(round_robin_evidence(original, old_ranked))
        rankings = [[] for _ in range(len(source["candidates"]) + 1)]
        for i, group in by_index.items():
            rankings[i + 1] = [dict(id=b["id"], score=b["score"]) for b in group]
        bank = dict(
            source,
            old_evidence=base,
            rankings=rankings,
            chunks={b["id"]: b for group in groups for b in group},
        )
        proposed = combinations.select(bank, prepared, packer)["headers"]
        accepted, comparison = combinations.compare(
            prepared, dict(indices=old, blocks=base), proposed
        )
        log["interface_comparison"] = comparison
        log["interface_combinations"] = [list(c) for c in prepared["combinations"]]
        blocks = base
        if accepted["specific_interface_gain"]:
            indices, blocks, branch = (
                proposed["indices"],
                proposed["blocks"],
                "interface",
            )
            tokens = len(
                packer.tokenizer.encode(
                    render_evidence(blocks), add_special_tokens=False
                )
            )
    return dict(
        indices=list(indices),
        groups=[[i] for i in indices],
        blocks=blocks,
        tokens=tokens,
        prompt_hash=digest(render_evidence(blocks)),
        branch=branch,
    )


def _keep_heads(source, backend, session, packer, current, log):
    if len(current['indices']) < 3:
        log['chain'] = dict(reason='fewer_than_three_supports')
        return current
    detail = prepare_chain(source, backend.parser, current)
    proposal = protected_chain(
        source,
        detail,
        current,
        source["cosines"],
        "protected_members",
        backend.tokenizer,
    )
    if proposal is None:
        log["chain"] = dict(reason="no_chain_proposal")
        return current
    queries = interfaces.queries(source, proposal["groups"])
    if not session.fits(queries):
        log["chain"] = dict(reason="query_budget_exhausted")
        return current
    blocks, tokens = _packed(session, queries, packer)
    proposal.update(
        blocks=blocks, tokens=tokens, prompt_hash=digest(render_evidence(blocks))
    )
    accepted, guard = interfaces.preserve_headers(
        interfaces.prepare_owner(source, session.files, backend.parser),
        current,
        proposal,
        backend.parser,
        "keep_heads",
    )
    log["chain"] = dict(guard, pair=proposal["pair"], accepted=accepted)
    return dict(proposal, branch="chain") if accepted else current


def _normalized(values):
    values = values.numpy() if hasattr(values, "numpy") else np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid API-first encoder output")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero API-first encoder vector")
    return values / np.maximum(1e-12, norms)


def _api_first(source, backend, session, packer, current, log, state=None, interface_policy=None):
    groups = current.get('groups') or [[i] for i in current['indices']]
    # The replacement rule below preserves the first two queries. With fewer
    # than three there is no replaceable position, regardless of API scores.
    if len(groups) < 3:
        log['api'] = dict(reason='fewer_than_three_queries')
        return current
    p = interfaces.prepare(
        source, session.files, backend.parser, current, backend.tokenizer
    )
    allowed = list(session.rankings)
    if (
        p["purpose_query"] is not None
        and p["purpose_query"] not in allowed
        and len(allowed) < session.limit
    ):
        allowed.append(p["purpose_query"])
    if not p["apis"] or not p["qs"]:
        log["api"] = dict(
            reason="no_resolved_interfaces" if not p["apis"] else "no_supports"
        )
        return current
    anchor_texts = [q["text"] for q in p["qs"][:2]]
    purpose = p["purpose_query"] if p["purpose_query"] in allowed else anchor_texts[0]
    anchor_texts = list(dict.fromkeys([*anchor_texts, purpose]))
    stats = {}
    if getattr(backend, "cache_query_vectors", False):
        anchors = cached_encode(backend, anchor_texts, statistics=stats)
    else:
        anchors = backend.encode(anchor_texts)
    by_text = dict(zip(anchor_texts, _normalized(anchors)))
    api_stats = {}
    api_vectors = _normalized(cached_encode(backend, p["texts"], statistics=api_stats))
    first = by_text[p["qs"][0]["text"]]
    second = by_text[p["qs"][min(1, len(p["qs"]) - 1)]["text"]]
    average = (first + second) / max(1e-12, np.linalg.norm(first + second))
    anchors = dict(
        api_first_support=first,
        api_two_supports=average,
        api_visible_purpose=by_text[purpose],
    )
    # All three hypotheses contribute candidates in the validated algorithm.
    # Only the first-support hypothesis decides the final replacement.
    scores = {v: np.clip(api_vectors @ anchors[v], -1, 1).tolist()
              for v in interfaces.VERSIONS}
    thresholds = (dict(minimum_similarity=interface_policy.minimum_similarity,
                       minimum_margin=interface_policy.minimum_margin)
                  if interface_policy is not None else {})
    predictions = {v: interfaces.predict(p, scores[v], **thresholds) for v in interfaces.VERSIONS}
    if state is not None:
        # Reuse this exact pre-API owner table and scores after the baseline
        # completes; never encode the same interface/query a second time.
        state.update(prepared=p, first_scores=scores['api_first_support'])
    log["api"] = dict(
        reason="no_confident_interface",
        hypotheses=predictions,
        anchor_encoding=stats,
        interface_encoding=api_stats,
    )
    if predictions['api_first_support'] is None:
        log['api']['reason'] = 'no_confident_first_support_interface'
        return current
    for i in interfaces.pool(source, p, current, predictions):
        text = source["candidates"][i]["query_text"]
        if text not in allowed and len(allowed) < session.limit:
            allowed.append(text)
    by_text = {}
    for i in sorted(
        range(len(source["candidates"])), key=lambda i: (i not in current["indices"], i)
    ):
        by_text.setdefault(source["candidates"][i]["query_text"], i)
    indices = [by_text[text] for text in allowed if text in by_text]
    _, indices = interfaces.replacement_candidates(source, current, indices)
    # Reserve the same bounded candidate slots as before, but retrieve only
    # candidates the final replacement rule can actually use. Purpose-only and
    # fixed-group queries cannot affect that decision. Current group rankings
    # are already present from earlier stages.
    needed = list(dict.fromkeys(source['candidates'][i]['query_text'] for i in indices))
    log['api']['reserved_query_count'] = len(allowed)
    log['api']['unused_query_count'] = len(set(allowed) - session.rankings.keys() - set(needed))
    session.retrieve(
        [
            dict(id="trial", text=text, role="support", segment_ids=[])
            for text in needed
        ],
        [30] * len(needed),
    )
    proposed = interfaces.select(
        source,
        p,
        current,
        predictions["api_first_support"],
        indices,
        session.rankings,
        packer,
        backend.parser,
    )
    log["api"].update(
        reason=(
            "improved_interface"
            if proposed is not None
            else "no_better_preserving_combination"
        ),
        pool=indices,
    )
    if proposed is None:
        return current
    log["api"].update(
        {
            key: proposed[key]
            for key in (
                "old_api_quality",
                "api_quality",
                "guard",
                "score",
                "drop_query_position",
            )
        }
    )
    return dict(proposed, branch="api_first_support")


def _callable_first(source, backend, session, packer, current, log, state, policy, interface_policy=None):
    if not state:
        log['callable'] = dict(configuration=asdict(policy),
                               reason='no_resolved_interface_or_replaceable_group',
                               packed_options=0, additional_queries=0,
                               queries=len(session.rankings))
        return current
    if 'prompt_hash' not in current:
        # Successful API replacement returns blocks
        # without a hash; the new stage's declaration guard needs this field.
        current = dict(current, prompt_hash=digest(render_evidence(current['blocks'])))
    from choruscoder_callable_support import select

    thresholds = (dict(minimum_similarity=interface_policy.minimum_similarity,
                       minimum_margin=interface_policy.minimum_margin)
                  if interface_policy is not None else {})
    proposed, detail = select(source, state['prepared'], current,
                              state['first_scores'], session, packer, backend.parser, **thresholds)
    log['callable'] = dict(detail, configuration=asdict(policy))
    return dict(proposed, branch='callable_replace_one') if proposed is not None else current


def materialize(source, groups, baseline, fused):
    """Represent joined raw pieces as a standard extensible support unit."""
    original = {c["candidate_id"]: c for c in baseline}
    chosen = []
    candidates = source["candidates"]
    for j, group in enumerate(groups):
        members = [candidates[i] for i in group]
        spans = merged([span for c in members for span in c["context_spans"]])
        if len(group) == 1:
            c = dict(original.get(members[0]["id"], members[0]))
            c.setdefault("candidate_id", members[0]["id"])
        else:
            c = dict(
                members[0],
                candidate_id="+".join(m["id"] for m in members),
                text=source_query(source["prefix"], spans),
                spans=spans,
                context_spans=spans,
                line_start=min(m["line_start"] for m in members),
                line_end=max(m["line_end"] for m in members),
                unit_kind="pair",
                member_ids=[m["id"] for m in members],
                query_text=source_query(
                    source["prefix"],
                    spans + source["structure"]["prefix_target_context_spans"],
                ),
            )
        c.update(id=f"S{j+1}", raw_candidate_indices=list(group))
        chosen.append(c)
    return chosen


def retrieve_supports(request, selection, backend, tokenizer, profile, semantic=None):
    """Run the frozen method through normal backends; errors always propagate."""
    policy = profile.api_support_config
    started = time.perf_counter()
    structure = selection["structure"]
    encoding = {}
    score_started = time.perf_counter()
    baseline, semantic_trace, fused_trace = score_fused_supports(
        selection["semantic_pool"],
        structure,
        lambda texts: cached_encode(backend, texts, statistics=encoding),
        semantic or profile.semantic_support_config,
        profile.fused_support_config,
    )
    finalize_fused_fallback(structure, request["prefix"], backend.tokenizer, baseline)
    # Same query constructor as the parent, including its 508-token fallback.
    from choruscoder_profiles import CALLABLE_REPLACE_ONE_METHOD

    original, _ = control_queries(
        CALLABLE_REPLACE_ONE_METHOD,
        selection["window"],
        structure,
        baseline,
        prefix=request["prefix"],
    )
    scoring_seconds = time.perf_counter() - score_started
    source = dict(
        prefix=request["prefix"],
        path=request["path"],
        target_T=structure["target"],
        structure=structure,
        window=selection["window"],
        candidates=selection["semantic_pool"]["candidates"],
        old_pool=selection["semantic_pool"],
        old_indices=semantic_trace["selected_indices"],
        old_cosines=semantic_trace["cosines"],
        cosines=semantic_trace["cosines"],
    )
    for c in source["candidates"]:
        if (
            source_query(source["prefix"], c["context_spans"]) != c["text"]
            or source_query(
                source["prefix"],
                c["context_spans"] + structure["prefix_target_context_spans"],
            )
            != c["query_text"]
        ):
            raise ValueError("API-first candidate differs from visible source")
    session = QuerySession(backend, request["files"], policy.trial_queries)
    packer = ContextPacker(tokenizer)
    log = dict(
        configuration=asdict(policy), original_indices=list(source["old_indices"])
    )
    stages = {}
    current = None
    callable_policy = profile.callable_support_config
    # Pass configurable thresholds only to the encoder comparison profiles.
    threshold_options = ({'interface_policy': policy}
                         if policy.version == 'api_first_support_thresholds_v2' else {})
    api_state = {} if callable_policy is not None else None
    try:
        steps = [
            (
                "source_first",
                lambda: _source_first(source, backend, session, packer, original, log),
            ),
            (
                "keep_heads",
                lambda: _keep_heads(source, backend, session, packer, current, log),
            ),
            (
                "api_first_support",
                lambda: (_api_first(source, backend, session, packer, current, log, **threshold_options)
                         if api_state is None else
                         _api_first(source, backend, session, packer, current, log, state=api_state, **threshold_options)),
            ),
        ]
        if callable_policy is not None:
            steps.append(('callable_replace_one', lambda: _callable_first(
                source, backend, session, packer, current, log, api_state, callable_policy, **threshold_options)))
        for name, step in steps:
            begin = time.perf_counter()
            current = step()
            stages[name] = time.perf_counter() - begin
        chosen = materialize(source, current["groups"], baseline, fused_trace)
        finalize_fused_fallback(structure, request["prefix"], backend.tokenizer, chosen)
        queries, _ = control_queries(
            CALLABLE_REPLACE_ONE_METHOD,
            selection["window"],
            structure,
            chosen,
            prefix=request["prefix"],
        )
        # All final queries must already have been tried; this is a pure lookup.
        if any(q["text"] not in session.rankings for q in queries):
            raise ValueError("API-first final query has no trial ranking")
        if profile.callable_ablation_config is not None:
            from choruscoder_callable_ablations import final_queries
            selection_queries = queries
            selection_count = len(session.rankings)
            queries, query_trace = final_queries(profile,
                request['prefix'], structure, chosen, selection_queries, backend.tokenizer)
            missing = {q['text'] for q in queries} - session.rankings.keys()
            merged_final = profile.callable_ablation_config.query_composition == 'merged'
            if len(missing) > int(merged_final):
                raise ValueError('Callable finalization requires unexpected new queries')
            # Selection has finished. Only the terminal merged query may use
            # this extra slot; all selection stages above keep the eight limit.
            session.limit = policy.trial_queries + int(merged_final)
            log['ablation'] = dict(configuration=asdict(profile.callable_ablation_config),
                selection_queries=selection_queries, selection_trial_query_count=selection_count,
                final_query_added=len(missing), queries=query_trace)
        ranked = session.retrieve(queries, QUOTAS[len(queries)])
        log.update(
            selected_indices=current["indices"],
            groups=current["groups"],
            raw_support_count=len(current["indices"]),
            final_query_count=len(queries),
            trial_query_count=len(session.rankings),
            trials=session.trials,
            branch=current["branch"],
            scoring_seconds=scoring_seconds,
            stage_seconds=stages,
            retrieval_calls=session.calls,
            query_seconds=session.query_seconds,
            total_seconds=time.perf_counter() - started,
        )
        return (
            dict(
                chosen=chosen,
                queries=queries,
                ranked=ranked,
                semantic=semantic_trace,
                fused=fused_trace,
                encoding=encoding,
            ),
            log,
        )
    finally:
        session.publish_timings()


def validate_selection(record, config, tokenizer=None):
    """Audit cached raw ranges, final ranking reuse, trial budget and policy."""
    from choruscoder_profiles import PROFILES, effective_api_support_config

    d = record["diagnostics"]
    trace = d["api_support"]
    configured = config.get('api_support_config') or {}
    if PROFILES[config['method']].encoder_config is not None:
        policy = asdict(effective_api_support_config(config['method'],
            configured.get('minimum_similarity'), configured.get('minimum_margin')))
    else:
        policy = asdict(PROFILES[config["method"]].api_support_config)
    if trace["configuration"] != policy or config.get("api_support_config") != policy:
        raise ValueError("API-first policy mismatch")
    ablation = PROFILES[config['method']].callable_ablation_config
    control = trace.get('ablation')
    selection_count = trace['trial_query_count']
    if ablation is not None:
        if (not control or control.get('configuration') != asdict(ablation) or
                config.get('callable_ablation_config') != asdict(ablation)):
            raise ValueError('Callable ablation policy mismatch')
        selection_count = control.get('selection_trial_query_count')
        added = control.get('final_query_added')
        if (type(selection_count) is not int or not 1 <= selection_count <= 8 or
                type(added) is not int or not 0 <= added <= int(ablation.query_composition == 'merged') or
                trace['trial_query_count'] != selection_count + added):
            raise ValueError('Callable selection/final query budgets invalid')
    elif control is not None:
        raise ValueError('Unexpected callable ablation log')
    callable_policy = PROFILES[config['method']].callable_support_config
    if callable_policy is not None:
        detail = trace.get('callable', {})
        if (config.get('callable_support_config') != asdict(callable_policy) or
                detail.get('configuration') != asdict(callable_policy) or
                type(detail.get('additional_queries')) is not int or
                not 0 <= detail['additional_queries'] <= callable_policy.maximum_extra_queries or
                type(detail.get('packed_options')) is not int or
                not 0 <= detail['packed_options'] <= callable_policy.maximum_combinations or
                detail.get('queries') != selection_count):
            raise ValueError('Callable replacement policy or bounded-work log mismatch')
    groups, indices = trace["groups"], trace["selected_indices"]
    candidates = d["semantic_support"]["candidates"]
    if (
        indices != [i for g in groups for i in g]
        or len(indices) > 4
        or len(set(indices)) != len(indices)
        or any(not g for g in groups)
        or any(type(i) is not int or not 0 <= i < len(candidates) for i in indices)
    ):
        raise ValueError("API-first raw support indices invalid")
    trials = trace["trials"]
    by_text = {t["query"]["text"]: t["ranked"] for t in trials}
    if (
        not 1 <= len(trials) <= 8 + int(ablation is not None and ablation.query_composition == 'merged')
        or len(by_text) != len(trials)
        or trace["trial_query_count"] != len(trials)
        or trace["raw_support_count"] != len(indices)
        or trace["final_query_count"] != (1 if ablation is not None and ablation.query_composition == 'merged' else max(1, len(groups)))
        or any(len(t["ranked"]) > 30 for t in trials)
    ):
        raise ValueError("API-first trial counts invalid")
    # The original GaussRank decision is still independently checked.
    from choruscoder_selection_controls import validate_fused_selection

    baseline = validate_fused_selection(record, config)
    prefix = record.get('prefix', record['window']['text'])
    source = dict(
        prefix=prefix, candidates=candidates, structure=d["ablation"]
    )
    chosen = materialize(source, groups, baseline, d["fused_support"])
    queries, _ = control_queries(
        config["method"],
        record["window"],
        dict(d["ablation"], fallback=record["fallback"]),
        chosen,
        prefix=prefix,
    )
    if ablation is not None:
        selection_texts = {t['query']['text'] for t in trials[:selection_count]}
        if control.get('selection_queries') != queries or any(q['text'] not in selection_texts for q in queries):
            raise ValueError('Callable selection queries do not reuse selection trials')
        from choruscoder_callable_ablations import final_queries
        queries, expected = final_queries(PROFILES[config['method']], prefix,
            dict(d['ablation'], fallback=record['fallback']), chosen, queries,
            tokenizer, control.get('queries'))
        if (control.get('queries') != expected or
                control['final_query_added'] != len({q['text'] for q in queries} - selection_texts) or
                {t['query']['text'] for t in trials[selection_count:]} != {q['text'] for q in queries} - selection_texts):
            raise ValueError('Callable terminal query trace mismatch')
    quotas = QUOTAS[len(queries)]
    if (
        record['queries'] != queries or d["recall_quotas"] != quotas
        or any(q["text"] not in by_text for q in queries)
        or d["ranked"] != [by_text[q["text"]][:n] for q, n in zip(queries, quotas)]
    ):
        raise ValueError("API-first final rankings were not reused from trials")
    return chosen
