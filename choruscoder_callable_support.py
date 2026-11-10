"""Bounded raw S replacement/priority, using the existing query-session memo.

Only visible prefix, allowed repository source and retrieved blocks are inputs.
Implements CallableReplaceOne after API-first support selection.
"""
import re

from choruscoder import round_robin_evidence
from choruscoder_chunk_controls import fully_covered
from choruscoder_api_support_combinations import QUOTAS, parameters
import choruscoder_api_support_interfaces as api


def select(source, prepared, current, first_scores, session, packer, parser, *,
           minimum_similarity=0.4, minimum_margin=0.015):
    eligible = [i for i, a in enumerate(prepared['apis'])
                if a['kind'] in {'function', 'class'}]
    prediction = api.predict(
        dict(apis=[prepared['apis'][i] for i in eligible]),
        [first_scores[i] for i in eligible], minimum_similarity=minimum_similarity,
        minimum_margin=minimum_margin) if eligible else None
    before = len(session.rankings)
    if prediction is None or len(current['groups']) < 3:
        return None, dict(reason='no_confident_target', queries=before,
                          packed_options=0, additional_queries=0)

    interface = prediction['api']
    call_cache = {}
    old_quality = api.quality(current['blocks'], interface, parser, call_cache)
    options, comparisons = [], []

    def consider(groups, reason):
        indices = [i for group in groups for i in group]
        if len(indices) > 4 or len(set(indices)) != len(indices):
            return
        qs = api.queries(source, groups)
        if any(q['text'] not in session.rankings for q in qs):
            return
        ranked = session.retrieve(qs, QUOTAS[len(qs)])
        blocks, tokens = packer.pack(round_robin_evidence(qs, ranked))
        quality = api.quality(blocks, interface, parser, call_cache)
        comparison = dict(groups=groups, proposal=reason, api_quality=quality,
                          old_api_quality=old_quality, accepted=False)
        comparisons.append(comparison)
        if not quality[0] or quality <= old_quality:
            return
        guard_ok, guard = api.preserve_headers(
            prepared['owner'], current,
            dict(prompt_hash='quota_proposal', blocks=blocks), parser, 'keep_heads')
        # Owner-header retention breaks ties after target-interface coverage.
        comparison.update(accepted=True, guard=guard)
        options.append(dict(indices=indices, groups=groups, blocks=blocks,
                            tokens=tokens, prediction=prediction, api_quality=quality,
                            old_api_quality=old_quality, guard=guard, proposal=reason,
                            score=[*quality, guard_ok]))

    for j in range(1, len(current['groups'])):
        consider([current['groups'][j]] + [g for k, g in enumerate(current['groups']) if k != j],
                 'promote_existing_support')

    cs = source['candidates']
    drop = min(range(len(current['groups'])), key=lambda j: (
        sum(source['cosines'][i] for i in current['groups'][j]) / len(current['groups'][j]), -j))
    fixed = [i for j, group in enumerate(current['groups']) if j != drop for i in group]
    occupied = source['structure']['prefix_target_context_spans'] + [
        span for i in fixed for span in cs[i]['context_spans']]
    # Encode only the provider file, not every file in the repository again.
    provider = next(f['text'].encode('utf-8') for f in session.files
                    if f['path'] == interface['path'])
    names = set().union(*(parameters(provider[a:z].decode('utf-8'))
                          for a, z in interface['headers']))
    names -= {'self', 'cls', 'data', 'value', 'args', 'kwargs', 'item', 'result'}
    candidates = [i for i, c in enumerate(cs) if i not in current['indices']
                  and not fully_covered(c['context_spans'], occupied)
                  and (len(session.rankings) < session.limit or c['query_text'] in session.rankings)]

    def score(i):
        identifiers = set(re.findall(r'\b[A-Za-z_]\w*\b', cs[i]['text']))
        return [interface['name'] in identifiers, len(names & identifiers),
                i in (prepared['purpose']['doc'], prepared['purpose']['head']),
                source['cosines'][i], -i]

    replacement = max(candidates, key=score) if candidates else None
    if replacement is not None:
        query = api.queries(source, [[replacement]])
        session.retrieve(query, [30])
        consider([[replacement]] + [g for j, g in enumerate(current['groups']) if j != drop],
                 'one_raw_replacement_promoted')
    if len(comparisons) > 4 or len(session.rankings) - before > 1:
        raise ValueError('Callable replacement exceeded its bounded work')
    selected = max(options, key=lambda x: x['score']) if options else None
    detail = dict(reason='selected' if selected is not None else 'no_better_combination',
                  prediction=prediction, queries=len(session.rankings),
                  packed_options=len(comparisons), comparisons=comparisons,
                  replacement=replacement, additional_queries=len(session.rankings) - before)
    if selected is not None:
        detail.update(selected_groups=selected['groups'], proposal=selected['proposal'],
                      api_quality=selected['api_quality'], old_api_quality=old_quality,
                      guard=selected['guard'])
    return selected, detail
