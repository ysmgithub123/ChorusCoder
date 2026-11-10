"""ReccEval's current-line protocol and the original DraCo metric definitions."""
import keyword
import re

from fuzzywuzzy import fuzz


EVALUATION_VERSION = 'recceval_first_physical_line_v1'
IDENTIFIER = re.compile(r'[_a-zA-Z][_a-zA-Z0-9]*')
STRINGS = re.compile(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'')
KEYWORDS = frozenset(k for k in keyword.kwlist if k not in ('True', 'False'))


def identifiers(code):
    # Match the order of operations in DraCo/experiments/evaluator.py.
    code = re.sub(r'#.*', '', code)
    code = STRINGS.sub('', code)
    return [token for token in re.findall(r'\w+', code)
            if IDENTIFIER.match(token) and token not in KEYWORDS]


def score_completion(raw_prediction, target):
    # Do not skip blank lines or complete an AST statement using future lines.
    prediction = re.split(r'\r\n|\n|\r', raw_prediction, maxsplit=1)[0]
    pred_ids, target_ids = identifiers(prediction), identifiers(target)
    pred_set, target_set = set(pred_ids), set(target_ids)
    denominator = len(pred_set) + len(target_set)
    return dict(pred=prediction, target=target,
                code_em=int(prediction.split() == target.split()),
                code_es=float(fuzz.ratio(prediction, target)),
                identifier_em=int(pred_ids == target_ids),
                identifier_f1=2 * len(pred_set & target_set) / denominator if denominator else 0.0)
