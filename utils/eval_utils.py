# Copyright Amazon.com, Inc. or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import timeout_decorator
from fuzzywuzzy import fuzz
from nltk.tokenize import RegexpTokenizer
from utils.keywords.keywordlist import get_language_keywords
IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')
string_pattern = '"([^"\\\\]*(\\\\.[^"\\\\]*)*)"|\\\'([^\\\'\\\\]*(\\\\.[^\\\'\\\\]*)*)\\\''
code_tokenizer = RegexpTokenizer('\\w+')

def cal_edit_sim(references, hypotheses):
    total = len(references)
    edit_sim = 0.0
    for pred, gt in zip(hypotheses, references):
        pred = pred.strip()
        gt = gt.strip()
        edit_sim += fuzz.ratio(pred, gt)
    return edit_sim / total

def is_identifier(token, lang=None):
    return IDENTIFIER_REGEX.match(token) and (lang is None or token not in get_language_keywords(lang))

def extract_identifiers(source_code, lang):
    source_code_without_strings = re.sub(string_pattern, '', source_code)
    _ids = [t for t in code_tokenizer.tokenize(source_code_without_strings) if is_identifier(t, lang)]
    return _ids

@timeout_decorator.timeout(5)
def get_ast(parser, code):
    assert isinstance(code, str) or isinstance(code, bytes)
    if isinstance(code, str):
        code = bytes(code, 'utf8')
    try:
        tree = parser.parse(code)
        return tree
    except Exception:
        return None

def remove_comments(code):
    code = re.sub('#.*', '', code)
    code = re.sub('//.*', '', code)
    return code

def is_parse_valid(parser, code):

    def syntax_error(node):
        if node.type == 'ERROR':
            return True
        try:
            for child in node.children:
                if syntax_error(child):
                    return True
        except RecursionError:
            return True
        return False
    tree = get_ast(parser, code)
    if tree is not None:
        return not syntax_error(tree.root_node)
    return False

def get_python_one_statement(prompt, completion, parser):
    for i in range(len(completion)):
        code = prompt + completion[:i + 1]
        if not is_parse_valid(parser, code):
            continue
        if completion[i + 1] == '\n':
            return completion[:i + 1].rstrip()
    return completion

def postprocess_code_lines(prompt, completion, parser, lang):
    try:
        if lang == 'python':
            return get_python_one_statement(prompt, completion, parser)
    except Exception:
        return completion
