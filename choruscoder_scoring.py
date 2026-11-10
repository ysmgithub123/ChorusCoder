"""Visible Python clues and pre-retrieval selection of independent S+T queries.

Window AST relationships are metadata, never extra query text or extra witnesses.
This is a bounded local approximation, not reaching-definition analysis.
"""
from __future__ import annotations
SCOPES = {'function_definition', 'class_definition', 'lambda', 'list_comprehension', 'set_comprehension', 'dictionary_comprehension', 'generator_expression'}
STATEMENTS = {'expression_statement', 'return_statement', 'raise_statement', 'assert_statement', 'delete_statement', 'import_statement', 'import_from_statement', 'ERROR'}
ASSIGNMENTS = {'assignment', 'augmented_assignment', 'named_expression'}
CONDITIONAL = {'if_statement', 'for_statement', 'while_statement', 'try_statement', 'match_statement', 'for_in_clause', 'conditional_expression'}

def _parents(node):
    while node is not None:
        yield node
        node = node.parent

def _walk(node):
    yield node
    for child in node.named_children:
        yield from _walk(child)

def _scope(node):
    return tuple((n.start_byte for n in reversed(list(_parents(node.parent))) if n.type in SCOPES))

def _read_identifier(node):
    child = node
    for parent in _parents(node.parent):
        if parent.type in {'string', 'comment', 'parameters', 'lambda_parameters', 'import_statement', 'import_from_statement', 'as_pattern_target', 'global_statement', 'nonlocal_statement'}:
            return False
        if parent.type == 'attribute' and parent.child_by_field_name('attribute') == child:
            return False
        if parent.type == 'keyword_argument' and parent.child_by_field_name('name') == child:
            return False
        if parent.type in ASSIGNMENTS and (parent.child_by_field_name('left') or parent.child_by_field_name('name')) == child:
            if any((p.type in {'attribute', 'subscript'} for p in _parents(node.parent) if p.start_byte >= child.start_byte and p.end_byte <= child.end_byte)):
                return True
            return parent.type == 'augmented_assignment'
        if parent.type in {'function_definition', 'class_definition'} and parent.child_by_field_name('name') == child:
            return False
        if parent.type in {'for_statement', 'for_in_clause'} and parent.child_by_field_name('left') == child:
            return any((p.type in {'attribute', 'subscript'} for p in _parents(node.parent) if p.start_byte >= child.start_byte and p.end_byte <= child.end_byte))
        if parent.type == 'ERROR':
            children = parent.children
            if children and children[0].type in {'def', 'class'}:
                opening = next((c.start_byte for c in children if c.type == '('), parent.end_byte)
                if node.start_byte < opening or children[0].type == 'def':
                    return False
            equals = next((i for i, c in enumerate(children) if c.type == '='), None)
            if equals is not None and any((c.type in {'identifier', 'tuple', 'list', 'pattern_list'} and c.start_byte <= node.start_byte < c.end_byte for c in children[:equals])):
                return False
        child = parent
    return True
