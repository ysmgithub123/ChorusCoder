"""Static receiver bindings and scope-aware names from the visible Python prefix."""
import ast
import builtins
import re
from choruscoder import make_parser
from choruscoder_far import PrefixFacts, path as node_path, path as dotted_path
from choruscoder_scoring import _scope, _parents, _read_identifier, CONDITIONAL
from choruscoder_dependency_support import cursor_statement, ReceiverBindings

PLACEHOLDER = '__dependency_audit_cursor__'
PADDING = '__offline_member_placeholder__'
RECEIVER = re.compile(r'(?<![\w.])(?P<receiver>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\.(?P<partial>[A-Za-z_]\w*)?\s*$')

def enclosing(node,kind):
    return next((p for p in _parents(node) if p.type==kind),None)

def unsafe(node):
    return node.has_error or any(p.type in CONDITIONAL|{'ERROR','except_clause','with_statement'} for p in _parents(node))

def key(entry):
    return (tuple(entry['scope']),entry['name'],entry['start']) if entry else None

def creator(index,node):
    rhs=node.child_by_field_name('right')
    return node_path(rhs.child_by_field_name('function'),index.raw) if rhs is not None and rhs.type=='call' else None

def extract(prefix,candidates,selected,parser=None):
    result={'receiver':None,'status':None,'bindings':[],'core_new_candidates':[],
            'expanded_new_candidates':[],'naive_definition':None,'naive_status':None}
    match=RECEIVER.search(prefix)
    if not match:
        result['status']='no_simple_terminal_receiver';return result
    receiver=match['receiver'];result['receiver']=receiver
    patched=prefix+PLACEHOLDER
    index=PrefixFacts(patched,parser or make_parser('python'))
    # Appending the placeholder repairs only the unknown final member, never
    # imports a future statement or a target answer into the selector.
    member=(match['partial'] or '')+PLACEHOLDER
    attr=next((n for n in reversed(index.nodes) if n.type=='attribute' and
        n.child_by_field_name('attribute') is not None and index.text(n.child_by_field_name('attribute'))==member),None)
    if attr is None:
        result['status']='receiver_ast_unresolved';return result
    node=attr.child_by_field_name('object')
    if node is None or index.text(node)!=receiver:
        result['status']='receiver_ast_mismatch';return result
    limit=len(prefix.encode('utf-8'))
    result['parse_has_error_after_placeholder']=index.root.has_error
    scope=_scope(node);owner=enclosing(node,'class_definition');function=enclosing(node,'function_definition')
    root=receiver.split('.')[0]
    bound=index.binding(root,node)
    entries=[]

    def add(entry,confidence,relation,tree_node=None):
        if entry and entry['span'][1]<=limit:
            item=dict(entry,scope=list(entry['scope']),confidence=confidence,relation=relation,
                code=index.raw[entry['span'][0]:entry['span'][1]].decode('utf-8'))
            if tree_node is not None:item['producer']=creator(index,tree_node)
            origin=item.get('origin')
            if origin and origin[0]=='import':
                item['provider']={'usage':'receiver_import','import_origin':origin,'bound_name':item['name'],'tail':[]}
            else:
                produced=item.get('producer') or (origin[1] if origin and origin[0]=='producer' else None)
                annotation=item.get('annotation')
                path=produced or annotation
                if path:
                    parts=path.split('.')
                    at=tree_node or index.root.descendant_for_byte_range(item['start'],item['start']+1)
                    imported=index.binding(parts[0],at)
                    if imported and imported.get('origin') and imported['origin'][0]=='import':
                        item['provider']={'usage':'constructor' if produced else 'annotation',
                            'import_origin':imported['origin'],'bound_name':imported['name'],'tail':parts[1:]}
            entries.append(item)

    if '.' not in receiver and receiver not in {'self','cls'}:
        if bound is None:
            result['status']='binding_unknown_or_unsafe'
        elif bound['parameter'] and bound.get('annotation') is None:
            result['status']='untyped_parameter_not_forced'
            add(bound,'weak_parameter','parameter')
        elif bound['parameter']:
            result['status']='typed_parameter_binding'
            add(bound,'core','typed_parameter')
        else:
            result['status']='lexical_binding'
            definition=next((n for n in index.nodes if n.start_byte==bound['start'] and n.end_byte>=bound['span'][1]),None)
            add(bound,'core','import' if bound.get('origin') and bound['origin'][0]=='import' else 'definition',definition)
    elif receiver.startswith('self.'):
        # A member's initializer is not proven by resolving the name "self".
        # Current-method writes are core; __init__ writes require an explicit
        # object-initialization assumption and are counted separately.
        writes=[]
        for n in index.nodes:
            if n.type not in {'assignment','augmented_assignment'} or n.end_byte>limit:continue
            left=n.child_by_field_name('left')
            if left is None or node_path(left,index.raw)!=receiver:continue
            if owner is None or enclosing(n,'class_definition')!=owner:continue
            fn=enclosing(n,'function_definition')
            if fn is None:continue
            fn_name=fn.child_by_field_name('name')
            writes.append((n,fn,index.text(fn_name) if fn_name is not None else None))
        current=[item for item in writes if item[1]==function]
        if bound is None or not bound['parameter']:
            result['status']='self_binding_not_parameter'
        elif current:
            n,fn,name=max(current,key=lambda z:z[0].end_byte)
            if unsafe(n):result['status']='current_member_write_conditional'
            else:
                result['status']='current_method_member_binding'
                add({'name':receiver,'scope':scope,'start':n.start_byte,'available':n.end_byte,
                    'span':[n.start_byte,n.end_byte],'unsafe':False,'origin':None,'annotation':None,'parameter':False},
                    'core','member_current_method',n)
        else:
            init=[item for item in writes if item[2]=='__init__']
            other=[item for item in writes if item[2]!='__init__']
            if len(init)==1 and not unsafe(init[0][0]) and not other:
                n,fn,name=init[0];result['status']='initializer_assumption_required'
                add({'name':receiver,'scope':_scope(n),'start':n.start_byte,'available':n.end_byte,
                    'span':[n.start_byte,n.end_byte],'unsafe':False,'origin':None,'annotation':None,'parameter':False},
                    'initializer_assumption','member_initializer',n)
            else:result['status']='member_binding_ambiguous_or_unavailable'
    elif receiver in {'self','cls'}:
        result['status']='self_is_not_a_useful_definition'
        if owner is not None:
            bases=owner.child_by_field_name('superclasses')
            if bases is not None:
                base_names=[node_path(n,index.raw) for n in bases.named_children]
                for name in base_names:
                    if not name or '.' in name or name in {'object','type'}:continue
                    base=index.binding(name,owner)
                    if base and base.get('origin'):
                        add(base,'inheritance_context','base_class')
    else:
        result['status']='attribute_chain_requires_further_analysis'

    # A deliberately loose comparator: nearest textual definition of a bare
    # receiver, regardless of scope/conditional control flow. It is not the
    # recommended rule. Only provable scope differences are called mismatches.
    if '.' not in receiver and receiver not in {'self','cls'}:
        possible=[e for (s,name),values in index.writes.items() if name==receiver for e in values
                  if e['available']<=node.start_byte and e['span'][1]<=limit and not e['parameter']]
        nearest=max(possible,key=lambda e:e['available']) if possible else None
        if nearest:
            naive=dict(nearest,scope=list(nearest['scope']),code=index.raw[nearest['span'][0]:nearest['span'][1]].decode('utf-8'))
            result['naive_definition']=naive
            if (tuple(nearest['scope']),receiver) in index.declared:
                # A global/nonlocal assignment in another function is not a
                # separate local binding. Its execution order is unknown.
                result['naive_status']='cannot_certify_binding'
                naive['declaration_requires_cross_function_execution_analysis']=True
            elif bound is not None and key(nearest)!=key(bound):
                result['naive_status']='different_from_visible_binding'
            elif bound is None:
                result['naive_status']='cannot_certify_binding'
            elif nearest['unsafe']:
                result['naive_status']='conditional_or_unsafe'
            else:result['naive_status']='same_visible_binding'
        else:result['naive_status']='no_definition'

    selected_ids={s.get('candidate_id') for s in selected}
    selected_spans=[span for s in selected for span in s['spans']]
    def covered(spans,a,b):return any(x<=a and b<=y for x,y in spans)
    unique={}
    for entry in entries:
        a,b=entry['span']
        matches=[(i,c) for i,c in enumerate(candidates,1) if covered(c['spans'],a,b)]
        entry['already_selected_witness']=covered(selected_spans,a,b)
        entry['mapped_candidates']=[f'C{i}' for i,c in matches]
        entry['candidate_pool_status']='mapped' if matches else 'definition_crosses_or_missing_candidate_boundary'
        for i,candidate in matches:
            ident=candidate['id']
            if ident not in unique:
                unique[ident]={'label':f'C{i}','candidate':candidate,'relations':[],
                    'already_selected':ident in selected_ids or any(candidate['spans']==s['spans'] for s in selected)}
            unique[ident]['relations'].append({'confidence':entry['confidence'],'relation':entry['relation'],
                'binding_code':entry['code'],'binding_span':entry['span'],'origin':entry.get('origin'),
                    'annotation':entry.get('annotation'),'producer':entry.get('producer'),
                    'provider':entry.get('provider'),'witness_already_selected':entry['already_selected_witness']})
    result['bindings']=entries
    result['dependency_candidates']=list(unique.values())
    result['core_new_candidates']=[v for v in unique.values() if not v['already_selected'] and
        any(r['confidence']=='core' and not r['witness_already_selected'] for r in v['relations'])]
    result['expanded_new_candidates']=[v for v in unique.values() if not v['already_selected'] and
        any(r['confidence'] in {'core','initializer_assumption','inheritance_context'} and not r['witness_already_selected'] for r in v['relations'])]
    if result['naive_definition']:
        a,b=result['naive_definition']['span']
        result['naive_definition']['candidate_labels']=[f'C{i}' for i,c in enumerate(candidates,1) if covered(c['spans'],a,b)]
        result['naive_definition']['already_selected_witness']=covered(selected_spans,a,b)
    return result

def current_statement(prefix):
    anchor=cursor_statement(prefix)
    if anchor['status']=='empty_cursor_line':
        trimmed=prefix.rstrip('\r\n \t')
        probe=cursor_statement(trimmed)
        if probe.get('error') and 'multi-line statement' in probe['error'] and probe.get('spans'):
            start=probe['spans'][0][0]
            anchor=dict(probe,text=prefix.encode('utf-8')[start:].decode('utf-8'),
                        spans=[[start,len(prefix.encode('utf-8'))]])
    return anchor

def in_spans(spans,a,b):return any(x<=a and b<=y for x,y in spans)

class VisibleNames:
    """Keep identical names in different functions apart, including global."""
    def __init__(self,prefix,parser):
        self.prefix=prefix;self.raw=prefix.encode('utf-8');self.limit=len(self.raw)
        patched=prefix+PADDING if re.search(r'\.\s*$',prefix) else prefix
        self.facts=ReceiverBindings(patched,parser).index
        self.nodes=self.facts.nodes
        self.declarations={}
        for n in self.nodes:
            if n.type in {'global_statement','nonlocal_statement'}:
                for c in n.named_children:
                    if c.type=='identifier':self.declarations[(_scope(n),self.facts.text(c))]=n.type

    def lexical_key(self,name,node,definition=False):
        scope=_scope(node)
        if definition and node.parent and node.parent.type in {'function_definition','class_definition'} and node.parent.child_by_field_name('name')==node:
            scope=_scope(node.parent)
        for size in range(len(scope),-1,-1):
            here=scope[:size]
            declared=self.declarations.get((here,name))
            if declared=='global_statement':return ('lexical',(),name)
            if declared=='nonlocal_statement':continue
            if size<len(scope) and size and self.facts.scope_type(here[-1])=='class_definition':continue
            if (here,name) in self.facts.writes:return ('lexical',here,name)
        # Unknown names stay within their current function instead of linking
        # arbitrary unresolved locals across unrelated functions.
        return ('unresolved',scope,name)

    def self_attribute(self,node):
        """self.client.send uses self.client, not the word self everywhere."""
        if self.facts.text(node) not in {'self','cls'}:return None
        first=node.parent
        if first is None or first.type!='attribute' or first.child_by_field_name('object')!=node:return None
        member=first.child_by_field_name('attribute')
        if member is None or member.end_byte>self.limit:return None
        name=dotted_path(first,self.facts.raw)
        if not name:return None
        owner=next((p for p in _parents(node) if p.type=='class_definition'),None)
        # Only combine class members across methods if self/cls is the actual
        # parameter; a locally rebound self must not impersonate that object.
        bound=self.facts.binding(self.facts.text(node),node)
        if owner is not None and bound and bound.get('parameter'):
            key=('class_member',owner.start_byte,name)
        else:key=('object_member',self.lexical_key(self.facts.text(node),node),name)
        return name,key,first

    def active_variables(self,anchor):
        spans=anchor.get('spans',[]);found={}
        for n in self.nodes:
            if n.type!='identifier' or n.is_missing or n.end_byte>self.limit:continue
            if not in_spans(spans,n.start_byte,n.end_byte) or not _read_identifier(n):continue
            name=self.facts.text(n)
            if name==PADDING:continue
            member=self.self_attribute(n)
            if member:
                name,key,position=member
            else:key=self.lexical_key(name,n);position=n
            if key[0]=='unresolved' and name in vars(builtins):continue
            encoded=repr(key)
            if encoded not in found:
                found[encoded]={'name':name,'key':key,'key_text':encoded,
                    'span':[position.start_byte,min(position.end_byte,self.limit)]}
        return list(found.values())

    def occurrences(self,variables):
        wanted={v['key_text']:v['name'] for v in variables};out=[]
        # Imports are counted by their bound aliases, not every name in the
        # package path. `import utils as tools` binds tools, not utils.
        for (scope,name),entries in self.facts.writes.items():
            for e in entries:
                if not e.get('origin') or e['origin'][0]!='import' or e['span'][1]>self.limit:continue
                node=self.facts.root.descendant_for_byte_range(e['start'],e['start']+1)
                key=self.lexical_key(name,node);encoded=repr(key)
                if encoded in wanted:
                    statement=self.facts.raw[e['span'][0]:e['span'][1]].decode('utf-8')
                    try:
                        parsed=ast.parse(statement).body[0]
                        line_starts=[0]
                        for line in statement.splitlines(keepends=True):line_starts.append(line_starts[-1]+len(line.encode('utf-8')))
                        for alias in parsed.names:
                            bound=alias.asname or (alias.name.split('.')[0] if isinstance(parsed,ast.Import) else alias.name)
                            if bound!=name:continue
                            a=e['start']+line_starts[alias.lineno-1]+alias.col_offset
                            b=e['start']+line_starts[alias.end_lineno-1]+alias.end_col_offset
                            identifiers=[n for n in self.nodes if n.type=='identifier' and a<=n.start_byte<n.end_byte<=b and self.facts.text(n)==name]
                            if identifiers:
                                actual=identifiers[-1] if alias.asname else identifiers[0]
                                out.append({'name':name,'key_text':encoded,'span':[actual.start_byte,actual.end_byte],'kind':'import'})
                    except (SyntaxError,ValueError,AttributeError):pass
        for n in self.nodes:
            if n.type!='identifier' or n.is_missing or n.end_byte>self.limit:continue
            parents=list(_parents(n.parent))
            if any(p.type in {'comment','import_statement','import_from_statement','global_statement','nonlocal_statement'} for p in parents):continue
            if n.parent and n.parent.type=='attribute' and n.parent.child_by_field_name('attribute')==n:continue
            if n.parent and n.parent.type=='keyword_argument' and n.parent.child_by_field_name('name')==n:continue
            name=self.facts.text(n)
            member=self.self_attribute(n)
            if member:name,key,position=member
            else:key=self.lexical_key(name,n,definition=True);position=n
            encoded=repr(key)
            if encoded in wanted:
                out.append({'name':name,'key_text':encoded,'span':[position.start_byte,min(position.end_byte,self.limit)],'kind':'name'})
        return out

    def non_code_spans(self):
        spans=[]
        for n in self.nodes:
            if n.end_byte>self.limit:continue
            if n.type=='comment':spans.append([n.start_byte,n.end_byte])
            elif n.type in {'string','concatenated_string'} and n.parent and n.parent.type=='expression_statement':
                # Only standalone literal strings; calls, assignments, f-string
                # interpolation and mixed expression statements remain code.
                if len(n.parent.named_children)==1 and not any(p.type=='interpolation' for p in walk(n)):
                    spans.append([n.start_byte,n.end_byte])
        return spans

def walk(node):
    yield node
    for c in node.named_children:yield from walk(c)
