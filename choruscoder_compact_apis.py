"""Resolve prefix imports and describe all APIs of their local classes/modules."""
import ast
import io
import re
import tokenize
from functools import lru_cache

STOP = set('self cls def class return assert test tests testing the a an to of in and or is None True False async await get set init call method function with from import for if else not as str int float bool list dict tuple np torch tf utils data value result'.lower().split())

@lru_cache(maxsize=128)
def source_info(text):
    result={'classes':{},'members':set(),'aliases':{},'parseable':False,'absolute_imports':set()}
    try:
        tree=ast.parse(text);result['parseable']=True
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                result['members'].add(node.name)
            elif isinstance(node,(ast.Assign,ast.AnnAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target,ast.Name):
                        result['members'].add(target.id)
                        value=node.value
                        if isinstance(value,ast.Call) and isinstance(value.func,ast.Name):
                            result['aliases'][target.id]=value.func.id
            elif isinstance(node,ast.ImportFrom):
                for a in node.names:result['members'].add(a.asname or a.name)
            elif isinstance(node,ast.Import):
                result['absolute_imports'].update(a.name for a in node.names)
            if isinstance(node,ast.ClassDef):
                members=set()
                for child in node.body:
                    if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):members.add(child.name)
                    if isinstance(child,(ast.Assign,ast.AnnAssign)):
                        targets=child.targets if isinstance(child,ast.Assign) else [child.target]
                        for target in targets:
                            if isinstance(target,ast.Name):members.add(target.id)
                for child in ast.walk(node):
                    if isinstance(child,ast.Attribute) and isinstance(child.ctx,ast.Store) and isinstance(child.value,ast.Name) and child.value.id=='self':
                        members.add(child.attr)
                result['classes'][node.name]={'members':members,'bases':len(node.bases),'decorated':bool(node.decorator_list)}
    except (SyntaxError,ValueError,RecursionError):pass
    return result

def suffix_paths(files,parts):
    if not parts:return []
    suffix='/'.join(parts)
    alternatives=[suffix+'.py',suffix+'/__init__.py']
    return [p for p in files if any(p==a or p.endswith('/'+a) for a in alternatives)]

def provider_lookup(descriptor,current_path,files):
    """Resolve explicit local imports; uncertain factories stay unknown."""
    if not descriptor:return {'status':'no_static_provider','paths':[]}
    _,level,module,imported=descriptor['import_origin']
    imported_parts=imported.split('.')
    if module is None and level==0:
        # For `import a.b` bound to `a`, retain the root-package interpretation.
        module_parts=imported_parts if descriptor['bound_name']!=imported_parts[0] else [imported_parts[0]]
        symbol=None
    else:
        if level:
            parent=current_path.split('/')[:-1]
            if level-1>len(parent):return {'status':'relative_import_out_of_range','paths':[]}
            module_parts=parent[:len(parent)-(level-1)]+(module.split('.') if module else [])
        else:module_parts=module.split('.') if module else []
        symbol=imported
    paths=[];class_name=None;mode='module'
    if symbol:
        sub=suffix_paths(files,module_parts+symbol.split('.'))
        if sub:paths=sub
        else:
            paths=suffix_paths(files,module_parts)
            if paths:
                classes=[p for p in paths if symbol in source_info(files[p])['classes']]
                instances=[(p,source_info(files[p])['aliases'][symbol]) for p in paths if symbol in source_info(files[p])['aliases']]
                if classes:paths=classes;class_name=symbol;mode='class'
                elif instances:
                    possible=[(p,n) for p,n in instances if n in source_info(files[p])['classes']]
                    if len(possible)==1:paths=[possible[0][0]];class_name=possible[0][1];mode='instance'
                    else:return {'status':'imported_instance_type_unknown','paths':paths}
                else:return {'status':'imported_symbol_or_reexport_unknown','paths':paths}
    else:paths=suffix_paths(files,module_parts)
    if not paths:return {'status':'provider_not_in_allowed_files','paths':[]}
    if len(paths)>1:return {'status':'multiple_candidate_source_paths','paths':paths}
    # A same-named local wrapper is not proof that a bare absolute import
    # refers to it (e.g. utils/numpy.py itself imports the external numpy).
    if module is None and level==0 and imported in source_info(files[paths[0]])['absolute_imports']:
        return {'status':'possible_external_module_name_collision','paths':paths}
    tail=descriptor['tail']
    if tail:
        if len(tail)==1 and tail[0] in source_info(files[paths[0]])['classes']:
            class_name=tail[0];mode='class'
        else:return {'status':'provider_attribute_chain_unresolved','paths':paths}
    if descriptor['usage'] in {'constructor','annotation'}:
        if class_name is None:return {'status':'factory_return_type_unknown','paths':paths}
        mode='instance'
    return {'status':'resolved_local_provider','paths':paths,'mode':mode,'class_name':class_name,
        'path_basis':'explicit_relative_import' if level else 'unique_absolute_module_suffix',
        'runtime_import_identity_unverified':not bool(level)}

def words(text):
    text=re.sub(r'([a-z0-9])([A-Z])',r'\1_\2',text)
    return set(re.findall(r'[a-zA-Z][a-zA-Z0-9]*',text.lower()))-STOP

def active_header(prefix,target):
    # Indentation is only a fallback-independent visible-prefix heuristic.
    last=target.splitlines()[-1] if target.splitlines() else ''
    indent=len(last)-len(last.lstrip())
    found=[]
    offsets=[0]
    for line in prefix.splitlines(keepends=True):offsets.append(offsets[-1]+len(line.encode('utf-8')))
    for m in re.finditer(r'(?m)^([ \t]*)(?:async\s+)?def\s+(\w+)\s*\(',prefix):
        if len(m[1].expandtabs(4))<indent:
            found.append((len(prefix[:m.start()].encode('utf-8')),m[2]))
    return found[-1] if found else (None,'')

def resolve_provider(bindings,path,files):
    out=[]
    mapping={f['path']:f['text'] for f in files}
    for b in bindings:
        p=provider_lookup(b.get('provider'),path,mapping)
        if p['status']=='resolved_local_provider':out.append(p)
    return out

def intervals(text):
    offsets=[0]
    for line in text.splitlines(keepends=True):offsets.append(offsets[-1]+len(line.encode()))
    def span(n):return [offsets[n.lineno-1]+n.col_offset,offsets[n.end_lineno-1]+n.end_col_offset]
    def header(n):
        a,b=span(n);section=text.encode()[a:b].decode();lines=section.splitlines(keepends=True);starts=[0]
        for line in lines:starts.append(starts[-1]+len(line.encode()))
        depth=0
        try:
            for tok in tokenize.generate_tokens(io.StringIO(section).readline):
                if tok.type!=tokenize.OP:continue
                if tok.string in '([{':depth+=1
                elif tok.string in ')]}':depth-=1
                elif tok.string==':' and depth==0:
                    return [a,a+starts[tok.end[0]-1]+len(lines[tok.end[0]-1][:tok.end[1]].encode())]
        except (tokenize.TokenError,IndentationError):pass
        return span(n)
    return span,header

def all_apis(providers,files):
    mapping={f['path']:f['text'] for f in files};found={}
    def add(name,path,region,signature,kind):
        key=(path,name)
        if key not in found:found[key]=dict(name=name,path=path,regions=[],headers=[],kind=kind)
        found[key]['regions'].append(region);found[key]['headers'].append(signature)
    for provider in providers:
        for path in provider['paths']:
            text=mapping[path];span,header=intervals(text)
            try:tree=ast.parse(text)
            except (SyntaxError,ValueError):continue
            owner=provider.get('class_name')
            container=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner),None) if owner else tree
            if container is None:continue
            for n in container.body:
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    add(n.name,path,span(n),header(n),'function')
                elif isinstance(n,ast.ClassDef):
                    add(n.name,path,header(n),header(n),'class')
                    for method in n.body:
                        if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)) and method.name=='__init__':
                            add(n.name,path,span(method),header(method),'class')
                elif isinstance(n,(ast.Assign,ast.AnnAssign)):
                    for t in n.targets if isinstance(n,ast.Assign) else [n.target]:
                        if isinstance(t,ast.Name):add(t.id,path,span(n),span(n),'assignment')
            if owner:
                for n in ast.walk(container):
                    if not isinstance(n,(ast.Assign,ast.AnnAssign)):continue
                    for t in n.targets if isinstance(n,ast.Assign) else [n.target]:
                        if isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self':
                            add(t.attr,path,span(n),span(n),'instance_field')
    return list(found.values())

def merged(spans):
    out=[]
    for a,b in sorted(spans):
        if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out

def overlap(first,second):
    return sum(max(0,min(b,z)-max(a,c)) for a,b in merged(first) for c,z in merged(second))

def hypothesis_scores(prefix,target,apis):
    start,header=active_header(prefix,target)
    current=prefix.encode()[start:].decode() if start is not None else prefix
    hw=words(header);recent=words(current[-1500:])
    quoted=' '.join(re.findall(r'(?s)(?:\"\"\"(.*?)\"\"\"|\x27\x27\x27(.*?)\x27\x27\x27)',current)[0]) if re.search(r'\"\"\"|\x27\x27\x27',current) and re.findall(r'(?s)(?:\"\"\"(.*?)\"\"\"|\x27\x27\x27(.*?)\x27\x27\x27)',current) else ''
    descriptions=quoted+'\n'+'\n'.join(l for l in current.splitlines() if l.lstrip().startswith('#'))
    scores=[]
    for api in apis:
        name=api['name'];nw=words(name)
        overlap=len(nw&hw)/max(1,len(nw))
        precision=len(nw&hw)/max(1,len(hw))
        exact=name.strip('_') in header and len(name.strip('_'))>=3
        in_doc=bool(re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',descriptions))
        similarity=(2*len(nw&hw)/(len(nw)+len(hw))) if nw and hw else 0
        score=5*exact+4*in_doc+2*overlap+precision+similarity+.25*len(nw&recent)/max(1,len(nw))
        scores.append(dict(name=name,score=score,header=header,exact=exact,in_doc=in_doc))
    return sorted(scores,key=lambda a:(-a['score'],a['name']))
