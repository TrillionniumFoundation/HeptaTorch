"""CFG authoring transform at a bounded, closed-function boundary.

The transformations are the parameter/forwarding rules of pinned HIA
tools.generate_hia_selfhost.compact_authoring_cfg. The aggregate source no
longer fits one Core document. Each actual body is instead bounded-decoded,
shape-checked and closed-body checked against immutable interfaces from the
actual source functions, before and after this transform. No synthetic callee
body is inserted. This author-time check is NOT an aggregate-program receipt.

Every emitted production module subsequently goes through the unchanged
native compiler, and the unchanged ABI3 native linker checks the actual object
interfaces, DAG, relocations and whole-program stack budget.
"""
from copy import deepcopy
import hashlib
import json


def compact(functions, selected):
    from hepta.hia_program import _decode, _function_shape, _function_interfaces, _verify_function
    from tools.generate_hia_selfhost import TYPES
    interfaces=_function_interfaces({fn['name']:fn for fn in functions})
    def checked(fn):
        decoded,_=_decode({'schema':'hepta.hia.program.v8','module':'bounded_authoring_body','functions':[fn]})
        detached=decoded['functions'][0]
        blocks=_function_shape(detached,recursive_chain=False,parameter_loan_returns=True)
        _verify_function('hepta.hia.program.v8',detached,blocks,interfaces)
        return detached
    def digest(fn):
        return hashlib.sha256(json.dumps(fn,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    results=[];records=[]
    for original in functions:
        # Unchanged bodies are not transformed or relabeled as newly checked.
        if original['name'] not in selected:continue
        before=checked(original);fn=deepcopy(before)
        def edges(block):
            term=block['term']
            return [term] if term['op']=='jump' else [term['then'],term['else']] if term['op']=='branch' else []
        blocks={block['id']:block for block in fn['blocks']}
        scalar=set(TYPES)-{'readbuf','mutbuf','buffer','chain'}
        live=set()
        for block in fn['blocks']:
            used={arg for op in block['ops'] for arg in op['args']}
            term=block['term']
            used.update(term['values'] if term['op']=='ret' else [term['cond']] if term['op']=='branch' else [])
            for param in block['params']:
                typ=param['type']
                if block['id']==fn['entry'] or not isinstance(typ,str) or typ not in scalar or param['name'] in used:
                    live.add((block['id'],param['name']))
        while True:
            previous=set(live)
            for block in fn['blocks']:
                params={p['name'] for p in block['params']}
                for edge in edges(block):
                    for actual,formal in zip(edge['args'],blocks[edge['target']]['params'],strict=True):
                        if (edge['target'],formal['name']) in live and actual in params:
                            live.add((block['id'],actual))
            if live==previous:break
        keep={block['id']:[i for i,param in enumerate(block['params']) if (block['id'],param['name']) in live] for block in fn['blocks']}
        pruned={block['id']:[p['name'] for i,p in enumerate(block['params']) if i not in keep[block['id']]] for block in fn['blocks']}
        for block in fn['blocks']:
            for edge in edges(block):edge['args']=[edge['args'][i] for i in keep[edge['target']]]
            block['params']=[block['params'][i] for i in keep[block['id']]]
        forwarded=[]
        while True:
            candidates={block['id']:block for block in fn['blocks'] if block['id']!=fn['entry'] and not block['ops']
                        and block['term']['op']=='jump' and block['term']['target']!=block['id']
                        and set(block['term']['args'])<={p['name'] for p in block['params']}}
            chosen=None
            for name,block in candidates.items():
                seen={name};target=block['term']['target']
                while target in candidates and target not in seen:
                    seen.add(target);target=candidates[target]['term']['target']
                if target not in seen:chosen=block;break
            if chosen is None:break
            names=[p['name'] for p in chosen['params']]
            forwarded.append({'block':chosen['id'],'parameters':names,
                              'target':chosen['term']['target'],'arguments':chosen['term']['args'][:]})
            for block in fn['blocks']:
                for edge in edges(block):
                    if edge['target']==chosen['id']:
                        actual=dict(zip(names,edge['args'],strict=True))
                        edge['args']=[actual[name] for name in chosen['term']['args']]
                        edge['target']=chosen['term']['target']
            fn['blocks']=[block for block in fn['blocks'] if block is not chosen]
        after=checked(fn)
        calls=sorted({op['attrs']['function'] for block in before['blocks'] for op in block['ops'] if op['op']=='call'})
        records.append({'name':fn['name'],'raw_structured_sha256':digest(before),'compacted_structured_sha256':digest(after),
                        'raw_blocks':len(before['blocks']),'compacted_blocks':len(after['blocks']),
                        'removed_scalar_parameters':{name:values for name,values in pruned.items() if values},
                        'forwarding_substitutions':forwarded,'callees':calls})
        results.append(after)
    return results,records
