#!/usr/bin/env python3
"""Read-only canonical HIA inventory for the paged owned tensor overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
sha=lambda b:hashlib.sha256(b).hexdigest()
BASE_SHA='264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233'
CONTRACT_SHA='8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    raw=a.source.read_bytes();assert sha(raw)==BASE_SHA
    contract_raw=(a.upstream/'docs/HIA_SELFHOST_CORE_SUPPORT.json').read_bytes()
    assert sha(contract_raw)==CONTRACT_SHA
    pins=json.loads(contract_raw)['source_identity']['files']
    authoring_inventory=Path(__file__).resolve().parent.parent/'storage-object-extension/package/build.py'
    inventory=authoring_inventory.read_text()
    files=re.search(r"AUTHORING_FILES = '''(.*?)'''\.splitlines\(\)",inventory,re.S).group(1).splitlines()
    assert len(files)==32
    for relative in files:
        assert sha((a.upstream/relative).read_bytes())==pins[relative],relative
    sys.path.insert(0,str(a.upstream.resolve()))
    from hepta.hia_agent_assembly import _parse_assembly_document
    document=_parse_assembly_document(raw.decode())
    mapping=json.loads(a.mapping.read_text())
    assert len(document['functions'])==143 and set(mapping)=={f['name'] for f in document['functions']}
    literal_hits={}
    direct_workspace_accesses=[]
    locations={m.group(1):raw[:m.start()].count(b'\n')+1
               for m in re.finditer(rb'^fn (\w+)\(',raw,re.M)}
    for function in document['functions']:
        name=function['name']
        constants={o['out'][0]['name']:o['attrs']['value'] for b in function['blocks'] for o in b['ops'] if o['op']=='const'}
        origins={p['name']:mapping[name]['params'][i]['name'] for i,p in enumerate(function['params'])}
        blocks={b['id']:b for b in function['blocks']}
        changed=True
        while changed:
            changed=False
            for block in function['blocks']:
                t=block['term']
                edges=([{'target':t['target'],'args':t['args']}] if t['op']=='jump'
                       else [t['then'],t['else']] if t['op']=='branch' else [])
                for edge in edges:
                    for arg,param in zip(edge['args'],blocks[edge['target']]['params']):
                        if arg in origins and param['name'] not in origins:
                            origins[param['name']]=origins[arg];changed=True
                for op in block['ops']:
                    if op['op'].startswith('borrow.') and op['args'][0] in origins:
                        for out in op['out']:
                            if out['name'] not in origins:
                                origins[out['name']]=origins[op['args'][0]];changed=True
        hits={str(value):sum(v==value for v in constants.values())
              for value in [48,56,255,256,4096,53248,262144,1048576,16777216]}
        hits={k:v for k,v in hits.items() if v}
        if hits:literal_hits[name]={'canonical_function_line':locations[name.encode()],'constants':hits}
        for block in function['blocks']:
            for op in block['ops']:
                if op['op']=='call' and op['attrs'].get('function') in ('getw','putw','get4','put4'):
                    args=op['args']
                    if origins.get(args[0])=='w' and args[1] in constants:
                        offset=constants[args[1]]
                        if offset<4096:
                            direct_workspace_accesses.append({'function':name,'block':block['id'],
                                'operation':op['attrs']['function'],'offset':offset,
                                'canonical_function_line':locations[name.encode()]})
    loaded={}
    root=a.upstream.resolve()
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_relative_to(root):
                relative=str(path.relative_to(root));actual=sha(path.read_bytes())
                assert relative in files and pins[relative]==actual
                loaded[relative]=actual
    header=[x for x in direct_workspace_accesses if x['offset']<128]
    slot_48_56=[x for x in header if x['offset'] in (48,56)]
    report={'schema':'heptatorch.owned-tensor-layout-source-inventory.v1',
            'canonical_source_sha256':BASE_SHA,'functions':len(document['functions']),
            'mapping_sha256':sha(a.mapping.read_bytes()),'upstream_contract_sha256':CONTRACT_SHA,
            'read_only_parser_inputs':loaded,'literal_inventory':literal_hits,
            'fixed_workspace_header_accesses':header,
            'fixed_workspace_scratch_accesses':direct_workspace_accesses,
            'fixed_header_48_or_56_accesses':slot_48_56,
            'header_48_56_no_direct_access':not slot_48_56,
            'scope':'Exact literal call offsets in canonical SSA, propagated workspace resource origins; does not prove arbitrary indirect writes absent.'}
    (a.output/'source-inventory.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'functions':report['functions'],'header_48_56_no_direct_access':not slot_48_56,
                      'header_offsets':sorted({x['offset'] for x in header}),
                      'scratch_offsets':sorted({x['offset'] for x in direct_workspace_accesses}),
                      'numel_limit_functions':[n for n,x in literal_hits.items() if '262144' in x['constants']]}))


if __name__=='__main__':main()
