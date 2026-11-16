#!/usr/bin/env python3
"""Source authoring adapter for a real borrowed-module native compiler.

No input-module compilation or linking is performed by this script. It authors
ordinary HIA compiler source from pinned upstream source, leaving upstream and
canonical HeptaTorch files untouched. Native compilation is a separate stage.
"""
from pathlib import Path
from types import FunctionType
import argparse, ast, hashlib, json, sys
sys.dont_write_bytecode = True
SPEC = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--capacity-adaptation', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
options = parser.parse_args()
ROOT = options.output.resolve()
UP = options.source_root.resolve()
BASE = options.capacity_adaptation.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
sha = lambda b:hashlib.sha256(b).hexdigest()
assert sha(BASE.read_bytes()) == '1926437c96c413b891b4c0b61d1555434598a40067855241e0d2a53b1bee1292'
receipt = json.loads(BASE.read_text())
assert receipt['upstream_commit'] == 'f33d8da316de8fdf27b504371f1a5310997ff548'
assert sha((UP/'examples/hia_bootstrap/selfhost.hepta').read_bytes()) == receipt['upstream_source_sha256']
assert sha((UP/'docs/HIA_SELFHOST_CORE_SUPPORT.json').read_bytes()) == receipt['upstream_contract_sha256']
for rel,digest in receipt['upstream_source_files_loaded'].items():
 assert sha((UP/rel).read_bytes())==digest,rel
sys.path.insert(0,str(UP))
from tools import generate_hia_selfhost as g
from hepta.hia_agent_assembly import render_assembly,assemble_program

# Reproduce the already admitted expanded source before the module changes.
source = g.SOURCE
for edit in receipt['source_changes'][:-4]:
 assert source.count(edit['old'])==edit['matches'],edit
 source=source.replace(edit['old'],edit['new'])
base_namespace=dict(g.source_program.__globals__)
base_namespace.update({k:v['new'] for k,v in receipt['layout_macros'].items()})
base_namespace.update(SOURCE=source,MAX_RELOCATIONS=14336)
def lower_source(text,keywords):
 ns=dict(base_namespace,SOURCE=text,W=keywords)
 raw=FunctionType(g.source_program.__code__,ns)()
 for at in (64,72):
  raw=raw.replace(f'byte(w, {at})',f'state_get32(w, {at})')
  raw=raw.replace(f'store(w, {at},',f'state_put32(w, {at},')
 return raw

def doc_for(raw):
 ns=dict(g.document.__globals__,source_program=lambda:raw)
 return FunctionType(g.document.__code__,ns)(compact=True)
baseline=lower_source(source,g.W)
base_doc,_=doc_for(baseline)
assert sha(render_assembly(base_doc).encode())=='5cf8a4286d0759d2368e2d8d5658ce7de0b2ad72435ce75915cf07c0a23df7bf'

edits=[]
def change(old,new,count=1):
 global source
 assert source.count(old)==count,(old,source.count(old),count)
 source=source.replace(old,new)
 edits.append({'old':old,'new':new,'matches':count})

def replace_function(name,new):
 global source
 start=source.index('\ndef '+name+'(')
 end=source.find('\ndef ',start+1)
 if end<0:end=len(source)
 old=source[start:end]
 source=source[:start]+'\n'+new.rstrip()+'\n'+source[end:]
 edits.append({'function':name,'old_sha256':sha(old.encode()),'new_sha256':sha(new.encode())})

# Explicit specialization: the tensor arena uses borrowed memory; this profile
# refuses HIA heap owners/chain opcodes, rather than dropping their checks.
change('if (t >= @buffer@ and t <= @chain@) and get(w, 232) != 1:',
       'if t >= @buffer@ and t <= @chain@:')
start=source.index('    if op >= @buffer.new@ and op <= @chain.head_mut@:\n')
end=source.index('    if op == @const@:',start)
removed=source[start:end]
source=source[:start]+'    if op >= @buffer.new@ and op <= @chain.head_mut@:\n        return fail(w, 64)\n'+source[end:]
edits.append({'region':'owned-op-dispatch','old_sha256':sha(removed.encode()),'replacement':'explicit INVALID refusal'})
change('            chain_drop_helper(o, w)\n','')
change('    if get(w, 344) != 0:\n        return owned_elf_finish(o, w)\n',
       '    if get(w, 344) != 0:\n        return fail(w, 69)\n')

# A declaration is entered into the exact ordinary name/signature table, but
# has no blocks and emits no executable body. Both semantic passes see it.
needle='    if get(w, 48) == 0:\n        put(w, fb + 176, effects)\n'
addition='''    if (effects & 1) != 0:
        return fail(w, 64)
    if get(w, 48) == 0:
        put(w, fb + 272, 0)
    if get(w, 8) == @external@:
        if get(w, 640) != 1:
            return fail(w, 69)
        separated(w)
        next_token(s, w)
        adjacent(w)
        expect(s, w, 40)
        adjacent(w)
        expect(s, w, 41)
        expect(s, w, 10)
        if get(w, 48) == 0:
            put(w, fb + 272, 1)
            put(w, fb + 16, 0)
            put(w, fb + 40, 0)
            put(w, fb + 216, 0)
        module_signature(w, fb)
        if export_loader_hook(s, get(w, fb + 240), get(w, fb + 248)) == 1:
            return fail(w, 69)
        return 0
'''
change(needle,needle+addition)
needle='''    if get(w, 48) == 0:
        put(w, fb + 16, get(w, 96))
        if get(w, 40) == 0 and get(w, 432) == 0:
            check_graph(s, o, w)
    return 0
'''
change(needle,needle.replace('    return 0\n','    if get(w, 48) > 0:\n        put(w, fb + 280, get(w, 56) - get(w, fb))\n    return 0\n'))
# The opcode is a real symbolic call; kind3 is deliberately skipped by the
# local fixup resolver and materialized as ELF PLT32 relocation metadata.
needle='''                    emit(o, w, 232)
                    relocation(o, w, 1, x)
'''
change(needle,'''                    emit(o, w, 232)
                    if get(w, fb + 272) == 1:
                        put(w, 344, get(w, 344) + 1)
                        relocation(o, w, 3, x)
                    else:
                        relocation(o, w, 1, x)
''')
replace_function('validate_exports', '''def validate_exports(s:readbuf, w:mutbuf) -> u64:
    i = 0
    if get(w, 640) == 0 or get(w, 432) == 1:
        return 0
    while i < get(w, 648):
        record = 1502736 + i * 48
        put(w, 456, get(w, record))
        put(w, 464, get(w, record + 8))
        put(w, 472, get(w, record + 16))
        f = lookup_identifier(s, w, FBASE + 240, FSTRIDE, get(w, 64))
        if f == get(w, 64):
            return fail(w, 68)
        put(w, record + 24, f)
        fb = FBASE + f * FSTRIDE
        if get(w, fb + 272) != 0:
            return fail(w, 69)
        module_signature(w, fb)
        if get(w, 40) != 0:
            return 0
        i = i + 1
    return 0
''')
# Replace stock scalar SysV publication with the explicit private descriptor ABI.
additions=(SPEC/'module_compiler_additions.txt').read_text()
section_names=b'\0.text\0.symtab\0.strtab\0.shstrtab\0.note.GNU-stack\0.rela.text\0.hia.interface\0'
assert section_names.index(b'.hia.interface')==60
literal_lines='\n    '.join(f'emit_bytes(o, w, {int.from_bytes(section_names[i:i+8],"little")}, {len(section_names[i:i+8])})' for i in range(0,len(section_names),8))
additions=additions.replace('MODULE_SECTION_NAMES',literal_lines).replace('MODULE_NAMES_LENGTH',str(len(section_names))).replace('MODULE_MAGIC',str(int.from_bytes(b'HIAOBJ01','little'))).replace('EXPORT_BASE','1502736')
addition_nodes=ast.parse(additions.replace('FBASE','4096').replace('FSTRIDE','320').replace('FIXBASE','939008')).body
# Preserve symbolic macros for the normal upstream source substitution stage.
for node in addition_nodes:
 body='\n'.join(additions.splitlines()[node.lineno-1:node.end_lineno])
 if node.name=='exports_finish':replace_function(node.name,body)
 else:source+='\n'+body+'\n'
keywords=dict(g.W);keywords['external']=max(keywords.values())+1
raw=lower_source(source,keywords)

# Remove only authoring functions unreachable from the compiler's entry after
# the explicit owned-op refusal; no admitted module body is optimized here.
nodes=ast.parse(raw).body
byname={n.name:n for n in nodes}
reachable={'compile'}
changed=True
while changed:
 changed=False
 for name in tuple(reachable):
  for n in ast.walk(byname[name]):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in byname and n.func.id not in reachable:
    reachable.add(n.func.id);changed=True
removed_functions=[n.name for n in nodes if n.name not in reachable]
raw='\n\n'.join(ast.unparse(n) for n in nodes if n.name in reachable)+'\n'
doc,names=doc_for(raw)
source_bytes=render_assembly(doc).encode()
assert sha(source_bytes)=='dfcd781b7e2986b4fb46adf77ad5895d14319378c92ae4c1533421c7b82967b8'
checked=assemble_program(source_bytes)
assert checked.as_dict()==doc
counts={'functions':len(doc['functions']),'blocks':sum(len(f['blocks']) for f in doc['functions']),'instructions':sum(len(b['ops']) for f in doc['functions'] for b in f['blocks']),'source_bytes':len(source_bytes)}
assert counts['functions']<=256 and counts['blocks']<=2560 and counts['instructions']<=8192 and len(source_bytes)<=1048576,counts
(ROOT/'compiler-authoring.txt').write_text(raw)
(ROOT/'module-compiler.hepta').write_bytes(source_bytes)
(ROOT/'compiler-functions.json').write_text(json.dumps(names,indent=2)+'\n')
loaded = {}
for module in tuple(sys.modules.values()):
 filename = getattr(module, '__file__', None)
 if filename is None:continue
 path=Path(filename).resolve()
 try:relative=path.relative_to(UP).as_posix()
 except ValueError:continue
 actual=sha(path.read_bytes());assert receipt['upstream_source_files_loaded'].get(relative)==actual,relative
 loaded[relative]=actual
assert set(loaded)==set(receipt['upstream_source_files_loaded'])
report={'authoring_adapter_sha256':sha(Path(__file__).read_bytes()),'module_additions_sha256':sha((SPEC/'module_compiler_additions.txt').read_bytes()),'upstream_source_files_verified_before_and_after':loaded,'profile':'heptatorch.hia.borrowed-modules.v1','source_sha256':sha(source_bytes),'source_structure':counts,'compiler_core_sha256':checked.sha256,'upstream_commit':receipt['upstream_commit'],'baseline_expanded_sha256':'5cf8a4286d0759d2368e2d8d5658ce7de0b2ad72435ce75915cf07c0a23df7bf','removed_unreachable_authoring_functions':removed_functions,'edits':edits,'native_bootstrap_performed':False}
(ROOT/'compiler-adaptation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('edits','upstream_source_files_verified_before_and_after')},indent=2))
