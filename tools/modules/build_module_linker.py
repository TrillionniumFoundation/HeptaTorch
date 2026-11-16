from pathlib import Path
from types import FunctionType
import argparse,ast,hashlib,inspect,json,struct,sys
sys.dont_write_bytecode=True
SPEC=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Author the HIA borrowed-module linker and generic HIA packer; native execution is separate.')
parser.add_argument('--source-root',type=Path,required=True)
parser.add_argument('--capacity-adaptation',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
options=parser.parse_args();R=options.output.resolve();UP=options.source_root.resolve();R.mkdir(parents=True,exist_ok=True)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
base=options.capacity_adaptation.read_bytes()
assert sha(base)=='1926437c96c413b891b4c0b61d1555434598a40067855241e0d2a53b1bee1292'
closure=set(json.loads(base)['upstream_source_files_loaded'])|{'tools/generate_hia_elf_linker.py','tools/generate_hia_native_filter.py'}
contract_raw=(UP/'docs/HIA_SELFHOST_CORE_SUPPORT.json').read_bytes()
assert sha(contract_raw)=='8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c'
contract=json.loads(contract_raw)['source_identity']['files']
for rel in sorted(closure):
 assert sha((UP/rel).read_bytes())==contract[rel]
sys.path.insert(0,str(UP))
from tools import generate_hia_elf_linker as g
from tools import generate_hia_native_filter as p
from hepta.hia_agent_assembly import render_assembly,assemble_program
from tools.generate_hia_selfhost import compact_authoring_cfg,total_scalar_functions
assert render_assembly(g.document()).encode()==(UP/'examples/hia_bootstrap/elf_static_linker.hepta').read_bytes()
assert render_assembly(p.document()).encode()==(UP/'examples/hia_bootstrap/native_filter_packer.hepta').read_bytes()

# Reuse the upstream syscall template's actual I/O/error paths. Only its call
# boundary changes from six SysV words to HIA descriptor argument/result areas.
raw_template=inspect.getsource(p.template)
old="""    emit('4c 89 e7 4c 89 ee 49 8d 94 24');word(OUTPUT_OFFSET)
    emit('b9');word(output_bytes)
    emit('4d 8d 84 24');word(workspace_offset)
    emit('41 b9');word(workspace_bytes)
    emit('e8');hole=len(code);word(0)
"""
new="""    emit('48 83 ec 40 4c 89 24 24 4c 89 6c 24 08')
    emit('49 8d 84 24');word(OUTPUT_OFFSET)
    emit('48 89 44 24 10 48 c7 44 24 18');word(output_bytes)
    emit('49 8d 84 24');word(workspace_offset)
    emit('48 89 44 24 20 48 c7 44 24 28');word(workspace_bytes)
    emit('48 89 e7 48 8d 74 24 30')
    emit('e8');hole=len(code);word(0)
    emit('8b 44 24 30 48 83 c4 40')
"""
assert raw_template.count(old)==1
ns=dict(p.template.__globals__,WIRE_BYTES=1<<20,OUTPUT_OFFSET=(1<<20)+4096)
exec(compile(raw_template.replace(old,new),'borrowed-entry-template','exec'),ns)
prefix,hole=ns['template'](output_bytes=1<<20,workspace_bytes=1<<20)
assert len(prefix)<1024
header_lines=['def headers(o:mutbuf, w:mutbuf) -> u64:']
for at in range(0,len(prefix),8):
 chunk=prefix[at:at+8];header_lines.append(f'    write(o, {at}, {int.from_bytes(chunk,"little")}, {len(chunk)})')
header_lines+=['    put(o, 96, get(w, 64))','    put(o, 104, get(w, 64))',f'    write(o, {hole}, get(w, 88) - {hole+4}, 4)','    return 0']
new_headers='\n'.join(header_lines)+'\n'
source=g.SOURCE.replace('MAGIC',str(int.from_bytes(b'HIABLK01','little')))
assert source.count('    put(w, 40, 256)')==1
source=source.replace('    put(w, 40, 256)','    put(w, 40, 1024)')
needle='    if symbols(s, w) == 0:\n        return 4294967295\n'
assert source.count(needle)==1
source=source.replace(needle,needle+'    if module_contracts(s, w) == 0:\n        return 4294967295\n')
needle='    record = get(w, found + 16)\n    flags = get(w, record + 24)\n'
assert source.count(needle)==1
source=source.replace(needle,'    record = get(w, found + 16)\n    if module_entry(s, w, record) == 0:\n        return 4294967295\n    flags = get(w, record + 24)\n')
start=source.index('\ndef headers(');end=source.index('\ndef property_note(',start)
source=source[:start]+'\n'+new_headers+source[end:]
additions=(SPEC/'module_linker_additions.txt').read_text()
names=b'\0.text\0.symtab\0.strtab\0.shstrtab\0.note.GNU-stack\0.rela.text\0.hia.interface\0'
checks='\n    '.join(f'if uint(s, at + {i}, {len(names[i:i+8])}) != {int.from_bytes(names[i:i+8],"little")}:\n        return 0' for i in range(0,len(names),8))
additions=additions.replace('MODULE_NAMES_CHECKS',checks).replace('MODULE_MAGIC',str(int.from_bytes(b'HIAOBJ01','little')))
source+='\n'+additions
(R/'linker-authoring.txt').write_text(source)
nodes=ast.parse(source).body;fns={n.name:'f'+str(i) for i,n in enumerate(nodes)}
sigs={n.name:([a.annotation.id for a in n.args.args],n.returns.id) for n in nodes}
total=total_scalar_functions(nodes);functions=[]
for n in nodes:
 b=g.LinkerBuilder(n,fns,sigs,total);b.statements(n.body)
 functions.append({'name':fns[n.name],'params':[b.param(v) for v in b.params],'returns':[b.returns],'effects':['read','write','trap'],'entry':'b0','blocks':b.blocks})
doc=compact_authoring_cfg({'schema':'hepta.hia.program.v8','module':'borrowed_module_linker','functions':functions})
raw=render_assembly(doc).encode();assert assemble_program(raw).as_dict()==doc
counts={'functions':len(functions),'blocks':sum(len(f['blocks']) for f in doc['functions']),'instructions':sum(len(b['ops']) for f in doc['functions'] for b in f['blocks']),'source_bytes':len(raw)}
assert counts['functions']<=256 and counts['blocks']<=2560 and counts['instructions']<=8192,counts
(R/'module-linker.hepta').write_bytes(raw)
(R/'linker-functions.json').write_text(json.dumps(fns,indent=2)+'\n')
assert sha(raw)=='5d32ecd85c32388b649e34038e423996c78011a3ccd8b2cbe17e4613c2fdc9d4'
pack_source=render_assembly(p.document(module='selfhost_exec')).encode()
(R/'linker-packer.hepta').write_bytes(pack_source)
loaded={}
for module in tuple(sys.modules.values()):
 filename=getattr(module,'__file__',None)
 if filename is None:continue
 path=Path(filename).resolve()
 try:relative=path.relative_to(UP).as_posix()
 except ValueError:continue
 assert relative in closure and sha(path.read_bytes())==contract[relative],relative
 loaded[relative]=contract[relative]
assert set(loaded)==closure
report={'source_sha256':sha(raw),'source_structure':counts,'packer_source_sha256':sha(pack_source),'native_compilation_performed':False,'native_packing_performed':False,'authoring_adapter_sha256':sha(Path(__file__).read_bytes()),'module_additions_sha256':sha((SPEC/'module_linker_additions.txt').read_bytes()),'upstream_source_files_verified_before_and_after':loaded,'application_startup_prefix_bytes':len(prefix),'application_startup_call_hole':hole,'application_buffers':{'input':1<<20,'output':1<<20,'workspace':1<<20},'linker_buffers':{'input':8<<20,'output':8<<20,'workspace':16<<20}}
(R/'linker-authoring.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='upstream_source_files_verified_before_and_after'},indent=2))
