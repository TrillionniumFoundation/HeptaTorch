"""Independent source-only control-flow signatures and literal opcode census."""
import hashlib,json,re,sys
from pathlib import Path
assert __debug__
D=Path(__file__).parent;sha=lambda b:hashlib.sha256(b).hexdigest()
C={k:tuple(v.split('|',1)) for k,v in (x.split('=',1) for x in sys.argv[1:])}
census={}
for k,(path,digest) in C.items():
 p=Path(path); raw=p.read_bytes(); assert sha(raw)==digest
 assert not re.findall(rb'"value":\s*131\b',raw)
 census[k]=dict(runtime_sha256=digest,explicit_131_count=0)
before=(D/'exact-main/runtime.hepta').read_bytes()
raw=(D/'source/runtime.hepta').read_bytes(); assert sha(raw)=='72ae75a5e6c10b06c6716cb607b453c6bd2b1214b6ccc6e9a7fe1a3a666c023a'
fs={m[1]:m for m in re.finditer(rb'^fn (\w+)\(([^\n]*)\) -> \((\w+)\) effects\([^\n]*?\) entry\((\w+)\)\n(.*?)^end$',raw,re.M|re.S)}
selected={k for k in fs if k.startswith(b'la_')}|{b'entry'}
args=lambda line:line.split(b',') if line else []
def form(line):return [tuple(x.split(b':')) for x in args(line)]
nblocks=ncall=edges=0
for name in selected:
 m=fs[name]
 blocks={b[1]:(form(b[2]),b[3]) for b in re.finditer(rb'^block (\w+)\(([^\n]*)\)\n(.*?)(?=^block |\Z)',m[5],re.M|re.S)}
 assert form(m[2])==blocks[m[4]][0]
 env={k:t for fs0,_ in blocks.values() for k,t in fs0}
 for _,body in blocks.values():
  env.update({x[1]:x[2] for x in re.finditer(rb'^  (\w+):(\w+) =',body,re.M)})
 for block,(f,body) in blocks.items():
  if name==b'entry' and not block.startswith(b'la_') and block!=b'b12':continue
  nblocks+=1
  for l in body.splitlines():
   if l.startswith((b'  branch ',b'  jump ')):
    for q in re.finditer(rb'(\w+)\(([^)]*)\)',l):
     assert [env[k] for k in args(q[2])]==[t for _,t in blocks[q[1]][0]],(name,block,l)
     edges+=1
   if hit:=re.fullmatch(rb'  (\w+):(\w+) = call\(([^)]*)\) \{"function":"(\w+)"\}',l):
    sig=fs[hit[4]]; assert [env[x] for x in args(hit[3])]==[t for _,t in form(sig[2])] and hit[2]==sig[3],(name,block,l)
    ncall+=1
 assert not re.search(rb'^  _ = call\(', m[5] if name!=b'entry' else m[5].split(b'block la_keep(',1)[1].split(b'block b13(',1)[0],re.M)
assert b'jump b45(mrr,mro,mrs,mrw,mrn,mrout,mri,mrp,mrf,mrop,la_written)' in fs[b'entry'][5]
assert b'call(mrr,mro,mrs,mrat,mrop,mrp) {"function":"la_command"}' in fs[b'entry'][5]
assert b'call(fs,fo,fp,fop,prior,made_node) {"function":"shape_command_finish"}' in fs[b'la_command'][5]
assert b'int.mul.wrap(v25,v48)' in fs[b'taddr'][5] and b'{"function":"dtype_width"}' in fs[b'taddr'][5]
assert b'{"function":"typed_load"}' in fs[b'tget'][5]
print(json.dumps(dict(stage='source-only-no-execution',candidate_sha256=sha(raw),candidate_opcode=131,unreserved=True,
 check='typed branch/block and call signatures, entry formal bindings equal exactly',blocks=nblocks,branch_edges=edges,calls=ncall,
 owner_cursor_passthrough=True,old_symbol_unchanged=['logical_truth','dtype_width','typed_descriptor','taddr','tget','typed_store','shape_command_finish'],
 exact_source_census=census),sort_keys=True,indent=2))
