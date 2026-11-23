"""Exact source-only projection for logical_or/xor extending qualified logical_and."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
D=Path(__file__).parent
SOURCE,BUILDER=map(Path,sys.argv[1:3]);SOURCE=SOURCE.resolve(strict=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
assert __debug__ and sha(BUILDER.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
mraw=(SOURCE/'manifest.json').read_bytes()
assert sha(mraw)=='b139fdc2648f5e7b49819b489096d97a7a9c4bb2aeb8a8ab0f538845b2e40bb6'
sp=importlib.util.spec_from_file_location('trusted_static_builder',BUILDER)
b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
m=json.loads(mraw,object_pairs_hook=b.unique_object);raw,units,_=b.check_sources(m,SOURCE,b.read_file)
assert sha(raw)=='72ae75a5e6c10b06c6716cb607b453c6bd2b1214b6ccc6e9a7fe1a3a666c023a'
u=next(z for z in m['units'] if z['name']=='runtime775_11')
rp,up=m['runtime']['path'],u['path']; rows={row['path']:value for row,value,_ in units}
assert sha(rows[up])=='92b3d040b56253da38e563837829800ac0daf1e419b0ca8c70a7470006f61cc0'
assert not re.search(rb'"value"\s*:\s*13[234]\b',raw)
def one(v,a,c):
 assert v.count(a)==1,a
 return v.replace(a,c,1)
old_route=b'  la_chosen:bool = compare.eq(v139,la_opcode)\n'
new_route=(D/'logical_route.hepta.inc').read_bytes()
oldstore=b'  lst_bits:u64 = bit.and(lst_l,lst_r)\n'
newstore=(D/'logical_select.hepta.inc').read_bytes()
old_cmd=b'  made:u64 = call(es,ex,em) {"function":"la_tensor"}\n'
new_cmd=b'  made:u64 = call(es,ex,em,eop) {"function":"la_tensor"}\n'
# Only blocks from which a value can ultimately reach the common store carry the opcode.
prefix={b'la_admit':b'la', b'la_capacity':b'lca', b'la_inputs':b'li', b'la_geometry':b'lg',
        b'la_allocate':b'llo', b'la_output':b'lot', b'la_output_count':b'loc', b'la_loop':b'llp', b'la_read':b'lrd', b'la_store':b'lst'}
def body(v,name):return re.search(rb'^fn '+name+rb'\([^\n]+\n.*?^end\n',v,re.M|re.S)[0]
t0=body(raw,b'la_tensor')
def thread(t,inverse=False):
 lines=t.splitlines(keepends=True); k=None; counts=[0,0,0]
 for i,l in enumerate(lines):
  if i==0:
   old=b'la_y:u64) ->'; new=b'la_y:u64,la_op:u64) ->'
   lines[i]=one(l,new if inverse else old,old if inverse else new);counts[0]+=1
  elif (r:=re.match(rb'^block (\w+)\(([^)]*)\)',l)):
   k=r[1]
   if k in prefix:
    su=b','+prefix[k]+b'_op:u64'
    before=r[0]
    if inverse:
     assert before.endswith(su+b')')
     after=before[:-len(su)-1]+b')'
    else: after=before[:-1]+su+b')'
    lines[i]=one(l,before,after);counts[1]+=1
  elif k is not None and (l.startswith(b'  jump ') or l.startswith(b'  branch ')):
   for r in re.finditer(rb'(\w+)\(([^)]*)\)',l):
    if r[1] not in prefix:continue
    assert k in prefix
    su=b','+prefix[k]+b'_op'; before=r[0]
    if inverse:
     assert before.endswith(su+b')'); after=before[:-len(su)-1]+b')'
    else: after=before[:-1]+su+b')'
    lines[i]=one(lines[i],before,after);counts[2]+=1
 assert counts==[1,10,10],counts
 return b''.join(lines)
t1=one(thread(t0),oldstore,newstore)
assert thread(one(t1,newstore,oldstore),True)==t0
ops=[(old_route,new_route),(old_cmd,new_cmd),(t0,t1)]
def transform(v,inverse=False):
 for a,c in reversed(ops) if inverse else ops:v=one(v,c,a) if inverse else one(v,a,c)
 return v
r1,u1=transform(raw),transform(rows[up])
assert transform(r1,True)==raw and transform(u1,True)==rows[up]
old=b.parse_source(raw); new=b.parse_source(r1)
assert set(new['definitions'])==set(old['definitions'])
changed={k for k in old['definitions'] if old['definitions'][k]!=new['definitions'][k]}
assert changed=={'entry','la_command','la_tensor'}
assert {k for k in old['signatures'] if old['signatures'][k]!=new['signatures'][k]}=={'la_tensor'}
prioru=b.parse_source(rows[up],unit=True); nextu=b.parse_source(u1,unit=True)
assert prioru['imports']==nextu['imports'] and prioru['exports']==nextu['exports']
assert r1.count(b'"value":132}')==0
# Op routing and selector only; no descriptor, load/store loops added.
assert r1.count(b'"value":133}')==r1.count(b'"value":134}')==2
for f in ('la_canonical','la_owned_layout','la_ordinary','taddr','tget','typed_descriptor','typed_store','logical_truth','shape_command_finish'):
 assert old['definitions'][f]==new['definitions'][f]
assert t1.count(b'{"function":"typed_descriptor"}')==t1.count(b'{"function":"typed_store"}')==1
m.pop('expected_native_sha256');u.pop('expected_object_sha256')
m['name']='opt_in_where_independent_same_shape_logical_and_or_xor_cpu'
m['runtime'].update(sha256=sha(r1),**{k:new[k] for k in ('functions','blocks','instructions')})
u['sha256']=sha(u1);rows.update({rp:r1,up:u1})
def bind(path,want):
 v=rows[path.relative_to(SOURCE).as_posix()];assert sha(v)==want;return v
b.check_sources(m,SOURCE,bind)
if len(sys.argv)==4:
 dest=Path(sys.argv[3]); dest.mkdir();(dest/'units').mkdir()
 for p,v in rows.items():(dest/p).write_bytes(v)
 (dest/'manifest.json').write_bytes(b.json_bytes(m));b.check_sources(m,dest,b.read_file)
print(json.dumps(dict(parent_commit='96e7d32a802a311abff1b17f5c3a2ceedb5eacb1',
 parent_tree='20cd3648f3a77afc268b12431e54ae36239e4be2',parent_combined_sha256=sha(raw),parent_manifest_sha256=sha(mraw),
 combined_sha256=sha(r1),unit11_sha256=sha(u1),manifest_sha256=sha(b.json_bytes(m)),
 changed=sorted(changed),exact_inverse=True,one_common_allocation_and_store=True,static_sources='PASS',
 candidate_opcodes=dict(logical_and=131,logical_or=133,logical_xor=134),opcode132_still_unsupported=True,
 native_or_compile_executed=False),indent=2,sort_keys=True))
