"""Immutable exact main-only opt-in masked_select source projection, no execution."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
P=Path(__file__).parent
src,checker=map(Path,sys.argv[1:3]); src=src.resolve(strict=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
assert __debug__
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checker',checker); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
raw=(src/'manifest.json').read_bytes()
assert sha(raw)=='348ed8e949256f985cdca0d9c612dc646ae738beee6013c27ab405d2df3e67c8'
m=json.loads(raw,object_pairs_hook=b.unique_object)
r,units,_=b.check_sources(m,src,b.read_file)
assert sha(r)=='0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43'
typedef=next(u for u in m['units'] if u['name']=='runtime775_11'); up=typedef['path']
rows={u['path']:raw for u,raw,_ in units}
assert sha(rows[up])=='8095595a1859fb2973f02bb567d56b4bc960c5b4666c8914f8a53079adfe7db9'
names=b'ms_error ms_tensor ms_geometry ms_output ms_command ms_owned_layout ms_ordinary'.split()
assert not re.findall(rb'"value":\s*130\b',r)
def replace(v,old,new):
 assert v.count(old)==1
 return v.replace(old,new,1)
KEEP=b'  branch v144 b13(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139) b14(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139)\n'
ROUTE=b'''  ms_opcode:u64 = const() {"value":130}
  ms_chosen:bool = compare.eq(v139,ms_opcode)
  branch ms_chosen ms_route(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139,v138) ms_keep(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139,v144)
block ms_keep(mkr:readbuf,mko:mutbuf,mks:mutbuf,mkw:mutbuf,mkn:u64,mkout:u64,mki:u64,mkp:u64,mkf:u64,mkop:u64,mka:bool)
  branch mka b13(mkr,mko,mks,mkw,mkn,mkout,mki,mkp,mkf,mkop) b14(mkr,mko,mks,mkw,mkn,mkout,mki,mkp,mkf,mkop)
block ms_route(mrr:readbuf,mro:mutbuf,mrs:mutbuf,mrw:mutbuf,mrn:u64,mrout:u64,mri:u64,mrp:u64,mrf:u64,mrop:u64,mrat:u64)
  ms_written:u64 = call(mrr,mro,mrs,mrat,mrop,mrp) {"function":"ms_command"}
  jump b45(mrr,mro,mrs,mrw,mrn,mrout,mri,mrp,mrf,mrop,ms_written)
'''
inc=b''.join((P/f'masked_{which}.hepta.inc').read_bytes() for which in ('core','support','owner'))
anchor=b'fn predicate_command('
def newly_import(name):
 q=re.search(rb'^fn '+name+rb'\([^)]+\) -> \(u64\) effects\(read,write,trap\)',r,re.M)[0]
 assert q not in rows[up]
 return q+b' external()\n'
additional=b''.join(map(newly_import,(b'incoming_same_shape',b'so_version_live')))
orig_header=b'exports(half_cast_admit,'
new_header=b'exports('+b','.join(names)+b',half_cast_admit,'
oldext=b'fn absf(v0:u64) -> (u64) effects(read,write,trap) external()\n'
def proj(v,split=False,inverse=False):
 pairs=[(KEEP,ROUTE),(anchor,inc+anchor)]
 if split:pairs.extend([(orig_header,new_header),(oldext,additional+oldext)])
 for a,c in reversed(pairs) if inverse else pairs:
  v=replace(v,c,a) if inverse else replace(v,a,c)
 return v
new=proj(r); split=proj(rows[up],True)
assert proj(new,inverse=True)==r and proj(split,True,True)==rows[up]
R=b.parse_source(r); N=b.parse_source(new); U=b.parse_source(split,unit=True)
assert (set(N['definitions'])-set(R['definitions']))=={n.decode() for n in names}
assert {n for n in R['definitions'] if R['definitions'][n]!=N['definitions'][n]}=={'entry'}
assert all(N['signatures'][n]==R['signatures'][n] for n in R['signatures'])
for f in (b'entry',*names):
 body=N['definitions'][f.decode()];bindings=[]
 for line in body.splitlines()[1:]:
  if line.startswith(b'block '):bindings+=re.findall(rb'\b(\w+):\w+',line)
  else:bindings+=re.findall(rb'^  (\w+):\w+ =',line)
 assert len(bindings)==len(set(bindings)),(f,[k for k in set(bindings) if bindings.count(k)>1])
 args=lambda v:v.count(b',')+1 if v else 0
 targets={q[1]:args(q[2]) for q in re.finditer(rb'^block (\w+)\(([^)]*)\)',body,re.M)}
 for t in re.findall(rb'^  (?:branch|jump) ([^\n]+)',body,re.M):
  for q in re.finditer(rb'(\w+)\(([^)]*)\)',t):
   assert q[1] in targets and targets[q[1]]==args(q[2]),(f,q[1])
core=N['definitions']['ms_tensor']
blocknames=set(re.findall(rb'^block (\w+)\(',core,re.M))
edges={}
for block,body in re.findall(rb'^block (\w+)\([^\n]*\)\n(.*?)(?=^block |^end)',core,re.M|re.S):
 term=re.findall(rb'^  (?:branch|jump) ([^\n]+)',body,re.M)
 edges[block]=set(re.findall(rb'(\w+)\(',term[0])) if term else set()
assert set(edges)==blocknames
dom={k:({k} if k==b'check' else set(blocknames)) for k in edges}
while True:
 nextdom=dict(dom)
 for k in blocknames-{b'check'}:
  pred=[dom[i] for i in edges if k in edges[i]]
  assert pred
  nextdom[k]={k}|set.intersection(*pred)
 if nextdom==dom:break
 dom=nextdom
assert {b'check',b'shape',b'cap',b'verify_shape',b'owner',b'count'}<=dom[b'bit']
assert {b'cap',b'owner',b'count'}<=dom[b'allocate']
assert {b'cap',b'owner',b'allocate',b'inspect',b'ready',b'choose',b'room',b'read_value'}<=dom[b'store']
assert {b'cap',b'owner',b'allocate',b'inspect',b'ready'}<=dom[b'done']
assert edges[b'capacity']==set() and edges[b'invalid']==set()
allocator=re.search(rb'^block allocate\([^\n]*\)\n(.*?)(?=^block )',core,re.M|re.S)[1]
before, after=allocator.split(b'{"function":"typed_descriptor"}',1)
for j in range(4):
 j=str(j).encode(); assert b'call(as,ap'+j+b',keep'+j+b') {"function":"putw"}' in after
assert b'call(as,ap0,ak) {"function":"putw"}' in before
assert b'_ = call(' not in inc
assert core.count(b'{"function":"typed_descriptor"}')==1
assert core.count(b'{"function":"typed_store"}')==1 and b'call(ts,target,t4,traw) {"function":"typed_store"}' in core
assert new.count(b'"value":130')==1
m.pop('expected_native_sha256');m['name']='opt_in_current_main_f32_masked_select_cpu'
typedef.pop('expected_object_sha256');typedef['sha256']=sha(split)
rp=m['runtime']['path'];m['runtime'].update(sha256=sha(new),**{k:N[k] for k in ('functions','blocks','instructions')})
rows.update({up:split,rp:new})
def reader(path,want):
 data=rows[path.relative_to(src).as_posix()]
 assert sha(data)==want
 return data
b.check_sources(m,src,reader)
if len(sys.argv)==4:
 dest=Path(sys.argv[3]);dest.mkdir();(dest/'units').mkdir()
 for path,data in rows.items():(dest/path).write_bytes(data)
 (dest/'manifest.json').write_bytes(b.json_bytes(m))
 b.check_sources(m,dest,b.read_file)
print(json.dumps(dict(parent_head='e9444902df9878060ab389a70ee957c1b7f4a887',parent_sha256=sha(r),runtime_sha256=sha(new),
 unit11_sha256=sha(split),manifest_sha256=sha(b.json_bytes(m)),function_count=N['functions'],
 check_sources='PASS',byte_exact_inverse=True,opcode_candidate=130,changed=['entry'],added=[n.decode() for n in names],
 native_or_torch_or_build_executed=False),sort_keys=True,indent=2))
