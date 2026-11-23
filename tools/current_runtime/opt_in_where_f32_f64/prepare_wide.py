"""Reversible explicit dimensioned F32/F64 value pair for one existing where opcode."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
src,checker=map(Path,sys.argv[1:3]);src=src.resolve(strict=True)
local=Path(__file__).parent
sha=lambda v:hashlib.sha256(v).hexdigest()
if not __debug__:raise ValueError('source checks require assertions')
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checker',checker);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(src/'manifest.json').read_bytes()
assert sha(mraw)=='8f24f10967e1c5ef6942bd33b923928d6a009260008547bf39461345805838e6'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
r,units,_=b.check_sources(m,src,b.read_file)
assert sha(r)=='2b6a32d43498f02af8addb4b3ee4187f33bc60cccdc4e86efb4ba24e5f8ac46e'
u=next(row for row in m['units'] if row['name']=='runtime800_12');up=u['path']
rows={row['path']:raw for row,raw,_ in units}
assert sha(rows[up])=='e926b92680fd396ed4adf55b50566d98cf263d17ee381cf5c8a18ee4d459ab8d'
from_body=lambda name:re.search(rb'^fn '+name+rb'\([^\n]+\n.*?^end\n',r,re.M|re.S)[0]
def replace(raw,old,new):
 assert raw.count(old)==1
 return raw.replace(old,new,1)
def region(body,begin,end):
 p=body.index(b'block '+begin+b'(');q=body.index(b'block '+end+b'(',p)
 return body[p:q]
a=from_body(b'where_self_tensor')
k=from_body(b'where_full_where_core')
prior_a=b'numeric_tag) bad(admit_ms)'
new_a=b'numeric_tag) wide_tags(admit_ms,admit_mm,admit_mx,admit_my,mxt,myt,mask_ok)'
a_new=replace(a,prior_a,new_a)
a_new=replace(a_new,b'block mixed_roots(', (local/'pair.hepta.inc').read_bytes()+b'block mixed_roots(')
prior_k=b') convert_bool(ps0,pm0,px0,py0,po0,pw0,pn0,pi0,pword,check_mix,is_bool,otype)'
new_k=b') pair_convert(ps0,pm0,px0,py0,po0,pw0,pn0,pi0,pword,check_mix,is_bool,stype,otype)'
k_new=replace(k,prior_k,new_k)
k_new=replace(k_new,b'block convert_bool(', (local/'widen.hepta.inc').read_bytes()+b'block convert_bool(')
assert region(a_new,b'mixed_roots',b'promote_pair')==region(a,b'mixed_roots',b'promote_pair')
assert region(k_new,b'convert_bool',b'store')==region(k,b'convert_bool',b'store')
new=replace(replace(r,a,a_new),k,k_new)
assert replace(replace(new,a_new,a),k_new,k)==r
split=replace(replace(rows[up],a,a_new),k,k_new)
head=b'fn is_dense('
extern=b'fn widen32(v0:u64) -> (u64) effects(read,write,trap) external()\n'
split=replace(split,head,extern+head)
assert replace(replace(replace(split,extern+head,head),a_new,a),k_new,k)==rows[up]
before=b.parse_source(r);after=b.parse_source(new)
changed={f for f in before['definitions'] if before['definitions'][f]!=after['definitions'][f]}
new_names=set(after['definitions'])-set(before['definitions'])
assert changed=={'where_self_tensor','where_full_where_core'} and not new_names
assert all(before['signatures'][f]==after['signatures'][f] for f in before['signatures'])
for f in changed|new_names:
 body=after['definitions'][f]
 bindings=[]
 for line in body.splitlines()[1:]:
  if line.startswith(b'block '):bindings+=re.findall(rb'\b(\w+):\w+',line)
  else:bindings+=re.findall(rb'^  (\w+):\w+ =',line)
 assert len(bindings)==len(set(bindings)), (f,[k for k in set(bindings) if bindings.count(k)>1])
 args=lambda v:v.count(b',')+1 if v else 0
 targets={hit[1]:args(hit[2]) for hit in re.finditer(rb'^block (\w+)\(([^)]*)\)',body,re.M)}
 for terminator in re.findall(rb'^  (?:branch|jump) ([^\n]+)',body,re.M):
  for hit in re.finditer(rb'(\w+)\(([^)]*)\)',terminator):
   assert hit[1] in targets and targets[hit[1]]==args(hit[2]),(f,hit[1])
rows[up]=split;rows[m['runtime']['path']]=new
m.pop('expected_native_sha256')
m['name']='opt_in_joint_where_explicit_dimensioned_f32_f64_cpu'
m['runtime'].update(sha256=sha(new),**{k:after[k] for k in ('functions','blocks','instructions')})
u.pop('expected_object_sha256');u['sha256']=sha(split)
def mem(path,expected):
 data=rows[path.relative_to(src).as_posix()]
 assert sha(data)==expected
 return data
b.check_sources(m,src,mem)
if len(sys.argv)==4:
 dest=Path(sys.argv[3]);dest.mkdir();(dest/'units').mkdir()
 for k,v in rows.items():(dest/k).write_bytes(v)
 (dest/'manifest.json').write_bytes(b.json_bytes(m))
 b.check_sources(m,dest,b.read_file)
print(json.dumps(dict(parent_head='9bc18fcaff165f3a5fd97258b6392935c1e64bcb',runtime_sha256=sha(new),unit12_sha256=sha(split),
 manifest_sha256=sha(b.json_bytes(m)),check_sources='PASS',byte_exact_inverse=True,
 changed=sorted(changed),added=sorted(new_names),native_torch_compile_executed=False),indent=2))
