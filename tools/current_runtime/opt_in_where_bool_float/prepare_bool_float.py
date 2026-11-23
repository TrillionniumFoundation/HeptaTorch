"""Reversible explicit Bool/float value pairs for one existing where opcode."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
src,checker=map(Path,sys.argv[1:3]);src=src.resolve(strict=True)
local=Path(__file__).parent
sha=lambda v:hashlib.sha256(v).hexdigest()
if not __debug__:raise ValueError('source checks require assertions')
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checker',checker);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(src/'manifest.json').read_bytes()
assert sha(mraw)=='18b1874d1fbcda826e73fe14aa85e70785b289fb57f69df2ae9e71bada5cfa14'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
r,units,_=b.check_sources(m,src,b.read_file)
assert sha(r)=='76dc718591af6f28ef6e929281f904c99dd3de14ac6541b053f1377b8e25acba'
u=next(row for row in m['units'] if row['name']=='runtime800_12');up=u['path']
rows={row['path']:raw for row,raw,_ in units}
assert sha(rows[up])=='9fbaec44200fcc2704da263cdddfd0e6d632f7a2c09089c6c5766f4c936721a6'
from_body=lambda name:re.search(rb'^fn '+name+rb'\([^\n]+\n.*?^end\n',r,re.M|re.S)[0]
def replace(raw,old,new):
 assert raw.count(old)==1
 return raw.replace(old,new,1)
def region(body,begin,end):
 p=body.index(b'block '+begin+b'(');q=body.index(b'block '+end+b'(',p)
 return body[p:q]
a=from_body(b'where_self_tensor')
k=from_body(b'where_full_where_core')
a_new=replace(a,region(a,b'mixed_tags',b'mixed_roots'),(local/'bool_float_tags.hepta.inc').read_bytes())
k_new=replace(k,region(k,b'check_word',b'store'),(local/'bool_float_value.hepta.inc').read_bytes())
bits=(local/'bool_float_bits.hepta.inc').read_bytes()
assert {int(v) for v in re.findall(rb'\{"value":(\d+)\}',bits)}>={0x3c00,0x3f80,0x3f800000,0x3ff0000000000000}
assert region(a_new,b'mixed_roots',b'promote_pair')==region(a,b'mixed_roots',b'promote_pair')
assert region(k_new,b'promote_word',b'check_word')==region(k,b'promote_word',b'check_word')
new=replace(replace(r,a,a_new),k,bits+k_new)
assert replace(replace(new,a_new,a),bits+k_new,k)==r
split=replace(replace(rows[up],a,a_new),k,bits+k_new)
e0=b'exports(where_full_where_core,'
e1=b'exports(where_bool_pair_true_bits,where_full_where_core,'
split=replace(split,e0,e1)
assert replace(replace(replace(split,e1,e0),a_new,a),bits+k_new,k)==rows[up]
before=b.parse_source(r);after=b.parse_source(new)
changed={f for f in before['definitions'] if before['definitions'][f]!=after['definitions'][f]}
new_names=set(after['definitions'])-set(before['definitions'])
assert changed=={'where_self_tensor','where_full_where_core'} and new_names=={'where_bool_pair_true_bits'}
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
m['name']='opt_in_joint_where_explicit_bool_floating_value_pairs_cpu'
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
print(json.dumps(dict(parent_head='86bb2bfee8913d1337911db43cfab90a0cab95a4',runtime_sha256=sha(new),unit12_sha256=sha(split),
 manifest_sha256=sha(b.json_bytes(m)),check_sources='PASS',byte_exact_inverse=True,
 changed=sorted(changed),added=sorted(new_names),native_torch_compile_executed=False),indent=2))
