"""Reversible explicit dimensioned I64/F32 value pair for one existing where opcode."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
src,checker=map(Path,sys.argv[1:3]);src=src.resolve(strict=True)
local=Path(__file__).parent
sha=lambda v:hashlib.sha256(v).hexdigest()
if not __debug__:raise ValueError('source checks require assertions')
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checker',checker);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(src/'manifest.json').read_bytes()
assert sha(mraw)=='9f9eda44e8b36a6339ef2c4d32d6b4fdcd1c17252aacb04bc127b038d8efaaff'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
r,units,_=b.check_sources(m,src,b.read_file)
assert sha(r)=='1db84bd51df9f422803d85a4bc2dbc23ccf8ba8d8cf9e2c6491e99b6147eee52'
u=next(row for row in m['units'] if row['name']=='runtime800_12');up=u['path']
rows={row['path']:raw for row,raw,_ in units}
assert sha(rows[up])=='cb2e377fe2962fb0b843d4066b10f58b7d89d26702a53361a3394c32289e8e07'
from_body=lambda name:re.search(rb'^fn '+name+rb'\([^\n]+\n.*?^end\n',r,re.M|re.S)[0]
def replace(raw,old,new):
 assert raw.count(old)==1
 return raw.replace(old,new,1)
def region(body,begin,end):
 p=body.index(b'block '+begin+b'(');q=body.index(b'block '+end+b'(',p)
 return body[p:q]
a=from_body(b'where_self_tensor')
k=from_body(b'where_full_where_core')
prior_a=b'hyt,hdt) bad(hps)'
new_a=b'hyt,hdt) i64_f32_pair(hps,hpm,hpx,hpy,hxt,hyt,hmask)'
a_new=replace(a,prior_a,new_a)
a_new=replace(a_new,b'block wide_ranks(', (local/'i64_f32_tags.hepta.inc').read_bytes()+b'block wide_ranks(')
prior_k=b') bf_pair_convert(hbs,hbm,hbx,hby,hbo,hbw,hbn,hbi,hbword,hbmode,hbbool,hbsrc,hbtgt,hbsel,hbsp)'
new_k=b') i64_f32_convert(hbs,hbm,hbx,hby,hbo,hbw,hbn,hbi,hbword,hbmode,hbbool,hbsrc,hbtgt,hbsel,hbsp)'
k_new=replace(k,prior_k,new_k)
k_new=replace(k_new,b'block bf_pair_convert(', (local/'i64_f32_value.hepta.inc').read_bytes()+b'block bf_pair_convert(')
assert region(a_new,b'wide_ranks',b'mixed_roots')==region(a,b'wide_ranks',b'mixed_roots')
assert region(k_new,b'convert_bool',b'store')==region(k,b'convert_bool',b'store')
assert region(k_new,b'f32_to_f64',b'i64_f64_convert')==region(k,b'f32_to_f64',b'i64_f64_convert')
assert region(k_new,b'bf_pair_convert',b'convert_bool')==region(k,b'bf_pair_convert',b'convert_bool')
assert region(k_new,b'f16_to_f64',b'i64_f32_convert')==region(k,b'f16_to_f64',b'bf_pair_convert')
new=replace(replace(r,a,a_new),k,k_new)
assert replace(replace(new,a_new,a),k_new,k)==r
split=replace(replace(rows[up],a,a_new),k,k_new)
extern=b'fn integer_i64_to_f32(v0:u64) -> (u64) effects(read,write,trap) external()\n'
anchor=b'fn integer_i64_to_f64(v0:u64) -> (u64) effects(read,write,trap) external()\n'
split=replace(split,anchor,extern+anchor)
assert replace(replace(replace(split,extern+anchor,anchor),a_new,a),k_new,k)==rows[up]
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
m['name']='opt_in_joint_where_explicit_dimensioned_i64_f32_cpu'
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
print(json.dumps(dict(parent_head='91ebfa9062d6b546e738da314b5d18ba933a87ef',parent_tree='a9f4ede72b77de831d9283ae50c4d3ab1862a1ec',runtime_sha256=sha(new),unit12_sha256=sha(split),
 manifest_sha256=sha(b.json_bytes(m)),check_sources='PASS',byte_exact_inverse=True,
 changed=sorted(changed),added=sorted(new_names),native_or_compile_executed=False),indent=2))
