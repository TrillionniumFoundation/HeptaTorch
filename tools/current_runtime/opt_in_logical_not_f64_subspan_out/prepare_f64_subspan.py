import importlib.util, json, re, sys
from hashlib import sha256
from pathlib import Path
src,dst,tool,pin=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('builder',tool); build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
sha=lambda x:sha256(x).hexdigest()
BASE='e3468379f448c7d123549c8c8cc6b1d6485a10dd983823f1f48d9bd4c4d7f4b1'
N='logical_out_permuted_f32_bool_i64_f64_subspan'
before=b'  p_offset_admissible:bool = bool.or(p_zero_offset,p_existing_or_aligned_i64)\n'
after=b'  p_f64_aligned:bool = bool.and(p_allow_f64,p_aligned8)\n  p_existing_or_aligned_f64:bool = bool.or(p_existing_or_aligned_i64,p_f64_aligned)\n  p_offset_admissible:bool = bool.or(p_zero_offset,p_existing_or_aligned_f64)\n'
def project(data):
 match=re.search(rb'^fn '+N.encode()+rb'\([^\n]+\n.*?^end\n',data,re.M|re.S); assert match
 old=match[0]; assert old.count(before)==1 and after not in data
 new=old.replace(before,after); out=data[:match.start()]+new+data[match.end():]
 assert out.replace(new,old,1)==data
 return out
m=json.loads((src/'manifest.json').read_bytes()); src_bytes=(src/'runtime.hepta').read_bytes();assert sha(src_bytes)==BASE
out=project(src_bytes); old=build.parse_source(src_bytes); parsed=build.parse_source(out)
assert set(old['definitions'])==set(parsed['definitions'])
assert set(n for n in parsed['definitions'] if parsed['definitions'][n]!=old['definitions'][n])=={N}
dst.mkdir();(dst/'units').mkdir();(dst/'runtime.hepta').write_bytes(out)
m['name']='opt_in_joint_logical_not_f32_bool_i64_f64_subspan_dense_permutation';m.pop('expected_native_sha256')
m['runtime'].update(sha256=sha(out),**{k:parsed[k] for k in ('functions','blocks','instructions')})
for row in m['units']:
 q=row['path'];got=(src/q).read_bytes();assert sha(got)==row['sha256']
 if row['name']=='runtime800_12':
  got=project(got);row['sha256']=sha(got);row.pop('expected_object_sha256')
 (dst/q).write_bytes(got)
if pin.is_file():
 b=json.loads(pin.read_bytes());assert b['source_sha256']==sha(out) and b['status']=='PASS_NATIVE_BUILD_ONLY'
 m['expected_native_sha256']=b['native_sha256']
 for row,prior in zip(m['units'],b['units']):
  assert row['name']==prior['name'] and row['sha256']==prior['source_sha256']
  row['expected_object_sha256']=prior['object_sha256']
(dst/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
build.check_sources(m,dst,build.read_file)
print(sha(out))
