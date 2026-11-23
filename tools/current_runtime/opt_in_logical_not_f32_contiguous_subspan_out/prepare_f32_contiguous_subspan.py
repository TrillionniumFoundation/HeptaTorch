import importlib.util, json, re, sys
from hashlib import sha256
from pathlib import Path
src,dst,tool,pin=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('builder',tool); build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
sha=lambda x:sha256(x).hexdigest()
BASE='dbc0822faac2c264712c0763618898aa14c66d36cebda6f8a539521bff6410fa'
N='logical_out_permuted_f32_bool_i64_f64_subspan'
before=b'  h_ok:bool = bool.and(h_dense_ok,h_permuted)\n'
after=b'  h_four:u64 = const() {"value":4}\n  h_dtype:u64 = call(h_s,h_node) {"function":"tensor_dtype"}\n  h_f32:bool = compare.eq(h_dtype,h_z)\n  h_width4:bool = compare.eq(h_width,h_four)\n  h_f32_width4:bool = bool.and(h_f32,h_width4)\n  h_has_offset:bool = compare.ne(h_offset,h_z)\n  h_allowed_contig:bool = bool.and(h_f32_width4,h_has_offset)\n  h_layout_ok:bool = bool.or(h_permuted,h_allowed_contig)\n  h_ok:bool = bool.and(h_dense_ok,h_layout_ok)\n'
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
m['name']='opt_in_joint_logical_not_f32_contiguous_subspan';m.pop('expected_native_sha256')
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
