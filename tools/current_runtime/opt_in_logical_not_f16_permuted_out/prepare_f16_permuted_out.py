import importlib.util, json, re, sys
from hashlib import sha256
from pathlib import Path
src,dst,tool,pin=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('builder',tool); build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
sha=lambda x:sha256(x).hexdigest()
BASE='29e3ef6794f1a670cabcc94de734a883738acf81c88adb0fd5f31b52782fdaea'
N='logical_out_permuted_f32_bool_i64_f64_subspan'
from project_zero_f16 import project
m=json.loads((src/'manifest.json').read_bytes()); src_bytes=(src/'runtime.hepta').read_bytes();assert sha(src_bytes)==BASE
out=project(src_bytes); old=build.parse_source(src_bytes); parsed=build.parse_source(out)
assert set(old['definitions'])==set(parsed['definitions'])
assert set(n for n in parsed['definitions'] if parsed['definitions'][n]!=old['definitions'][n])=={N}
dst.mkdir();(dst/'units').mkdir();(dst/'runtime.hepta').write_bytes(out)
m['name']='opt_in_joint_logical_not_f16_zero_offset_permuted_out';m.pop('expected_native_sha256')
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
