import importlib.util,json,re,sys
from pathlib import Path
src,dst,tool,pin=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('fixed_build',tool);build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
from hashlib import sha256
def sha(s):return sha256(s).hexdigest()
from project_subspan import project as proposal, OLD as BASE
def project(raw):return proposal(raw)[0]
m=json.loads((src/'manifest.json').read_bytes()); raw=(src/'runtime.hepta').read_bytes(); assert sha(raw)==BASE
out=project(raw); parsed=build.parse_source(out); old=build.parse_source(raw)
for name,v in old['definitions'].items():
 if name not in {'logical_out_permuted_f32_bool_i64_f64','logical_out_apply'}:assert v==parsed['definitions'][name]
assert parsed['definitions']['logical_out_apply']==old['definitions']['logical_out_apply'].replace(b'logical_out_permuted_f32_bool_i64_f64',b'logical_out_permuted_f32_bool_i64_f64_subspan')
dst.mkdir();(dst/'units').mkdir();(dst/'runtime.hepta').write_bytes(out)
m['name']='opt_in_joint_logical_not_f32_subspan_dense_permutation';m.pop('expected_native_sha256',None)
m['runtime'].update(sha256=sha(out),**{k:parsed[k] for k in ('functions','blocks','instructions')})
for row in m['units']:
 q=row['path']; got=(src/q).read_bytes();assert sha(got)==row['sha256']
 if row['name']=='runtime800_12':
  got=project(got);row['sha256']=sha(got);row.pop('expected_object_sha256',None)
 (dst/q).write_bytes(got)
if pin.is_file():
 b=json.loads(pin.read_bytes());assert b['source_sha256']==sha(out) and b['status']=='PASS_NATIVE_BUILD_ONLY'
 m['expected_native_sha256']=b['native_sha256']
 for row,prior in zip(m['units'],b['units']):
  assert row['name']==prior['name'] and row['sha256']==prior['source_sha256']
  row['expected_object_sha256']=prior['object_sha256']
(dst/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
build.check_sources(m,dst,build.read_file)
