import importlib.util,json,re,sys
from pathlib import Path
src,dst,tool,pin=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('fixed_build',tool);build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
from hashlib import sha256
def sha(s):return sha256(s).hexdigest()
BASE='a7152179ba5b3669f6e81aa1bc9da5d548874de353978186cdb09eddbe0b33e0'
def project(raw):
 def rep(a,b):
  nonlocal raw
  assert raw.count(a.encode())==1,a
  raw=raw.replace(a.encode(),b.encode())
 start=raw.index(b'fn logical_out_permuted_f32_bool(');end=raw.index(b'\nend\n',start)+5
 prefix,raw,suffix=raw[:start],raw[start:end],raw[end:]
 rep('  p_type_ok:bool = bool.or(p_allow_f32,p_allow_bool)','''  p_eight:u64 = const() {"value":8}
  p_i64:bool = compare.eq(p_dtype,p_two)
  p_i64_width:bool = compare.eq(p_width,p_eight)
  p_allow_i64:bool = bool.and(p_i64,p_i64_width)
  p_prior:bool = bool.or(p_allow_f32,p_allow_bool)
  p_type_ok:bool = bool.or(p_prior,p_allow_i64)''')
 return (prefix+raw+suffix).replace(b'logical_out_permuted_f32_bool',b'logical_out_permuted_f32_bool_i64')
m=json.loads((src/'manifest.json').read_bytes()); raw=(src/'runtime.hepta').read_bytes(); assert sha(raw)==BASE
out=project(raw); parsed=build.parse_source(out); old=build.parse_source(raw)
for name,v in old['definitions'].items():
 if name not in {'logical_out_permuted_f32_bool','logical_out_apply'}:assert v==parsed['definitions'][name]
assert parsed['definitions']['logical_out_apply']==old['definitions']['logical_out_apply'].replace(b'logical_out_permuted_f32_bool',b'logical_out_permuted_f32_bool_i64')
dst.mkdir();(dst/'units').mkdir();(dst/'runtime.hepta').write_bytes(out)
m['name']='opt_in_joint_logical_not_f32_bool_i64_dense_permutation';m.pop('expected_native_sha256',None)
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
