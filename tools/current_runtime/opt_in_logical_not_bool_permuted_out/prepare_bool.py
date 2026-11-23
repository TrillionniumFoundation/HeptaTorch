import importlib.util,json,re,sys
from pathlib import Path
src,dst,tool,pin=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('fixed_build',tool);build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
from hashlib import sha256
def sha(s):return sha256(s).hexdigest()
BASE='97f5e000faa8bd779807d5b683dbfaff795767a0385b10039ba175a07d4d0f25'
def project(raw):
 def rep(a,b):
  nonlocal raw
  assert raw.count(a.encode())==1,a
  raw=raw.replace(a.encode(),b.encode())
 start=raw.index(b'fn logical_out_permuted_f32(');end=raw.index(b'\nend\n',start)+5
 prefix,raw,suffix=raw[:start],raw[start:end],raw[end:]
 rep('  p_kill0:u64 = bit.or(p_dtype,p_grad)','''  p_f32:bool = compare.eq(p_dtype,p_z)
  p_bool:bool = compare.eq(p_dtype,p_one)
  p_width:u64 = call(p_dtype) {"function":"dtype_width"}
  p_f32_width:bool = compare.eq(p_width,p_four)
  p_bool_width:bool = compare.eq(p_width,p_one)
  p_allow_f32:bool = bool.and(p_f32,p_f32_width)
  p_allow_bool:bool = bool.and(p_bool,p_bool_width)
  p_type_ok:bool = bool.or(p_allow_f32,p_allow_bool)
  p_kill0:u64 = bit.or(p_z,p_grad)''')
 rep('  p_guard0:bool = bool.and(p_clean,p_is_view)','''  p_clean_view:bool = bool.and(p_clean,p_is_view)
  p_guard0:bool = bool.and(p_clean_view,p_type_ok)''')
 for s in ('p','r'):
  rep(f'{s}_rank,{s}_count,{s}_z,{s}_one,{s}_four',f'{s}_rank,{s}_count,{s}_z,{s}_one,{s}_width')
 rep('r_one:u64,r_four:u64','r_one:u64,r_width:u64')
 rep('v_one:u64,v_four:u64','v_one:u64,v_width:u64')
 rep('v_store,v_version,v_four','v_store,v_version,v_width')
 # Carry width only to Storage extent, with no additional admission path.
 for a,b in [('v_z,v_z,v_one)', 'v_z,v_z,v_one,v_width)'),('d_axis:u64,d_product:u64)','d_axis:u64,d_product:u64,d_width:u64)'),('d_axis,d_product) bad','d_axis,d_product) bad'),('d_axis,d_product) density','d_axis,d_product,d_width) density'),('d_count,d_z,d_product)','d_count,d_z,d_product,d_width)'),('a_axis:u64,a_product:u64)','a_axis:u64,a_product:u64,a_width:u64)'),('a_axis,a_product,a_shape)','a_axis,a_product,a_shape,a_width)'),('m_product:u64,m_shape:u64)','m_product:u64,m_shape:u64,m_width:u64)'),('m_product,m_shape)','m_product,m_shape,m_width)'),('n_product:u64,n_shape:u64)','n_product:u64,n_shape:u64,n_width:u64)'),('n_next_axis,n_next_product)','n_next_axis,n_next_product,n_width)'),('y_product:u64)','y_product:u64,y_width:u64)'),('y_count,y_z) bad','y_count,y_z,y_width) bad'),('h_count:u64,h_z:u64)','h_count:u64,h_z:u64,h_width:u64)'),('h_count,h_z) bad','h_count,h_z,h_width) bad'),('s_count:u64,s_z:u64)','s_count:u64,s_z:u64,s_width:u64)'),('s_count,s_z) bad','s_count,s_z,s_width) bad'),('e_count:u64,e_z:u64)','e_count:u64,e_z:u64,e_width:u64)'),('  e_four:u64 = const() {"value":4}\n',''),('e_z,e_four,e_tag','e_z,e_width,e_tag'),('b_z:u64,b_four:u64','b_z:u64,b_width:u64'),('b_count,b_four','b_count,b_width')]:
  if a!=b:rep(a,b)
 return (prefix+raw+suffix).replace(b'logical_out_permuted_f32',b'logical_out_permuted_f32_bool')
m=json.loads((src/'manifest.json').read_bytes()); raw=(src/'runtime.hepta').read_bytes(); assert sha(raw)==BASE
out=project(raw); parsed=build.parse_source(out); old=build.parse_source(raw)
for name,v in old['definitions'].items():
 if name not in {'logical_out_permuted_f32','logical_out_apply'}:assert v==parsed['definitions'][name]
assert parsed['definitions']['logical_out_apply']==old['definitions']['logical_out_apply'].replace(b'logical_out_permuted_f32',b'logical_out_permuted_f32_bool')
dst.mkdir();(dst/'units').mkdir();(dst/'runtime.hepta').write_bytes(out)
m['name']='opt_in_joint_logical_not_f32_bool_dense_permutation';m.pop('expected_native_sha256',None)
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
