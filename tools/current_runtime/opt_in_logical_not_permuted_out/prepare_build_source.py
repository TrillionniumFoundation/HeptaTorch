import importlib.util,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_source_review import sha,transition,PENDING

src,dst,tool,pin=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('fixed_build',tool); build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
m=json.loads((src/'manifest.json').read_bytes()); raw=(src/'runtime.hepta').read_bytes(); assert sha(raw)==PENDING
out,_,_=transition(raw); parsed=build.parse_source(out)
dst.mkdir();(dst/'units').mkdir();(dst/'runtime.hepta').write_bytes(out)
m['name']='opt_in_joint_logical_not_f32_dense_permutation'
m.pop('expected_native_sha256',None)
m['runtime'].update(sha256=sha(out),**{k:parsed[k] for k in ('functions','blocks','instructions')})
for row in m['units']:
 q=row['path']; got=(src/q).read_bytes();assert sha(got)==row['sha256']
 if row['name']=='runtime800_12':
  changed=transition(got)[0]
  old_exp=b'exports(logical_out_layout,logical_out_apply,'
  assert changed.count(old_exp)==1
  new_exp=b'exports(logical_out_layout,logical_out_permuted_f32,logical_out_apply,'
  changed=changed.replace(old_exp,new_exp,1)
  local=build.parse_source(got,unit=True)
  definitions=set(local['definitions']) | {'logical_out_permuted_f32'}
  signatures=local['signatures']
  missing=set()
  for body in [*local['definitions'].values(), parsed['definitions']['logical_out_permuted_f32']]:
   for n in build.CALL.findall(body):
    n=n.decode()
    if n not in definitions and n not in signatures:missing.add(n)
  decl=b''.join(re.sub(rb' entry\(\w+\)$',b' external()',parsed['definitions'][n].split(b'\n')[0])+b'\n' for n in sorted(missing))
  sp=changed.index(b'\n',changed.index(b'exports('))+1
  changed=changed[:sp]+decl+changed[sp:]
  assert changed[:sp]+changed[sp+len(decl):]==transition(got)[0].replace(old_exp,new_exp,1)
  row['sha256']=sha(changed);row.pop('expected_object_sha256',None)
  print('additional split imports:',sorted(missing),flush=True)
  got=changed
 (dst/q).write_bytes(got)
if pin.is_file():
 b=json.loads(pin.read_bytes());assert b['source_sha256']==sha(out) and b['status']=='PASS_NATIVE_BUILD_ONLY'
 m['expected_native_sha256']=b['native_sha256']
 for row,prior in zip(m['units'],b['units']):
  assert row['name']==prior['name'] and row['sha256']==prior['source_sha256']
  row['expected_object_sha256']=prior['object_sha256']
(dest:=dst/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
build.check_sources(m,dst,build.read_file)
