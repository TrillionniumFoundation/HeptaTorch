"""Source-only reversible canonical-empty where projection; no compilation."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
D=Path(__file__).parent;S,B=map(Path,sys.argv[1:3]);S=S.resolve(strict=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
assert __debug__ and sha(B.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('source_checker',B)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mb=(S/'manifest.json').read_bytes()
assert sha(mb)=='22f5e5831a42ed65b1e8fffdd891690cbd8805d9d0a157fdfcd6ff0045baed9f'
m=json.loads(mb,object_pairs_hook=b.unique_object)
old,units,_=b.check_sources(m,S,b.read_file); rpath=m['runtime']['path']
assert sha(old)=='1ad950ecae208c3ccc60debd47393d727680111da205d9e2ef103612dc0ca273'
rows={row['path']:value for row,value,_ in units}
u=next(row for row in m['units'] if row['name']=='runtime800_12');up=u['path']
assert sha(rows[up])=='04b68ae230accc0c85952a10d293e56ea021ce8ac965827bb6581d13910d0b88'
inc=(D/'empty_canonical.hepta.inc').read_bytes()
names=re.findall(rb'^fn (\w+)\(',inc,re.M)
assert len(names)==5 and len(set(names))==5
anchor=b'fn where_self_tensor('
old_edge=b'  branch znboth mixed_roots(zns,znm,znx,zny,znxt,znyt,znone,znf64) bad(zns)\n'
edge=old_edge.replace(b'bad(zns)',b'zero_wide_canonical_empty(zns,znm,znx,zny,znxt,znyt,znone,znf64,zndim,znmaskdesc)')+(D/'empty_scope.hepta.inc').read_bytes()
export=b'exports(where_f16_fbgemm_profile_word,'
export_new=b'exports('+b','.join(names)+b',where_f16_fbgemm_profile_word,'
def match(raw,a,c):
 assert raw.count(a)==1
 return raw.replace(a,c,1)
for path,orig in ((rpath,old),(up,rows[up])):
 pairs=[(old_edge,edge),(anchor,inc+anchor)]+([(export,export_new)] if path==up else [])
 dat=orig
 for a,z in pairs:dat=match(dat,a,z)
 back=dat
 for a,z in reversed(pairs):back=match(back,z,a)
 assert back==orig
 rows[path]=dat
before=b.parse_source(old);after=b.parse_source(rows[rpath])
mod={k for k in before['definitions'] if before['definitions'][k]!=after['definitions'][k]}
added=set(after['definitions'])-set(before['definitions'])
assert mod=={'where_self_tensor'} and added=={x.decode() for x in names}
assert set(before['definitions'])<=set(after['definitions'])
assert all(before['signatures'][k]==after['signatures'][k] for k in before['signatures'])
args=lambda a:a.count(b',')+bool(a)
for k in mod|added:
 body=after['definitions'][k]
 bind=[]
 for ln in body.splitlines()[1:]:
  if ln.startswith(b'block '):bind+=re.findall(rb'\b(\w+):\w+',ln)
  else:bind+=re.findall(rb'^  (\w+):\w+ =',ln)
 assert len(bind)==len(set(bind)),(k,bind)
 targets={p[1]:args(p[2]) for p in re.finditer(rb'^block (\w+)\(([^)]*)\)',body,re.M)}
 for ln in re.findall(rb'^  (?:branch|jump) ([^\n]+)',body,re.M):
  for p in re.finditer(rb'(\w+)\(([^)]*)\)',ln):assert targets[p[1]]==args(p[2]),(k,p[1])
for k in added:
 body=after['definitions'][k]
 assert not any(i in body for i in (b'putw',b'so_uput',b'buffer.store',b'alloc',b'terror'))
m.pop('expected_native_sha256')
m['name']='opt_in_joint_where_zero_dim_source_f64_canonical_empty_cpu'
m['runtime'].update(sha256=sha(rows[rpath]),**{k:after[k] for k in ('functions','blocks','instructions')})
u.pop('expected_object_sha256');u['sha256']=sha(rows[up])
def memory(p,h):
 raw=rows[p.relative_to(S).as_posix()];assert sha(raw)==h;return raw
b.check_sources(m,S,memory)
if len(sys.argv)==4:
 out=Path(sys.argv[3]);out.mkdir();(out/'units').mkdir()
 for name,data in rows.items():(out/name).write_bytes(data)
 (out/'manifest.json').write_bytes(b.json_bytes(m))
 b.check_sources(m,out,b.read_file)
print(json.dumps(dict(parent_head='8c7f5b84cd1280d8a4868036fba88abd8d95d6aa',parent_tree='71165daa05849e3cfac262d5e58437ad07df444a',
 parent_runtime_sha256=sha(old),runtime_sha256=sha(rows[rpath]),unit12_sha256=sha(rows[up]),manifest_sha256=sha(b.json_bytes(m)),
 modified=sorted(mod),added=sorted(added),byte_exact_inverse=True,check_sources='PASS',native_or_compile_executed=False),indent=2))
