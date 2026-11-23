"""Pure reversible admission-only projection for genuine scalar-shaped F64 where promotion."""
import hashlib, importlib.util, json, re, sys
from pathlib import Path
D=Path(__file__).parent
S,B=map(Path,sys.argv[1:3]);S=S.resolve(strict=True)
sha=lambda x:hashlib.sha256(x).hexdigest()
assert __debug__ and sha(B.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checked_builder',B)
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
mraw=(S/'manifest.json').read_bytes()
assert sha(mraw) in {'49ecefa7622b93ca566c5ccc9a03b2489e4280bf569d7e79a766d4562c8b2def',
                     '51c2a04cb532f0817eb3996f4eff5d28fd6bc72642cf94160c20bb5cb33c8660'}
m=json.loads(mraw,object_pairs_hook=checker.unique_object)
r,units,_=checker.check_sources(m,S,checker.read_file)
assert sha(r)=='9ce43810f5280ee6390728351e01fe2f29bd5ad874c6e56a06eae3a0c884cab8'
rows={row['path']:raw for row,raw,_ in units}
u=next(z for z in m['units'] if z['name']=='runtime800_12');up=u['path']
assert sha(rows[up])=='c5d3ad9bee2f00393efad07b0159e85780a330d63db356a492c1b557025136ac'
rpath=m['runtime']['path']
old=b'  branch row_both mixed_roots(wrs,wrm,wrx,wry,wrxt,wryt,mask_width_w,wrt) bad(wrs)\n'
rep=b'  branch row_both mixed_roots(wrs,wrm,wrx,wry,wrxt,wryt,mask_width_w,wrt) zero_wide_scope(wrs,wrm,wrx,wry,wrxt,wryt,wrt,wrxd,wryd,wrxr,wryr)\n'+(D/'zero_wide.hepta.inc').read_bytes()
for f,raw in ((rpath,r),(up,rows[up])):
    assert raw.count(old)==1
    new=raw.replace(old,rep,1)
    assert new.count(rep)==1 and new.replace(rep,old,1)==raw
    rows[f]=new
before=checker.parse_source(r);after=checker.parse_source(rows[rpath])
assert before['signatures']==after['signatures']
assert set(before['definitions'])==set(after['definitions'])
modified={k for k in before['definitions'] if before['definitions'][k]!=after['definitions'][k]}
assert modified=={'where_self_tensor'}
body=after['definitions']['where_self_tensor']
bindings=[]
for line in body.splitlines()[1:]:
    if line.startswith(b'block '):bindings+=re.findall(rb'\b(\w+):\w+',line)
    else:bindings+=re.findall(rb'^  (\w+):\w+ =',line)
assert len(bindings)==len(set(bindings))
args=lambda v:v.count(b',')+1 if v else 0
targets={hit[1]:args(hit[2]) for hit in re.finditer(rb'^block (\w+)\(([^)]*)\)',body,re.M)}
for line in re.findall(rb'^  (?:branch|jump) ([^\n]+)',body,re.M):
    for hit in re.finditer(rb'(\w+)\(([^)]*)\)',line):assert targets[hit[1]]==args(hit[2]),hit[1]
m.pop('expected_native_sha256',None)
m['name']='opt_in_joint_where_nonempty_zero_dim_source_wide_cpu'
m['runtime'].update(sha256=sha(rows[rpath]),**{k:after[k] for k in ('functions','blocks','instructions')})
u.pop('expected_object_sha256',None);u['sha256']=sha(rows[up])
def mem(p,h):
    v=rows[p.relative_to(S).as_posix()]
    assert sha(v)==h
    return v
checker.check_sources(m,S,mem)
if len(sys.argv)==4:
    dest=Path(sys.argv[3]);dest.mkdir();(dest/'units').mkdir()
    for path,data in rows.items():(dest/path).write_bytes(data)
    (dest/'manifest.json').write_bytes(checker.json_bytes(m))
    checker.check_sources(m,dest,checker.read_file)
print(json.dumps(dict(parent_head='5c0afe4de62fd440fd9b74e1b2ce624a5f68b87e',parent_tree='72ff4675ac42a20806ef58919c706fcff1cb2b57',
 parent_runtime_sha256=sha(r),parent_manifest_sha256=sha(mraw),runtime_sha256=sha(rows[rpath]),unit12_sha256=sha(rows[up]),
 manifest_sha256=sha(checker.json_bytes(m)),changed=['where_self_tensor'],byte_exact_inverse=True,
 check_sources='PASS',native_or_compile_executed=False),indent=2))
