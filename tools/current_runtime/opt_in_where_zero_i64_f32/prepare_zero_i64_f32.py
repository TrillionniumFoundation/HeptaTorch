"""Pure reversible admission-only projection for genuine rank-zero I64 with dimensional F32 where promotion."""
import hashlib, importlib.util, json, re, sys
from pathlib import Path
D=Path(__file__).parent
S,B=map(Path,sys.argv[1:3]);S=S.resolve(strict=True)
sha=lambda x:hashlib.sha256(x).hexdigest()
assert __debug__ and sha(B.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checked_builder',B)
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
mraw=(S/'manifest.json').read_bytes()
assert sha(mraw) in {'85f5748e73ab5961c2b4a9f7c98efee12e1a19572b6c7ab31d4ec319f00edb97'}
m=json.loads(mraw,object_pairs_hook=checker.unique_object)
r,units,_=checker.check_sources(m,S,checker.read_file)
assert sha(r)=='400cd399360d1b846fc043932766ef9e121dfdb96795fc7bfde240fb0ec73e9d'
rows={row['path']:raw for row,raw,_ in units}
u=next(z for z in m['units'] if z['name']=='runtime800_12');up=u['path']
assert sha(rows[up])=='98656eb27764f44beccb2785f07daa3fa87a0c9707f19253f72c488fdf5cc4b7'
rpath=m['runtime']['path']
old=b'  branch wz_check zero_wide_count(zws,zwm,zwx,zwy,zwxt,zwyt,wz_one,wz_f64,wz_dptr,wz_dim_desc) bad(zws)\n'
rep=old.replace(b'bad(zws)',b'zero_i64_f32_scope(zws,zwm,zwx,zwy,zwxt,zwyt,zwt,zwxd,zwyd,zwxr,zwyr)')+(D/'zero_i64_f32.hepta.inc').read_bytes()
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
m['name']='opt_in_joint_where_zero_dim_i64_f32_canonical_nonempty_cpu'
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
print(json.dumps(dict(parent_head='aedb958dd9ed3b86bc1b45886442f2099fdac4c8',parent_tree='4eadcda7cad57e975f7d27193cd4be08f150e078',
 parent_runtime_sha256=sha(r),parent_manifest_sha256=sha(mraw),runtime_sha256=sha(rows[rpath]),unit12_sha256=sha(rows[up]),
 manifest_sha256=sha(checker.json_bytes(m)),changed=['where_self_tensor'],byte_exact_inverse=True,
 check_sources='PASS',native_or_compile_executed=False),indent=2))
