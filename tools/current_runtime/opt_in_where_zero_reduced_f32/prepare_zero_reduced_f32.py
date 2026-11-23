"""Pure reversible admission-only projection for genuine rank-zero F16/BF16 with dimensional F32 where promotion."""
import hashlib, importlib.util, json, re, sys
from pathlib import Path
D=Path(__file__).parent
S,B=map(Path,sys.argv[1:3]);S=S.resolve(strict=True)
sha=lambda x:hashlib.sha256(x).hexdigest()
assert __debug__ and sha(B.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checked_builder',B)
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
mraw=(S/'manifest.json').read_bytes()
assert sha(mraw) in {'ea588ba995b73540c62283be18f30563be7d81e1c43bf27b11b077668faf3cce'}
m=json.loads(mraw,object_pairs_hook=checker.unique_object)
r,units,_=checker.check_sources(m,S,checker.read_file)
assert sha(r)=='3521d8a8312b32b9a6f64c8dca922d4214839c20df17cb4fed5138c86e33092d'
rows={row['path']:raw for row,raw,_ in units}
u=next(z for z in m['units'] if z['name']=='runtime800_12');up=u['path']
assert sha(rows[up])=='6808a351b8b30b0b736d8e454d49e32c6758dac68b3464facbffc481d92fb8da'
rpath=m['runtime']['path']
old=b'  zf_x_scalar:bool = bool.and(zf_x0,zf_xi)\n  zf_y_scalar:bool = bool.and(zf_y0,zf_yi)\n'
rep=(D/'reduced_tags.hepta.inc').read_bytes()
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
# The conversion blocks are in this unchanged function (and only accept their exact tags).
assert before['definitions']['where_full_where_core']==after['definitions']['where_full_where_core']
assert before['definitions']['where_f16_fbgemm_profile_word']==after['definitions']['where_f16_fbgemm_profile_word']
assert before['definitions']['half_widen_bf16']==after['definitions']['half_widen_bf16']

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
m['name']='opt_in_joint_where_zero_dim_reduced_f32_canonical_nonempty_cpu'
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
print(json.dumps(dict(parent_head='7e0cca2ecd1afd63190ae9f1545d6ea21ab35ad7',parent_tree='af1c96727ca2f69496743f48692b6f7bc7f84dd6',
 parent_runtime_sha256=sha(r),parent_manifest_sha256=sha(mraw),runtime_sha256=sha(rows[rpath]),unit12_sha256=sha(rows[up]),
 manifest_sha256=sha(checker.json_bytes(m)),changed=['where_self_tensor'],byte_exact_inverse=True,
 check_sources='PASS',native_or_compile_executed=False),indent=2))
