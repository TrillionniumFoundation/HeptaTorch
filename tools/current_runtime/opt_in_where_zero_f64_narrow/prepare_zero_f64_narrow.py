"""Pure reversible projection for genuine rank-zero F64 with dimensional F32 where promotion."""
import hashlib, importlib.util, json, re, sys
from pathlib import Path
D=Path(__file__).parent
S,B=map(Path,sys.argv[1:3]);S=S.resolve(strict=True)
sha=lambda x:hashlib.sha256(x).hexdigest()
assert __debug__ and sha(B.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checked_builder',B)
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
mraw=(S/'manifest.json').read_bytes()
assert sha(mraw) in {'f135c1704b178d04ddf6eaf63ce34b71db185393cfcdda59d52a92bc18d8bee4'}
m=json.loads(mraw,object_pairs_hook=checker.unique_object)
r,units,_=checker.check_sources(m,S,checker.read_file)
assert sha(r)=='1e49af9ac6c3d9b03fe5965965f756a48efec4aa2e135cec02baeeafefd2b700'
rows={row['path']:raw for row,raw,_ in units}
u=next(z for z in m['units'] if z['name']=='runtime800_12');up=u['path']
assert sha(rows[up])=='e3879063eb5a866433d86090d505f865260037e7592142a87e6b12bc721d4110'
rpath=m['runtime']['path']
before_gate=(b'  zf_allowed:bool = bool.and(zf_pair,zf_target)\n'
             b'  zf_rd:u64 = select(zf_left,zfxd,zfyd)\n'
             b'  zf_dd:u64 = select(zf_left,zfyd,zfxd)\n')
after_gate=b'  zf_existing:bool = bool.and(zf_pair,zf_target)\n'+(D/'narrow_admit.hepta.inc').read_bytes()
edge=b' i64_reduced_convert(ifs,ifm,ifx,ify,ifo,ifw,ifn,ifi0,ifword,ifmode,ifbool,ifsrc,iftgt,ifselected,ifsp)\n'
post_edge=edge.replace(b' i64_reduced_convert(',b' rank0_f64_f32_convert(')
anchor=b'block i64_to_f32('
blocks=(D/'narrow_gate.hepta.inc').read_bytes()
ext=b'fn narrow64(v0:u64) -> (u64) effects(read,write,trap) external()\n'
other=b'fn widen32(v0:u64) -> (u64) effects(read,write,trap) external()\n'
ops0=[(before_gate,after_gate),(edge,post_edge),(anchor,blocks+anchor)]
for f,raw in ((rpath,r),(up,rows[up])):
    ops=ops0+([(other,other+ext)] if f==up else [])
    new=raw
    for a,b in ops:
        assert new.count(a)==1
        new=new.replace(a,b,1)
    inv=new
    for a,b in reversed(ops):
        assert inv.count(b)==1
        inv=inv.replace(b,a,1)
    assert inv==raw
    rows[f]=new
before=checker.parse_source(r);after=checker.parse_source(rows[rpath])
assert before['signatures']==after['signatures']
assert set(before['definitions'])==set(after['definitions'])
modified={k for k in before['definitions'] if before['definitions'][k]!=after['definitions'][k]}
assert modified=={'where_self_tensor','where_full_where_core'}
assert before['definitions']['narrow64']==after['definitions']['narrow64']
assert before['definitions']['where_f16_fbgemm_profile_word']==after['definitions']['where_f16_fbgemm_profile_word']
assert before['definitions']['half_widen_bf16']==after['definitions']['half_widen_bf16']
preunit=checker.parse_source(next(raw for row,raw,_ in units if row['path']==up),unit=True)
postunit=checker.parse_source(rows[up],unit=True)
assert preunit['exports']==postunit['exports'] and set(postunit['imports'])-set(preunit['imports'])=={'narrow64'}
for f in sorted(modified):
    body=after['definitions'][f]
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
m['name']='opt_in_joint_where_zero_dim_f64_narrow_f32_canonical_nonempty_cpu'
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
print(json.dumps(dict(parent_head='768f7819b0b1248a9e5fb656e3431e4365b009b4',parent_tree='6fa2a0165395c522ea474ce3102b60ad1f018b8d',
 parent_runtime_sha256=sha(r),parent_manifest_sha256=sha(mraw),runtime_sha256=sha(rows[rpath]),unit12_sha256=sha(rows[up]),
 manifest_sha256=sha(checker.json_bytes(m)),changed=sorted(modified),import_added='narrow64',global_narrow64_unchanged=True,byte_exact_inverse=True,
 check_sources='PASS',native_or_compile_executed=False),indent=2))
