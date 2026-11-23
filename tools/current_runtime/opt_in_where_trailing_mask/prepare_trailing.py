"""Project trailing singleton condition broadcasting only; source-checks in memory."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
S,B=map(Path,sys.argv[1:3]);S=S.resolve(strict=True)
h=lambda b:hashlib.sha256(b).hexdigest()
assert h(B.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('builder',B);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(S/'manifest.json').read_bytes()
assert h(mraw)=='4feb0a8d48755c5a4ac2226eba420c1476f570b4aa3dca3e3076c2f83a2d39ed'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
runtime, units,_=b.check_sources(m,S,b.read_file)
assert h(runtime)=='c1a7eef6c1ce03fa435f147e8c499eb6584a24a09801eabca406cb10cdb87083'
u=next(row for row in m['units'] if row['name']=='runtime800_12');up=u['path']
rows={row['path']: raw for row,raw,_ in units}
assert h(rows[up])=='60d16441a6f5afb7a34d56b3540320dd986d153da78fb6d856c1328bebaba3de'
pat=rb'^fn where_self_tensor\([^\n]+\n.*?^end\n'
original=re.search(pat,runtime,re.M|re.S)[0]
proj=Path(__file__).with_name('geometry.hepta.inc').read_bytes()
old=b"  branch setup_shapes_ok allocate(ss,sm,sx,sy,st,sw,mask_step) bad(ss)\n"
pairs=[(old,proj),
(b'  mask_index:u64 = int.mul.wrap(qi,qstep)\n',b"""
  linear_one:u64 = const() {"value":1}
  divide:bool = compare.gt(qstep,linear_one)
  branch divide trailing_index(qs,qm,qx,qy,qo,qn,qw,qi,qstep) linear_index(qs,qm,qx,qy,qo,qn,qw,qi,qstep)
block trailing_index(is0:mutbuf,im:u64,ix:u64,iy:u64,io:u64,in0:u64,iw:u64,ii:u64,idivisor:u64)
  quotient:u64 = int.div.checked(ii,idivisor)
  jump mask_at(is0,im,ix,iy,io,in0,iw,ii,idivisor,quotient)
block linear_index(js:mutbuf,jm:u64,jx:u64,jy:u64,jo:u64,jn:u64,jw:u64,ji:u64,jstep:u64)
  linear:u64 = int.mul.wrap(ji,jstep)
  jump mask_at(js,jm,jx,jy,jo,jn,jw,ji,jstep,linear)
block mask_at(qs0:mutbuf,qm0:u64,qx0:u64,qy0:u64,qo0:u64,qn0:u64,qw0:u64,qi0:u64,qstep0:u64,mask_index:u64)
""".lstrip(b'\n'))]
def sub(raw,old,new):
 assert raw.count(old)==1
 return raw.replace(old,new,1)
# Block parameters in the old choose remain dedicated to that block.
new=original
tail_start=new.index(b'  raw_mask:u64')
tail_end=new.index(b'block load(',tail_start)
tail=new[tail_start:tail_end]
newtail=re.sub(rb'\b(qs|qm|qx|qy|qo|qn|qw|qi|qstep)\b',lambda m:m[0]+b'0',tail)
pairs.append((tail,newtail))
for x,y in pairs:new=sub(new,x,y)
back=new
for x,y in reversed(pairs):back=sub(back,y,x)
assert back==original
assert b'int.div.checked(ii,idivisor)' in new and b'call(ts,addr,byte_width,bits)' in new
agg=sub(runtime,original,new)
assert sub(agg,new,original)==runtime
projected=sub(rows[up],original,new)
assert sub(projected,new,original)==rows[up]
before=b.parse_source(runtime);after=b.parse_source(agg)
assert before['signatures']==after['signatures']
assert {k for k in before['definitions'] if before['definitions'][k]!=after['definitions'][k]}=={'where_self_tensor'}
rows[up]=projected;rows[m['runtime']['path']]=agg
m.pop('expected_native_sha256')
m['name']='opt_in_joint_where_trailing_singleton_condition_cpu'
m['runtime'].update(sha256=h(agg),**{k:after[k] for k in ('functions','blocks','instructions')})
u.pop('expected_object_sha256');u['sha256']=h(projected)
def mem(path,expected):
 raw=rows[path.relative_to(S).as_posix()]
 assert h(raw)==expected
 return raw
b.check_sources(m,S,mem)
if len(sys.argv)==4:
 D=Path(sys.argv[3]);D.mkdir();(D/'units').mkdir()
 for k,v in rows.items():(D/k).write_bytes(v)
 (D/'manifest.json').write_bytes(b.json_bytes(m))
 b.check_sources(m,D,b.read_file)
print(json.dumps(dict(parent_head='a38fb62a3838ea8c4072ee4cf1722216392bac66', runtime_sha256=h(agg),unit12_sha256=h(projected),
 manifest_sha256=h(b.json_bytes(m)),old_body_sha256=h(original),new_body_sha256=h(new),check_sources='PASS',byte_exact_inverse=True,
 changed=['where_self_tensor'],native_torch_compile_executed=False),indent=2))
