"""Project scalar-condition broadcasting only; source-checks in memory."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
S,B=map(Path,sys.argv[1:3]);S=S.resolve(strict=True)
h=lambda b:hashlib.sha256(b).hexdigest()
assert h(B.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('builder',B);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(S/'manifest.json').read_bytes()
assert h(mraw)=='52418fe42d97540ce5407e3f5761e300608c60ed265ee8b82d6496c459d29c02'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
runtime, units,_=b.check_sources(m,S,b.read_file)
assert h(runtime)=='babddb5d0fc9e5e5e6eb0f923a4a25144d01e2ef29357c86acbd3faf88e68022'
u=next(row for row in m['units'] if row['name']=='runtime800_12');up=u['path']
rows={row['path']: raw for row,raw,_ in units}
assert h(rows[up])=='e7cf1fef3ffdaa90bf413c7e346acd0fbd7e399f170183232323b43dcc4024eb'
pat=rb'^fn where_self_tensor\([^\n]+\n.*?^end\n'
original=re.search(pat,runtime,re.M|re.S)[0]
start=original.index(b'block shapes(')
last=original.index(b'block allocate(')
proj=Path(__file__).with_name('shapes.hepta.inc').read_bytes()
old=original[start:last]
pairs=[
(old,proj),
(b'at:u64,aw:u64)',b'at:u64,aw:u64,astep:u64)'),
(b'ao,an,aw,at)',b'ao,an,aw,at,astep)'),
(b'cw:u64,ct:u64)',b'cw:u64,ct:u64,cstep:u64)'),
(b'cw,ct,zresult)',b'cw,ct,zresult,cstep)'),
(b'mtag:u64,mz:u64)',b'mtag:u64,mz:u64,mstep:u64)'),
(b'mo,mn,mw,mz)',b'mo,mn,mw,mz,mstep)'),
(b'lw:u64,li:u64)',b'lw:u64,li:u64,lstep:u64)'),
(b'lo,ln,lw,li)',b'lo,ln,lw,li,lstep)'),
(b'qw:u64,qi:u64)\n  raw_mask:u64 = call(qs,qm,qi)',b'qw:u64,qi:u64,qstep:u64)\n  mask_index:u64 = int.mul.wrap(qi,qstep)\n  raw_mask:u64 = call(qs,qm,mask_index)'),
(b'qo,qn,qw,qi)',b'qo,qn,qw,qi,qstep)'),
(b'vw:u64,vi:u64)',b'vw:u64,vi:u64,vstep:u64)'),
(b'vn,vw,vi,word)',b'vn,vw,vi,word,vstep)'),
(b'ti:u64,bits:u64)',b'ti:u64,bits:u64,tstep:u64)'),
(b'tn,byte_width,next_i)',b'tn,byte_width,next_i,tstep)'),
]
def sub(raw,old,new):
 assert raw.count(old)==1
 return raw.replace(old,new,1)
new=original
for x,y in pairs:new=sub(new,x,y)
back=new
for x,y in reversed(pairs):back=sub(back,y,x)
assert back==original
assert b'int.mul.wrap(qi,qstep)' in new and b'call(ts,addr,byte_width,bits)' in new
agg=sub(runtime,original,new)
assert sub(agg,new,original)==runtime
projected=sub(rows[up],original,new)
assert sub(projected,new,original)==rows[up]
before=b.parse_source(runtime);after=b.parse_source(agg)
assert before['signatures']==after['signatures']
assert {k for k in before['definitions'] if before['definitions'][k]!=after['definitions'][k]}=={'where_self_tensor'}
rows[up]=projected;rows[m['runtime']['path']]=agg
m.pop('expected_native_sha256')
m['name']='opt_in_joint_where_scalar_condition_cpu'
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
print(json.dumps(dict(parent_head='3b5061f59bdd752841163ddf22087fc9b03a6f73', runtime_sha256=h(agg),unit12_sha256=h(projected),
 manifest_sha256=h(b.json_bytes(m)),old_body_sha256=h(original),new_body_sha256=h(new),check_sources='PASS',byte_exact_inverse=True,
 changed=['where_self_tensor'],native_torch_compile_executed=False),indent=2))
