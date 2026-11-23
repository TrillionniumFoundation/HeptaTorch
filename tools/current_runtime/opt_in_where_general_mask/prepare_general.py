"""Reversible current66 where fallback for right-aligned Bool mask only."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
S,B=map(Path,sys.argv[1:3]);S=S.resolve(strict=True)
h=lambda b:hashlib.sha256(b).hexdigest()
assert h(B.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('builder',B);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(S/'manifest.json').read_bytes()
assert h(mraw)=='f3768dc20fcf10d16ea19d40c155c713c6b74ee90f48f50275b56ad87aa7181a'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
runtime,units,_=b.check_sources(m,S,b.read_file)
assert h(runtime)=='1e4430612b11941c465ec2b21fc7d67c0f672858617517407b586b982a832386'
u=next(row for row in m['units'] if row['name']=='runtime800_12');up=u['path']
rows={row['path']:raw for row,raw,_ in units}
assert h(rows[up])=='b5d7a98deb472d437915643950619c2e43e41dda5a9082f4f9da0fe6de1e7b41'
inc=Path(__file__).with_name('general.hepta.inc').read_bytes()
pat=rb'^fn where_self_tensor\([^\n]+\n.*?^end\n'
original=re.search(pat,runtime,re.M|re.S)[0]
fallback=b'''
block general_try(gs0:mutbuf,gm0:u64,gx0:u64,gy0:u64,gt0:u64,gw0:u64)
  g_equal:u64 = call(gs0,gx0,gy0) {"function":"incoming_same_shape"}
  g_geo:u64 = call(gs0,gm0,gx0) {"function":"where_general_geometry"}
  g_z:u64 = const() {"value":0}
  g_mode:u64 = const() {"value":262145}
  g_both:u64 = bit.and(g_equal,g_geo)
  g_admit:bool = compare.ne(g_both,g_z)
  branch g_admit allocate(gs0,gm0,gx0,gy0,gt0,gw0,g_mode) bad(gs0)
'''.lstrip(b'\n')
decode=b'''
  general_mode:u64 = const() {"value":262145}
  general_decode:bool = compare.eq(qstep,general_mode)
  branch general_decode general_at(qs,qm,qx,qy,qo,qn,qw,qi,qstep) fast_index(qs,qm,qx,qy,qo,qn,qw,qi,qstep)
block general_at(gas:mutbuf,gam:u64,gax:u64,gay:u64,gao:u64,gan:u64,gaw:u64,gai:u64,gamode:u64)
  gaindex:u64 = call(gas,gam,gax,gai) {"function":"where_general_index"}
  gaok:bool = compare.lt(gaindex,gamode)
  branch gaok mask_at(gas,gam,gax,gay,gao,gan,gaw,gai,gamode,gaindex) bad(gas)
block fast_index(fis:mutbuf,fim:u64,fix:u64,fiy:u64,fio:u64,fin:u64,fiw:u64,fii:u64,fistep:u64)
  linear_one:u64 = const() {"value":1}
  divide:bool = compare.gt(fistep,linear_one)
  branch divide trailing_index(fis,fim,fix,fiy,fio,fin,fiw,fii,fistep) linear_index(fis,fim,fix,fiy,fio,fin,fiw,fii,fistep)
'''.lstrip(b'\n')
old_fast=b'''  linear_one:u64 = const() {"value":1}
  divide:bool = compare.gt(qstep,linear_one)
  branch divide trailing_index(qs,qm,qx,qy,qo,qn,qw,qi,qstep) linear_index(qs,qm,qx,qy,qo,qn,qw,qi,qstep)
'''
pairs=[(b'prefix(hs,hm,hx,hy,ht,hw,hd,vd,hr,hc,hz) bad(hs)',b'prefix(hs,hm,hx,hy,ht,hw,hd,vd,hr,hc,hz) general_try(hs,hm,hx,hy,ht,hw)'),
(b'prefix(fs,fm,fx,fy,ft,fw,fmd,fvd,fr,fc,next_axis) bad(fs)',b'prefix(fs,fm,fx,fy,ft,fw,fmd,fvd,fr,fc,next_axis) general_try(fs,fm,fx,fy,ft,fw)'),
(b'trailing_count(ds,dm,dx,dy,dt,dw,dd,dc,db) bad(ds)',b'trailing_count(ds,dm,dx,dy,dt,dw,dd,dc,db) general_try(ds,dm,dx,dy,dt,dw)'),
(b'block allocate(',fallback+b'block allocate('),
(old_fast,decode)]
def sub(raw,old,new):
 assert raw.count(old)==1
 return raw.replace(old,new,1)
new=original
for x,y in pairs:new=sub(new,x,y)
back=new
for x,y in reversed(pairs):back=sub(back,y,x)
assert back==original
agg=sub(runtime,original,inc+new)
assert sub(agg,inc+new,original)==runtime
split=sub(rows[up],original,inc+new)
export0=b'exports(where_ordinary,'
export1=b'exports(where_general_geometry,where_general_index,where_ordinary,'
split=sub(split,export0,export1)
assert sub(sub(split,export1,export0),inc+new,original)==rows[up]
before=b.parse_source(runtime);after=b.parse_source(agg)
assert all(before['signatures'][k]==after['signatures'][k] for k in before['signatures'])
assert set(after['definitions'])-set(before['definitions'])=={'where_general_geometry','where_general_index'}
assert {k for k in before['definitions'] if before['definitions'][k]!=after['definitions'][k]}=={'where_self_tensor'}
rows[up]=split;rows[m['runtime']['path']]=agg
m.pop('expected_native_sha256')
m['name']='opt_in_joint_where_right_aligned_bool_condition_cpu'
m['runtime'].update(sha256=h(agg),**{k:after[k] for k in ('functions','blocks','instructions')})
u.pop('expected_object_sha256');u['sha256']=h(split)
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
print(json.dumps(dict(parent_head='d47f75e9c78e2523468edf11d5f10c7ea6566cf5',runtime_sha256=h(agg),unit12_sha256=h(split),
 manifest_sha256=h(b.json_bytes(m)),old_body_sha256=h(original),new_body_sha256=h(new),check_sources='PASS',byte_exact_inverse=True,
 changed=['where_self_tensor'],added=['where_general_geometry','where_general_index'],native_torch_compile_executed=False),indent=2))
