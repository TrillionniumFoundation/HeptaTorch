"""Source-only projection from exact b545f2a logical family."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
D=Path(__file__).parent; S,B=map(Path,sys.argv[1:3]);S=S.resolve(strict=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
assert __debug__ and sha(B.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('builder',B);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(S/'manifest.json').read_bytes()
assert sha(mraw)=='3248f2f185461b6901ff5f6c42fc03dbcd2f1f262cce32cc8a4e1aed04cecce2'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
r,units,_=b.check_sources(m,S,b.read_file)
assert sha(r)=='e37dd3c347cb6143a53d283c39c9e2ff637640d568c8347095117c43b9ab6144'
u=next(x for x in m['units'] if x['name']=='runtime775_11');up=u['path'];rp=m['runtime']['path']
rows={x['path']:v for x,v,_ in units};assert sha(rows[up])=='d0278d8a7acf0965e75c7f42f6f02910a9b0eb3d9c1b8182f9de1f939ec82709'
def one(s,a,c):
 assert s.count(a)==1,a
 return s.replace(a,c,1)
def fn(v,n):return re.search(rb'^fn '+n+rb'\([^\n]+\n.*?^end\n',v,re.M|re.S)[0]
t0=fn(r,b'la_tensor');t=t0
t=one(t,b'  li_equal:bool = compare.eq(li_n,li_m)\n',b'  li_y_nonempty:bool = compare.gt(li_m,li_z)\n')
t=one(t,b'bool.and(li_equal,li_nonempty)',b'bool.and(li_y_nonempty,li_nonempty)')
t=one(t,b'li_xd,li_yd,li_n,li_op)',b'li_xd,li_yd,li_n,li_m,li_op)')
t=one(t,b'lg_n:u64,lg_op:u64)',b'lg_n:u64,lg_m:u64,lg_op:u64)')
t=one(t,b'lg_yd,lg_yr,lg_n)',b'lg_yd,lg_yr,lg_m)')
t=one(t,b'''  lg_all:u64 = bit.and(lg_both,lg_same)
  lg_ok:bool = compare.ne(lg_all,lg_z)
  branch lg_ok la_allocate(lg_s,lg_x,lg_y,lg_t,lg_xd,lg_xr,lg_n,lg_op) la_bad(lg_s)
''',b'''  lg_ok:bool = compare.ne(lg_both,lg_z)
  branch lg_ok la_choose(lg_s,lg_x,lg_y,lg_t,lg_xd,lg_yd,lg_xr,lg_yr,lg_n,lg_op,lg_same,lg_z) la_bad(lg_s)
block la_choose(lch_s:mutbuf,lch_x:u64,lch_y:u64,lch_t:u64,lch_xd:u64,lch_yd:u64,lch_xr:u64,lch_yr:u64,lch_n:u64,lch_op:u64,lch_same:u64,lch_z:u64)
  lch_direct:bool = compare.ne(lch_same,lch_z)
  branch lch_direct la_allocate(lch_s,lch_x,lch_y,lch_t,lch_xd,lch_xr,lch_n,lch_op) la_bcast(lch_s,lch_x,lch_y,lch_t,lch_xd,lch_yd,lch_xr,lch_yr,lch_op)
'''+(D/'broadcast_path.hepta.inc').read_bytes())
t=one(t,b'  branch loc_ok la_loop(loc_s,loc_x,loc_y,loc_t,loc_out,loc_n,loc_z,loc_op)',b'  loc_direct:bool = compare.ne(loc_z,loc_z)\n  branch loc_ok la_loop(loc_s,loc_x,loc_y,loc_t,loc_out,loc_n,loc_z,loc_op,loc_direct)')
for p0 in (b'llp',b'lrd',b'lst'):
 t=one(t,p0+b'_op:u64)\n',p0+b'_op:u64,'+p0+b'_bcast:bool)\n')
t=one(t,b'llp_out,llp_n,llp_i,llp_op) la_done',b'llp_out,llp_n,llp_i,llp_op,llp_bcast) la_done')
t=one(t,b'lst_out,lst_n,lst_next,lst_op) la_zero',b'lst_out,lst_n,lst_next,lst_op,lst_bcast) la_zero')
start=t.index(b'  lrd_z:u64 ='); end=t.index(b'block la_store',start)
piece=t[start:end].replace(b'lrd_',b'lv_')
piece=one(piece,b'call(lv_s,lv_x,lv_i)',b'call(lv_s,lv_x,lv_xi)')
piece=one(piece,b'call(lv_s,lv_y,lv_i)',b'call(lv_s,lv_y,lv_yi)')
piece=one(piece,b'lv_i,lv_a,lv_b,lv_op) la_read_failure',b'lv_i,lv_a,lv_b,lv_op,lv_bcast) la_read_failure')
t=t[:start]+(D/'index_path.hepta.inc').read_bytes()+piece+t[end:]
inc=(D/'geometry.hepta.inc').read_bytes();names=re.findall(rb'^fn (\w+)\(',inc,re.M)
assert set(names)==set(b'la_bc_aligned la_bc_join la_bc_product la_bc_alloc la_bc_extent la_bc_plan'.split())
export_old=b'exports(la_error,'
export_new=b'exports('+b','.join(names)+b',la_error,'
imports0=re.search(rb'^fn bindex\([^\n]+effects\(read,write,trap\)',r,re.M)[0]+b' external()\n'
im_old=b'fn absf(v0:u64) -> (u64) effects(read,write,trap) external()\n'
assert imports0 not in rows[up]
steps=[(t0,t),(b'fn la_command(',inc+b'fn la_command(')]
def trans(v,split,inv=False):
 seq=steps+([(export_old,export_new),(im_old,imports0+im_old)] if split else [])
 for a,c in reversed(seq) if inv else seq:v=one(v,c,a) if inv else one(v,a,c)
 return v
R=trans(r,False);U=trans(rows[up],True)
assert trans(R,False,True)==r and trans(U,True,True)==rows[up]
old=b.parse_source(r); now=b.parse_source(R); old_u=b.parse_source(rows[up],unit=True);new_u=b.parse_source(U,unit=True)
assert set(now['definitions'])-set(old['definitions'])==set(x.decode() for x in names)
assert {x for x in old['definitions'] if old['definitions'][x]!=now['definitions'][x]}=={'la_tensor'}
assert set(new_u['imports'])-set(old_u['imports'])=={'bindex'}
assert set(new_u['exports'])-set(old_u['exports'])==set(x.decode() for x in names)
assert all(now['signatures'][k]==old['signatures'][k] for k in old['signatures'])
m.pop('expected_native_sha256');u.pop('expected_object_sha256')
m['name']='opt_in_logical_family_same_dtype_nonempty_broadcast_cpu'
m['runtime'].update(sha256=sha(R),**{k:now[k] for k in ('functions','blocks','instructions')})
u['sha256']=sha(U);rows.update({rp:R,up:U})
def read(p,h):
 v=rows[p.relative_to(S).as_posix()];assert sha(v)==h;return v
b.check_sources(m,S,read)
if len(sys.argv)==4:
 dest=Path(sys.argv[3]);dest.mkdir();(dest/'units').mkdir()
 for path,v in rows.items():(dest/path).write_bytes(v)
 (dest/'manifest.json').write_bytes(b.json_bytes(m));b.check_sources(m,dest,b.read_file)
print(json.dumps(dict(parent_head='b545f2a50540390f6fa86300418f83fddff41c96',parent_tree='1b234b74507ab0c4ad2dbc1cd86427a9e2cae4f9',
 parent_source=sha(r),source_sha256=sha(R),unit11=sha(U),manifest=sha(b.json_bytes(m)),inverse_combined_and_split=True,
 existing_changed=['la_tensor'],added=[k.decode() for k in names],split_imports_added=['bindex'],wire_entry_unchanged=True,
 source_static='PASS',native_or_compile_executed=False),indent=2,sort_keys=True))
