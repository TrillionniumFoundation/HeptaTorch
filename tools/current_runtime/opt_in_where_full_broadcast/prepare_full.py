"""Source-only reversible three-way where fallback from the exact published CPU candidate."""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
S,B=map(Path,sys.argv[1:3]);S=S.resolve(strict=True)
h=lambda b:hashlib.sha256(b).hexdigest()
if not __debug__: raise ValueError('source checks require assertions')
assert h(B.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('builder',B);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(S/'manifest.json').read_bytes()
assert h(mraw)=='3cf7b033a467cea99bbc5fed0ee6685beecbefc80612688efc93153551c7c575'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
runtime,units,_=b.check_sources(m,S,b.read_file)
assert h(runtime)=='30cd4e10394bc82d7197ba5b1a707ce6d7806e05c0674b8e171addd6de9d34da'
u=next(row for row in m['units'] if row['name']=='runtime800_12');up=u['path']
rows={row['path']:raw for row,raw,_ in units}
assert h(rows[up])=='ae2d2bbfb96224645acaada95300a881d0a193682dbe247f75c79a5b907e3839'
inc=Path(__file__).with_name('full.hepta.inc').read_bytes()
pat=rb'^fn where_self_tensor\([^\n]+\n.*?^end\n'
original=re.search(pat,runtime,re.M|re.S)[0]
src=b'''  branch g_admit allocate(gs0,gm0,gx0,gy0,gt0,gw0,g_mode) bad(gs0)
'''
dst=b'''  branch g_admit allocate(gs0,gm0,gx0,gy0,gt0,gw0,g_mode) full_try(gs0,gm0,gx0,gy0,gt0,gw0)
block full_try(ws0:mutbuf,wm0:u64,wx0:u64,wy0:u64,wt0:u64,ww0:u64)
  full_out:u64 = call(ws0,wm0,wx0,wy0,wt0,ww0) {"function":"where_full_where"}
  ret(full_out)
'''
def sub(raw,old,new):
 assert raw.count(old)==1
 return raw.replace(old,new,1)
new=sub(original,src,dst)
assert sub(new,dst,src)==original
agg=sub(runtime,original,inc+new)
assert sub(agg,inc+new,original)==runtime
split=sub(rows[up],original,inc+new)
export0=b'exports(where_general_geometry,'
exports=b'where_full_aligned,where_full_join,where_full_extent,where_full_product,where_full_alloc,where_full_where'
export1=b'exports('+exports+b',where_general_geometry,'
split=sub(split,export0,export1)
assert sub(sub(split,export1,export0),inc+new,original)==rows[up]
before=b.parse_source(runtime);after=b.parse_source(agg)
assert all(before['signatures'][k]==after['signatures'][k] for k in before['signatures'])
assert set(after['definitions'])-set(before['definitions'])==set(exports.decode().split(','))
assert {k for k in before['definitions'] if before['definitions'][k]!=after['definitions'][k]}=={'where_self_tensor'}
rows[up]=split;rows[m['runtime']['path']]=agg
m.pop('expected_native_sha256')
m['name']='opt_in_joint_where_three_operand_cpu'
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
print(json.dumps(dict(parent_head='c2fac4373c8047b7a9a6acf5fadf2f3aeffa2c8b',runtime_sha256=h(agg),unit12_sha256=h(split),
 manifest_sha256=h(b.json_bytes(m)),old_body_sha256=h(original),new_body_sha256=h(new),check_sources='PASS',byte_exact_inverse=True,
 changed=['where_self_tensor'],added=exports.decode().split(','),native_torch_compile_executed=False),indent=2))
