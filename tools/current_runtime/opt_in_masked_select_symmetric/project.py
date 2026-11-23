"""Exact source-only bounded symmetric masked_select, never launches HIA."""
from pathlib import Path
import hashlib,json,re,sys,importlib.util
src,mp,checker,dest=map(Path,sys.argv[1:]); p=Path(__file__).parent
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('check',checker); B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
assert sha(mp.read_bytes())=='3bb8e64a5c594936d49e2d88966b32885c908e0dadb0a80760656461d0b1a193'
m=json.loads(mp.read_bytes(),object_pairs_hook=B.unique_object); R0,rows,_=B.check_sources(m,src,B.read_file)
assert sha(R0)=='25a495092c57ade642c4d3648a16bc42ded060a88473c06b60d22080b99ee73f'
nodes={j['path']:v for j,v,_ in rows};unit=next(j for j in m['units'] if j['name']=='runtime775_11');u=unit['path'];U0=nodes[u]
assert sha(U0)=='609715ee8249646b0559edb21f63b5f3c7ae26d1120822c36512ca91a2335c3a'
def repl(v,a,b):
 assert v.count(a)==1,a
 return v.replace(a,b,1)
O=B.parse_source(R0)
old=O['definitions']['ms_tensor']; v=old.decode()
v=repl(v,'rank_fits:bool = compare.le(mrank,xrank)','rank_fits:bool = compare.le(mrank,four)')
v=repl(v,'''  uok:u64 = call(us,uxd,umd,uxr,umr) {"function":"ms_bshape"}
  pass:bool = compare.ne(uok,uz)
  branch pass cap(us,ux,um,uxd,un,uok,uwidth) invalid(us)
''','''  fixed_rank:bool = compare.le(umr,uxr)
  branch fixed_rank fixed_extent(us,ux,um,uxd,umd,un,uz,uxr,umr) sym_join(us,ux,um,uxd,umd,un,uxr,umr)
block fixed_extent(gs:mutbuf,gx:u64,gm:u64,gxd:u64,gmd:u64,gn:u64,gz:u64,gxr:u64,gmr:u64)
  gok:u64 = call(gs,gxd,gmd,gxr,gmr) {"function":"ms_bshape"}
  gpass:bool = compare.ne(gok,gz)
  branch gpass cap(gs,gx,gm,gxd,gn,gok) sym_join(gs,gx,gm,gxd,gmd,gn,gxr,gmr)
block sym_join(ys:mutbuf,yx:u64,ym:u64,yxd:u64,ymd:u64,yn:u64,yxr:u64,ymr:u64)
  enc:u64 = call(ys,yxd,ymd,yxr,ymr) {"function":"ms_scount"}
  yz:u64 = const() {"value":0}
  yvalid:bool = compare.ne(enc,yz)
  branch yvalid sym_bounds(ys,yx,ym,yxd,ymd,yn,yxr,ymr,enc) invalid(ys)
block sym_bounds(zs1:mutbuf,zx:u64,zm:u64,zxd:u64,zmd:u64,zxn:u64,zxr:u64,zmr:u64,zenc:u64)
  z8:u64 = const() {"value":8}
  zmax:u64 = const() {"value":262144}
  zmp:u64 = int.add.wrap(zmd,z8)
  zmn:u64 = call(zs1,zmp) {"function":"getw"}
  zmcap:bool = compare.le(zmn,zmax)
  branch zmcap sym_limit(zs1,zx,zm,zxd,zmd,zxn,zmn,zxr,zmr,zenc,zmax) invalid(zs1)
block sym_limit(ls:mutbuf,lx:u64,lm:u64,lxd:u64,lmd:u64,lxn:u64,lmn:u64,lxr:u64,lmr:u64,lenc:u64,lmax:u64)
  l1:u64 = const() {"value":1}
  joined:u64 = int.sub.wrap(lenc,l1)
  jc:bool = compare.le(joined,lmax)
  xc:bool = compare.le(lxn,lmax)
  bothcap:bool = bool.and(jc,xc)
  branch bothcap sym_valid(ls,lx,lm,lxd,lmd,lxn,lmn,lxr,lmr,joined) capacity(ls)
block sym_valid(qs:mutbuf,qx:u64,qm:u64,qxd:u64,qmd:u64,qxn:u64,qmn:u64,qxr:u64,qmr:u64,qlattice:u64)
  q0:u64 = const() {"value":0}
  q2:u64 = const() {"value":2}
  xeq:u64 = call(qs,qxd,qxr,qxn) {"function":"ms_geometry"}
  meq:u64 = call(qs,qmd,qmr,qmn) {"function":"ms_geometry"}
  proof:u64 = bit.and(xeq,meq)
  precise:bool = compare.ne(proof,q0)
  branch precise owner(qs,qx,qm,qlattice,q2) invalid(qs)
''')
# Add one width parameter to each new geometry block, no extra state writes.
forward={'fixed_extent':'gwidth','sym_join':'ywidth','sym_bounds':'zwidth','sym_limit':'lwidth','sym_valid':'qwidth'}
b=None;lines=[]
for line in v.splitlines(keepends=True):
 if mm:=re.match(r'^block (\w+)\(',line):
  b=mm[1]
  if b in forward:line=line.replace(')',','+forward[b]+':u64)',1)
 elif b in {**forward,'expand_mask':'uwidth'} and re.match(r'^  branch ',line):
  for dst in [*forward,'cap','owner']:
   # cap/owner already called with width in existing unrelated bodies, only these rewrites reach them.
   line=re.sub(r'\b'+dst+r'\(([^)]*)\)',lambda q:dst+'('+q[1]+','+({**forward,'expand_mask':'uwidth'})[b]+')',line)
 lines.append(line)
v=''.join(lines)
assert v.count('{"function":"ms_bindex"}')==2
v=v.replace('{"function":"ms_bindex"}','{"function":"ms_mask_index"}')
v=repl(v,'''  raw:u64 = call(ds,dx,di) {"function":"tget"}
''','''  xidx:u64 = call(ds,dx,dm,di,dmode) {"function":"ms_value_index"}
  xlim:u64 = const() {"value":262144}
  valid_x:bool = compare.lt(xidx,xlim)
  branch valid_x value_load(ds,dx,dm,dn,dk,do,dj,dnext,dmode,dwidth,xidx) invalid(ds)
block value_load(vls:mutbuf,vlx:u64,vlm:u64,vln:u64,vlk:u64,vlo:u64,vlj:u64,vlnext:u64,vlmode:u64,vlwidth:u64,vlidx:u64)
  raw:u64 = call(vls,vlx,vlidx) {"function":"tget"}
''')
a=v.index('block value_load'); b=v.index('block selected_word',a); body=v[a:b]
rename=dict(ds='vls',dx='vlx',dm='vlm',dn='vln',dk='vlk',do='vlo',dj='vlj',dnext='vlnext',dmode='vlmode',dwidth='vlwidth')
body=re.sub(r'\b\w+\b',lambda q:rename.get(q[0],q[0]),body)
assert 'call(vls,d0)' in body
body=body.replace('call(vls,d0)','call(vls,vd0)')
v=v[:a]+body[:body.index('  raw:u64')]+'''  vd0:u64 = const() {"value":0}
'''+body[body.index('  raw:u64'):]+v[b:]
v=repl(v,'''  d0:u64 = const() {"value":0}
  xidx''','''  xidx''')
v=repl(v,'compare.eq(rd_err,d0)','compare.eq(rd_err,vd0)')
extras=(p/'helpers.hepta.inc').read_bytes();new=v.encode()+extras
names=['ms_align','ms_merge_dim','ms_scount','ms_sindex','ms_mask_index','ms_value_index']
R=repl(R0,old,new); U=repl(U0,old,new)
ex0=b'exports(ms_bshape,';ex1=b'exports('+','.join(names).encode()+b',ms_bshape,'
U=repl(U,ex0,ex1)
assert repl(R,new,old)==R0 and repl(repl(U,ex1,ex0),new,old)==U0
N=B.parse_source(R)
assert set(N['definitions'])-set(O['definitions'])==set(names)
assert {x for x in O['definitions'] if O['definitions'][x]!=N['definitions'][x]}=={'ms_tensor'}
assert all(N['signatures'][x]==O['signatures'][x] for x in O['signatures'])
for name in ['ms_tensor',*names]:
 f=N['definitions'][name].decode()
 args=lambda a:a.split(',') if a else []
 rng=re.match(r'^fn \w+\(([^\n]*)\) -> .* entry\((\w+)\)\nblock (\w+)\(([^\n]+)\)\n',f)
 assert rng and rng[1]==rng[4] and rng[2]==rng[3],name
 params=re.findall(r'\b(\w+):(?:u64|bool|readbuf|mutbuf|usize)\b',f.split('\n',1)[1]); assert len(params)==len(set(params)),name
 dests={g[1]:len(args(g[2])) for g in re.finditer(r'^block (\w+)\(([^)]*)\)',f,re.M)}
 for t in re.findall(r'^  (?:branch|jump) ([^\n]+)',f,re.M):
  for gg in re.finditer(r'(\w+)\(([^)]*)\)',t):assert dests[gg[1]]==len(args(gg[2])),(name,gg[0])
 assert '_ = call(' not in f
assert b'call(mrr,mro,mrs,mrat,mrop,mrp)' in N['definitions']['entry']
assert 'call(as,a1,ap0,a0,a1,value_tag)' in v and 'call(ts,target,twidth,traw)' in v
assert b'"function":"putw"' not in extras and b'"function":"typed_descriptor"' not in extras
m.pop('expected_native_sha256');unit.pop('expected_object_sha256')
unit['sha256']=sha(U);m['name']='opt_in_main_public_value_masked_select_symmetric_bool_mask'
rp=m['runtime']['path'];m['runtime'].update(sha256=sha(R),**{k:N[k] for k in ('functions','blocks','instructions')})
nodes.update({rp:R,u:U});dest.mkdir();(dest/'units').mkdir()
for k,v in nodes.items():(dest/k).write_bytes(v)
(dest/'manifest.json').write_bytes(B.json_bytes(m));B.check_sources(m,dest,B.read_file)
print(json.dumps(dict(head='0489d49f4f497f02a218a06d64d80081ae004a00',tree='58ba08dbb8f14ac045e836f664b841e250280f7c',
 runtime=sha(R),unit11=sha(U),manifest=sha(B.json_bytes(m)),inverse=True,static_split=True,actual_native=False),sort_keys=True))
