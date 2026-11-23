"""Exact, reversible source-only mask broadcast projection atop #67."""
import re,hashlib,sys,json,importlib.util
from pathlib import Path
p=Path(__file__).parent; src,checker,dst=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
def repl(v,a,c):
 assert v.count(a)==1,a
 return v.replace(a,c,1)
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checker',checker); B=importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
raw=(src/'manifest.json').read_bytes();assert sha(raw)=='9c660b55d73f51abbf9286c15057a9b92d09fc72ba450db45b52e63829184a25'
m=json.loads(raw,object_pairs_hook=B.unique_object)
r,units,_=B.check_sources(m,src,B.read_file)
assert sha(r)=='baa3b155e5965e2c452fae534921a378c26d8b5befafd69b68d1f542b5e249ee'
unit=next(u for u in m['units'] if u['name']=='runtime775_11'); up=unit['path']
rows={u['path']:v for u,v,_ in units}
oldunit=rows[up];assert sha(oldunit)=='551a5c4e8ad7cad3c0d7bf009e1e1a8bd85ac0b93f2984cf2815f3ff5522c230'
O=B.parse_source(r); old=O['definitions']['ms_tensor']
core=old.decode()
a='''  r_same:bool = compare.eq(xrank,mrank)
'''
core=repl(core,a,'''  rank_fits:bool = compare.le(mrank,xrank)
''')
core=repl(core,'bool.and(rx,r_same)','bool.and(rx,rank_fits)')
core=repl(core,'''  branch geometry cap(ss,sx,sm,sxd,sn) invalid(ss)
''','''  branch geometry cap(ss,sx,sm,sxd,sn,sz) expand_mask(ss,sx,sm,sxd,smd,sn,sz)
block expand_mask(us:mutbuf,ux:u64,um:u64,uxd:u64,umd:u64,un:u64,uz:u64)
  uxr:u64 = call(us,uxd) {"function":"getw"}
  umr:u64 = call(us,umd) {"function":"getw"}
  uok:u64 = call(us,uxd,umd,uxr,umr) {"function":"ms_bshape"}
  pass:bool = compare.ne(uok,uz)
  branch pass cap(us,ux,um,uxd,un,uok) invalid(us)
''')
mode={'verify_shape':'vgmode','cap':'cmode','owner':'omode','count':'cmode0','bit':'bmode',
 'allocate':'amode','inspect':'vmode','ready':'rmode','copy':'pmode','choose':'hmode',
 'selected':'jmode','room':'mmode','read_value':'dmode','store':'tmode'}
# Precisely thread a geometry mode (0 exact, 1 checked broadcast) through the two immutable passes.
block=None; lines=[]
for l in core.splitlines(keepends=True):
 if q:=re.match(r'^block (\w+)\(',l):
  block=q[1]
  if block in mode:l=l.replace(')',','+mode[block]+':u64)',1)
 elif block in mode and re.match(r'^  (branch|jump) ',l):
  for dest in mode:
   l=re.sub(r'\b'+dest+r'\(([^)]*)\)',lambda q:dest+'('+q[1]+','+mode[block]+')',l)
 lines.append(l)
core=''.join(lines)
# Before either unchecked Bool element loader, refuse sentinel. Direct exact index remains i.
for block,after,sv,xv,mv,nv,iv,kv,extras in (
 ('bit','mask_word','bs','bx','bm','bn','bi','bk',[]),
 ('choose','hbit','hs','hx','hm','hn','hi','hk',[('ho','uo'),('hj','uj')])):
 h=re.search(r'^block '+block+r'\(([^\n]+)\)\n',core,re.M); header=h[0]
 params=[s.split(':')[0] for s in h[1].split(',')]
 types=[s.split(':')[1] for s in h[1].split(',')]
 prefix=block[:1]+'wire';ix=prefix+'i'; lim=prefix+'lim'; good=prefix+'ok'; nextb=block+'_load'
 core=repl(core,header,header+f'''  {ix}:u64 = call({sv},{xv},{mv},{iv},{mode[block]}) {{"function":"ms_bindex"}}
  {lim}:u64 = const() {{"value":262144}}
  {good}:bool = compare.lt({ix},{lim})
  branch {good} {nextb}({','.join(params+[ix])}) invalid({sv})
block {nextb}({','.join([a+':'+t for a,t in zip(params,types,strict=True)]+[prefix+'j:u64'])})
''')
 # Function-wide SSA identifiers must remain distinct across block parameters.
 piece=re.search(r'^block '+nextb+r'\([^\n]+\)\n.*?(?=^block )',core,re.S|re.M)[0]
 maps={a:prefix+a for a in params}
 new=re.sub(r'\b(?:'+'|'.join(sorted(maps,key=len,reverse=True))+r')\b',lambda m:maps[m[0]],piece)
 new=repl(new,f'call({maps[sv]},{maps[mv]},{maps[iv]})',f'call({maps[sv]},{maps[mv]},{prefix}j)')
 core=repl(core,piece,new)
inc=(p/'masked_broadcast.hepta.inc').read_bytes()
newcore=core.encode()
# Insert before other definitions; same split unit so no imports.
before=old
after=newcore+inc
R=repl(r,before,after); U=repl(oldunit,before,after)
names={'ms_bshape','ms_bindex'}
exp0=b'exports(ms_error,'
exp1=b'exports(ms_bshape,ms_bindex,ms_error,'
U=repl(U,exp0,exp1)
assert repl(R,after,before)==r and repl(repl(U,exp1,exp0),after,before)==oldunit
N=B.parse_source(R)
assert set(N['definitions'])-set(O['definitions'])==names
assert {k for k in O['definitions'] if O['definitions'][k]!=N['definitions'][k]}=={'ms_tensor'}
assert all(N['signatures'][k]==v for k,v in O['signatures'].items())
assert R.count(b'"value":130')==1
assert b'call(mrr,mro,mrs,mrat,mrop,mrp)' in N['definitions']['entry']
assert b'"function":"typed_store"' in newcore and b'call(ts,target,t4,traw)' in newcore
for key in ('ms_tensor',*sorted(names)):
 f=N['definitions'][key].decode()
 args=lambda s:s.split(',') if s else []
 vars=re.findall(r'\b(\w+):(?:u64|bool|readbuf|mutbuf|usize)\b',f.split('\n',1)[1])
 assert len(vars)==len(set(vars)),(key,[v for v in vars if vars.count(v)>1])
 targets={m[1]:len(args(m[2])) for m in re.finditer(r'^block (\w+)\(([^)]*)\)',f,re.M)}
 for t in re.findall(r'^  (?:branch|jump) ([^\n]+)',f,re.M):
  for g in re.finditer(r'(\w+)\(([^)]*)\)',t):assert targets[g[1]]==len(args(g[2])),(key,g[0])
 assert '_ = call(' not in f
m.pop('expected_native_sha256')
unit.pop('expected_object_sha256')
unit['sha256']=sha(U);m['name']='opt_in_main_f32_masked_select_one_way_bool_mask'
rp=m['runtime']['path']; m['runtime'].update(sha256=sha(R),**{k:N[k] for k in ('functions','blocks','instructions')})
rows.update({up:U,rp:R})
dst.mkdir();(dst/'units').mkdir()
for k,v in rows.items():(dst/k).write_bytes(v)
(dst/'manifest.json').write_bytes(B.json_bytes(m)); B.check_sources(m,dst,B.read_file)
print(json.dumps(dict(parent='ad964ebaf3f7b3913f039d3ee44802aaff447198',runtime=sha(R),unit11=sha(U),manifest=sha(B.json_bytes(m)),inverse=True,static_split=True),sort_keys=True))
