"""Source-only reversible exact #69 rank1 public-dtype nonzero projection."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
src,checker,dest=map(Path,sys.argv[1:])
src=src.resolve(strict=True)
def sha(v):return hashlib.sha256(v).hexdigest()
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('checker',checker)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
m=json.loads((src/'manifest.json').read_bytes(),object_pairs_hook=b.unique_object)
r,units,_=b.check_sources(m,src,b.read_file)
assert sha(r)=='42cadd9e3f1c7aea5ea391ef32b020efecbc972030a9df3c44a1a689c9ff9f91'
unit=next(u for u in m['units'] if u['name']=='runtime775_11');up=unit['path']
rows={u['path']:rb for u,rb,_ in units}
assert sha(rows[up])=='681b3509f2e5d689e73aaa66e4ab32a4afd145fc873f384b192ee9768d82d914'
R=b.parse_source(r)
old=R['definitions']['nz_tensor'].decode()
new=old
def rep(a,z):
 global new
 assert new.count(a)==1,a
 new=new.replace(a,z,1)
rep('  tag_ok:bool = compare.eq(tag,one)', '''  z:u64 = const() {"value":0}
  priv:u64 = const() {"value":3}
  wid:u64 = call(tag) {"function":"dtype_width"}
  pub:bool = compare.ne(tag,priv)
  known_dtype:bool = compare.ne(wid,z)
  tag_ok:bool = bool.and(pub,known_dtype)''')
# Predicate depends only on cached admitted dtype and current raw input word.
rep('branch gate cap(s,x,n,one) invalid(s)','branch gate cap(s,x,n,wid,tag) invalid(s)')
rep('block cap(cs:mutbuf,cx:u64,cn:u64,co:u64)','block cap(cs:mutbuf,cx:u64,cn:u64,co:u64,ct:u64)')
rep('owned(cs,cx,cn,co)','owned(cs,cx,cn,co,ct)')
rep('block owned(os1:mutbuf,ox1:u64,on:u64,ow1:u64)','block owned(os1:mutbuf,ox1:u64,on:u64,ow1:u64,ot:u64)')
rep('count(os1,ox1,on,oz,oz)','count(os1,ox1,on,oz,oz,ot)')
rep('block count(cs0:mutbuf,cx0:u64,cn0:u64,ci:u64,ck:u64)','block count(cs0:mutbuf,cx0:u64,cn0:u64,ci:u64,ck:u64,ct0:u64)')
rep('bit(cs0,cx0,cn0,ci,ck) alloc_result(cs0,cx0,cn0,ck)','bit(cs0,cx0,cn0,ci,ck,ct0) alloc_result(cs0,cx0,cn0,ck,ct0)')
first=new.index('block bit(');last=new.index('block alloc_result(')
new=new[:first]+'''block bit(bs:mutbuf,bx:u64,bn:u64,bi:u64,bk:u64,bt:u64)
  bitval:u64 = call(bs,bx,bi,bt) {"function":"nz_truth"}
  b1:u64 = const() {"value":1}
  nxt:u64 = int.add.wrap(bi,b1)
  cnt:u64 = int.add.wrap(bk,bitval)
  ok:bool = compare.le(bitval,b1)
  branch ok count(bs,bx,bn,nxt,cnt,bt) invalid(bs)
'''+new[last:]
for a,z in [
 ('block alloc_result(as:mutbuf,ax:u64,an:u64,ak:u64)','block alloc_result(as:mutbuf,ax:u64,an:u64,ak:u64,at:u64)'),
 ('verify_out(as,ax,an,ak,new_t,a0)','verify_out(as,ax,an,ak,new_t,a0,at)'),
 ('block verify_out(vs:mutbuf,vx:u64,vn:u64,vk:u64,vo:u64,vz:u64)','block verify_out(vs:mutbuf,vx:u64,vn:u64,vk:u64,vo:u64,vz:u64,vt:u64)'),
 ('tested(vs,vx,vn,vk,vo,vz)','tested(vs,vx,vn,vk,vo,vz,vt)'),
 ('block tested(ts1:mutbuf,tx1:u64,tn1:u64,tk1:u64,to1:u64,tz1:u64)','block tested(ts1:mutbuf,tx1:u64,tn1:u64,tk1:u64,to1:u64,tz1:u64,tt:u64)'),
 ('loop(ts1,tx1,tn1,tk1,to1,tz1,tz1)','loop(ts1,tx1,tn1,tk1,to1,tz1,tz1,tt)'),
 ('block loop(ls:mutbuf,lx:u64,ln:u64,lk:u64,lo:u64,li:u64,lj:u64)','block loop(ls:mutbuf,lx:u64,ln:u64,lk:u64,lo:u64,li:u64,lj:u64,ltag:u64)'),
 ('choose(ls,lx,ln,lk,lo,li,lj)','choose(ls,lx,ln,lk,lo,li,lj,ltag)'),
 ('block choose(hs:mutbuf,hx:u64,hn:u64,hk:u64,ho:u64,hi:u64,hj:u64)','block choose(hs:mutbuf,hx:u64,hn:u64,hk:u64,ho:u64,hi:u64,hj:u64,ht:u64)'),
 ('  hv:u64 = call(hs,hx,hi) {"function":"tget"}\n  herr:u64 = call(hs,h0) {"function":"getw"}\n  sane:bool = compare.le(hv,h1)\n  hclean:bool = compare.eq(herr,h0)\n  hsane:bool = bool.and(sane,hclean)','  hv:u64 = call(hs,hx,hi,ht) {"function":"nz_truth"}\n  hsane:bool = compare.le(hv,h1)'),
 ('place(hs,hx,hn,hk,ho,hi,hj,hv,h0,h1)','place(hs,hx,hn,hk,ho,hi,hj,hv,h0,h1,ht)'),
 ('pz1:u64,pone:u64)','pz1:u64,pone:u64,pt:u64)'),
 ('room(ps1,px1,pn1,pk1,po1,pi1,pj1,step) loop(ps1,px1,pn1,pk1,po1,step,pj1)','room(ps1,px1,pn1,pk1,po1,pi1,pj1,step,pt) loop(ps1,px1,pn1,pk1,po1,step,pj1,pt)'),
 ('mj:u64,mnext:u64)','mj:u64,mnext:u64,mt:u64)'),
 ('save_word(ms,mx,mn,mk,mo,mi,mj,mnext)','save_word(ms,mx,mn,mk,mo,mi,mj,mnext,mt)'),
 ('wj:u64,wnext:u64)','wj:u64,wnext:u64,wt:u64)'),
 ('loop(ws,wx,wn,wk,wo,wnext,jnext)','loop(ws,wx,wn,wk,wo,wnext,jnext,wt)'),
]:rep(a,z)
# Private local helper. 2 sentinel denotes checked read error or noncanonical Bool.
helper='''fn nz_truth(s:mutbuf,x:u64,i:u64,t:u64) -> (u64) effects(read,write,trap) entry(rd)
block rd(s:mutbuf,x:u64,i:u64,t:u64)
  z:u64 = const() {"value":0}
  o:u64 = const() {"value":1}
  w:u64 = call(s,x,i) {"function":"tget"}
  st:u64 = call(s,z) {"function":"getw"}
  clean:bool = compare.eq(st,z)
  isb:bool = compare.eq(t,o)
  high:bool = compare.gt(w,o)
  badb:bool = bool.and(isb,high)
  branch clean chk(w,t,badb) bad()
block chk(cw:u64,ct:u64,cb:bool)
  branch cb bad() truth(cw,ct)
block truth(tw:u64,tt:u64)
  ans:u64 = call(tw,tt) {"function":"logical_truth"}
  ret(ans)
block bad()
  no:u64 = const() {"value":2}
  ret(no)
end
'''
new=(helper+new).encode();old=old.encode()
extern=b'fn logical_truth(v0:u64,v1:u64) -> (u64) effects(read,write,trap) external()\n'
at=b'fn so_version_live(v0:mutbuf,v1:u64) -> (u64) effects(read,write,trap) external()\n'
def project(x,split=False,inverse=False):
 changes=[(old,new)]+([(at,at+extern),(b'exports(nz_error,',b'exports(nz_truth,nz_error,')] if split else [])
 for a,z in (reversed(changes) if inverse else changes):
  if inverse:a,z=z,a
  assert x.count(a)==1
  x=x.replace(a,z,1)
 return x
nr=project(r);nu=project(rows[up],True)
assert project(nr,inverse=True)==r and project(nu,True,True)==rows[up]
N=b.parse_source(nr);U=b.parse_source(nu,unit=True)
assert set(N['definitions'])-set(R['definitions'])=={'nz_truth'}
assert {n for n in R['definitions'] if N['definitions'][n]!=R['definitions'][n]}=={'nz_tensor'}
assert all(N['signatures'][n]==R['signatures'][n] for n in R['signatures'])
def arity(xs):return len(xs.split(b',')) if xs else 0
for f in ('nz_tensor','nz_truth'):
 v=N['definitions'][f]
 bindings=[]
 for line in v.splitlines()[1:]:
  if line.startswith(b'block '):bindings+=re.findall(rb'\b(\w+):\w+',line)
  else:bindings+=re.findall(rb'^  (\w+):\w+ =',line)
 assert len(bindings)==len(set(bindings)),(f,[k for k in set(bindings) if bindings.count(k)>1])
 assert all(len(k)<=99 for k in bindings)
 targets={q[1]:arity(q[2]) for q in re.finditer(rb'^block (\w+)\(([^)]*)\)',v,re.M)}
 for line in re.findall(rb'^  (?:branch|jump) ([^\n]+)',v,re.M):
  for q in re.finditer(rb'(\w+)\(([^)]*)\)',line):
   assert q[1] in targets and targets[q[1]]==arity(q[2]),(f,q[1])
 for q in re.finditer(rb'^  (\w+):u64 = call\(([^)]*)\) \{"function":"(\w+)"\}$',v,re.M):
  decl=N['definitions'][q[3].decode()].splitlines()[0]
  assert arity(q[2])==arity(decl.split(b'(',1)[1].split(b')',1)[0]),(f,q[3])
for name in (b'typed_descriptor',b'typed_store',b'taddr',b'nz_output',b'nz_ordinary'):
 anchor=rb'^  .*\{"function":"'+name+rb'"\}$'
 assert re.findall(anchor,old,re.M)==re.findall(anchor,new,re.M)
m['name']='opt_in_exact_nonzero_rank1_public_dtype'
m.pop('expected_native_sha256',None);unit.pop('expected_object_sha256',None)
unit['sha256']=sha(nu)
rp=m['runtime']['path'];m['runtime'].update(sha256=sha(nr),**{k:N[k] for k in ('functions','blocks','instructions')})
rows.update({up:nu,rp:nr})
def reader(path,want):
 data=rows[path.relative_to(src).as_posix()];assert sha(data)==want;return data
b.check_sources(m,src,reader)
dest.mkdir();(dest/'units').mkdir()
for path,data in rows.items():(dest/path).write_bytes(data)
(dest/'manifest.json').write_bytes(b.json_bytes(m));b.check_sources(m,dest,b.read_file)
print(json.dumps(dict(parent_git='cb83670b272a6a9816b12c4ac8b2ef6f5473a087',parent_tree='2a76d48da8ddd23a45df734c6020491e28fed88a',
 parent=sha(r),combined=sha(nr),unit11=sha(nu),manifest=sha(b.json_bytes(m)),inverse=True,split=True,source_only=True,
 opcode_unchanged=135,changed_bodies=['nz_tensor'],added_local_bodies=['nz_truth']),sort_keys=True))
