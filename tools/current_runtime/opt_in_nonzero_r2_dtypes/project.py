"""Reversible exact 2b5b2fc rank2 public dtype nonzero source-only projection."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
src,checker,out=map(Path,sys.argv[1:]);src=src.resolve(strict=True)
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('source_checker',checker)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(src/'manifest.json').read_bytes()
assert sha(mraw)=='f1bc7e013769a1b36010245652e9ccbf8a64207c43ea835043746db9af01f921'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
a,units,_=b.check_sources(m,src,b.read_file)
assert sha(a)=='8420d6599d484e2b6844576a0f60207d6b318cb21bda269f0d60bf135bac10db'
unit=next(x for x in m['units'] if x['name']=='runtime775_11'); p=unit['path']
rows={row['path']:raw for row,raw,_ in units}
assert sha(rows[p])=='8d151d5b8b8fe40e04bafdfeb3e373bd54b1a43156438898a6a9ffd778d08ce3'
A=b.parse_source(a)
old=A['definitions']['nz_rank2']; body=old.decode()
def rep(q,z):
 global body
 assert body.count(q)==1,q
 body=body.replace(q,z,1)
rep('  tp:bool = compare.eq(dt,one)', '''  priv:u64 = const() {"value":3}
  wid:u64 = call(dt) {"function":"dtype_width"}
  pub:bool = compare.ne(dt,priv)
  width_known:bool = compare.ne(wid,z)
  tp:bool = bool.and(pub,width_known)''')
for q,z in [
 ('geom(s,x,n,n0,n1,lim,one)','geom(s,x,n,n0,n1,lim,wid,dt)'),
 ('gl:u64,gw:u64)','gl:u64,gw:u64,gt:u64)'),
 ('limit(gs,gx,gn,gc,gl,gw)','limit(gs,gx,gn,gc,gl,gw,gt)'),
 ('ll:u64,lw:u64)','ll:u64,lw:u64,ltag:u64)'),
 ('owned(ls,lx,ln,lc,lw)','owned(ls,lx,ln,lc,lw,ltag)'),
 ('wc:u64,ww:u64)','wc:u64,ww:u64,wtag:u64)'),
 ('cnt(ws,wx,wn,wc,wz,wz,ww)','cnt(ws,wx,wn,wc,wz,wz,wtag)'),
 ('ck:u64,cw:u64)','ck:u64,ct:u64)'),
 ('scan(cs,cx,cn,cc,ci,ck,cw) allocate(cs,cx,cn,cc,ck)','scan(cs,cx,cn,cc,ci,ck,ct) allocate(cs,cx,cn,cc,ck,ct)'),
 ('bk:u64,bw:u64)','bk:u64,bt:u64)'),
 ('  bv:u64 = call(bs,bx,bi,bw) {"function":"nz_truth"}\n  canon:bool = compare.le(bv,bw)\n  nexti:u64 = int.add.wrap(bi,bw)',
  '  bv:u64 = call(bs,bx,bi,bt) {"function":"nz_truth"}\n  b1:u64 = const() {"value":1}\n  canon:bool = compare.le(bv,b1)\n  nexti:u64 = int.add.wrap(bi,b1)'),
 ('cnt(bs,bx,bn,bc,nexti,nextk,bw)','cnt(bs,bx,bn,bc,nexti,nextk,bt)'),
 ('alc:u64,alk:u64)','alc:u64,alk:u64,alt:u64)'),
 ('fresh(als,alx,aln,alc,alk)','fresh(als,alx,aln,alc,alk,alt)'),
 ('fc:u64,fk:u64)','fc:u64,fk:u64,ft:u64)'),
 ('verify(fs,fx,fnn,fc,fk,fo)','verify(fs,fx,fnn,fc,fk,fo,ft)'),
 ('vk:u64,vo:u64)','vk:u64,vo:u64,vtag:u64)'),
 ('ready(vs,vx,vn,vc,vk,vo,vz,empty)','ready(vs,vx,vn,vc,vk,vo,vz,empty,vtag)'),
 ('tz:u64,te:bool)','tz:u64,te:bool,tt:u64)'),
 ('loop(ts,tx,tn,tc,tk,to,tz,tz)','loop(ts,tx,tn,tc,tk,to,tz,tz,tt)'),
 ('yo:u64,yi:u64,yj:u64)','yo:u64,yi:u64,yj:u64,yt:u64)'),
 ('take(ys,yx,yn,yc,yk,yo,yi,yj)','take(ys,yx,yn,yc,yk,yo,yi,yj,yt)'),
 ('po:u64,pi:u64,pj:u64)','po:u64,pi:u64,pj:u64,pt:u64)'),
 ('call(ps,px,pi,p1) {"function":"nz_truth"}','call(ps,px,pi,pt) {"function":"nz_truth"}'),
 ('write_if(ps,px,pn,pc,pk,po,pi,pj,step,skip)','write_if(ps,px,pn,pc,pk,po,pi,pj,step,skip,pt)'),
 ('ui2:u64,uski:bool)','ui2:u64,uski:bool,ut:u64)'),
 ('loop(us,ux,un,uc,uk,uo,ui2,uj) room(us,ux,un,uc,uk,uo,ui,uj,ui2)','loop(us,ux,un,uc,uk,uo,ui2,uj,ut) room(us,ux,un,uc,uk,uo,ui,uj,ui2,ut)'),
 ('mi:u64,mj:u64,mi2:u64)','mi:u64,mj:u64,mi2:u64,mtt:u64)'),
 ('spill(ms,mx,mn,mc,mk,mo,mi,mj,mi2)','spill(ms,mx,mn,mc,mk,mo,mi,mj,mi2,mtt)'),
 ('si:u64,sj:u64,si2:u64)','si:u64,sj:u64,si2:u64,stag:u64)'),
 ('loop(ss,sx,sn,sc,sk,so,si2,sj2)','loop(ss,sx,sn,sc,sk,so,si2,sj2,stag)'),
]:rep(q,z)
new=body.encode()
def change(v,inverse=False):
 x,y=(new,old) if inverse else (old,new)
 assert v.count(x)==1
 return v.replace(x,y,1)
c=change(a);u=change(rows[p]);assert change(c,True)==a and change(u,True)==rows[p]
C=b.parse_source(c);U=b.parse_source(u,unit=True)
assert C['definitions'].keys()==A['definitions'].keys()
assert {f for f in A['definitions'] if A['definitions'][f]!=C['definitions'][f]}=={'nz_rank2'}
assert C['signatures']==A['signatures']
def decl(v):
 return tuple(tuple(x.split(b':',1)) for x in v.split(b',') if x)
def args(v):
 return tuple(x for x in v.split(b',') if x)
for f in [b'nz_rank2']:
 body=C['definitions'][f.decode()]
 targets={x[1]:decl(x[2]) for x in re.finditer(rb'^block (\w+)\(([^)]*)\)',body,re.M)}
 global_keys=[]
 for nm,pv in targets.items():global_keys.extend(k for k,_ in pv)
 global_keys+=re.findall(rb'^  (\w+):\w+ =',body,re.M)
 assert len(global_keys)==len(set(global_keys)),(f,[k for k in set(global_keys) if global_keys.count(k)>1])
 for block in re.split(rb'(?m)(?=^block )',body)[1:]:
  label=block.split(b'(',1)[0][6:];loc=dict(targets[label])
  for line in block.splitlines()[1:]:
   if x:=re.fullmatch(rb'  (\w+):(\w+) = call\(([^)]*)\) \{"function":"(\w+)"\}',line):
    parameters,returns,_=C['signatures'][x[4].decode()]
    assert tuple(loc[k] for k in args(x[3]))==parameters and (x[2],)==returns,(f,line)
   if line.startswith((b'  branch ',b'  jump ')):
    if line.startswith(b'  branch '):assert loc[line.split()[1]]==b'bool'
    for z in re.finditer(rb'(\w+)\(([^)]*)\)',line):
     assert tuple(loc[k] for k in args(z[2]))==tuple(t for _,t in targets[z[1]]),(f,line)
   if x:=re.fullmatch(rb'  (\w+):(\w+) = .*',line):loc[x[1]]=x[2]
assert b'_ = call(' not in new
assert new.count(b'{"function":"nz_truth"}')==2
assert b'call(ws,wx,ww) {"function":"nz_ordinary"}' in new
assert b'call(bs,bx,bi,bt) {"function":"nz_truth"}' in new and b'call(ps,px,pi,pt) {"function":"nz_truth"}' in new
sp=lambda b:b.split(b'block spill(',1)[1]
assert sp(old)==sp(new).replace(b',stag:u64)',b')').replace(b',sj2,stag)',b',sj2)')
m['name']='opt_in_exact_nonzero_rank2_public_dtypes'
m.pop('expected_native_sha256',None);unit.pop('expected_object_sha256',None)
unit['sha256']=sha(u)
rp=m['runtime']['path'];m['runtime'].update(sha256=sha(c),**{k:C[k] for k in ('functions','blocks','instructions')})
rows.update({p:u,rp:c})
def reader(path,wanted):
 data=rows[path.relative_to(src).as_posix()]
 assert sha(data)==wanted
 return data
b.check_sources(m,src,reader)
out.mkdir();(out/'units').mkdir()
for path,raw in rows.items():(out/path).write_bytes(raw)
(out/'manifest.json').write_bytes(b.json_bytes(m));b.check_sources(m,out,b.read_file)
print(json.dumps(dict(base_git='2b5b2fcd8b7937d5441317da1becf9cbef360a1b',base_tree='e4ca3d95e47728428380adf5c3f4710d54ae62f1',
 base=sha(a),combined=sha(c),unit11=sha(u),manifest=sha(b.json_bytes(m)),unit11_counts={k:U[k] for k in ('functions','blocks','instructions')},
 exact_inverse=True,only_changed_body='nz_rank2',added_bodies=[],other_14_units_exact=True,wire_unchanged=135,
 proven_ssa_call_and_branch_types=True),sort_keys=True))
