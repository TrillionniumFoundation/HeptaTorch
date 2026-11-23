"""Reversible exact cf89ed3 rank3 Bool nonzero source-only projection."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
src,checker,out=map(Path,sys.argv[1:]);src=src.resolve(strict=True)
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha(checker.read_bytes())=='9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d'
spec=importlib.util.spec_from_file_location('source_checker',checker)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mraw=(src/'manifest.json').read_bytes()
assert sha(mraw)=='ec4080d5a1ce169f9cd77fa5ee13df1f61dd169e9474b5b72d993ef23a2c0079'
m=json.loads(mraw,object_pairs_hook=b.unique_object)
a,units,_=b.check_sources(m,src,b.read_file)
assert sha(a)=='c76588477d02e69e96fd86c8677430602c0c0f6afae198c07a0e60e906360155'
unit=next(x for x in m['units'] if x['name']=='runtime775_11'); p=unit['path']
rows={row['path']:raw for row,raw,_ in units}
assert sha(rows[p])=='df3047ab845986cdbf9d158ffe7a26ef79b4768f0e4cae312c7295f59df63b3d'
A=b.parse_source(a)
names=['nz_rank_route','nz_rank2','nz_r2_alloc','nz_r2_own_out']
edited={}
def original(n):return A['definitions'][n].decode()
def replace(v,a,z):
 assert v.count(a)==1,a
 return v.replace(a,z,1)
def edits(v,xs):
 for a,z in xs:v=replace(v,a,z)
 return v
v=original('nz_rank_route')
v=replace(v,'  branch second r2(s,x,d) r1(s,x)','''  rthree:u64 = const() {"value":3}
  third_rank:bool = compare.eq(r,rthree)
  shared:bool = bool.or(second,third_rank)
  branch shared r2(s,x,d) r1(s,x)''')
edited['nz_rank_route']=v.encode()
v=original('nz_rank2')
v=replace(v,'fn nz_rank2(s:mutbuf,x:u64,d:u64)','fn nz_rank2(is0:mutbuf,ix0:u64,id0:u64)')
v=replace(v,'block start(s:mutbuf,x:u64,d:u64)\n', '''block start(is0:mutbuf,ix0:u64,id0:u64)
  rk0:u64 = call(is0,id0) {"function":"getw"}
  i3:u64 = const() {"value":3}
  ir3:bool = compare.eq(rk0,i3)
  branch ir3 g3(is0,ix0,id0) g2(is0,ix0,id0)
block g2(s:mutbuf,x:u64,d:u64)
''')
v=replace(v,'block limit(',Path(__file__).with_name('geometry.inc').read_text()+'block limit(')
v=replace(v,'  branch exact limit(gs,gx,gn,gc,gl,gw,gt)','  cols2:u64 = const() {"value":2}\n  branch exact limit(gs,gx,gn,gc,gl,gw,gt,cols2,gc)')
v=edits(v,[
 ('ltag:u64)','ltag:u64,lcols:u64,lplane:u64)'),
 ('owned(ls,lx,ln,lc,lw,ltag)','owned(ls,lx,ln,lc,lw,ltag,lcols,lplane)'),
 ('wtag:u64)','wtag:u64,wcols:u64,wplane:u64)'),
 ('cnt(ws,wx,wn,wc,wz,wz,wtag)','cnt(ws,wx,wn,wc,wz,wz,wtag,wcols,wplane)'),
 ('ct:u64)','ct:u64,ccols:u64,cplane:u64)'),
 ('scan(cs,cx,cn,cc,ci,ck,ct) allocate(cs,cx,cn,cc,ck,ct)','scan(cs,cx,cn,cc,ci,ck,ct,ccols,cplane) allocate(cs,cx,cn,cc,ck,ct,ccols,cplane)'),
 ('bt:u64)','bt:u64,bcols:u64,bplane:u64)'),
 ('cnt(bs,bx,bn,bc,nexti,nextk,bt)','cnt(bs,bx,bn,bc,nexti,nextk,bt,bcols,bplane)'),
 ('alt:u64)','alt:u64,alcols:u64,alplane:u64)'),
 ('  maxk:u64 = const() {"value":131072}', '''  max2:u64 = const() {"value":131072}
  max3:u64 = const() {"value":87381}
  ac3:u64 = const() {"value":3}
  isthree:bool = compare.eq(alcols,ac3)
  maxk:u64 = select(isthree,max3,max2)'''),
 ('fresh(als,alx,aln,alc,alk,alt)','fresh(als,alx,aln,alc,alk,alt,alcols,alplane)'),
 ('ft:u64)','ft:u64,fcols:u64,fplane:u64)'),
 ('call(fs,fk) {"function":"nz_r2_alloc"}','call(fs,fk,fcols) {"function":"nz_r2_alloc"}'),
 ('verify(fs,fx,fnn,fc,fk,fo,ft)','verify(fs,fx,fnn,fc,fk,fo,ft,fcols,fplane)'),
 ('vtag:u64)','vtag:u64,vcols:u64,vplane:u64)'),
 ('call(vs,vx,vo,vk) {"function":"nz_r2_own_out"}','call(vs,vx,vo,vk,vcols) {"function":"nz_r2_own_out"}'),
 ('ready(vs,vx,vn,vc,vk,vo,vz,empty,vtag)','ready(vs,vx,vn,vc,vk,vo,vz,empty,vtag,vcols,vplane)'),
 (',tt:u64)',',tt:u64,tcols:u64,tplane:u64)'),
 ('loop(ts,tx,tn,tc,tk,to,tz,tz,tt)','loop(ts,tx,tn,tc,tk,to,tz,tz,tt,tcols,tplane)'),
 ('yt:u64)','yt:u64,ycols:u64,yplane:u64)'),
 ('take(ys,yx,yn,yc,yk,yo,yi,yj,yt)','take(ys,yx,yn,yc,yk,yo,yi,yj,yt,ycols,yplane)'),
 ('pt:u64)','pt:u64,ptcols:u64,ptplane:u64)'),
 ('write_if(ps,px,pn,pc,pk,po,pi,pj,step,skip,pt)','write_if(ps,px,pn,pc,pk,po,pi,pj,step,skip,pt,ptcols,ptplane)'),
 ('ut:u64)','ut:u64,ucols:u64,uplane:u64)'),
 ('loop(us,ux,un,uc,uk,uo,ui2,uj,ut) room(us,ux,un,uc,uk,uo,ui,uj,ui2,ut)','loop(us,ux,un,uc,uk,uo,ui2,uj,ut,ucols,uplane) room(us,ux,un,uc,uk,uo,ui,uj,ui2,ut,ucols,uplane)'),
 (',mtt:u64)',',mtt:u64,mcols:u64,mplane:u64)'),
 ('  go:bool = bool.and(col_nonzero,jfits)','  plane_nonzero:bool = compare.ne(mplane,zero)\n  axes_nonzero:bool = bool.and(col_nonzero,plane_nonzero)\n  go:bool = bool.and(axes_nonzero,jfits)'),
 ('spill(ms,mx,mn,mc,mk,mo,mi,mj,mi2,mtt)','spill(ms,mx,mn,mc,mk,mo,mi,mj,mi2,mtt,mcols,mplane)'),
])
a0=v.index('block spill('); a1=v.index('block finish(')
v=v[:a0]+Path(__file__).with_name('copy.inc').read_text()+v[a1:]
edited['nz_rank2']=v.encode()
v=edits(original('nz_r2_alloc'),[
 ('fn nz_r2_alloc(s:mutbuf,k:u64)','fn nz_r2_alloc(s:mutbuf,k:u64,oc:u64)'),
 ('block start(s:mutbuf,k:u64)','block start(s:mutbuf,k:u64,oc:u64)'),
 ('call(s,p1,two) {"function":"putw"}','call(s,p1,oc) {"function":"putw"}'),
])
edited['nz_r2_alloc']=v.encode()
v=edits(original('nz_r2_own_out'),[
 ('fn nz_r2_own_out(s:mutbuf,x:u64,o:u64,k:u64)','fn nz_r2_own_out(s:mutbuf,x:u64,o:u64,k:u64,oc:u64)'),
 ('block start(s:mutbuf,x:u64,o:u64,k:u64)','block start(s:mutbuf,x:u64,o:u64,k:u64,oc:u64)'),
 ('shape(s,x,o,k,w)','shape(s,x,o,k,w,oc)'),
 ('sw:u64)','sw:u64,scount:u64)'),
 ('int.mul.wrap(sk,two)','int.mul.wrap(sk,scount)'),
 ('compare.eq(e1,two)','compare.eq(e1,scount)'),
 ('compare.eq(sg0,two)','compare.eq(sg0,scount)'),
])
edited['nz_r2_own_out']=v.encode()
def change(v,inverse=False):
 for n in names:
  x,y=original(n).encode(),edited[n]
  if inverse:x,y=y,x
  assert v.count(x)==1,n
  v=v.replace(x,y,1)
 return v
c=change(a);u=change(rows[p]);assert change(c,True)==a and change(u,True)==rows[p]
C=b.parse_source(c);U=b.parse_source(u,unit=True)
assert C['definitions'].keys()==A['definitions'].keys()
assert {f for f in A['definitions'] if A['definitions'][f]!=C['definitions'][f]}==set(names)
assert {f for f in A['signatures'] if A['signatures'][f]!=C['signatures'][f]}=={'nz_r2_alloc','nz_r2_own_out'}
def decl(v):
 return tuple(tuple(x.split(b':',1)) for x in v.split(b',') if x)
def args(v):
 return tuple(x for x in v.split(b',') if x)
for f in [n.encode() for n in names]:
 body=C['definitions'][f.decode()]
 targets={x[1]:decl(x[2]) for x in re.finditer(rb'^block (\w+)\(([^)]*)\)',body,re.M)}
 # Pinned native parser requires identical value-name IDs in the entry.
 head=body.splitlines()[0]
 sig=re.fullmatch(rb'fn (\w+)\(([^)]*)\) -> \([^)]*\) effects\([^)]*\) entry\((\w+)\)',head)
 assert sig and sig[1]==f and decl(sig[2])==targets[sig[3]],(f,head,targets)
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

assert all(b'_ = call(' not in v for v in edited.values())
assert edited['nz_rank2'].count(b'{"function":"nz_truth"}')==2
assert edited['nz_rank2'].count(b'{"function":"typed_store"}')==3
assert b'branch te layout(ts,to,tk)' in edited['nz_rank2']
assert b'int.add.wrap(thaddr,thk)' in edited['nz_rank2']
m['name']='opt_in_exact_nonzero_rank3_bool'
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
print(json.dumps(dict(base_git='cf89ed3b9969ca2f548738b211e7b756ae2e0913',base_tree='c028424ac03c7e4900a450042efc8a4edfb5263b',
 base=sha(a),combined=sha(c),unit11=sha(u),manifest=sha(b.json_bytes(m)),unit11_counts={k:U[k] for k in ('functions','blocks','instructions')},
 exact_inverse=True,changed_bodies=names,added_bodies=[],other_14_units_exact=True,wire_unchanged=135,
 proven_ssa_call_and_branch_types=True),sort_keys=True))
