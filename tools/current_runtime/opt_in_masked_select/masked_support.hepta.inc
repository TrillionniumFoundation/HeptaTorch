fn ms_geometry(s:mutbuf,d:u64,r:u64,n:u64) -> (u64) effects(read,write,trap) entry(start)
block start(s:mutbuf,d:u64,r:u64,n:u64)
  g0:u64 = const() {"value":0}
  g1:u64 = const() {"value":1}
  jump axes(s,d,r,n,g0,g1)
block axes(as:mutbuf,ad:u64,ar:u64,an:u64,ai:u64,ap:u64)
  more:bool = compare.lt(ai,ar)
  branch more extent(as,ad,ar,an,ai,ap) finish(an,ap)
block extent(es:mutbuf,ed:u64,er:u64,en:u64,ei:u64,ep:u64)
  e1:u64 = const() {"value":1}
  e8:u64 = const() {"value":8}
  e96:u64 = const() {"value":96}
  max:u64 = const() {"value":262144}
  capped:u64 = const() {"value":262145}
  p0:u64 = int.mul.wrap(ei,e8)
  p1:u64 = int.add.wrap(ed,e96)
  p2:u64 = int.add.wrap(p0,p1)
  dim:u64 = call(es,p2) {"function":"getw"}
  dim_ok:bool = compare.le(dim,max)
  branch dim_ok product(es,ed,er,en,ei,ep,dim,max,capped,e1) no()
block product(ps:mutbuf,pd:u64,pr:u64,pn:u64,pi:u64,pp:u64,pdim:u64,pmax:u64,pcap:u64,pone:u64)
  num:u64 = int.mul.wrap(pp,pdim)
  exceeds:bool = compare.gt(num,pmax)
  sat:u64 = select(exceeds,pcap,num)
  next:u64 = int.add.wrap(pi,pone)
  jump axes(ps,pd,pr,pn,next,sat)
block finish(final_n:u64,fp:u64)
  same:bool = compare.eq(final_n,fp)
  one:u64 = const() {"value":1}
  zero:u64 = const() {"value":0}
  out:u64 = select(same,one,zero)
  ret(out)
block no()
  bad:u64 = const() {"value":0}
  ret(bad)
end
fn ms_output(s:mutbuf,o:u64,k:u64) -> (u64) effects(read,write,trap) entry(check)
block check(s:mutbuf,o:u64,k:u64)
  four:u64 = const() {"value":4}
  z:u64 = const() {"value":0}
  root:u64 = call(s,o,four) {"function":"ms_ordinary"}
  good:bool = compare.ne(root,z)
  branch good meta(s,o,k) no()
block meta(ms:mutbuf,mo:u64,mk:u64)
  m0:u64 = const() {"value":0}
  m1:u64 = const() {"value":1}
  m4:u64 = const() {"value":4}
  m8:u64 = const() {"value":8}
  m24:u64 = const() {"value":24}
  m216:u64 = const() {"value":216}
  m16:u64 = const() {"value":16}
  m96:u64 = const() {"value":96}
  m128:u64 = const() {"value":128}
  m224:u64 = const() {"value":224}
  d:u64 = call(ms,mo) {"function":"dptr"}
  dt:u64 = call(ms,mo) {"function":"tensor_dtype"}
  r:u64 = call(ms,d) {"function":"getw"}
  np:u64 = int.add.wrap(d,m8)
  dp:u64 = int.add.wrap(d,m96)
  sp:u64 = int.add.wrap(d,m128)
  vp:u64 = int.add.wrap(d,m224)
  stp:u64 = int.add.wrap(d,m216)
  n:u64 = call(ms,np) {"function":"getw"}
  extent:u64 = call(ms,dp) {"function":"getw"}
  element_stride:u64 = call(ms,sp) {"function":"getw"}
  sh:u64 = call(ms,stp) {"function":"getw"}
  sl:u64 = call(ms,sh) {"function":"so_storage_live"}
  slen_at:u64 = int.add.wrap(sl,m24)
  slen:u64 = call(ms,slen_at) {"function":"so_uget"}
  exact_len:u64 = int.mul.wrap(mk,m4)
  len_ok:bool = compare.eq(slen,exact_len)
  vh:u64 = call(ms,vp) {"function":"getw"}
  vl:u64 = call(ms,vh) {"function":"so_version_live"}
  vc_at:u64 = int.add.wrap(vl,m16)
  vc:u64 = call(ms,vc_at) {"function":"so_uget"}
  isf:bool = compare.eq(dt,m0)
  rank:bool = compare.eq(r,m1)
  count:bool = compare.eq(n,mk)
  dim:bool = compare.eq(extent,mk)
  stride:bool = compare.eq(element_stride,m1)
  version:bool = compare.eq(vc,m0)
  a:bool = bool.and(isf,rank)
  b:bool = bool.and(count,dim)
  c:bool = bool.and(stride,version)
  ab:bool = bool.and(a,b)
  abc:bool = bool.and(ab,c)
  ok:bool = bool.and(abc,len_ok)
  retv:u64 = select(ok,m1,m0)
  ret(retv)
block no()
  zero:u64 = const() {"value":0}
  ret(zero)
end
fn ms_command(r:readbuf,o:mutbuf,s:mutbuf,a:u64,op:u64,p:u64) -> (u64) effects(read,write,trap) entry(wire)
block wire(r:readbuf,o:mutbuf,s:mutbuf,a:u64,op:u64,p:u64)
  w8:u64 = const() {"value":8}
  w16:u64 = const() {"value":16}
  w24:u64 = const() {"value":24}
  w56:u64 = const() {"value":56}
  w0:u64 = const() {"value":0}
  xa:u64 = int.add.wrap(a,w8)
  ma:u64 = int.add.wrap(a,w16)
  pa:u64 = int.add.wrap(a,w24)
  pe:u64 = int.add.wrap(a,w56)
  xt:u64 = call(r,xa) {"function":"getr"}
  mt:u64 = call(r,ma) {"function":"getr"}
  jump padding(r,o,s,op,p,xt,mt,pa,pe,w0,w8)
block padding(pr:readbuf,po:mutbuf,ps:mutbuf,pop:u64,pp:u64,px:u64,pm:u64,pa0:u64,pend:u64,pz:u64,pstep:u64)
  more:bool = compare.le(pa0,pend)
  branch more pad_word(pr,po,ps,pop,pp,px,pm,pa0,pend,pz,pstep) resolve_x(po,ps,pop,pp,px,pm)
block pad_word(qr:readbuf,qo:mutbuf,qs:mutbuf,qop:u64,qp:u64,qx:u64,qm:u64,qa:u64,qend:u64,qz:u64,qstep:u64)
  raw:u64 = call(qr,qa) {"function":"getr"}
  zero:bool = compare.eq(raw,qz)
  next:u64 = int.add.wrap(qa,qstep)
  branch zero padding(qr,qo,qs,qop,qp,qx,qm,next,qend,qz,qstep) reject_pad(qo,qs,qop,qp)
block reject_pad(bo:mutbuf,bs:mutbuf,bop:u64,bp:u64)
  b7:u64 = const() {"value":7}
  failure:u64 = call(bs,b7) {"function":"ms_error"}
  jump finish(bo,bs,bop,bp,failure)
block resolve_x(xo:mutbuf,xs:mutbuf,xop:u64,xp:u64,xtok:u64,xmt:u64)
  x0:u64 = const() {"value":0}
  xn:u64 = call(xs,xtok) {"function":"public_node"}
  xpass:bool = compare.ne(xn,x0)
  branch xpass resolve_m(xo,xs,xop,xp,xn,xmt) finish(xo,xs,xop,xp,x0)
block resolve_m(mo:mutbuf,ms:mutbuf,mop:u64,mp:u64,mx:u64,mtok:u64)
  m0:u64 = const() {"value":0}
  mn:u64 = call(ms,mtok) {"function":"public_node"}
  mpass:bool = compare.ne(mn,m0)
  branch mpass execute(mo,ms,mop,mp,mx,mn) finish(mo,ms,mop,mp,m0)
block execute(eo:mutbuf,es:mutbuf,eop:u64,ep:u64,ex:u64,em:u64)
  made:u64 = call(es,ex,em) {"function":"ms_tensor"}
  jump finish(eo,es,eop,ep,made)
block finish(fo:mutbuf,fs:mutbuf,fop:u64,fp:u64,made_node:u64)
  off:u64 = const() {"value":8}
  prior:u64 = call(fs,off) {"function":"getw"}
  written:u64 = call(fs,fo,fp,fop,prior,made_node) {"function":"shape_command_finish"}
  ret(written)
end
