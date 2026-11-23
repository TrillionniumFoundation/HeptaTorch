fn ms_error(s:mutbuf,status:u64) -> (u64) effects(read,write,trap) entry(fail)
block fail(s:mutbuf,status:u64)
  r:u64 = call(s,status) {"function":"terror"}
  ret(r)
end
fn ms_tensor(s:mutbuf,x:u64,m:u64) -> (u64) effects(read,write,trap) entry(check)
block check(s:mutbuf,x:u64,m:u64)
  z:u64 = const() {"value":0}
  one:u64 = const() {"value":1}
  four:u64 = const() {"value":4}
  e8:u64 = const() {"value":8}
  xd:u64 = call(s,x) {"function":"dptr"}
  md:u64 = call(s,m) {"function":"dptr"}
  xrank:u64 = call(s,xd) {"function":"getw"}
  mrank:u64 = call(s,md) {"function":"getw"}
  xt:u64 = call(s,x) {"function":"tensor_dtype"}
  mt:u64 = call(s,m) {"function":"tensor_dtype"}
  xlo:bool = compare.ge(xrank,one)
  xhi:bool = compare.le(xrank,four)
  r_same:bool = compare.eq(xrank,mrank)
  xt_ok:bool = compare.eq(xt,z)
  mt_ok:bool = compare.eq(mt,one)
  rx:bool = bool.and(xlo,xhi)
  rr:bool = bool.and(rx,r_same)
  rt:bool = bool.and(xt_ok,mt_ok)
  rt_ok:bool = bool.and(rr,rt)
  branch rt_ok shape(s,x,m,xd,md,e8) invalid(s)
block shape(ss:mutbuf,sx:u64,sm:u64,sxd:u64,smd:u64,se:u64)
  snp:u64 = int.add.wrap(sxd,se)
  smp:u64 = int.add.wrap(smd,se)
  sn:u64 = call(ss,snp) {"function":"getw"}
  smn:u64 = call(ss,smp) {"function":"getw"}
  same:u64 = call(ss,sx,sm) {"function":"incoming_same_shape"}
  sz:u64 = const() {"value":0}
  same_ok:bool = compare.ne(same,sz)
  equal_n:bool = compare.eq(sn,smn)
  geometry:bool = bool.and(same_ok,equal_n)
  branch geometry cap(ss,sx,sm,sxd,sn) invalid(ss)
block verify_shape(vgs:mutbuf,vgx:u64,vgm:u64,vgd:u64,vgn:u64)
  vg_z:u64 = const() {"value":0}
  vg_r:u64 = call(vgs,vgd) {"function":"getw"}
  vg_match:u64 = call(vgs,vgd,vg_r,vgn) {"function":"ms_geometry"}
  vg_ok:bool = compare.ne(vg_match,vg_z)
  branch vg_ok owner(vgs,vgx,vgm,vgn) invalid(vgs)
block cap(cs:mutbuf,cx:u64,cm:u64,cd:u64,cn:u64)
  lim:u64 = const() {"value":262144}
  within:bool = compare.le(cn,lim)
  branch within verify_shape(cs,cx,cm,cd,cn) capacity(cs)
block owner(os:mutbuf,ox:u64,om:u64,on:u64)
  ow1:u64 = const() {"value":1}
  ow4:u64 = const() {"value":4}
  oz:u64 = const() {"value":0}
  oxl:u64 = call(os,ox,ow4) {"function":"ms_ordinary"}
  oml:u64 = call(os,om,ow1) {"function":"ms_ordinary"}
  ol:u64 = bit.and(oxl,oml)
  opass:bool = compare.ne(ol,oz)
  branch opass count(os,ox,om,on,oz,oz) invalid(os)
block count(cs0:mutbuf,cx0:u64,cm0:u64,cn0:u64,ci:u64,ck:u64)
  cmore:bool = compare.lt(ci,cn0)
  branch cmore bit(cs0,cx0,cm0,cn0,ci,ck) allocate(cs0,cx0,cm0,cn0,ck)
block bit(bs:mutbuf,bx:u64,bm:u64,bn:u64,bi:u64,bk:u64)
  mask_word:u64 = call(bs,bm,bi) {"function":"tget"}
  bzero:u64 = const() {"value":0}
  bone:u64 = const() {"value":1}
  berr:u64 = call(bs,bzero) {"function":"getw"}
  good:bool = compare.eq(berr,bzero)
  canon:bool = compare.le(mask_word,bone)
  bi_next:u64 = int.add.wrap(bi,bone)
  bk_next:u64 = int.add.wrap(bk,mask_word)
  can_count:bool = bool.and(good,canon)
  branch can_count count(bs,bx,bm,bn,bi_next,bk_next) invalid(bs)
block allocate(as:mutbuf,ax:u64,am:u64,an:u64,ak:u64)
  a0:u64 = const() {"value":0}
  a1:u64 = const() {"value":1}
  ap0:u64 = const() {"value":128}
  ap1:u64 = const() {"value":136}
  ap2:u64 = const() {"value":144}
  ap3:u64 = const() {"value":152}
  keep0:u64 = call(as,ap0) {"function":"getw"}
  keep1:u64 = call(as,ap1) {"function":"getw"}
  keep2:u64 = call(as,ap2) {"function":"getw"}
  keep3:u64 = call(as,ap3) {"function":"getw"}
  ms_used_ack1:u64 = call(as,ap0,ak) {"function":"putw"}
  ms_used_ack2:u64 = call(as,ap1,a0) {"function":"putw"}
  ms_used_ack3:u64 = call(as,ap2,a0) {"function":"putw"}
  ms_used_ack4:u64 = call(as,ap3,a0) {"function":"putw"}
  out:u64 = call(as,a1,ap0,a0,a1,a0) {"function":"typed_descriptor"}
  ms_used_ack5:u64 = call(as,ap0,keep0) {"function":"putw"}
  ms_used_ack6:u64 = call(as,ap1,keep1) {"function":"putw"}
  ms_used_ack7:u64 = call(as,ap2,keep2) {"function":"putw"}
  ms_used_ack8:u64 = call(as,ap3,keep3) {"function":"putw"}
  aerr:u64 = call(as,a0) {"function":"getw"}
  aok:bool = compare.eq(aerr,a0)
  aout:bool = compare.ne(out,a0)
  alive:bool = bool.and(aok,aout)
  branch alive inspect(as,ax,am,an,ak,out) capacity(as)
block inspect(vs:mutbuf,vx:u64,vm:u64,vn:u64,vk:u64,vo:u64)
  okout:u64 = call(vs,vo,vk) {"function":"ms_output"}
  vz:u64 = const() {"value":0}
  vp:bool = compare.ne(okout,vz)
  branch vp ready(vs,vx,vm,vn,vk,vo,vz) invalid(vs)
block ready(rs:mutbuf,rx0:u64,rm:u64,rn:u64,rk:u64,ro:u64,r0:u64)
  no_copy:bool = compare.eq(rk,r0)
  branch no_copy done(ro) copy(rs,rx0,rm,rn,rk,ro,r0,r0)
block copy(ps:mutbuf,px:u64,pm:u64,pn:u64,pk:u64,po:u64,pi:u64,pj:u64)
  more_copy:bool = compare.lt(pi,pn)
  branch more_copy choose(ps,px,pm,pn,pk,po,pi,pj) finish(ps,pk,po,pj)
block choose(hs:mutbuf,hx:u64,hm:u64,hn:u64,hk:u64,ho:u64,hi:u64,hj:u64)
  hzero:u64 = const() {"value":0}
  hone:u64 = const() {"value":1}
  hbit:u64 = call(hs,hm,hi) {"function":"tget"}
  herr:u64 = call(hs,hzero) {"function":"getw"}
  hclean:bool = compare.eq(herr,hzero)
  hcanonical:bool = compare.le(hbit,hone)
  hsafe:bool = bool.and(hclean,hcanonical)
  branch hsafe selected(hs,hx,hm,hn,hk,ho,hi,hj,hbit) invalid(hs)
block selected(js:mutbuf,jx:u64,jm:u64,jn:u64,jk:u64,jo:u64,ji:u64,jj:u64,jbit:u64)
  j0:u64 = const() {"value":0}
  j1:u64 = const() {"value":1}
  ni:u64 = int.add.wrap(ji,j1)
  skip:bool = compare.eq(jbit,j0)
  branch skip copy(js,jx,jm,jn,jk,jo,ni,jj) room(js,jx,jm,jn,jk,jo,ji,jj,ni)
block room(ms:mutbuf,mx:u64,mm:u64,mn:u64,mk:u64,mo:u64,mi:u64,mj:u64,mnext:u64)
  fits:bool = compare.lt(mj,mk)
  branch fits read_value(ms,mx,mm,mn,mk,mo,mi,mj,mnext) invalid(ms)
block read_value(ds:mutbuf,dx:u64,dm:u64,dn:u64,dk:u64,do:u64,di:u64,dj:u64,dnext:u64)
  d0:u64 = const() {"value":0}
  raw:u64 = call(ds,dx,di) {"function":"tget"}
  rd_err:u64 = call(ds,d0) {"function":"getw"}
  rd_ok:bool = compare.eq(rd_err,d0)
  branch rd_ok store(ds,dx,dm,dn,dk,do,dj,dnext,raw) already_failed(ds)
block store(ts:mutbuf,tx:u64,tm:u64,tn:u64,tk:u64,to:u64,tj:u64,tnext:u64,traw:u64)
  t1:u64 = const() {"value":1}
  t4:u64 = const() {"value":4}
  target:u64 = call(ts,to,tj) {"function":"taddr"}
  ms_used_ack9:u64 = call(ts,target,t4,traw) {"function":"typed_store"}
  tj_next:u64 = int.add.wrap(tj,t1)
  jump copy(ts,tx,tm,tn,tk,to,tnext,tj_next)
block finish(fs:mutbuf,fk:u64,fo:u64,fj:u64)
  exact:bool = compare.eq(fk,fj)
  branch exact done(fo) invalid(fs)
block done(result:u64)
  ret(result)
block invalid(es:mutbuf)
  reason:u64 = const() {"value":7}
  rejected:u64 = call(es,reason) {"function":"ms_error"}
  ret(rejected)
block capacity(caps:mutbuf)
  cap_status:u64 = const() {"value":3}
  cap_fail:u64 = call(caps,cap_status) {"function":"ms_error"}
  ret(cap_fail)
block already_failed(efs:mutbuf)
  zero_ret:u64 = const() {"value":0}
  ret(zero_ret)
end
