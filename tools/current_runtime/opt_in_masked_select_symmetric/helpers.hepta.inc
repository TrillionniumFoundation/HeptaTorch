fn ms_align(s:mutbuf,d:u64,r:u64,trail:u64) -> (u64) effects(read,write,trap) entry(ax0)
block ax0(s:mutbuf,d:u64,r:u64,trail:u64)
  present:bool = compare.lt(trail,r)
  branch present axread(s,d,r,trail) axone()
block axread(qs:mutbuf,qd:u64,qr:u64,qi:u64)
  e8:u64 = const() {"value":8}
  e96:u64 = const() {"value":96}
  one:u64 = const() {"value":1}
  sub:u64 = int.sub.wrap(qr,one)
  axis:u64 = int.sub.wrap(sub,qi)
  off:u64 = int.mul.wrap(axis,e8)
  start:u64 = int.add.wrap(qd,e96)
  addr:u64 = int.add.wrap(start,off)
  axis_val:u64 = call(qs,addr) {"function":"getw"}
  ret(axis_val)
block axone()
  val:u64 = const() {"value":1}
  ret(val)
end
fn ms_merge_dim(a:u64,b:u64) -> (u64) effects(read,write,trap) entry(mg0)
block mg0(a:u64,b:u64)
  one:u64 = const() {"value":1}
  fail_dim:u64 = const() {"value":262145}
  same:bool = compare.eq(a,b)
  aone:bool = compare.eq(a,one)
  bone:bool = compare.eq(b,one)
  keep_a:bool = bool.or(same,bone)
  keep_b:u64 = select(aone,b,fail_dim)
  dim:u64 = select(keep_a,a,keep_b)
  ret(dim)
end
fn ms_scount(s:mutbuf,xd:u64,md:u64,xr:u64,mr:u64) -> (u64) effects(read,write,trap) entry(ct0)
block ct0(s:mutbuf,xd:u64,md:u64,xr:u64,mr:u64)
  xbig:bool = compare.gt(xr,mr)
  rankmax:u64 = select(xbig,xr,mr)
  z:u64 = const() {"value":0}
  one:u64 = const() {"value":1}
  jump ctloop(s,xd,md,xr,mr,rankmax,z,one)
block ctloop(ls:mutbuf,lx:u64,lm:u64,lr:u64,lr2:u64,lrmax:u64,li:u64,lp:u64)
  more:bool = compare.lt(li,lrmax)
  branch more ctaxis(ls,lx,lm,lr,lr2,lrmax,li,lp) ctdone(lp)
block ctaxis(as1:mutbuf,ax:u64,am:u64,ar:u64,ar2:u64,armax:u64,ai:u64,ap:u64)
  one2:u64 = const() {"value":1}
  last:u64 = int.sub.wrap(armax,one2)
  ct_trail:u64 = int.sub.wrap(last,ai)
  ae:u64 = call(as1,ax,ar,ct_trail) {"function":"ms_align"}
  be:u64 = call(as1,am,ar2,ct_trail) {"function":"ms_align"}
  res:u64 = call(ae,be) {"function":"ms_merge_dim"}
  max:u64 = const() {"value":262144}
  stop:u64 = const() {"value":262145}
  valid_dim:bool = compare.le(res,max)
  prod:u64 = int.mul.wrap(ap,res)
  large:bool = compare.gt(prod,max)
  sat:u64 = select(large,stop,prod)
  step:u64 = int.add.wrap(ai,one2)
  branch valid_dim ctloop(as1,ax,am,ar,ar2,armax,step,sat) ctfail()
block ctdone(cp:u64)
  one3:u64 = const() {"value":1}
  enc:u64 = int.add.wrap(cp,one3)
  ret(enc)
block ctfail()
  fail:u64 = const() {"value":0}
  ret(fail)
end
fn ms_sindex(s:mutbuf,a:u64,b:u64,i:u64) -> (u64) effects(read,write,trap) entry(si0)
block si0(s:mutbuf,a:u64,b:u64,i:u64)
  ad:u64 = call(s,a) {"function":"dptr"}
  bd:u64 = call(s,b) {"function":"dptr"}
  ar:u64 = call(s,ad) {"function":"getw"}
  br:u64 = call(s,bd) {"function":"getw"}
  wider:bool = compare.gt(ar,br)
  rankmax:u64 = select(wider,ar,br)
  eight:u64 = const() {"value":8}
  zero:u64 = const() {"value":0}
  one:u64 = const() {"value":1}
  ap:u64 = int.add.wrap(ad,eight)
  ac:u64 = call(s,ap) {"function":"getw"}
  jump silo(s,ad,bd,ar,br,rankmax,i,zero,zero,one,ac)
block silo(ls:mutbuf,la:u64,lb:u64,lr:u64,lr2:u64,lmax:u64,lq:u64,li:u64,lj:u64,lf:u64,lc:u64)
  more:bool = compare.lt(li,lmax)
  branch more siaxis(ls,la,lb,lr,lr2,lmax,lq,li,lj,lf,lc) sidone(lj,lc)
block siaxis(as2:mutbuf,aa:u64,ab:u64,ar1:u64,ar2:u64,amax:u64,aq:u64,ai:u64,aj:u64,af:u64,acount:u64)
  ade:u64 = call(as2,aa,ar1,ai) {"function":"ms_align"}
  bde:u64 = call(as2,ab,ar2,ai) {"function":"ms_align"}
  total:u64 = call(ade,bde) {"function":"ms_merge_dim"}
  a0:u64 = const() {"value":0}
  allcap:u64 = const() {"value":262144}
  gt0:bool = compare.gt(total,a0)
  in_cap:bool = compare.le(total,allcap)
  valid_axis:bool = bool.and(gt0,in_cap)
  branch valid_axis sidigit(as2,aa,ab,ar1,ar2,amax,aq,ai,aj,af,acount,ade,total) sibad()
block sidigit(ds:mutbuf,da:u64,db:u64,dr:u64,dr2:u64,dmax:u64,dq:u64,di:u64,dj:u64,df:u64,dc:u64,dd:u64,span:u64)
  d0:u64 = const() {"value":0}
  d1:u64 = const() {"value":1}
  coord:u64 = int.rem.checked(dq,span)
  rest:u64 = int.div.checked(dq,span)
  single:bool = compare.eq(dd,d1)
  mapped:u64 = select(single,d0,coord)
  plus:u64 = int.mul.wrap(mapped,df)
  accum:u64 = int.add.wrap(dj,plus)
  srcstep:u64 = int.mul.wrap(df,dd)
  next:u64 = int.add.wrap(di,d1)
  jump silo(ds,da,db,dr,dr2,dmax,rest,next,accum,srcstep,dc)
block sidone(j:u64,c:u64)
  bad_idx:u64 = const() {"value":262145}
  valid_idx:bool = compare.lt(j,c)
  chosen:u64 = select(valid_idx,j,bad_idx)
  ret(chosen)
block sibad()
  missing:u64 = const() {"value":262145}
  ret(missing)
end
fn ms_mask_index(s:mutbuf,x:u64,m:u64,i:u64,mp:u64) -> (u64) effects(read,write,trap) entry(mi0)
block mi0(s:mutbuf,x:u64,m:u64,i:u64,mp:u64)
  two:u64 = const() {"value":2}
  both:bool = compare.eq(mp,two)
  branch both mipair(s,m,x,i) miold(s,x,m,i,mp)
block mipair(ps:mutbuf,pm:u64,px:u64,pi:u64)
  val:u64 = call(ps,pm,px,pi) {"function":"ms_sindex"}
  ret(val)
block miold(os:mutbuf,ox:u64,om:u64,oi:u64,op:u64)
  oldval:u64 = call(os,ox,om,oi,op) {"function":"ms_bindex"}
  ret(oldval)
end
fn ms_value_index(s:mutbuf,x:u64,m:u64,i:u64,mp:u64) -> (u64) effects(read,write,trap) entry(vi0)
block vi0(s:mutbuf,x:u64,m:u64,i:u64,mp:u64)
  two:u64 = const() {"value":2}
  both:bool = compare.eq(mp,two)
  branch both vipair(s,x,m,i) viold(i)
block vipair(ps:mutbuf,px:u64,pm:u64,pi:u64)
  val:u64 = call(ps,px,pm,pi) {"function":"ms_sindex"}
  ret(val)
block viold(oi:u64)
  ret(oi)
end
