fn where_empty_input_canonical(s:mutbuf,node:u64) -> (u64) effects(read,write,trap) entry(start)
block start(s:mutbuf,node:u64)
  d:u64 = call(s,node) {"function":"dptr"}
  rank:u64 = call(s,d) {"function":"getw"}
  four:u64 = const() {"value":4}
  one:u64 = const() {"value":1}
  ok:bool = compare.le(rank,four)
  branch ok next(s,d,rank,one) no()
block next(ns:mutbuf,nd:u64,nr:u64,expected:u64)
  one_n:u64 = const() {"value":1}
  z:u64 = const() {"value":0}
  more:bool = compare.ne(nr,z)
  branch more axis(ns,nd,nr,expected,one_n) yes()
block axis(as:mutbuf,ad:u64,ar:u64,ex:u64,one_a:u64)
  i:u64 = int.sub.wrap(ar,one_a)
  b:u64 = const() {"value":8}
  dims:u64 = const() {"value":96}
  strides:u64 = const() {"value":128}
  ix:u64 = int.mul.wrap(i,b)
  dp0:u64 = int.add.wrap(ad,dims)
  sp0:u64 = int.add.wrap(ad,strides)
  dp:u64 = int.add.wrap(dp0,ix)
  sp:u64 = int.add.wrap(sp0,ix)
  n:u64 = call(as,dp) {"function":"getw"}
  actual:u64 = call(as,sp) {"function":"getw"}
  cap:u64 = const() {"value":262144}
  bounded:bool = compare.le(n,cap)
  same:bool = compare.eq(actual,ex)
  legal:bool = bool.and(bounded,same)
  branch legal extend(as,ad,i,ex,n,one_a) no()
block extend(es:mutbuf,ed:u64,ei:u64,ee:u64,en:u64,eo:u64)
  ez:u64 = const() {"value":0}
  top:bool = compare.eq(ei,ez)
  nonzero:bool = compare.ne(en,ez)
  factor:u64 = select(nonzero,en,eo)
  branch top yes() multiply(es,ed,ei,ee,factor)
block multiply(ms:mutbuf,md:u64,mi:u64,me:u64,mf:u64)
  v:u64 = int.mul.wrap(me,mf)
  jump next(ms,md,mi,v)
block no()
  n0:u64 = const() {"value":0}
  ret(n0)
block yes()
  y1:u64 = const() {"value":1}
  ret(y1)
end
fn where_empty_effective_stride(s:mutbuf,d:u64,r:u64,i:u64,e:u64) -> (u64) effects(read,write,trap) entry(start)
block start(s:mutbuf,d:u64,r:u64,i:u64,e:u64)
  rt:u64 = call(s,d) {"function":"getw"}
  k:u64 = int.add.wrap(rt,i)
  present:bool = compare.ge(k,r)
  branch present aligned(s,d,k,r,e) zero()
block aligned(as:mutbuf,ad:u64,ak:u64,ar:u64,ae:u64)
  ix:u64 = int.sub.wrap(ak,ar)
  eight:u64 = const() {"value":8}
  shape_at:u64 = const() {"value":96}
  stride_at:u64 = const() {"value":128}
  n0:u64 = int.add.wrap(ad,shape_at)
  s0:u64 = int.add.wrap(ad,stride_at)
  off:u64 = int.mul.wrap(ix,eight)
  na:u64 = int.add.wrap(n0,off)
  sa:u64 = int.add.wrap(s0,off)
  n:u64 = call(as,na) {"function":"getw"}
  st:u64 = call(as,sa) {"function":"getw"}
  one:u64 = const() {"value":1}
  z:u64 = const() {"value":0}
  solo:bool = compare.eq(n,one)
  expanded:bool = compare.ne(ae,one)
  zeroed:bool = bool.and(solo,expanded)
  res:u64 = select(zeroed,z,st)
  ret(res)
block zero()
  z0:u64 = const() {"value":0}
  ret(z0)
end
fn where_empty_cmp_operand(s:mutbuf,d:u64,r:u64,a:u64,b:u64,ea:u64,eb:u64) -> (u64) effects(read,write,trap) entry(start)
block start(s:mutbuf,d:u64,r:u64,a:u64,b:u64,ea:u64,eb:u64)
  a_stride:u64 = call(s,d,r,a,ea) {"function":"where_empty_effective_stride"}
  b_stride:u64 = call(s,d,r,b,eb) {"function":"where_empty_effective_stride"}
  z:u64 = const() {"value":0}
  a_zero:bool = compare.eq(a_stride,z)
  b_zero:bool = compare.eq(b_stride,z)
  skipped:bool = bool.or(a_zero,b_zero)
  branch skipped ambiguous() compare_stride(a_stride,b_stride,ea,eb)
block compare_stride(sa:u64,sb:u64,se:u64,sf:u64)
  greater:bool = compare.gt(sa,sb)
  lower:bool = compare.lt(sa,sb)
  tie:bool = compare.eq(sa,sb)
  extent_greater:bool = compare.gt(se,sf)
  tie_swap:bool = bool.and(tie,extent_greater)
  swapped:bool = bool.or(greater,tie_swap)
  zc:u64 = const() {"value":0}
  one:u64 = const() {"value":1}
  two:u64 = const() {"value":2}
  less_result:u64 = select(lower,one,zc)
  answer:u64 = select(swapped,two,less_result)
  ret(answer)
block ambiguous()
  a0:u64 = const() {"value":0}
  ret(a0)
end
fn where_empty_cmp(s:mutbuf,m:u64,x:u64,y:u64,d:u64,r:u64,a:u64,b:u64) -> (u64) effects(read,write,trap) entry(start)
block start(s:mutbuf,m:u64,x:u64,y:u64,d:u64,r:u64,a:u64,b:u64)
  e:u64 = call(s,m,x,y,r,a) {"function":"where_full_extent"}
  f:u64 = call(s,m,x,y,r,b) {"function":"where_full_extent"}
  cm:u64 = call(s,m,r,a,b,e,f) {"function":"where_empty_cmp_operand"}
  z:u64 = const() {"value":0}
  decided:bool = compare.ne(cm,z)
  branch decided answer(cm) values(s,d,r,a,b,e,f)
block values(vs:mutbuf,vd:u64,vr:u64,va:u64,vb:u64,ve:u64,vf:u64)
  cv:u64 = call(vs,vd,vr,va,vb,ve,vf) {"function":"where_empty_cmp_operand"}
  ret(cv)
block answer(c:u64)
  ret(c)
end
fn where_empty_reverse_plan(s:mutbuf,m:u64,x:u64,y:u64,d:u64,r:u64) -> (u64) effects(read,write,trap) entry(shape)
block shape(s:mutbuf,m:u64,x:u64,y:u64,d:u64,r:u64)
  z:u64 = const() {"value":0}
  jump axes(s,m,x,y,d,r,z,z)
block axes(as:mutbuf,am:u64,ax:u64,ay:u64,ad:u64,ar:u64,ai:u64,zeros:u64)
  more:bool = compare.lt(ai,ar)
  branch more proof(as,am,ax,ay,ad,ar,ai,zeros) init(as,am,ax,ay,ad,ar,zeros)
block proof(ps:mutbuf,pm:u64,px:u64,py:u64,pd:u64,pr:u64,pi:u64,pz:u64)
  e:u64 = call(ps,pm,px,py,pr,pi) {"function":"where_full_extent"}
  cap:u64 = const() {"value":262144}
  one_p:u64 = const() {"value":1}
  z_p:u64 = const() {"value":0}
  valid:bool = compare.le(e,cap)
  vacant:bool = compare.eq(e,z_p)
  zero_bit:u64 = select(vacant,one_p,z_p)
  new_zero:u64 = bit.or(pz,zero_bit)
  next_p:u64 = int.add.wrap(pi,one_p)
  branch valid axes(ps,pm,px,py,pd,pr,next_p,new_zero) no()
block init(is:mutbuf,im:u64,ix:u64,iy:u64,id:u64,ir:u64,iz:u64)
  one_i:u64 = const() {"value":1}
  t_i:u64 = const() {"value":2}
  th_i:u64 = const() {"value":3}
  z_i:u64 = const() {"value":0}
  p2:u64 = const() {"value":1}
  p3:u64 = const() {"value":6}
  p4:u64 = const() {"value":27}
  two_rank:bool = compare.eq(ir,t_i)
  three_rank:bool = compare.eq(ir,th_i)
  one_rank:bool = compare.eq(ir,one_i)
  r2:u64 = select(two_rank,p2,p4)
  r3:u64 = select(three_rank,p3,r2)
  initial:u64 = select(one_rank,z_i,r3)
  empty:bool = compare.eq(iz,one_i)
  branch empty outer(is,im,ix,iy,id,ir,one_i,initial,initial) no()
block outer(os:mutbuf,om:u64,ox:u64,oy:u64,od:u64,orr:u64,oi:u64,op:u64,base:u64)
  further:bool = compare.lt(oi,orr)
  branch further inner(os,om,ox,oy,od,orr,oi,oi,oi,op,base) finish(op,base)
block inner(ns:mutbuf,nm:u64,nx:u64,ny:u64,nd:u64,nr:u64,ni:u64,left_cursor:u64,pivot:u64,np:u64,nb:u64)
  zero_n:u64 = const() {"value":0}
  more_n:bool = compare.ne(left_cursor,zero_n)
  branch more_n inspect(ns,nm,nx,ny,nd,nr,ni,left_cursor,pivot,np,nb) next(ns,nm,nx,ny,nd,nr,ni,np,nb)
block inspect(ts:mutbuf,tm:u64,tx:u64,ty:u64,td:u64,tr:u64,ti:u64,lc:u64,tp:u64,bits:u64,tb:u64)
  one_t:u64 = const() {"value":1}
  two_t:u64 = const() {"value":2}
  mask_t:u64 = const() {"value":3}
  j:u64 = int.sub.wrap(lc,one_t)
  shj:u64 = int.mul.wrap(j,two_t)
  shp:u64 = int.mul.wrap(tp,two_t)
  wj:u64 = int.shr.checked(bits,shj)
  wp:u64 = int.shr.checked(bits,shp)
  aj:u64 = bit.and(wj,mask_t)
  ap:u64 = bit.and(wp,mask_t)
  cmp:u64 = call(ts,tm,tx,ty,td,tr,aj,ap) {"function":"where_empty_cmp"}
  swap_it:bool = compare.eq(cmp,two_t)
  stop:bool = compare.eq(cmp,one_t)
  branch swap_it swap(ts,tm,tx,ty,td,tr,ti,j,bits,tb,aj,ap,shj,shp) noswap(ts,tm,tx,ty,td,tr,ti,j,tp,bits,tb,stop)
block swap(ss:mutbuf,sm:u64,sx:u64,sy:u64,sd:u64,sr:u64,si:u64,sj:u64,sp:u64,sb:u64,sa:u64,sf:u64,sh1:u64,sh2:u64)
  delta:u64 = bit.xor(sa,sf)
  v1:u64 = int.shl.checked(delta,sh1)
  v2:u64 = int.shl.checked(delta,sh2)
  b1:u64 = bit.xor(sp,v1)
  result:u64 = bit.xor(b1,v2)
  jump inner(ss,sm,sx,sy,sd,sr,si,sj,sj,result,sb)
block noswap(cs:mutbuf,cm:u64,cx:u64,cy:u64,cd:u64,cr:u64,ci:u64,cj:u64,cp:u64,cw:u64,cb:u64,csorted:bool)
  branch csorted next(cs,cm,cx,cy,cd,cr,ci,cw,cb) inner(cs,cm,cx,cy,cd,cr,ci,cj,cp,cw,cb)
block next(xs:mutbuf,xm:u64,xx:u64,xy:u64,xd:u64,xr:u64,xi:u64,xw:u64,xb:u64)
  one_x:u64 = const() {"value":1}
  next_x:u64 = int.add.wrap(xi,one_x)
  jump outer(xs,xm,xx,xy,xd,xr,next_x,xw,xb)
block finish(fw:u64,fb:u64)
  unchanged:bool = compare.eq(fw,fb)
  one_f:u64 = const() {"value":1}
  z_f:u64 = const() {"value":0}
  passed:u64 = select(unchanged,one_f,z_f)
  ret(passed)
block no()
  n0:u64 = const() {"value":0}
  ret(n0)
end
