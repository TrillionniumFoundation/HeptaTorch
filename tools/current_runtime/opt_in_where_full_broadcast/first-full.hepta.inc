fn where_full_aligned(s:mutbuf,d:u64,r:u64,i:u64) -> (u64) effects(read,write,trap) entry(start)
block start(s:mutbuf,d:u64,r:u64,i:u64)
  dr:u64 = call(s,d) {"function":"getw"}
  ri:u64 = int.add.wrap(dr,i)
  present:bool = compare.ge(ri,r)
  branch present axis(s,d,ri,r) unit()
block axis(as0:mutbuf,ad:u64,ari:u64,ar:u64)
  slot:u64 = int.sub.wrap(ari,ar)
  qword:u64 = const() {"value":8}
  shapes:u64 = const() {"value":96}
  base:u64 = int.add.wrap(ad,shapes)
  delta:u64 = int.mul.wrap(slot,qword)
  p:u64 = int.add.wrap(base,delta)
  dim:u64 = call(as0,p) {"function":"getw"}
  ret(dim)
block unit()
  one:u64 = const() {"value":1}
  ret(one)
end
fn where_full_join(a:u64,b:u64) -> (u64) effects(read,write,trap) entry(start)
block start(a:u64,b:u64)
  one:u64 = const() {"value":1}
  ba:bool = compare.eq(b,a)
  bo:bool = compare.eq(b,one)
  keep:bool = bool.or(ba,bo)
  branch keep ans(a) other(a,b)
block other(oa:u64,ob:u64)
  uno:u64 = const() {"value":1}
  sentinel:u64 = const() {"value":262145}
  ok:bool = compare.eq(oa,uno)
  v:u64 = select(ok,ob,sentinel)
  ret(v)
block ans(result:u64)
  ret(result)
end
fn where_full_extent(s:mutbuf,md:u64,xd:u64,yd:u64,r:u64,i:u64) -> (u64) effects(read,write,trap) entry(check)
block check(s:mutbuf,md:u64,xd:u64,yd:u64,r:u64,i:u64)
  active:bool = compare.lt(i,r)
  branch active axes(s,md,xd,yd,r,i) pad()
block axes(as0:mutbuf,am:u64,ax:u64,ay:u64,ar:u64,ai:u64)
  ma:u64 = call(as0,am,ar,ai) {"function":"where_full_aligned"}
  xa:u64 = call(as0,ax,ar,ai) {"function":"where_full_aligned"}
  ya:u64 = call(as0,ay,ar,ai) {"function":"where_full_aligned"}
  j1:u64 = call(ma,xa) {"function":"where_full_join"}
  j2:u64 = call(j1,ya) {"function":"where_full_join"}
  ret(j2)
block pad()
  unit:u64 = const() {"value":1}
  ret(unit)
end
fn where_full_product(a:u64,b:u64) -> (u64) effects(read,write,trap) entry(start)
block start(a:u64,b:u64)
  maxp:u64 = const() {"value":262145}
  mul:u64 = int.mul.wrap(a,b)
  large:bool = compare.gt(mul,maxp)
  res:u64 = select(large,maxp,mul)
  ret(res)
end
fn where_full_alloc(s:mutbuf,r:u64,t:u64,w:u64,n:u64,d0:u64,d1:u64,d2:u64,d3:u64) -> (u64) effects(read,write,trap) entry(stage)
block stage(s:mutbuf,r:u64,t:u64,w:u64,n:u64,d0:u64,d1:u64,d2:u64,d3:u64)
  p0:u64 = const() {"value":128}
  p1:u64 = const() {"value":136}
  p2:u64 = const() {"value":144}
  p3:u64 = const() {"value":152}
  old0:u64 = call(s,p0) {"function":"getw"}
  old1:u64 = call(s,p1) {"function":"getw"}
  old2:u64 = call(s,p2) {"function":"getw"}
  old3:u64 = call(s,p3) {"function":"getw"}
  staged0:u64 = call(s,p0,d0) {"function":"putw"}
  staged1:u64 = call(s,p1,d1) {"function":"putw"}
  staged2:u64 = call(s,p2,d2) {"function":"putw"}
  staged3:u64 = call(s,p3,d3) {"function":"putw"}
  z:u64 = const() {"value":0}
  one:u64 = const() {"value":1}
  out:u64 = call(s,r,p0,z,one,t) {"function":"typed_descriptor"}
  restore0:u64 = call(s,p0,old0) {"function":"putw"}
  restore1:u64 = call(s,p1,old1) {"function":"putw"}
  restore2:u64 = call(s,p2,old2) {"function":"putw"}
  restore3:u64 = call(s,p3,old3) {"function":"putw"}
  made:bool = compare.ne(out,z)
  branch made verify(s,r,t,w,n,d0,d1,d2,d3,out) no()
block verify(vs:mutbuf,vr:u64,vt:u64,vw:u64,vn:u64,v0:u64,v1:u64,v2:u64,v3:u64,vo:u64)
  layout:u64 = call(vs,vo,vw) {"function":"where_ordinary"}
  des:u64 = call(vs,vo) {"function":"dptr"}
  dtype:u64 = call(vs,vo) {"function":"tensor_dtype"}
  rank:u64 = call(vs,des) {"function":"getw"}
  at_count:u64 = const() {"value":8}
  countptr:u64 = int.add.wrap(des,at_count)
  count:u64 = call(vs,countptr) {"function":"getw"}
  zero_v:u64 = const() {"value":0}
  dtype_ok:bool = compare.eq(dtype,vt)
  rank_ok:bool = compare.eq(rank,vr)
  count_ok:bool = compare.eq(count,vn)
  owner_ok:bool = compare.ne(layout,zero_v)
  first_ok:bool = bool.and(dtype_ok,rank_ok)
  second_ok:bool = bool.and(count_ok,owner_ok)
  ok:bool = bool.and(first_ok,second_ok)
  branch ok dims(vs,des,vr,v0,v1,v2,v3,vo) bad(vs)
block dims(ds:mutbuf,dd:u64,dr:u64,da:u64,db:u64,dc:u64,df:u64,dy:u64)
  off0:u64 = const() {"value":96}
  off1:u64 = const() {"value":104}
  off2:u64 = const() {"value":112}
  off3:u64 = const() {"value":120}
  dp0:u64 = int.add.wrap(dd,off0)
  dp1:u64 = int.add.wrap(dd,off1)
  dp2:u64 = int.add.wrap(dd,off2)
  dp3:u64 = int.add.wrap(dd,off3)
  got0:u64 = call(ds,dp0) {"function":"getw"}
  got1:u64 = call(ds,dp1) {"function":"getw"}
  got2:u64 = call(ds,dp2) {"function":"getw"}
  got3:u64 = call(ds,dp3) {"function":"getw"}
  zero_d:u64 = const() {"value":0}
  one_d:u64 = const() {"value":1}
  two_d:u64 = const() {"value":2}
  three_d:u64 = const() {"value":3}
  unused0:bool = compare.le(dr,zero_d)
  unused1:bool = compare.le(dr,one_d)
  unused2:bool = compare.le(dr,two_d)
  unused3:bool = compare.le(dr,three_d)
  match0:bool = compare.eq(got0,da)
  match1:bool = compare.eq(got1,db)
  match2:bool = compare.eq(got2,dc)
  match3:bool = compare.eq(got3,df)
  good0:bool = bool.or(unused0,match0)
  good1:bool = bool.or(unused1,match1)
  good2:bool = bool.or(unused2,match2)
  good3:bool = bool.or(unused3,match3)
  ga:bool = bool.and(good0,good1)
  gb:bool = bool.and(good2,good3)
  all:bool = bool.and(ga,gb)
  branch all ok_result(dy) bad(ds)
block ok_result(result:u64)
  ret(result)
block no()
  nil:u64 = const() {"value":0}
  ret(nil)
block bad(bs:mutbuf)
  unsupported:u64 = const() {"value":7}
  fail:u64 = call(bs,unsupported) {"function":"terror"}
  ret(fail)
end
fn where_full_where(s:mutbuf,m:u64,x:u64,y:u64,t:u64,w:u64) -> (u64) effects(read,write,trap) entry(plan)
block plan(s:mutbuf,m:u64,x:u64,y:u64,t:u64,w:u64)
  md:u64 = call(s,m) {"function":"dptr"}
  xd:u64 = call(s,x) {"function":"dptr"}
  yd:u64 = call(s,y) {"function":"dptr"}
  mr:u64 = call(s,md) {"function":"getw"}
  xr:u64 = call(s,xd) {"function":"getw"}
  yr:u64 = call(s,yd) {"function":"getw"}
  x_larger:bool = compare.gt(xr,yr)
  val_rank:u64 = select(x_larger,xr,yr)
  mask_larger:bool = compare.gt(mr,val_rank)
  rank:u64 = select(mask_larger,mr,val_rank)
  four:u64 = const() {"value":4}
  rank_ok:bool = compare.le(rank,four)
  branch rank_ok prove(s,m,x,y,t,w,md,xd,yd,rank) bad(s)
block prove(ps:mutbuf,pm:u64,px:u64,py:u64,pt:u64,pw:u64,pmd:u64,pxd:u64,pyd:u64,pr:u64)
  i0:u64 = const() {"value":0}
  i1:u64 = const() {"value":1}
  i2:u64 = const() {"value":2}
  i3:u64 = const() {"value":3}
  cap:u64 = const() {"value":262144}
  d0:u64 = call(ps,pmd,pxd,pyd,pr,i0) {"function":"where_full_extent"}
  d1:u64 = call(ps,pmd,pxd,pyd,pr,i1) {"function":"where_full_extent"}
  d2:u64 = call(ps,pmd,pxd,pyd,pr,i2) {"function":"where_full_extent"}
  d3:u64 = call(ps,pmd,pxd,pyd,pr,i3) {"function":"where_full_extent"}
  c0:bool = compare.le(d0,cap)
  c1:bool = compare.le(d1,cap)
  c2:bool = compare.le(d2,cap)
  c3:bool = compare.le(d3,cap)
  ca:bool = bool.and(c0,c1)
  cb:bool = bool.and(c2,c3)
  c_all:bool = bool.and(ca,cb)
  branch c_all capacity(ps,pm,px,py,pt,pw,pr,d0,d1,d2,d3) bad(ps)
block capacity(cs:mutbuf,cm:u64,cx:u64,cy:u64,ct:u64,cw:u64,cr:u64,c0:u64,c1:u64,c2:u64,c3:u64)
  a:u64 = call(c0,c1) {"function":"where_full_product"}
  b:u64 = call(a,c2) {"function":"where_full_product"}
  n:u64 = call(b,c3) {"function":"where_full_product"}
  bound:u64 = const() {"value":262144}
  supported:bool = compare.le(n,bound)
  branch supported allocate(cs,cm,cx,cy,ct,cw,cr,c0,c1,c2,c3,n) over_cap(cs)
block allocate(as0:mutbuf,am:u64,ax:u64,ay:u64,at:u64,aw:u64,ar:u64,a0:u64,a1:u64,a2:u64,a3:u64,an:u64)
  out:u64 = call(as0,ar,at,aw,an,a0,a1,a2,a3) {"function":"where_full_alloc"}
  z:u64 = const() {"value":0}
  ok:bool = compare.ne(out,z)
  branch ok loop(as0,am,ax,ay,out,aw,an,z) zero_out()
block loop(ls:mutbuf,lm:u64,lx:u64,ly:u64,lo:u64,lw:u64,ln:u64,li:u64)
  more:bool = compare.lt(li,ln)
  branch more maskpos(ls,lm,lx,ly,lo,lw,ln,li) done(lo)
block maskpos(ms:mutbuf,mm:u64,mx:u64,my:u64,mo:u64,mw:u64,mn:u64,mi:u64)
  p:u64 = call(ms,mm,mo,mi) {"function":"where_general_index"}
  invalid:u64 = const() {"value":262145}
  p_ok:bool = compare.lt(p,invalid)
  branch p_ok maskword(ms,mm,mx,my,mo,mw,mn,mi,p) bad(ms)
block maskword(ws:mutbuf,wm:u64,wx:u64,wy:u64,wo:u64,ww:u64,wn:u64,wi:u64,wp:u64)
  raw:u64 = call(ws,wm,wp) {"function":"tget"}
  zero_m:u64 = const() {"value":0}
  one_m:u64 = const() {"value":1}
  status:u64 = call(ws,zero_m) {"function":"getw"}
  good_read:bool = compare.eq(status,zero_m)
  legal:bool = compare.le(raw,one_m)
  good:bool = bool.and(good_read,legal)
  which:bool = compare.eq(raw,one_m)
  src:u64 = select(which,wx,wy)
  branch good sourcepos(ws,wm,wx,wy,wo,ww,wn,wi,src) bad(ws)
block sourcepos(vs:mutbuf,vm:u64,vx:u64,vy:u64,vo:u64,vw:u64,vn:u64,vi:u64,src0:u64)
  pos:u64 = call(vs,src0,vo,vi) {"function":"where_general_index"}
  invalid_pos:u64 = const() {"value":262145}
  pos_ok:bool = compare.lt(pos,invalid_pos)
  branch pos_ok value(vs,vm,vx,vy,vo,vw,vn,vi,src0,pos) bad(vs)
block value(rs:mutbuf,rm:u64,rx:u64,ry:u64,ro:u64,rw:u64,rn:u64,ri:u64,rsource:u64,rp:u64)
  word:u64 = call(rs,rsource,rp) {"function":"tget"}
  zero_r:u64 = const() {"value":0}
  rstatus:u64 = call(rs,zero_r) {"function":"getw"}
  read_ok:bool = compare.eq(rstatus,zero_r)
  branch read_ok store(rs,rm,rx,ry,ro,rw,rn,ri,word) bad(rs)
block store(ss:mutbuf,sm:u64,sx:u64,sy:u64,so:u64,byte_width:u64,sn:u64,si:u64,bits:u64)
  addr:u64 = call(ss,so,si) {"function":"taddr"}
  stored:u64 = call(ss,addr,byte_width,bits) {"function":"typed_store"}
  one_s:u64 = const() {"value":1}
  store_ok:bool = compare.eq(stored,one_s)
  next:u64 = int.add.wrap(si,one_s)
  branch store_ok loop(ss,sm,sx,sy,so,byte_width,sn,next) bad(ss)
block over_cap(os:mutbuf)
  code:u64 = const() {"value":3}
  over:u64 = call(os,code) {"function":"terror"}
  ret(over)
block bad(bs:mutbuf)
  unsupported:u64 = const() {"value":7}
  fail:u64 = call(bs,unsupported) {"function":"terror"}
  ret(fail)
block zero_out()
  no_value:u64 = const() {"value":0}
  ret(no_value)
block done(d:u64)
  ret(d)
end
