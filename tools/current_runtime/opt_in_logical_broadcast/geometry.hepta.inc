fn la_bc_aligned(s:mutbuf,d:u64,r:u64,i:u64) -> (u64) effects(read,write,trap) entry(lbf_start)
block lbf_start(s:mutbuf,d:u64,r:u64,i:u64)
  dr:u64 = call(s,d) {"function":"getw"}
  ri:u64 = int.add.wrap(dr,i)
  present:bool = compare.ge(ri,r)
  branch present lbf_axis(s,d,ri,r) lbf_unit()
block lbf_axis(as0:mutbuf,ad:u64,ari:u64,ar:u64)
  slot:u64 = int.sub.wrap(ari,ar)
  qword:u64 = const() {"value":8}
  shapes:u64 = const() {"value":96}
  base:u64 = int.add.wrap(ad,shapes)
  delta:u64 = int.mul.wrap(slot,qword)
  p:u64 = int.add.wrap(base,delta)
  dim:u64 = call(as0,p) {"function":"getw"}
  ret(dim)
block lbf_unit()
  one:u64 = const() {"value":1}
  ret(one)
end
fn la_bc_join(a:u64,b:u64) -> (u64) effects(read,write,trap) entry(lbf_start)
block lbf_start(a:u64,b:u64)
  one:u64 = const() {"value":1}
  ba:bool = compare.eq(b,a)
  bo:bool = compare.eq(b,one)
  keep:bool = bool.or(ba,bo)
  branch keep lbf_ans(a) lbf_other(a,b)
block lbf_other(oa:u64,ob:u64)
  uno:u64 = const() {"value":1}
  sentinel:u64 = const() {"value":0}
  ok:bool = compare.eq(oa,uno)
  v:u64 = select(ok,ob,sentinel)
  ret(v)
block lbf_ans(result:u64)
  ret(result)
end
fn la_bc_product(a:u64,b:u64) -> (u64) effects(read,write,trap) entry(lbf_start)
block lbf_start(a:u64,b:u64)
  maxp:u64 = const() {"value":262145}
  mul:u64 = int.mul.wrap(a,b)
  large:bool = compare.gt(mul,maxp)
  res:u64 = select(large,maxp,mul)
  ret(res)
end
fn la_bc_alloc(s:mutbuf,r:u64,t:u64,w:u64,n:u64,d0:u64,d1:u64,d2:u64,d3:u64) -> (u64) effects(read,write,trap) entry(lbf_stage)
block lbf_stage(s:mutbuf,r:u64,t:u64,w:u64,n:u64,d0:u64,d1:u64,d2:u64,d3:u64)
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
  branch made lbf_verify(s,r,t,w,n,d0,d1,d2,d3,out) lbf_no()
block lbf_verify(vs:mutbuf,vr:u64,vt:u64,vw:u64,vn:u64,v0:u64,v1:u64,v2:u64,v3:u64,vo:u64)
  layout:u64 = call(vs,vo,vw) {"function":"la_ordinary"}
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
  branch ok lbf_dims(vs,des,vr,v0,v1,v2,v3,vo) lbf_bad(vs)
block lbf_dims(ds:mutbuf,dd:u64,dr:u64,da:u64,db:u64,dc:u64,df:u64,dy:u64)
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
  branch all lbf_canonical_check(ds,dd,dr,dy) lbf_bad(ds)
block lbf_canonical_check(cc_s:mutbuf,cc_d:u64,cc_rank:u64,cc_out:u64)
  cc_eight:u64 = const() {"value":8}
  cc_at:u64 = int.add.wrap(cc_d,cc_eight)
  cc_n:u64 = call(cc_s,cc_at) {"function":"getw"}
  cc_lay:u64 = call(cc_s,cc_d,cc_rank,cc_n) {"function":"la_canonical"}
  cc_zero:u64 = const() {"value":0}
  cc_ok:bool = compare.ne(cc_lay,cc_zero)
  branch cc_ok lbf_ok_result(cc_out) lbf_bad(cc_s)
block lbf_ok_result(result:u64)
  ret(result)
block lbf_no()
  nil:u64 = const() {"value":0}
  ret(nil)
block lbf_bad(bs:mutbuf)
  unsupported:u64 = const() {"value":7}
  fail:u64 = call(bs,unsupported) {"function":"terror"}
  ret(fail)
end
fn la_bc_extent(be_s:mutbuf,be_x:u64,be_y:u64,be_rank:u64,be_i:u64) -> (u64) effects(read,write,trap) entry(lbf_be_begin)
block lbf_be_begin(be_s:mutbuf,be_x:u64,be_y:u64,be_rank:u64,be_i:u64)
  be_active:bool = compare.lt(be_i,be_rank)
  branch be_active lbf_be_join(be_s,be_x,be_y,be_rank,be_i) lbf_be_pad()
block lbf_be_join(bj_s:mutbuf,bj_x:u64,bj_y:u64,bj_rank:u64,bj_i:u64)
  bj_l:u64 = call(bj_s,bj_x,bj_rank,bj_i) {"function":"la_bc_aligned"}
  bj_r:u64 = call(bj_s,bj_y,bj_rank,bj_i) {"function":"la_bc_aligned"}
  bj_d:u64 = call(bj_l,bj_r) {"function":"la_bc_join"}
  ret(bj_d)
block lbf_be_pad()
  bp_one:u64 = const() {"value":1}
  ret(bp_one)
end
fn la_bc_plan(bl_s:mutbuf,bl_xd:u64,bl_yd:u64,bl_xr:u64,bl_yr:u64) -> (u64) effects(read,write,trap) entry(lbf_bl_plan)
block lbf_bl_plan(bl_s:mutbuf,bl_xd:u64,bl_yd:u64,bl_xr:u64,bl_yr:u64)
  bl_left_large:bool = compare.gt(bl_xr,bl_yr)
  bl_rank:u64 = select(bl_left_large,bl_xr,bl_yr)
  bl_zero:u64 = const() {"value":0}
  bl_one:u64 = const() {"value":1}
  bl_two:u64 = const() {"value":2}
  bl_three:u64 = const() {"value":3}
  bl_d0:u64 = call(bl_s,bl_xd,bl_yd,bl_rank,bl_zero) {"function":"la_bc_extent"}
  bl_d1:u64 = call(bl_s,bl_xd,bl_yd,bl_rank,bl_one) {"function":"la_bc_extent"}
  bl_d2:u64 = call(bl_s,bl_xd,bl_yd,bl_rank,bl_two) {"function":"la_bc_extent"}
  bl_d3:u64 = call(bl_s,bl_xd,bl_yd,bl_rank,bl_three) {"function":"la_bc_extent"}
  bl_k0:bool = compare.gt(bl_d0,bl_zero)
  bl_k1:bool = compare.gt(bl_d1,bl_zero)
  bl_k2:bool = compare.gt(bl_d2,bl_zero)
  bl_k3:bool = compare.gt(bl_d3,bl_zero)
  bl_p01:bool = bool.and(bl_k0,bl_k1)
  bl_p23:bool = bool.and(bl_k2,bl_k3)
  bl_ok:bool = bool.and(bl_p01,bl_p23)
  branch bl_ok lbf_bl_cap(bl_s,bl_rank,bl_d0,bl_d1,bl_d2,bl_d3) lbf_bl_bad(bl_s)
block lbf_bl_cap(bc_s:mutbuf,bc_rank:u64,bc_d0:u64,bc_d1:u64,bc_d2:u64,bc_d3:u64)
  bc_p01:u64 = call(bc_d0,bc_d1) {"function":"la_bc_product"}
  bc_p23:u64 = call(bc_d2,bc_d3) {"function":"la_bc_product"}
  bc_n:u64 = call(bc_p01,bc_p23) {"function":"la_bc_product"}
  bc_limit:u64 = const() {"value":262144}
  bc_ok:bool = compare.le(bc_n,bc_limit)
  branch bc_ok lbf_bl_make(bc_s,bc_rank,bc_d0,bc_d1,bc_d2,bc_d3,bc_n) lbf_bl_too_large(bc_s)
block lbf_bl_make(bm_s:mutbuf,bm_rank:u64,bm_d0:u64,bm_d1:u64,bm_d2:u64,bm_d3:u64,bm_n:u64)
  bm_one:u64 = const() {"value":1}
  bm_o:u64 = call(bm_s,bm_rank,bm_one,bm_one,bm_n,bm_d0,bm_d1,bm_d2,bm_d3) {"function":"la_bc_alloc"}
  ret(bm_o)
block lbf_bl_bad(bb_s:mutbuf)
  bb_code:u64 = const() {"value":7}
  bb_ret:u64 = call(bb_s,bb_code) {"function":"la_error"}
  ret(bb_ret)
block lbf_bl_too_large(bt_s:mutbuf)
  bt_code:u64 = const() {"value":3}
  bt_ret:u64 = call(bt_s,bt_code) {"function":"la_error"}
  ret(bt_ret)
end
