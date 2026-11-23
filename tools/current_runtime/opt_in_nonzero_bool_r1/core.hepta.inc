fn nz_error(s:mutbuf,code:u64) -> (u64) effects(read,write,trap) entry(err0)
block err0(s:mutbuf,code:u64)
  r:u64 = call(s,code) {"function":"terror"}
  ret(r)
end
fn nz_output(s:mutbuf,x:u64,o:u64,k:u64) -> (u64) effects(read,write,trap) entry(out0)
block out0(s:mutbuf,x:u64,o:u64,k:u64)
  z:u64 = const() {"value":0}
  w:u64 = const() {"value":8}
  own:u64 = call(s,o,w) {"function":"nz_ordinary"}
  yes:bool = compare.ne(own,z)
  branch yes out_shape(s,x,o,k,w) out_no()
block out_shape(os:mutbuf,ox:u64,oo:u64,ok:u64,ow:u64)
  z0:u64 = const() {"value":0}
  one:u64 = const() {"value":1}
  two:u64 = const() {"value":2}
  a96:u64 = const() {"value":96}
  a104:u64 = const() {"value":104}
  a128:u64 = const() {"value":128}
  a136:u64 = const() {"value":136}
  d:u64 = call(os,oo) {"function":"dptr"}
  dt:u64 = call(os,oo) {"function":"tensor_dtype"}
  rank:u64 = call(os,d) {"function":"getw"}
  np:u64 = int.add.wrap(d,ow)
  xp:u64 = int.add.wrap(d,a96)
  yp:u64 = int.add.wrap(d,a104)
  xpstride:u64 = int.add.wrap(d,a128)
  ypstride:u64 = int.add.wrap(d,a136)
  n:u64 = call(os,np) {"function":"getw"}
  extent0:u64 = call(os,xp) {"function":"getw"}
  extent1:u64 = call(os,yp) {"function":"getw"}
  stride0:u64 = call(os,xpstride) {"function":"getw"}
  stride1:u64 = call(os,ypstride) {"function":"getw"}
  type_ok:bool = compare.eq(dt,two)
  rank_ok:bool = compare.eq(rank,two)
  count_ok:bool = compare.eq(n,ok)
  ext0_ok:bool = compare.eq(extent0,ok)
  ext1_ok:bool = compare.eq(extent1,one)
  st0_ok:bool = compare.eq(stride0,one)
  st1_ok:bool = compare.eq(stride1,one)
  nd_ok:bool = compare.ne(ox,oo)
  p0:bool = bool.and(type_ok,rank_ok)
  p1:bool = bool.and(count_ok,ext0_ok)
  p2:bool = bool.and(ext1_ok,st0_ok)
  p3:bool = bool.and(st1_ok,nd_ok)
  p4:bool = bool.and(p0,p1)
  p5:bool = bool.and(p2,p3)
  proof:bool = bool.and(p4,p5)
  branch proof out_ids(os,ox,oo,d,ok,ow,ypstride,z0) out_no()
block out_ids(ids:mutbuf,ix:u64,io:u64,id:u64,ik:u64,iw:u64,ip:u64,iz:u64)
  st_off:u64 = const() {"value":216}
  vers_off:u64 = const() {"value":224}
  bytes_off:u64 = const() {"value":24}
  cnt_off:u64 = const() {"value":16}
  xd:u64 = call(ids,ix) {"function":"dptr"}
  stp:u64 = int.add.wrap(id,st_off)
  vstp:u64 = int.add.wrap(id,vers_off)
  xstp:u64 = int.add.wrap(xd,st_off)
  xvstp:u64 = int.add.wrap(xd,vers_off)
  st:u64 = call(ids,stp) {"function":"getw"}
  dst_ver:u64 = call(ids,vstp) {"function":"getw"}
  xst:u64 = call(ids,xstp) {"function":"getw"}
  xver:u64 = call(ids,xvstp) {"function":"getw"}
  ss:u64 = call(ids,st) {"function":"so_storage_live"}
  vv:u64 = call(ids,dst_ver) {"function":"so_version_live"}
  slp:u64 = int.add.wrap(ss,bytes_off)
  vcp:u64 = int.add.wrap(vv,cnt_off)
  bytes_actual:u64 = call(ids,slp) {"function":"so_uget"}
  counter:u64 = call(ids,vcp) {"function":"so_uget"}
  bytes_expected:u64 = int.mul.wrap(ik,iw)
  size_ok:bool = compare.eq(bytes_actual,bytes_expected)
  ver_ok:bool = compare.eq(counter,iz)
  dis_s:bool = compare.ne(st,xst)
  dis_v:bool = compare.ne(dst_ver,xver)
  f0:bool = bool.and(size_ok,ver_ok)
  f1:bool = bool.and(dis_s,dis_v)
  ready:bool = bool.and(f0,f1)
  branch ready out_publishable(ids,ip,ik,io) out_no()
block out_publishable(ps:mutbuf,pp:u64,pk:u64,po:u64)
  set_stride_ack:u64 = call(ps,pp,pk) {"function":"putw"}
  actual:u64 = call(ps,pp) {"function":"getw"}
  exact:bool = compare.eq(actual,pk)
  z2:u64 = const() {"value":0}
  ans:u64 = select(exact,po,z2)
  ret(ans)
block out_no()
  no:u64 = const() {"value":0}
  ret(no)
end
fn nz_tensor(s:mutbuf,x:u64) -> (u64) effects(read,write,trap) entry(check)
block check(s:mutbuf,x:u64)
  one:u64 = const() {"value":1}
  e8:u64 = const() {"value":8}
  e96:u64 = const() {"value":96}
  e128:u64 = const() {"value":128}
  d:u64 = call(s,x) {"function":"dptr"}
  rank:u64 = call(s,d) {"function":"getw"}
  tag:u64 = call(s,x) {"function":"tensor_dtype"}
  np:u64 = int.add.wrap(d,e8)
  dp:u64 = int.add.wrap(d,e96)
  sp:u64 = int.add.wrap(d,e128)
  n:u64 = call(s,np) {"function":"getw"}
  dim:u64 = call(s,dp) {"function":"getw"}
  strd:u64 = call(s,sp) {"function":"getw"}
  rank_ok:bool = compare.eq(rank,one)
  tag_ok:bool = compare.eq(tag,one)
  dim_ok:bool = compare.eq(n,dim)
  str_ok:bool = compare.eq(strd,one)
  g0:bool = bool.and(rank_ok,tag_ok)
  g1:bool = bool.and(dim_ok,str_ok)
  gate:bool = bool.and(g0,g1)
  branch gate cap(s,x,n,one) invalid(s)
block cap(cs:mutbuf,cx:u64,cn:u64,co:u64)
  capn:u64 = const() {"value":262144}
  ok_n:bool = compare.le(cn,capn)
  branch ok_n owned(cs,cx,cn,co) capacity(cs)
block owned(os1:mutbuf,ox1:u64,on:u64,ow1:u64)
  oz:u64 = const() {"value":0}
  safe:u64 = call(os1,ox1,ow1) {"function":"nz_ordinary"}
  ready:bool = compare.ne(safe,oz)
  branch ready count(os1,ox1,on,oz,oz) invalid(os1)
block count(cs0:mutbuf,cx0:u64,cn0:u64,ci:u64,ck:u64)
  more:bool = compare.lt(ci,cn0)
  branch more bit(cs0,cx0,cn0,ci,ck) alloc_result(cs0,cx0,cn0,ck)
block bit(bs:mutbuf,bx:u64,bn:u64,bi:u64,bk:u64)
  bitval:u64 = call(bs,bx,bi) {"function":"tget"}
  b0:u64 = const() {"value":0}
  b1:u64 = const() {"value":1}
  berr:u64 = call(bs,b0) {"function":"getw"}
  clean:bool = compare.eq(berr,b0)
  canon:bool = compare.le(bitval,b1)
  nxt:u64 = int.add.wrap(bi,b1)
  cnt:u64 = int.add.wrap(bk,bitval)
  ok:bool = bool.and(clean,canon)
  branch ok count(bs,bx,bn,nxt,cnt) invalid(bs)
block alloc_result(as:mutbuf,ax:u64,an:u64,ak:u64)
  a0:u64 = const() {"value":0}
  a1:u64 = const() {"value":1}
  a2:u64 = const() {"value":2}
  p0:u64 = const() {"value":128}
  p1:u64 = const() {"value":136}
  p2:u64 = const() {"value":144}
  p3:u64 = const() {"value":152}
  keep0:u64 = call(as,p0) {"function":"getw"}
  keep1:u64 = call(as,p1) {"function":"getw"}
  keep2:u64 = call(as,p2) {"function":"getw"}
  keep3:u64 = call(as,p3) {"function":"getw"}
  st0:u64 = call(as,p0,ak) {"function":"putw"}
  st1:u64 = call(as,p1,a1) {"function":"putw"}
  st2:u64 = call(as,p2,a0) {"function":"putw"}
  st3:u64 = call(as,p3,a0) {"function":"putw"}
  new_t:u64 = call(as,a2,p0,a0,a1,a2) {"function":"typed_descriptor"}
  rs0:u64 = call(as,p0,keep0) {"function":"putw"}
  rs1:u64 = call(as,p1,keep1) {"function":"putw"}
  rs2:u64 = call(as,p2,keep2) {"function":"putw"}
  rs3:u64 = call(as,p3,keep3) {"function":"putw"}
  err:u64 = call(as,a0) {"function":"getw"}
  clean_alloc:bool = compare.eq(err,a0)
  has_out:bool = compare.ne(new_t,a0)
  live:bool = bool.and(clean_alloc,has_out)
  branch live verify_out(as,ax,an,ak,new_t,a0) capacity(as)
block verify_out(vs:mutbuf,vx:u64,vn:u64,vk:u64,vo:u64,vz:u64)
  validated:u64 = call(vs,vx,vo,vk) {"function":"nz_output"}
  known:bool = compare.eq(validated,vo)
  branch known tested(vs,vx,vn,vk,vo,vz) invalid(vs)
block tested(ts1:mutbuf,tx1:u64,tn1:u64,tk1:u64,to1:u64,tz1:u64)
  no_data:bool = compare.eq(tk1,tz1)
  branch no_data done(to1) loop(ts1,tx1,tn1,tk1,to1,tz1,tz1)
block loop(ls:mutbuf,lx:u64,ln:u64,lk:u64,lo:u64,li:u64,lj:u64)
  lmore:bool = compare.lt(li,ln)
  branch lmore choose(ls,lx,ln,lk,lo,li,lj) finish(ls,lk,lo,lj)
block choose(hs:mutbuf,hx:u64,hn:u64,hk:u64,ho:u64,hi:u64,hj:u64)
  h0:u64 = const() {"value":0}
  h1:u64 = const() {"value":1}
  hv:u64 = call(hs,hx,hi) {"function":"tget"}
  herr:u64 = call(hs,h0) {"function":"getw"}
  sane:bool = compare.le(hv,h1)
  hclean:bool = compare.eq(herr,h0)
  hsane:bool = bool.and(sane,hclean)
  branch hsane place(hs,hx,hn,hk,ho,hi,hj,hv,h0,h1) invalid(hs)
block place(ps1:mutbuf,px1:u64,pn1:u64,pk1:u64,po1:u64,pi1:u64,pj1:u64,pv1:u64,pz1:u64,pone:u64)
  step:u64 = int.add.wrap(pi1,pone)
  keep:bool = compare.ne(pv1,pz1)
  branch keep room(ps1,px1,pn1,pk1,po1,pi1,pj1,step) loop(ps1,px1,pn1,pk1,po1,step,pj1)
block room(ms:mutbuf,mx:u64,mn:u64,mk:u64,mo:u64,mi:u64,mj:u64,mnext:u64)
  fits:bool = compare.lt(mj,mk)
  branch fits save_word(ms,mx,mn,mk,mo,mi,mj,mnext) invalid(ms)
block save_word(ws:mutbuf,wx:u64,wn:u64,wk:u64,wo:u64,wi:u64,wj:u64,wnext:u64)
  w1:u64 = const() {"value":1}
  w8:u64 = const() {"value":8}
  loc:u64 = call(ws,wo,wj) {"function":"taddr"}
  put_ack:u64 = call(ws,loc,w8,wi) {"function":"typed_store"}
  jnext:u64 = int.add.wrap(wj,w1)
  jump loop(ws,wx,wn,wk,wo,wnext,jnext)
block finish(fs:mutbuf,fk:u64,fo:u64,fj:u64)
  fin:bool = compare.eq(fk,fj)
  branch fin done(fo) invalid(fs)
block done(result:u64)
  ret(result)
block invalid(is1:mutbuf)
  code1:u64 = const() {"value":7}
  fail1:u64 = call(is1,code1) {"function":"nz_error"}
  ret(fail1)
block capacity(cs1:mutbuf)
  code2:u64 = const() {"value":3}
  fail2:u64 = call(cs1,code2) {"function":"nz_error"}
  ret(fail2)
end
