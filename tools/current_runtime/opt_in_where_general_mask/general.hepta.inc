fn where_general_geometry(gs:mutbuf,gm:u64,gx:u64) -> (u64) effects(read,write,trap) entry(check)
block check(gs:mutbuf,gm:u64,gx:u64)
  gd_m:u64 = call(gs,gm) {"function":"dptr"}
  gd_x:u64 = call(gs,gx) {"function":"dptr"}
  gr_m:u64 = call(gs,gd_m) {"function":"getw"}
  gr_x:u64 = call(gs,gd_x) {"function":"getw"}
  one:u64 = const() {"value":1}
  four:u64 = const() {"value":4}
  nonzero:bool = compare.ge(gr_m,one)
  no_expand:bool = compare.le(gr_m,gr_x)
  in_cap:bool = compare.le(gr_x,four)
  first:bool = bool.and(nonzero,no_expand)
  ok:bool = bool.and(first,in_cap)
  branch ok repeat(gs,gd_m,gd_x,gr_m,gr_x,one,one) no()
block repeat(rs:mutbuf,rm:u64,rx:u64,rma:u64,rxa:u64,rm_prod:u64,rx_prod:u64)
  zr:u64 = const() {"value":0}
  more:bool = compare.ne(rxa,zr)
  branch more x_axis(rs,rm,rx,rma,rxa,rm_prod,rx_prod) counts(rs,rm,rx,rm_prod,rx_prod)
block x_axis(xs:mutbuf,xm:u64,xx:u64,xma:u64,xxa:u64,xm_prod:u64,xx_prod:u64)
  xo:u64 = const() {"value":1}
  xz:u64 = const() {"value":0}
  x_axis_num:u64 = int.sub.wrap(xxa,xo)
  x8:u64 = const() {"value":8}
  x96:u64 = const() {"value":96}
  xscale:u64 = int.mul.wrap(x_axis_num,x8)
  xoffset:u64 = int.add.wrap(x96,xscale)
  xptr:u64 = int.add.wrap(xx,xoffset)
  xdim:u64 = call(xs,xptr) {"function":"getw"}
  mask_present:bool = compare.ne(xma,xz)
  branch mask_present m_axis(xs,xm,xx,xma,x_axis_num,xm_prod,xx_prod,xdim) checked(xs,xm,xx,xma,x_axis_num,xm_prod,xx_prod,xdim,xo)
block m_axis(ms:mutbuf,mm:u64,mx:u64,mma:u64,mxa:u64,mm_prod:u64,mx_prod:u64,mxd:u64)
  mo:u64 = const() {"value":1}
  m_axis_num:u64 = int.sub.wrap(mma,mo)
  m8:u64 = const() {"value":8}
  m96:u64 = const() {"value":96}
  mscale:u64 = int.mul.wrap(m_axis_num,m8)
  moffset:u64 = int.add.wrap(m96,mscale)
  mptr:u64 = int.add.wrap(mm,moffset)
  mdim:u64 = call(ms,mptr) {"function":"getw"}
  jump checked(ms,mm,mx,m_axis_num,mxa,mm_prod,mx_prod,mxd,mdim)
block checked(cs:mutbuf,cm:u64,cx:u64,cma:u64,cxa:u64,cm_prod:u64,cx_prod:u64,cxd:u64,cmd:u64)
  co:u64 = const() {"value":1}
  cap:u64 = const() {"value":262144}
  xa_ok:bool = compare.le(cxd,cap)
  ma_ok:bool = compare.le(cmd,cap)
  both_caps:bool = bool.and(xa_ok,ma_ok)
  equal_extent:bool = compare.eq(cmd,cxd)
  singleton:bool = compare.eq(cmd,co)
  shape_ok:bool = bool.or(equal_extent,singleton)
  proven:bool = bool.and(both_caps,shape_ok)
  branch proven products(cs,cm,cx,cma,cxa,cm_prod,cx_prod,cxd,cmd) no()
block products(ps:mutbuf,pm:u64,px:u64,pma:u64,pxa:u64,pm_prod:u64,px_prod:u64,pxd:u64,pmd:u64)
  sat:u64 = const() {"value":262145}
  vm:u64 = int.mul.wrap(pm_prod,pmd)
  vx:u64 = int.mul.wrap(px_prod,pxd)
  mov:bool = compare.gt(vm,sat)
  xov:bool = compare.gt(vx,sat)
  mn:u64 = select(mov,sat,vm)
  xn:u64 = select(xov,sat,vx)
  jump repeat(ps,pm,px,pma,pxa,mn,xn)
block counts(fs:mutbuf,fm:u64,fx:u64,fm_prod:u64,fx_prod:u64)
  f8:u64 = const() {"value":8}
  fmpos:u64 = int.add.wrap(fm,f8)
  fxpos:u64 = int.add.wrap(fx,f8)
  fmc:u64 = call(fs,fmpos) {"function":"getw"}
  fxc:u64 = call(fs,fxpos) {"function":"getw"}
  fm_ok:bool = compare.eq(fm_prod,fmc)
  fx_ok:bool = compare.eq(fx_prod,fxc)
  full:bool = bool.and(fm_ok,fx_ok)
  branch full yes() no()
block yes()
  one_out:u64 = const() {"value":1}
  ret(one_out)
block no()
  zero_out:u64 = const() {"value":0}
  ret(zero_out)
end
fn where_general_index(gs:mutbuf,gm:u64,gx:u64,gi:u64) -> (u64) effects(read,write,trap) entry(check)
block check(gs:mutbuf,gm:u64,gx:u64,gi:u64)
  gmd:u64 = call(gs,gm) {"function":"dptr"}
  gxd:u64 = call(gs,gx) {"function":"dptr"}
  gma:u64 = call(gs,gmd) {"function":"getw"}
  gxa:u64 = call(gs,gxd) {"function":"getw"}
  g8:u64 = const() {"value":8}
  gmp:u64 = int.add.wrap(gmd,g8)
  gxp:u64 = int.add.wrap(gxd,g8)
  gmc:u64 = call(gs,gmp) {"function":"getw"}
  gxc:u64 = call(gs,gxp) {"function":"getw"}
  one:u64 = const() {"value":1}
  zero:u64 = const() {"value":0}
  in_x:bool = compare.lt(gi,gxc)
  nonempty_m:bool = compare.ge(gmc,one)
  ok:bool = bool.and(in_x,nonempty_m)
  branch ok walk(gs,gmd,gxd,gma,gxa,gi,gmc,one,zero) bad()
block walk(ws:mutbuf,wm:u64,wx:u64,wma:u64,wxa:u64,wi:u64,wn:u64,wstride:u64,windex:u64)
  wz:u64 = const() {"value":0}
  more:bool = compare.ne(wma,wz)
  branch more axes(ws,wm,wx,wma,wxa,wi,wn,wstride,windex) finish(wn,wstride,windex)
block axes(as0:mutbuf,am:u64,ax:u64,ama:u64,axa:u64,ai:u64,an:u64,astride:u64,aindex:u64)
  ao:u64 = const() {"value":1}
  a8:u64 = const() {"value":8}
  a96:u64 = const() {"value":96}
  a_m:u64 = int.sub.wrap(ama,ao)
  a_x:u64 = int.sub.wrap(axa,ao)
  a_mword:u64 = int.mul.wrap(a_m,a8)
  a_xword:u64 = int.mul.wrap(a_x,a8)
  a_mb:u64 = int.add.wrap(am,a96)
  a_xb:u64 = int.add.wrap(ax,a96)
  a_mp:u64 = int.add.wrap(a_mb,a_mword)
  a_xp:u64 = int.add.wrap(a_xb,a_xword)
  amd:u64 = call(as0,a_mp) {"function":"getw"}
  axd:u64 = call(as0,a_xp) {"function":"getw"}
  md_nonzero:bool = compare.ge(amd,ao)
  xd_nonzero:bool = compare.ge(axd,ao)
  geom:bool = bool.and(md_nonzero,xd_nonzero)
  branch geom index(as0,am,ax,a_m,a_x,ai,an,astride,aindex,amd,axd) bad()
block index(is0:mutbuf,im:u64,ix:u64,ima:u64,ixa:u64,ii:u64,in0:u64,istride:u64,iindex:u64,imd:u64,ixd:u64)
  coord:u64 = int.rem.checked(ii,ixd)
  remain:u64 = int.div.checked(ii,ixd)
  iz:u64 = const() {"value":0}
  io:u64 = const() {"value":1}
  broadcast:bool = compare.eq(imd,io)
  vcoord:u64 = select(broadcast,iz,coord)
  ok_coord:bool = compare.lt(vcoord,imd)
  scaled:u64 = int.mul.wrap(vcoord,istride)
  next:u64 = int.add.wrap(iindex,scaled)
  newstride:u64 = int.mul.wrap(istride,imd)
  bounded_stride:bool = compare.le(newstride,in0)
  bounded_index:bool = compare.lt(next,in0)
  bounds:bool = bool.and(bounded_stride,bounded_index)
  ok_all:bool = bool.and(ok_coord,bounds)
  branch ok_all walk(is0,im,ix,ima,ixa,remain,in0,newstride,next) bad()
block finish(fn0:u64,fstride:u64,findex:u64)
  stride_matches:bool = compare.eq(fstride,fn0)
  index_matches:bool = compare.lt(findex,fn0)
  both:bool = bool.and(stride_matches,index_matches)
  branch both answer(findex) bad()
block answer(result:u64)
  ret(result)
block bad()
  sentinel:u64 = const() {"value":262145}
  ret(sentinel)
end
