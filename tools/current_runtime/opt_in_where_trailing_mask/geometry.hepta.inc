  branch setup_shapes_ok allocate(ss,sm,sx,sy,st,sw,mask_step) alternate(ss,sm,sx,sy,st,sw,mask_desc,mask_rank,mask_count,has_equal_values)
block alternate(hs:mutbuf,hm:u64,hx:u64,hy:u64,ht:u64,hw:u64,hd:u64,hr:u64,hc:u64,hvalues:bool)
  vd:u64 = call(hs,hx) {"function":"dptr"}
  vr:u64 = call(hs,vd) {"function":"getw"}
  hz:u64 = const() {"value":0}
  nonzero_rank:bool = compare.ne(hr,hz)
  same_rank:bool = compare.eq(hr,vr)
  rank_ok:bool = bool.and(nonzero_rank,same_rank)
  admission:bool = bool.and(rank_ok,hvalues)
  branch admission prefix(hs,hm,hx,hy,ht,hw,hd,vd,hr,hc,hz) bad(hs)
block prefix(ps:mutbuf,pm:u64,px:u64,py:u64,pt:u64,pw:u64,pmd:u64,pvd:u64,pr:u64,pc:u64,pi:u64)
  unit:u64 = const() {"value":1}
  last:u64 = int.sub.wrap(pr,unit)
  last_axis:bool = compare.eq(pi,last)
  dim_base:u64 = const() {"value":96}
  dim_word:u64 = const() {"value":8}
  pos_word:u64 = int.mul.wrap(pi,dim_word)
  axis_off:u64 = int.add.wrap(dim_base,pos_word)
  mp:u64 = int.add.wrap(pmd,axis_off)
  vp:u64 = int.add.wrap(pvd,axis_off)
  mask_extent:u64 = call(ps,mp) {"function":"getw"}
  val_extent:u64 = call(ps,vp) {"function":"getw"}
  branch last_axis last_admit(ps,pm,px,py,pt,pw,pvd,pc,mask_extent,val_extent) prefix_compare(ps,pm,px,py,pt,pw,pmd,pvd,pr,pc,pi,mask_extent,val_extent)
block prefix_compare(fs:mutbuf,fm:u64,fx:u64,fy:u64,ft:u64,fw:u64,fmd:u64,fvd:u64,fr:u64,fc:u64,fi:u64,fa:u64,fb:u64)
  match:bool = compare.eq(fa,fb)
  next_one:u64 = const() {"value":1}
  next_axis:u64 = int.add.wrap(fi,next_one)
  branch match prefix(fs,fm,fx,fy,ft,fw,fmd,fvd,fr,fc,next_axis) bad(fs)
block last_admit(ds:mutbuf,dm:u64,dx:u64,dy:u64,dt:u64,dw:u64,dd:u64,dc:u64,da:u64,db:u64)
  done_one:u64 = const() {"value":1}
  max_extent:u64 = const() {"value":262144}
  exact_mask:bool = compare.eq(da,done_one)
  bounded_extent:bool = compare.le(db,max_extent)
  good_extent:bool = bool.and(exact_mask,bounded_extent)
  branch good_extent trailing_count(ds,dm,dx,dy,dt,dw,dd,dc,db) bad(ds)
block trailing_count(ns:mutbuf,nm:u64,nx:u64,ny:u64,nt:u64,nw:u64,nd:u64,nc:u64,ne:u64)
  numel_word:u64 = const() {"value":8}
  numel_pos:u64 = int.add.wrap(nd,numel_word)
  values_count:u64 = call(ns,numel_pos) {"function":"getw"}
  expected_count:u64 = int.mul.wrap(nc,ne)
  count_exact:bool = compare.eq(values_count,expected_count)
  d_zero:u64 = const() {"value":0}
  d_one:u64 = const() {"value":1}
  zero_extent:bool = compare.eq(ne,d_zero)
  safe_divisor:u64 = select(zero_extent,d_one,ne)
  branch count_exact allocate(ns,nm,nx,ny,nt,nw,safe_divisor) bad(ns)
