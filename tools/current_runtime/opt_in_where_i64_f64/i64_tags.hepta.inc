block i64_f64_pair(ips:mutbuf,ipm:u64,ipx:u64,ipy:u64,ixt:u64,iyt:u64,i_mask:bool)
  i64_tag:u64 = const() {"value":2}
  f64_tag:u64 = const() {"value":4}
  ix_int:bool = compare.eq(ixt,i64_tag)
  iy_int:bool = compare.eq(iyt,i64_tag)
  ix_f64:bool = compare.eq(ixt,f64_tag)
  iy_f64:bool = compare.eq(iyt,f64_tag)
  i_forward:bool = bool.and(ix_int,iy_f64)
  i_reverse:bool = bool.and(ix_f64,iy_int)
  i_pair:bool = bool.or(i_forward,i_reverse)
  i_admit:bool = bool.and(i_mask,i_pair)
  branch i_admit wide_ranks(ips,ipm,ipx,ipy,ixt,iyt,f64_tag) bad(ips)
