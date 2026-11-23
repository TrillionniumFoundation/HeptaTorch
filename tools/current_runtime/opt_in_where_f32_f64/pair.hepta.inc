block wide_tags(wps:mutbuf,wpm:u64,wpx:u64,wpy:u64,wxt:u64,wyt:u64,wmask_ok:bool)
  zero_w:u64 = const() {"value":0}
  f64_w:u64 = const() {"value":4}
  wx_f32:bool = compare.eq(wxt,zero_w)
  wy_f32:bool = compare.eq(wyt,zero_w)
  wx_f64:bool = compare.eq(wxt,f64_w)
  wy_f64:bool = compare.eq(wyt,f64_w)
  forward_w:bool = bool.and(wx_f32,wy_f64)
  reverse_w:bool = bool.and(wx_f64,wy_f32)
  pair_w:bool = bool.or(forward_w,reverse_w)
  accept_w:bool = bool.and(wmask_ok,pair_w)
  branch accept_w wide_ranks(wps,wpm,wpx,wpy,wxt,wyt,f64_w) bad(wps)
block wide_ranks(wrs:mutbuf,wrm:u64,wrx:u64,wry:u64,wrxt:u64,wryt:u64,wrt:u64)
  wrxd:u64 = call(wrs,wrx) {"function":"dptr"}
  wryd:u64 = call(wrs,wry) {"function":"dptr"}
  wrxr:u64 = call(wrs,wrxd) {"function":"getw"}
  wryr:u64 = call(wrs,wryd) {"function":"getw"}
  z_rank:u64 = const() {"value":0}
  mask_width_w:u64 = const() {"value":1}
  rowx:bool = compare.gt(wrxr,z_rank)
  rowy:bool = compare.gt(wryr,z_rank)
  row_both:bool = bool.and(rowx,rowy)
  branch row_both mixed_roots(wrs,wrm,wrx,wry,wrxt,wryt,mask_width_w,wrt) bad(wrs)
