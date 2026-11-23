block mixed_tags(admit_ms:mutbuf,admit_mm:u64,admit_mx:u64,admit_my:u64,mxt:u64,myt:u64,mask_ok:bool)
  bool_m:u64 = const() {"value":1}
  i64_m:u64 = const() {"value":2}
  x_bool:bool = compare.eq(mxt,bool_m)
  y_bool:bool = compare.eq(myt,bool_m)
  x_i64:bool = compare.eq(mxt,i64_m)
  y_i64:bool = compare.eq(myt,i64_m)
  first_order:bool = bool.and(x_bool,y_i64)
  second_order:bool = bool.and(x_i64,y_bool)
  exact_pair:bool = bool.or(first_order,second_order)
  admit_pair:bool = bool.and(mask_ok,exact_pair)
  branch admit_pair mixed_roots(admit_ms,admit_mm,admit_mx,admit_my,mxt,myt,bool_m,i64_m) bad(admit_ms)
block mixed_roots(ms0:mutbuf,mm0:u64,mx0:u64,my0:u64,mxt0:u64,myt0:u64,mone:u64,mout_type:u64)
  mx_width:u64 = call(mxt0) {"function":"dtype_width"}
  my_width:u64 = call(myt0) {"function":"dtype_width"}
  out_width:u64 = call(mout_type) {"function":"dtype_width"}
  mm_ok:u64 = call(ms0,mm0,mone) {"function":"where_ordinary"}
  mx_ok:u64 = call(ms0,mx0,mx_width) {"function":"where_ordinary"}
  my_ok:u64 = call(ms0,my0,my_width) {"function":"where_ordinary"}
  mix_ok0:u64 = bit.and(mm_ok,mx_ok)
  mix_ok1:u64 = bit.and(mix_ok0,my_ok)
  mzero:u64 = const() {"value":0}
  m_ready:bool = compare.ne(mix_ok1,mzero)
  branch m_ready promote_pair(ms0,mm0,mx0,my0,mout_type,out_width) bad(ms0)
block promote_pair(bs0:mutbuf,bm0:u64,bx0:u64,by0:u64,bt0:u64,bw0:u64)
  yes_mode:u64 = const() {"value":0}
  mixed_mode:bool = compare.eq(yes_mode,yes_mode)
  bout:u64 = call(bs0,bm0,bx0,by0,bt0,bw0,mixed_mode) {"function":"where_full_where_core"}
  ret(bout)
