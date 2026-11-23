block mixed_tags(admit_ms:mutbuf,admit_mm:u64,admit_mx:u64,admit_my:u64,mxt:u64,myt:u64,mask_ok:bool)
  bool_m:u64 = const() {"value":1}
  x_bool:bool = compare.eq(mxt,bool_m)
  y_bool:bool = compare.eq(myt,bool_m)
  different_tags:bool = compare.ne(mxt,myt)
  at_least_one_bool:bool = bool.or(x_bool,y_bool)
  exactly_one_bool:bool = bool.and(different_tags,at_least_one_bool)
  numeric_tag:u64 = select(x_bool,myt,mxt)
  t_f32:u64 = const() {"value":0}
  t_i64:u64 = const() {"value":2}
  t_f64:u64 = const() {"value":4}
  t_f16:u64 = const() {"value":5}
  t_bf16:u64 = const() {"value":6}
  is_f32_m:bool = compare.eq(numeric_tag,t_f32)
  is_i64_m:bool = compare.eq(numeric_tag,t_i64)
  is_f64_m:bool = compare.eq(numeric_tag,t_f64)
  is_f16_m:bool = compare.eq(numeric_tag,t_f16)
  is_bf16_m:bool = compare.eq(numeric_tag,t_bf16)
  full_m:bool = bool.or(is_f32_m,is_f64_m)
  half_m:bool = bool.or(is_f16_m,is_bf16_m)
  float_m:bool = bool.or(full_m,half_m)
  public_pair_target:bool = bool.or(float_m,is_i64_m)
  pair0:bool = bool.and(mask_ok,exactly_one_bool)
  admit_pair:bool = bool.and(pair0,public_pair_target)
  branch admit_pair mixed_roots(admit_ms,admit_mm,admit_mx,admit_my,mxt,myt,bool_m,numeric_tag) bad(admit_ms)
