block zero_wide_scope(zws:mutbuf,zwm:u64,zwx:u64,zwy:u64,zwxt:u64,zwyt:u64,zwt:u64,zwxd:u64,zwyd:u64,zwxr:u64,zwyr:u64)
  wz_zero:u64 = const() {"value":0}
  wz_one:u64 = const() {"value":1}
  wz_i64:u64 = const() {"value":2}
  wz_f64:u64 = const() {"value":4}
  wz_f16:u64 = const() {"value":5}
  wz_bf16:u64 = const() {"value":6}
  wz_target:bool = compare.eq(zwt,wz_f64)
  wz_x_scalar:bool = compare.eq(zwxr,wz_zero)
  wz_y_scalar:bool = compare.eq(zwyr,wz_zero)
  wz_x_dim:bool = compare.gt(zwxr,wz_zero)
  wz_y_dim:bool = compare.gt(zwyr,wz_zero)
  wz_x_f64:bool = compare.eq(zwxt,wz_f64)
  wz_y_f64:bool = compare.eq(zwyt,wz_f64)
  wz_x_role:bool = bool.and(wz_x_dim,wz_x_f64)
  wz_y_role:bool = bool.and(wz_y_dim,wz_y_f64)
  wz_sx:bool = bool.and(wz_x_scalar,wz_y_role)
  wz_sy:bool = bool.and(wz_y_scalar,wz_x_role)
  wz_shape:bool = bool.or(wz_sx,wz_sy)
  wz_possible:bool = bool.and(wz_target,wz_shape)
  wz_dtype:u64 = select(wz_sx,zwxt,zwyt)
  wz_dptr:u64 = select(wz_sx,zwxd,zwyd)
  wz_dim_desc:u64 = select(wz_sx,zwyd,zwxd)
  wz_f32_ok:bool = compare.eq(wz_dtype,wz_zero)
  wz_i64_ok:bool = compare.eq(wz_dtype,wz_i64)
  wz_f16_ok:bool = compare.eq(wz_dtype,wz_f16)
  wz_bf16_ok:bool = compare.eq(wz_dtype,wz_bf16)
  wz_left:bool = bool.or(wz_f32_ok,wz_i64_ok)
  wz_right:bool = bool.or(wz_f16_ok,wz_bf16_ok)
  wz_tag:bool = bool.or(wz_left,wz_right)
  wz_check:bool = bool.and(wz_possible,wz_tag)
  branch wz_check zero_wide_count(zws,zwm,zwx,zwy,zwxt,zwyt,wz_one,wz_f64,wz_dptr,wz_dim_desc) bad(zws)
block zero_wide_count(zcs:mutbuf,zcm:u64,zcx:u64,zcy:u64,zcxt:u64,zcyt:u64,zcone:u64,zcf64:u64,zcd:u64,zcdim:u64)
  count_offset:u64 = const() {"value":8}
  count_address:u64 = int.add.wrap(zcd,count_offset)
  scalar_count:u64 = call(zcs,count_address) {"function":"getw"}
  count_one:bool = compare.eq(scalar_count,zcone)
  branch count_one zero_wide_nonempty(zcs,zcm,zcx,zcy,zcxt,zcyt,zcone,zcf64,zcdim) bad(zcs)
block zero_wide_nonempty(zns:mutbuf,znm:u64,znx:u64,zny:u64,znxt:u64,znyt:u64,znone:u64,znf64:u64,zndim:u64)
  zncd:u64 = const() {"value":8}
  znzero:u64 = const() {"value":0}
  znmaskdesc:u64 = call(zns,znm) {"function":"dptr"}
  znmaskcountat:u64 = int.add.wrap(znmaskdesc,zncd)
  zndimcountat:u64 = int.add.wrap(zndim,zncd)
  znmaskcount:u64 = call(zns,znmaskcountat) {"function":"getw"}
  zndimcount:u64 = call(zns,zndimcountat) {"function":"getw"}
  znmaskok:bool = compare.gt(znmaskcount,znzero)
  zndimok:bool = compare.gt(zndimcount,znzero)
  znboth:bool = bool.and(znmaskok,zndimok)
  branch znboth mixed_roots(zns,znm,znx,zny,znxt,znyt,znone,znf64) bad(zns)
