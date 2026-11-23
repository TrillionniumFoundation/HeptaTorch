block zero_i64_f32_scope(zfs:mutbuf,zfm:u64,zfx:u64,zfy:u64,zfxt:u64,zfyt:u64,zft:u64,zfxd:u64,zfyd:u64,zfxr:u64,zfyr:u64)
  zf_f32:u64 = const() {"value":0}
  zf_i64:u64 = const() {"value":2}
  zf_one:u64 = const() {"value":1}
  zf_target:bool = compare.eq(zft,zf_f32)
  zf_x0:bool = compare.eq(zfxr,zf_f32)
  zf_y0:bool = compare.eq(zfyr,zf_f32)
  zf_xdim:bool = compare.gt(zfxr,zf_f32)
  zf_ydim:bool = compare.gt(zfyr,zf_f32)
  zf_xi:bool = compare.eq(zfxt,zf_i64)
  zf_yi:bool = compare.eq(zfyt,zf_i64)
  zf_xf:bool = compare.eq(zfxt,zf_f32)
  zf_yf:bool = compare.eq(zfyt,zf_f32)
  zf_x_scalar:bool = bool.and(zf_x0,zf_xi)
  zf_y_scalar:bool = bool.and(zf_y0,zf_yi)
  zf_x_value:bool = bool.and(zf_xdim,zf_xf)
  zf_y_value:bool = bool.and(zf_ydim,zf_yf)
  zf_left:bool = bool.and(zf_x_scalar,zf_y_value)
  zf_right:bool = bool.and(zf_y_scalar,zf_x_value)
  zf_pair:bool = bool.or(zf_left,zf_right)
  zf_allowed:bool = bool.and(zf_pair,zf_target)
  zf_rd:u64 = select(zf_left,zfxd,zfyd)
  zf_dd:u64 = select(zf_left,zfyd,zfxd)
  branch zf_allowed zero_i64_f32_count(zfs,zfm,zfx,zfy,zfxt,zfyt,zf_one,zf_f32,zf_rd,zf_dd) bad(zfs)
block zero_i64_f32_count(zis:mutbuf,zim:u64,zix:u64,ziy:u64,zixt:u64,ziyt:u64,zione:u64,zizero:u64,zisd:u64,zidd:u64)
  zi_off:u64 = const() {"value":8}
  zi_md:u64 = call(zis,zim) {"function":"dptr"}
  zi_sca:u64 = int.add.wrap(zisd,zi_off)
  zi_dca:u64 = int.add.wrap(zidd,zi_off)
  zi_mca:u64 = int.add.wrap(zi_md,zi_off)
  zi_sc:u64 = call(zis,zi_sca) {"function":"getw"}
  zi_dc:u64 = call(zis,zi_dca) {"function":"getw"}
  zi_mc:u64 = call(zis,zi_mca) {"function":"getw"}
  zi_true_scalar:bool = compare.eq(zi_sc,zione)
  zi_ndim:bool = compare.gt(zi_dc,zizero)
  zi_nmask:bool = compare.gt(zi_mc,zizero)
  zi_nonempty:bool = bool.and(zi_ndim,zi_nmask)
  zi_ready:bool = bool.and(zi_true_scalar,zi_nonempty)
  branch zi_ready zero_i64_f32_canonical(zis,zim,zix,ziy,zixt,ziyt,zione,zizero) bad(zis)
block zero_i64_f32_canonical(if0_zks:mutbuf,if0_zkm:u64,if0_zkx:u64,if0_zky:u64,if0_zkxt:u64,if0_zkyt:u64,if0_zkone:u64,if0_zkzero:u64)
  if0_zk_m:u64 = call(if0_zks,if0_zkm) {"function":"where_empty_input_canonical"}
  if0_zk_l:u64 = call(if0_zks,if0_zkx) {"function":"where_empty_input_canonical"}
  if0_zk_r:u64 = call(if0_zks,if0_zky) {"function":"where_empty_input_canonical"}
  if0_zk_pair:u64 = bit.and(if0_zk_l,if0_zk_r)
  if0_zk_all:u64 = bit.and(if0_zk_pair,if0_zk_m)
  if0_zk_ready:bool = compare.eq(if0_zk_all,if0_zkone)
  branch if0_zk_ready mixed_roots(if0_zks,if0_zkm,if0_zkx,if0_zky,if0_zkxt,if0_zkyt,if0_zkone,if0_zkzero) bad(if0_zks)
