block la_bcast(lbc_s:mutbuf,lbc_x:u64,lbc_y:u64,lbc_t:u64,lbc_xd:u64,lbc_yd:u64,lbc_xr:u64,lbc_yr:u64,lbc_op:u64)
  lbc_o:u64 = call(lbc_s,lbc_xd,lbc_yd,lbc_xr,lbc_yr) {"function":"la_bc_plan"}
  lbc_z:u64 = const() {"value":0}
  lbc_ok:bool = compare.ne(lbc_o,lbc_z)
  branch lbc_ok la_bcast_go(lbc_s,lbc_x,lbc_y,lbc_t,lbc_o,lbc_op,lbc_z) la_zero()
block la_bcast_go(lbg_s:mutbuf,lbg_x:u64,lbg_y:u64,lbg_t:u64,lbg_out:u64,lbg_op:u64,lbg_z:u64)
  lbg_desc:u64 = call(lbg_s,lbg_out) {"function":"dptr"}
  lbg_eight:u64 = const() {"value":8}
  lbg_at:u64 = int.add.wrap(lbg_desc,lbg_eight)
  lbg_n:u64 = call(lbg_s,lbg_at) {"function":"getw"}
  lbg_mode:bool = compare.eq(lbg_z,lbg_z)
  jump la_loop(lbg_s,lbg_x,lbg_y,lbg_t,lbg_out,lbg_n,lbg_z,lbg_op,lbg_mode)
