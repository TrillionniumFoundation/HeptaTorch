  branch lrd_bcast la_bc_index(lrd_s,lrd_x,lrd_y,lrd_t,lrd_out,lrd_n,lrd_i,lrd_op,lrd_bcast) la_values(lrd_s,lrd_x,lrd_y,lrd_t,lrd_out,lrd_n,lrd_i,lrd_i,lrd_i,lrd_op,lrd_bcast)
block la_bc_index(lbi_s:mutbuf,lbi_x:u64,lbi_y:u64,lbi_t:u64,lbi_out:u64,lbi_n:u64,lbi_i:u64,lbi_op:u64,lbi_bcast:bool)
  lbi_xi:u64 = call(lbi_s,lbi_out,lbi_x,lbi_i) {"function":"bindex"}
  lbi_yi:u64 = call(lbi_s,lbi_out,lbi_y,lbi_i) {"function":"bindex"}
  jump la_values(lbi_s,lbi_x,lbi_y,lbi_t,lbi_out,lbi_n,lbi_i,lbi_xi,lbi_yi,lbi_op,lbi_bcast)
block la_values(lv_s:mutbuf,lv_x:u64,lv_y:u64,lv_t:u64,lv_out:u64,lv_n:u64,lv_i:u64,lv_xi:u64,lv_yi:u64,lv_op:u64,lv_bcast:bool)
