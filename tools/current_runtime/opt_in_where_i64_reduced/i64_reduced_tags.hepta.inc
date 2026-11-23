block i64_reduced_pair(irps:mutbuf,irpm:u64,irpx:u64,irpy:u64,irxt:u64,iryt:u64,irmask:bool)
  ir_i:u64 = const() {"value":2}
  ir_f:u64 = const() {"value":5}
  ir_b:u64 = const() {"value":6}
  ir_xi:bool = compare.eq(irxt,ir_i)
  ir_yi:bool = compare.eq(iryt,ir_i)
  ir_xf:bool = compare.eq(irxt,ir_f)
  ir_yf:bool = compare.eq(iryt,ir_f)
  ir_xb:bool = compare.eq(irxt,ir_b)
  ir_yb:bool = compare.eq(iryt,ir_b)
  ir_xh:bool = bool.or(ir_xf,ir_xb)
  ir_yh:bool = bool.or(ir_yf,ir_yb)
  ir_left:bool = bool.and(ir_xi,ir_yh)
  ir_right:bool = bool.and(ir_xh,ir_yi)
  ir_pair:bool = bool.or(ir_left,ir_right)
  ir_admit:bool = bool.and(irmask,ir_pair)
  ir_out:u64 = select(ir_left,iryt,irxt)
  branch ir_admit wide_ranks(irps,irpm,irpx,irpy,irxt,iryt,ir_out) bad(irps)
