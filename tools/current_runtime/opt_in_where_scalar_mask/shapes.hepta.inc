block shapes(ss:mutbuf,sm:u64,sx:u64,sy:u64,st:u64,sw:u64)
  mx:u64 = call(ss,sm,sx) {"function":"incoming_same_shape"}
  xy:u64 = call(ss,sx,sy) {"function":"incoming_same_shape"}
  mask_desc:u64 = call(ss,sm) {"function":"dptr"}
  mask_rank:u64 = call(ss,mask_desc) {"function":"getw"}
  mask_count_off:u64 = const() {"value":8}
  mask_count_addr:u64 = int.add.wrap(mask_desc,mask_count_off)
  mask_count:u64 = call(ss,mask_count_addr) {"function":"getw"}
  zshape:u64 = const() {"value":0}
  one_shape:u64 = const() {"value":1}
  has_equal_values:bool = compare.ne(xy,zshape)
  has_equal_condition:bool = compare.ne(mx,zshape)
  has_scalar_rank:bool = compare.eq(mask_rank,zshape)
  has_scalar_count:bool = compare.eq(mask_count,one_shape)
  has_scalar_condition:bool = bool.and(has_scalar_rank,has_scalar_count)
  condition_shape_ok:bool = bool.or(has_equal_condition,has_scalar_condition)
  setup_shapes_ok:bool = bool.and(has_equal_values,condition_shape_ok)
  mask_step:u64 = select(has_scalar_condition,zshape,one_shape)
  branch setup_shapes_ok allocate(ss,sm,sx,sy,st,sw,mask_step) bad(ss)
