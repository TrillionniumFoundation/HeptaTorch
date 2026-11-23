block check_word(ps0:mutbuf,pm0:u64,px0:u64,py0:u64,po0:u64,pw0:u64,pn0:u64,pi0:u64,psrc:u64,pword:u64,check_mix:bool)
  stype:u64 = call(ps0,psrc) {"function":"tensor_dtype"}
  otype:u64 = call(ps0,po0) {"function":"tensor_dtype"}
  bool_src:u64 = const() {"value":1}
  same_raw_type:bool = compare.eq(stype,otype)
  is_bool:bool = compare.eq(stype,bool_src)
  branch same_raw_type store(ps0,pm0,px0,py0,po0,pw0,pn0,pi0,pword,check_mix) convert_bool(ps0,pm0,px0,py0,po0,pw0,pn0,pi0,pword,check_mix,is_bool,otype)
block convert_bool(zs:mutbuf,zm:u64,zx:u64,zy:u64,zo:u64,zw:u64,zn:u64,zi:u64,zword:u64,z_mix:bool,zis_bool:bool,ztarget:u64)
  one_bool:u64 = const() {"value":1}
  zero_bool:u64 = const() {"value":0}
  true_bits:u64 = call(ztarget) {"function":"where_bool_pair_true_bits"}
  target_known:bool = compare.ne(true_bits,zero_bool)
  canonical:bool = compare.le(zword,one_bool)
  legal_input:bool = bool.and(canonical,zis_bool)
  legal_bool:bool = bool.and(legal_input,target_known)
  positive:bool = compare.eq(zword,one_bool)
  converted_raw:u64 = select(positive,true_bits,zero_bool)
  branch legal_bool store(zs,zm,zx,zy,zo,zw,zn,zi,converted_raw,z_mix) bad(zs)
