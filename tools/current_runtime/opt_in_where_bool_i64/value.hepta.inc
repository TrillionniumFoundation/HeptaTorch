  branch read_ok promote_word(rs,rm,rx,ry,ro,rw,rn,ri,rsource,word,r_mix) bad(rs)
block promote_word(ns:mutbuf,nm:u64,nx:u64,ny:u64,no:u64,nw:u64,nn:u64,ni:u64,nsource:u64,nword:u64,n_mix:bool)
  branch n_mix check_word(ns,nm,nx,ny,no,nw,nn,ni,nsource,nword,n_mix) store(ns,nm,nx,ny,no,nw,nn,ni,nword,n_mix)
block check_word(ps0:mutbuf,pm0:u64,px0:u64,py0:u64,po0:u64,pw0:u64,pn0:u64,pi0:u64,psrc:u64,pword:u64,check_mix:bool)
  stype:u64 = call(ps0,psrc) {"function":"tensor_dtype"}
  bool_src:u64 = const() {"value":1}
  i64_src:u64 = const() {"value":2}
  is_i64:bool = compare.eq(stype,i64_src)
  is_bool:bool = compare.eq(stype,bool_src)
  branch is_i64 store(ps0,pm0,px0,py0,po0,pw0,pn0,pi0,pword,check_mix) convert_bool(ps0,pm0,px0,py0,po0,pw0,pn0,pi0,pword,check_mix,is_bool)
block convert_bool(zs:mutbuf,zm:u64,zx:u64,zy:u64,zo:u64,zw:u64,zn:u64,zi:u64,zword:u64,z_mix:bool,zis_bool:bool)
  one_bool:u64 = const() {"value":1}
  zero_bool:u64 = const() {"value":0}
  canonical:bool = compare.le(zword,one_bool)
  legal_bool:bool = bool.and(canonical,zis_bool)
  positive:bool = compare.eq(zword,one_bool)
  bool_as_i64_raw:u64 = select(positive,one_bool,zero_bool)
  branch legal_bool store(zs,zm,zx,zy,zo,zw,zn,zi,bool_as_i64_raw,z_mix) bad(zs)
