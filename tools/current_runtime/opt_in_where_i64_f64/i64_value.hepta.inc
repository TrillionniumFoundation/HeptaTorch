block i64_f64_convert(ics:mutbuf,icm:u64,icx:u64,icy:u64,ico:u64,icw:u64,icn:u64,ici:u64,icword:u64,icmode:bool,icbool:bool,icsrc:u64,ictgt:u64,icselected:u64,icsp:u64)
  int_src:u64 = const() {"value":2}
  double_tgt:u64 = const() {"value":4}
  exact_int:bool = compare.eq(icsrc,int_src)
  exact_double:bool = compare.eq(ictgt,double_tgt)
  exact_pair_out:bool = bool.and(exact_int,exact_double)
  branch exact_pair_out i64_into_f64(ics,icm,icx,icy,ico,icw,icn,ici,icword,icmode) bf_pair_convert(ics,icm,icx,icy,ico,icw,icn,ici,icword,icmode,icbool,icsrc,ictgt,icselected,icsp)
block i64_into_f64(ivs:mutbuf,ivm:u64,ivx:u64,ivy:u64,ivo:u64,ivw:u64,ivn:u64,ivi:u64,ivword:u64,ivmode:bool)
  f64_bits:u64 = call(ivword) {"function":"integer_i64_to_f64"}
  jump store(ivs,ivm,ivx,ivy,ivo,ivw,ivn,ivi,f64_bits,ivmode)
