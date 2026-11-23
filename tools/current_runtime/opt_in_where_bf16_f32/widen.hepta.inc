block bf_pair_convert(bcs:mutbuf,bcm:u64,bcx:u64,bcy:u64,bco:u64,bcw:u64,bcn:u64,bci:u64,bcword:u64,bcmode:bool,bcbool:bool,bcsrc:u64,bctgt:u64)
  bf16_src:u64 = const() {"value":6}
  f32_tgt:u64 = const() {"value":0}
  src_bf16:bool = compare.eq(bcsrc,bf16_src)
  dst_f32:bool = compare.eq(bctgt,f32_tgt)
  exact_bf:bool = bool.and(src_bf16,dst_f32)
  branch exact_bf bf16_to_f32(bcs,bcm,bcx,bcy,bco,bcw,bcn,bci,bcword,bcmode) convert_bool(bcs,bcm,bcx,bcy,bco,bcw,bcn,bci,bcword,bcmode,bcbool,bctgt)
block bf16_to_f32(bws:mutbuf,bwm:u64,bwx:u64,bwy:u64,bwo:u64,bww:u64,bwn:u64,bwi:u64,bwword:u64,bwmode:bool)
  wide_bf_word:u64 = call(bwword) {"function":"half_widen_bf16"}
  jump store(bws,bwm,bwx,bwy,bwo,bww,bwn,bwi,wide_bf_word,bwmode)
