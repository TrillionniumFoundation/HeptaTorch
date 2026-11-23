block pair_convert(ws1:mutbuf,wm1:u64,wx1:u64,wy1:u64,wo1:u64,ww1:u64,wn1:u64,wi1:u64,wword:u64,wmode:bool,wbool:bool,wsrc:u64,wtgt:u64)
  f32_src:u64 = const() {"value":0}
  f64_tgt:u64 = const() {"value":4}
  src32:bool = compare.eq(wsrc,f32_src)
  dst64:bool = compare.eq(wtgt,f64_tgt)
  exact_wide:bool = bool.and(src32,dst64)
  branch exact_wide f32_to_f64(ws1,wm1,wx1,wy1,wo1,ww1,wn1,wi1,wword,wmode) convert_bool(ws1,wm1,wx1,wy1,wo1,ww1,wn1,wi1,wword,wmode,wbool,wtgt)
block f32_to_f64(ws2:mutbuf,wm2:u64,wx2:u64,wy2:u64,wo2:u64,ww2:u64,wn2:u64,wi2:u64,wword2:u64,wmode2:bool)
  wide_word:u64 = call(wword2) {"function":"widen32"}
  jump store(ws2,wm2,wx2,wy2,wo2,ww2,wn2,wi2,wide_word,wmode2)
