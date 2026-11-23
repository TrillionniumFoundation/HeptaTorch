block i64_f32_convert(ifs:mutbuf,ifm:u64,ifx:u64,ify:u64,ifo:u64,ifw:u64,ifn:u64,ifi0:u64,ifword:u64,ifmode:bool,ifbool:bool,ifsrc:u64,iftgt:u64,ifselected:u64,ifsp:u64)
  ifi64:u64 = const() {"value":2}
  iff32:u64 = const() {"value":0}
  iffrom:bool = compare.eq(ifsrc,ifi64)
  ifinto:bool = compare.eq(iftgt,iff32)
  ifok:bool = bool.and(iffrom,ifinto)
  branch ifok i64_to_f32(ifs,ifm,ifx,ify,ifo,ifw,ifn,ifi0,ifword,ifmode) bf_pair_convert(ifs,ifm,ifx,ify,ifo,ifw,ifn,ifi0,ifword,ifmode,ifbool,ifsrc,iftgt,ifselected,ifsp)
block i64_to_f32(its:mutbuf,itm:u64,itx:u64,ity:u64,ito:u64,itw:u64,itn:u64,iti:u64,itword:u64,itmode:bool)
  itf32:u64 = call(itword) {"function":"integer_i64_to_f32"}
  jump store(its,itm,itx,ity,ito,itw,itn,iti,itf32,itmode)
