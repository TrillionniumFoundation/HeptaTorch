block i64_f32_pair(ifps:mutbuf,ifpm:u64,ifpx:u64,ifpy:u64,ifxt:u64,ifyt:u64,ifmask:bool)
  ifi:u64 = const() {"value":2}
  iff:u64 = const() {"value":0}
  ifxi:bool = compare.eq(ifxt,ifi)
  ifyi:bool = compare.eq(ifyt,ifi)
  ifxf:bool = compare.eq(ifxt,iff)
  ifyf:bool = compare.eq(ifyt,iff)
  ifforward:bool = bool.and(ifxi,ifyf)
  ifreverse:bool = bool.and(ifxf,ifyi)
  ifpair:bool = bool.or(ifforward,ifreverse)
  ifadmit:bool = bool.and(ifmask,ifpair)
  branch ifadmit wide_ranks(ifps,ifpm,ifpx,ifpy,ifxt,ifyt,iff) bad(ifps)
