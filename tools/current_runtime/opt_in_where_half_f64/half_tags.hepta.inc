block half_f64_pair(hps:mutbuf,hpm:u64,hpx:u64,hpy:u64,hxt:u64,hyt:u64,hmask:bool)
  hft:u64 = const() {"value":5}
  hbt:u64 = const() {"value":6}
  hdt:u64 = const() {"value":4}
  hxf:bool = compare.eq(hxt,hft)
  hxb:bool = compare.eq(hxt,hbt)
  hyf:bool = compare.eq(hyt,hft)
  hyb:bool = compare.eq(hyt,hbt)
  hxh:bool = bool.or(hxf,hxb)
  hyh:bool = bool.or(hyf,hyb)
  hxd:bool = compare.eq(hxt,hdt)
  hyd:bool = compare.eq(hyt,hdt)
  hleft:bool = bool.and(hxh,hyd)
  hright:bool = bool.and(hxd,hyh)
  hpair:bool = bool.or(hleft,hright)
  hok:bool = bool.and(hmask,hpair)
  branch hok wide_ranks(hps,hpm,hpx,hpy,hxt,hyt,hdt) bad(hps)
