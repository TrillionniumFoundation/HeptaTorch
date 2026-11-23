block zero_wide_canonical_empty(zes:mutbuf,zem:u64,zex:u64,zey:u64,zext:u64,zeyt:u64,zeone:u64,zef64:u64,zedim:u64,zemd:u64)
  ze_xd:u64 = call(zes,zex) {"function":"dptr"}
  ze_yd:u64 = call(zes,zey) {"function":"dptr"}
  ze_mr:u64 = call(zes,zemd) {"function":"getw"}
  ze_dr:u64 = call(zes,zedim) {"function":"getw"}
  ze_four:u64 = const() {"value":4}
  ze_mlarger:bool = compare.gt(ze_mr,ze_dr)
  ze_rank:u64 = select(ze_mlarger,ze_mr,ze_dr)
  ze_ok_rank:bool = compare.le(ze_rank,ze_four)
  branch ze_ok_rank zero_wide_canonical_roots(zes,zem,zex,zey,zext,zeyt,zeone,zef64,zedim,zemd,ze_xd,ze_yd,ze_rank) bad(zes)
block zero_wide_canonical_roots(zks:mutbuf,zkm:u64,zkx:u64,zky:u64,zkxt:u64,zkyt:u64,zkone:u64,zkf64:u64,zkdim:u64,zkmd:u64,zkxd:u64,zkyd:u64,zkrank:u64)
  zk_left:u64 = call(zks,zkx) {"function":"where_empty_input_canonical"}
  zk_right:u64 = call(zks,zky) {"function":"where_empty_input_canonical"}
  zk_mask:u64 = call(zks,zkm) {"function":"where_empty_input_canonical"}
  zk_pair:u64 = bit.and(zk_left,zk_right)
  zk_all:u64 = bit.and(zk_pair,zk_mask)
  zk_roots:bool = compare.eq(zk_all,zkone)
  branch zk_roots zero_wide_canonical_plan(zks,zkm,zkx,zky,zkxt,zkyt,zkone,zkf64,zkdim,zkmd,zkxd,zkyd,zkrank) bad(zks)
block zero_wide_canonical_plan(zps:mutbuf,zpm:u64,zpx:u64,zpy:u64,zpxt:u64,zpyt:u64,zpone:u64,zpf64:u64,zpdim:u64,zpmd:u64,zpxd:u64,zpyd:u64,zprank:u64)
  zw_empty_plan:u64 = call(zps,zpmd,zpxd,zpyd,zpdim,zprank) {"function":"where_empty_reverse_plan"}
  zw_empty_ok:bool = compare.eq(zw_empty_plan,zpone)
  branch zw_empty_ok mixed_roots(zps,zpm,zpx,zpy,zpxt,zpyt,zpone,zpf64) bad(zps)
