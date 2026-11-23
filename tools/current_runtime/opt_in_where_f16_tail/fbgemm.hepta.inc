fn where_f16_fbgemm_profile_word(raw:u64,source_index:u64,source_count:u64) -> (u64) effects(read,write,trap) entry(start)
block start(raw:u64,source_index:u64,source_count:u64)
  zero:u64 = const() {"value":0}
  lanes:u64 = const() {"value":8}
  tail:u64 = int.rem.checked(source_count,lanes)
  cutoff:u64 = int.sub.wrap(source_count,tail)
  scalar_tail:bool = compare.ge(source_index,cutoff)
  exponent_mask:u64 = const() {"value":31744}
  fraction_mask:u64 = const() {"value":1023}
  raw_exp:u64 = bit.and(raw,exponent_mask)
  raw_frac:u64 = bit.and(raw,fraction_mask)
  exp_all:bool = compare.eq(raw_exp,exponent_mask)
  frac_nonzero:bool = compare.ne(raw_frac,zero)
  is_nan:bool = bool.and(exp_all,frac_nonzero)
  scalar_nan:bool = bool.and(scalar_tail,is_nan)
  branch scalar_nan canonical() general(raw)
block canonical()
  positive_nan:u64 = const() {"value":2147483647}
  ret(positive_nan)
block general(input:u64)
  widened:u64 = call(input) {"function":"widen_f16"}
  ret(widened)
end
