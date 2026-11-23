block i64_reduced_convert(urs:mutbuf,urm:u64,urx:u64,ury:u64,uro:u64,urw:u64,urn:u64,uri:u64,urword:u64,urmode:bool,urbool:bool,ursrc:u64,urtgt:u64,ursel:u64,ursp:u64)
  ur_i:u64 = const() {"value":2}
  ur_f:u64 = const() {"value":5}
  ur_b:u64 = const() {"value":6}
  ur_from:bool = compare.eq(ursrc,ur_i)
  ur_into_f:bool = compare.eq(urtgt,ur_f)
  ur_into_b:bool = compare.eq(urtgt,ur_b)
  ur_into_half:bool = bool.or(ur_into_f,ur_into_b)
  ur_admit:bool = bool.and(ur_from,ur_into_half)
  branch ur_admit i64_to_reduced(urs,urm,urx,ury,uro,urw,urn,uri,urword,urmode,ur_into_f) bf_pair_convert(urs,urm,urx,ury,uro,urw,urn,uri,urword,urmode,urbool,ursrc,urtgt,ursel,ursp)
block i64_to_reduced(uis:mutbuf,uim:u64,uix:u64,uiy:u64,uio:u64,uiw:u64,uin:u64,uii:u64,uiword:u64,uimode:bool,uito_f:bool)
  uif32:u64 = call(uiword) {"function":"integer_i64_to_f32"}
  branch uito_f i64_f16_bits(uis,uim,uix,uiy,uio,uiw,uin,uii,uif32,uimode) i64_bf16_bits(uis,uim,uix,uiy,uio,uiw,uin,uii,uif32,uimode)
block i64_f16_bits(ufs:mutbuf,ufm:u64,ufx:u64,ufy:u64,ufo:u64,ufw:u64,ufn:u64,ufi:u64,ufword:u64,ufmode:bool)
  uf_bits:u64 = call(ufword) {"function":"round_f16"}
  jump store(ufs,ufm,ufx,ufy,ufo,ufw,ufn,ufi,uf_bits,ufmode)
block i64_bf16_bits(ubs:mutbuf,ubm:u64,ubx:u64,uby:u64,ubo:u64,ubw:u64,ubn:u64,ubi:u64,ubword:u64,ubmode:bool)
  ub_bits:u64 = call(ubword) {"function":"round_bf16"}
  jump store(ubs,ubm,ubx,uby,ubo,ubw,ubn,ubi,ub_bits,ubmode)
