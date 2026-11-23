  la_still_and:bool = compare.eq(v139,la_opcode)
  la_or_opcode:u64 = const() {"value":133}
  la_xor_opcode:u64 = const() {"value":134}
  la_is_or:bool = compare.eq(v139,la_or_opcode)
  la_is_xor:bool = compare.eq(v139,la_xor_opcode)
  la_either_new:bool = bool.or(la_is_or,la_is_xor)
  la_chosen:bool = bool.or(la_still_and,la_either_new)
