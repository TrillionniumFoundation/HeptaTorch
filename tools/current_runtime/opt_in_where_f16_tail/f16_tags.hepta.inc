block f16_pair_tags(tps:mutbuf,tpm:u64,tpx:u64,tpy:u64,txt:u64,tyt:u64,t_mask:bool)
  f16_t:u64 = const() {"value":5}
  f32_t:u64 = const() {"value":0}
  bf16_t:u64 = const() {"value":6}
  tx_f16:bool = compare.eq(txt,f16_t)
  ty_f16:bool = compare.eq(tyt,f16_t)
  tx_f32:bool = compare.eq(txt,f32_t)
  ty_f32:bool = compare.eq(tyt,f32_t)
  tx_bf16:bool = compare.eq(txt,bf16_t)
  ty_bf16:bool = compare.eq(tyt,bf16_t)
  left_usable:bool = bool.or(tx_f32,tx_bf16)
  right_usable:bool = bool.or(ty_f32,ty_bf16)
  forward_t:bool = bool.and(tx_f16,right_usable)
  reverse_t:bool = bool.and(ty_f16,left_usable)
  pair_t:bool = bool.or(forward_t,reverse_t)
  accept_t:bool = bool.and(t_mask,pair_t)
  f16_value:u64 = select(tx_f16,tpx,tpy)
  branch accept_t f16_scope(tps,tpm,tpx,tpy,txt,tyt,f32_t,f16_value) bad(tps)
block f16_scope(fss:mutbuf,fsm:u64,fsx:u64,fsy:u64,fsxt:u64,fsyt:u64,fst:u64,fsrc:u64)
  fsdesc:u64 = call(fss,fsrc) {"function":"dptr"}
  fs_count_off:u64 = const() {"value":8}
  fs_count_addr:u64 = int.add.wrap(fsdesc,fs_count_off)
  fs_count:u64 = call(fss,fs_count_addr) {"function":"getw"}
  fs_grain:u64 = const() {"value":32768}
  one_cast_chunk:bool = compare.lt(fs_count,fs_grain)
  branch one_cast_chunk wide_ranks(fss,fsm,fsx,fsy,fsxt,fsyt,fst) bad(fss)
