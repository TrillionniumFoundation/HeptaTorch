block bf_pair_tags(bps:mutbuf,bpm:u64,bpx:u64,bpy:u64,bxt:u64,byt:u64,bmask_ok:bool)
  bf32:u64 = const() {"value":0}
  bf16:u64 = const() {"value":6}
  bx_f32:bool = compare.eq(bxt,bf32)
  by_f32:bool = compare.eq(byt,bf32)
  bx_bf16:bool = compare.eq(bxt,bf16)
  by_bf16:bool = compare.eq(byt,bf16)
  forward_b:bool = bool.and(bx_f32,by_bf16)
  reverse_b:bool = bool.and(bx_bf16,by_f32)
  pair_b:bool = bool.or(forward_b,reverse_b)
  accept_b:bool = bool.and(bmask_ok,pair_b)
  branch accept_b wide_ranks(bps,bpm,bpx,bpy,bxt,byt,bf32) bad(bps)
