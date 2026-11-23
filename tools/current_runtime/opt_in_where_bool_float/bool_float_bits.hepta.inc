fn where_bool_pair_true_bits(t:u64) -> (u64) effects(read,write,trap) entry(start)
block start(t:u64)
  t0:u64 = const() {"value":0}
  t2:u64 = const() {"value":2}
  t4:u64 = const() {"value":4}
  t5:u64 = const() {"value":5}
  t6:u64 = const() {"value":6}
  o0:u64 = const() {"value":1065353216}
  o2:u64 = const() {"value":1}
  o4:u64 = const() {"value":4607182418800017408}
  o5:u64 = const() {"value":15360}
  o6:u64 = const() {"value":16256}
  s0:bool = compare.eq(t,t0)
  s2:bool = compare.eq(t,t2)
  s4:bool = compare.eq(t,t4)
  s5:bool = compare.eq(t,t5)
  s6:bool = compare.eq(t,t6)
  v0:u64 = select(s0,o0,t0)
  v2:u64 = select(s2,o2,v0)
  v4:u64 = select(s4,o4,v2)
  v5:u64 = select(s5,o5,v4)
  v6:u64 = select(s6,o6,v5)
  ret(v6)
end
