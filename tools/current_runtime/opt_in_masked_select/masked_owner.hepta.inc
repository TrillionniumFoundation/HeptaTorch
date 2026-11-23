fn ms_owned_layout(lo_layout_s:mutbuf,lo_layout_desc:u64,lo_layout_node:u64,lo_layout_storage:u64,lo_layout_version:u64,lo_layout_width:u64) -> (u64) effects(read,write,trap) entry(layout)
block layout(lo_layout_s:mutbuf,lo_layout_desc:u64,lo_layout_node:u64,lo_layout_storage:u64,lo_layout_version:u64,lo_layout_width:u64)
  lo_layout_z:u64 = const() {"value":0}
  lo_layout_one:u64 = const() {"value":1}
  lo_layout_ranklimit:u64 = const() {"value":4}
  lo_layout_countlimit:u64 = const() {"value":262144}
  lo_layout_tag:u64 = const() {"value":9223372036854775808}
  lo_layout_rank:u64 = call(lo_layout_s,lo_layout_desc) {"function":"getw"}
  lo_layout_count_off:u64 = const() {"value":8}
  lo_layout_count_at:u64 = int.add.wrap(lo_layout_desc,lo_layout_count_off)
  lo_layout_count:u64 = call(lo_layout_s,lo_layout_count_at) {"function":"getw"}
  lo_layout_start_off:u64 = const() {"value":16}
  lo_layout_start_at:u64 = int.add.wrap(lo_layout_desc,lo_layout_start_off)
  lo_layout_start:u64 = call(lo_layout_s,lo_layout_start_at) {"function":"getw"}
  lo_layout_requires_off:u64 = const() {"value":32}
  lo_layout_requires_at:u64 = int.add.wrap(lo_layout_desc,lo_layout_requires_off)
  lo_layout_requires:u64 = call(lo_layout_s,lo_layout_requires_at) {"function":"getw"}
  lo_layout_edge_off:u64 = const() {"value":40}
  lo_layout_edge_at:u64 = int.add.wrap(lo_layout_desc,lo_layout_edge_off)
  lo_layout_edge:u64 = call(lo_layout_s,lo_layout_edge_at) {"function":"getw"}
  lo_layout_offset_off:u64 = const() {"value":232}
  lo_layout_offset_at:u64 = int.add.wrap(lo_layout_desc,lo_layout_offset_off)
  lo_layout_offset:u64 = call(lo_layout_s,lo_layout_offset_at) {"function":"getw"}
  lo_layout_viewflag_off:u64 = const() {"value":664}
  lo_layout_viewflag_at:u64 = int.add.wrap(lo_layout_desc,lo_layout_viewflag_off)
  lo_layout_viewflag:u64 = call(lo_layout_s,lo_layout_viewflag_at) {"function":"getw"}
  lo_layout_live:u64 = call(lo_layout_s,lo_layout_storage) {"function":"so_storage_live"}
  lo_layout_v_live:u64 = call(lo_layout_s,lo_layout_version) {"function":"so_version_live"}
  lo_layout_test0:bool = compare.le(lo_layout_rank,lo_layout_ranklimit)
  lo_layout_test1:bool = compare.le(lo_layout_count,lo_layout_countlimit)
  lo_layout_gate1:bool = bool.and(lo_layout_test0,lo_layout_test1)
  lo_layout_test2:bool = compare.eq(lo_layout_requires,lo_layout_z)
  lo_layout_gate2:bool = bool.and(lo_layout_gate1,lo_layout_test2)
  lo_layout_test3:bool = compare.eq(lo_layout_edge,lo_layout_z)
  lo_layout_gate3:bool = bool.and(lo_layout_gate2,lo_layout_test3)
  lo_layout_test4:bool = compare.eq(lo_layout_offset,lo_layout_z)
  lo_layout_gate4:bool = bool.and(lo_layout_gate3,lo_layout_test4)
  lo_layout_test5:bool = compare.eq(lo_layout_viewflag,lo_layout_z)
  lo_layout_gate5:bool = bool.and(lo_layout_gate4,lo_layout_test5)
  lo_layout_test6:bool = compare.ne(lo_layout_live,lo_layout_z)
  lo_layout_gate6:bool = bool.and(lo_layout_gate5,lo_layout_test6)
  lo_layout_test7:bool = compare.ne(lo_layout_v_live,lo_layout_z)
  lo_layout_gate7:bool = bool.and(lo_layout_gate6,lo_layout_test7)
  branch lo_layout_gate7 extent(lo_layout_s,lo_layout_node,lo_layout_width,lo_layout_count,lo_layout_start,lo_layout_live,lo_layout_tag,lo_layout_z) bad(lo_layout_z)
block extent(lo_extent_s:mutbuf,lo_extent_node:u64,lo_extent_width:u64,lo_extent_count:u64,lo_extent_start:u64,lo_extent_live:u64,lo_extent_tag:u64,lo_extent_z:u64)
  lo_extent_base_off:u64 = const() {"value":16}
  lo_extent_base_at:u64 = int.add.wrap(lo_extent_live,lo_extent_base_off)
  lo_extent_base:u64 = call(lo_extent_s,lo_extent_base_at) {"function":"so_uget"}
  lo_extent_size_off:u64 = const() {"value":24}
  lo_extent_size_at:u64 = int.add.wrap(lo_extent_live,lo_extent_size_off)
  lo_extent_size:u64 = call(lo_extent_s,lo_extent_size_at) {"function":"so_uget"}
  lo_extent_expected_start:u64 = int.add.wrap(lo_extent_tag,lo_extent_base)
  lo_extent_needed:u64 = int.mul.wrap(lo_extent_count,lo_extent_width)
  lo_extent_contig:u64 = call(lo_extent_s,lo_extent_node) {"function":"is_contiguous"}
  lo_extent_test0:bool = compare.ne(lo_extent_contig,lo_extent_z)
  lo_extent_test1:bool = compare.eq(lo_extent_start,lo_extent_expected_start)
  lo_extent_gate1:bool = bool.and(lo_extent_test0,lo_extent_test1)
  lo_extent_test2:bool = compare.le(lo_extent_needed,lo_extent_size)
  lo_extent_gate2:bool = bool.and(lo_extent_gate1,lo_extent_test2)
  branch lo_extent_gate2 ok() bad(lo_extent_z)
block ok()
  lo_ok_success:u64 = const() {"value":1}
  ret(lo_ok_success)
block bad(lo_bad_fail_value:u64)
  ret(lo_bad_fail_value)
end
fn ms_ordinary(ws:mutbuf,wn:u64,ww:u64) -> (u64) effects(read,write,trap) entry(check)
block check(ws:mutbuf,wn:u64,ww:u64)
  wd:u64 = call(ws,wn) {"function":"dptr"}
  so:u64 = const() {"value":216}
  vo:u64 = const() {"value":224}
  sa:u64 = int.add.wrap(wd,so)
  va:u64 = int.add.wrap(wd,vo)
  stor:u64 = call(ws,sa) {"function":"getw"}
  vers:u64 = call(ws,va) {"function":"getw"}
  layout:u64 = call(ws,wd,wn,stor,vers,ww) {"function":"ms_owned_layout"}
  z:u64 = const() {"value":0}
  good_layout:bool = compare.ne(layout,z)
  branch good_layout extent(ws,wd,stor,ww) no()
block extent(es:mutbuf,ed:u64,estore:u64,ew:u64)
  live:u64 = call(es,estore) {"function":"so_storage_live"}
  count_off:u64 = const() {"value":8}
  base_off:u64 = const() {"value":16}
  count_addr:u64 = int.add.wrap(ed,count_off)
  base_addr:u64 = int.add.wrap(live,base_off)
  count:u64 = call(es,count_addr) {"function":"getw"}
  base:u64 = call(es,base_addr) {"function":"so_uget"}
  tag:u64 = const() {"value":9223372036854775808}
  lower:bool = compare.lt(base,tag)
  branch lower bounded(es,base,count,ew,tag) no()
block bounded(bstate:mutbuf,bb:u64,bn:u64,bw:u64,bt:u64)
  available:u64 = int.sub.wrap(bt,bb)
  need:u64 = int.mul.wrap(bn,bw)
  fit:bool = compare.le(need,available)
  branch fit decode(bstate,bb,bt,need) no()
block decode(ds:mutbuf,db:u64,dt:u64,need_bytes:u64)
  zero_bytes:u64 = const() {"value":0}
  is_empty:bool = compare.eq(need_bytes,zero_bytes)
  branch is_empty yes() byte_bounds(ds,db,dt,need_bytes)
block byte_bounds(bs0:mutbuf,db0:u64,dt0:u64,size_bytes:u64)
  tagged_start:u64 = int.add.wrap(db0,dt0)
  physical_start:u64 = call(bs0,tagged_start) {"function":"object_address"}
  physical_limit_usize:usize = buffer.len(bs0)
  physical_limit:u64 = int.cast(physical_limit_usize) {"mode":"wrap"}
  owner_min:u64 = const() {"value":139264}
  beyond_header:bool = compare.ge(physical_start,owner_min)
  within:bool = compare.le(physical_start,physical_limit)
  start_ok:bool = bool.and(beyond_header,within)
  branch start_ok byte_span(physical_start,physical_limit,size_bytes) no()
block byte_span(byte_start:u64,byte_limit:u64,byte_len:u64)
  physical_room:u64 = int.sub.wrap(byte_limit,byte_start)
  physical_fit:bool = compare.le(byte_len,physical_room)
  branch physical_fit yes() no()
block yes()
  one:u64 = const() {"value":1}
  ret(one)
block no()
  zero:u64 = const() {"value":0}
  ret(zero)
end
