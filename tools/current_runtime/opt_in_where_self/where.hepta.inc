fn where_ordinary(ws:mutbuf,wn:u64,ww:u64) -> (u64) effects(read,write,trap) entry(check)
block check(ws:mutbuf,wn:u64,ww:u64)
  wd:u64 = call(ws,wn) {"function":"dptr"}
  so:u64 = const() {"value":216}
  vo:u64 = const() {"value":224}
  sa:u64 = int.add.wrap(wd,so)
  va:u64 = int.add.wrap(wd,vo)
  stor:u64 = call(ws,sa) {"function":"getw"}
  vers:u64 = call(ws,va) {"function":"getw"}
  layout:u64 = call(ws,wd,wn,stor,vers,ww) {"function":"logical_out_layout"}
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
fn where_self_tensor(s:mutbuf,m:u64,x:u64,y:u64) -> (u64) effects(read,write,trap) entry(tags)
block tags(s:mutbuf,m:u64,x:u64,y:u64)
  mt:u64 = call(s,m) {"function":"tensor_dtype"}
  xt:u64 = call(s,x) {"function":"tensor_dtype"}
  yt:u64 = call(s,y) {"function":"tensor_dtype"}
  bool_tag:u64 = const() {"value":1}
  private_tag:u64 = const() {"value":3}
  max_public:u64 = const() {"value":6}
  mask_bool:bool = compare.eq(mt,bool_tag)
  values_match:bool = compare.eq(xt,yt)
  values_not_private:bool = compare.ne(xt,private_tag)
  values_bounded:bool = compare.le(xt,max_public)
  valid0:bool = bool.and(mask_bool,values_match)
  valid1:bool = bool.and(values_not_private,values_bounded)
  valid_tags:bool = bool.and(valid0,valid1)
  branch valid_tags roots(s,m,x,y,xt,bool_tag) bad(s)
block roots(rs:mutbuf,rm:u64,rx:u64,ry:u64,rt:u64,rb:u64)
  width:u64 = call(rt) {"function":"dtype_width"}
  mok:u64 = call(rs,rm,rb) {"function":"where_ordinary"}
  xok:u64 = call(rs,rx,width) {"function":"where_ordinary"}
  yok:u64 = call(rs,ry,width) {"function":"where_ordinary"}
  all01:u64 = bit.and(mok,xok)
  all012:u64 = bit.and(all01,yok)
  zero:u64 = const() {"value":0}
  roots_ok:bool = compare.ne(all012,zero)
  branch roots_ok shapes(rs,rm,rx,ry,rt,width) bad(rs)
block shapes(ss:mutbuf,sm:u64,sx:u64,sy:u64,st:u64,sw:u64)
  mx:u64 = call(ss,sm,sx) {"function":"incoming_same_shape"}
  xy:u64 = call(ss,sx,sy) {"function":"incoming_same_shape"}
  both:u64 = bit.and(mx,xy)
  zshape:u64 = const() {"value":0}
  shapes_ok:bool = compare.ne(both,zshape)
  branch shapes_ok allocate(ss,sm,sx,sy,st,sw) bad(ss)
block allocate(as:mutbuf,am:u64,ax:u64,ay:u64,at:u64,aw:u64)
  desc:u64 = call(as,ax) {"function":"dptr"}
  rank:u64 = call(as,desc) {"function":"getw"}
  count_off:u64 = const() {"value":8}
  count_addr:u64 = int.add.wrap(desc,count_off)
  an:u64 = call(as,count_addr) {"function":"getw"}
  dims_off:u64 = const() {"value":96}
  dims:u64 = int.add.wrap(desc,dims_off)
  zero_alloc:u64 = const() {"value":0}
  one_alloc:u64 = const() {"value":1}
  ao:u64 = call(as,rank,dims,zero_alloc,one_alloc,at) {"function":"typed_descriptor"}
  has_result:bool = compare.ne(ao,zero_alloc)
  branch has_result check_result(as,am,ax,ay,ao,an,aw,at) zero_return()
block check_result(cs:mutbuf,cm:u64,cx:u64,cy:u64,co:u64,cn:u64,cw:u64,ct:u64)
  ro:u64 = call(cs,co,cw) {"function":"where_ordinary"}
  zresult:u64 = const() {"value":0}
  checked:bool = compare.ne(ro,zresult)
  branch checked result_metadata(cs,cm,cx,cy,co,cn,cw,ct,zresult) bad(cs)
block result_metadata(ms:mutbuf,mm:u64,mx_node:u64,my:u64,mo:u64,mn:u64,mw:u64,mtag:u64,mz:u64)
  result_dtype:u64 = call(ms,mo) {"function":"tensor_dtype"}
  result_desc:u64 = call(ms,mo) {"function":"dptr"}
  result_numel_off:u64 = const() {"value":8}
  result_numel_addr:u64 = int.add.wrap(result_desc,result_numel_off)
  result_numel:u64 = call(ms,result_numel_addr) {"function":"getw"}
  same_result_shape:u64 = call(ms,mx_node,mo) {"function":"incoming_same_shape"}
  dtype_ok:bool = compare.eq(result_dtype,mtag)
  count_ok:bool = compare.eq(result_numel,mn)
  shape_ok:bool = compare.ne(same_result_shape,mz)
  raw_meta_ok:bool = bool.and(dtype_ok,count_ok)
  result_meta_ok:bool = bool.and(raw_meta_ok,shape_ok)
  branch result_meta_ok loop(ms,mm,mx_node,my,mo,mn,mw,mz) bad(ms)
block loop(ls:mutbuf,lm:u64,lx:u64,ly:u64,lo:u64,ln:u64,lw:u64,li:u64)
  more:bool = compare.lt(li,ln)
  branch more choose(ls,lm,lx,ly,lo,ln,lw,li) done(lo)
block choose(qs:mutbuf,qm:u64,qx:u64,qy:u64,qo:u64,qn:u64,qw:u64,qi:u64)
  raw_mask:u64 = call(qs,qm,qi) {"function":"tget"}
  qz:u64 = const() {"value":0}
  qone:u64 = const() {"value":1}
  mask_status:u64 = call(qs,qz) {"function":"getw"}
  ok_read:bool = compare.eq(mask_status,qz)
  canonical:bool = compare.le(raw_mask,qone)
  ok_mask:bool = bool.and(ok_read,canonical)
  take_x:bool = compare.eq(raw_mask,qone)
  chosen:u64 = select(take_x,qx,qy)
  branch ok_mask load(qs,chosen,qx,qy,qm,qo,qn,qw,qi) bad(qs)
block load(vs:mutbuf,chosen_node:u64,vx:u64,vy:u64,vm:u64,vo_node:u64,vn:u64,vw:u64,vi:u64)
  word:u64 = call(vs,chosen_node,vi) {"function":"tget"}
  vz:u64 = const() {"value":0}
  value_status:u64 = call(vs,vz) {"function":"getw"}
  value_ok:bool = compare.eq(value_status,vz)
  branch value_ok store(vs,vx,vy,vm,vo_node,vn,vw,vi,word) bad(vs)
block store(ts:mutbuf,tx:u64,ty:u64,tm:u64,to:u64,tn:u64,byte_width:u64,ti:u64,bits:u64)
  addr:u64 = call(ts,to,ti) {"function":"taddr"}
  ok_store:u64 = call(ts,addr,byte_width,bits) {"function":"typed_store"}
  one_store:u64 = const() {"value":1}
  store_ok:bool = compare.eq(ok_store,one_store)
  next_i:u64 = int.add.wrap(ti,one_store)
  branch store_ok loop(ts,tm,tx,ty,to,tn,byte_width,next_i) bad(ts)
block bad(bs:mutbuf)
  unsupported:u64 = const() {"value":7}
  err:u64 = call(bs,unsupported) {"function":"terror"}
  ret(err)
block zero_return()
  none:u64 = const() {"value":0}
  ret(none)
block done(result:u64)
  ret(result)
end
fn where_self_command(wr:readbuf,wo:mutbuf,ws0:mutbuf,wat:u64,wop:u64,wp:u64) -> (u64) effects(read,write,trap) entry(wire)
block wire(wr:readbuf,wo:mutbuf,ws0:mutbuf,wat:u64,wop:u64,wp:u64)
  offset8:u64 = const() {"value":8}
  offset16:u64 = const() {"value":16}
  offset24:u64 = const() {"value":24}
  offset32:u64 = const() {"value":32}
  offset40:u64 = const() {"value":40}
  offset48:u64 = const() {"value":48}
  offset56:u64 = const() {"value":56}
  mpos:u64 = int.add.wrap(wat,offset8)
  xpos:u64 = int.add.wrap(wat,offset16)
  ypos:u64 = int.add.wrap(wat,offset24)
  p4:u64 = int.add.wrap(wat,offset32)
  p5:u64 = int.add.wrap(wat,offset40)
  p6:u64 = int.add.wrap(wat,offset48)
  p7:u64 = int.add.wrap(wat,offset56)
  wmask:u64 = call(wr,mpos) {"function":"getr"}
  wx:u64 = call(wr,xpos) {"function":"getr"}
  wy:u64 = call(wr,ypos) {"function":"getr"}
  reserved4:u64 = call(wr,p4) {"function":"getr"}
  reserved5:u64 = call(wr,p5) {"function":"getr"}
  reserved6:u64 = call(wr,p6) {"function":"getr"}
  reserved7:u64 = call(wr,p7) {"function":"getr"}
  group45:u64 = bit.or(reserved4,reserved5)
  group67:u64 = bit.or(reserved6,reserved7)
  reserved:u64 = bit.or(group45,group67)
  nought:u64 = const() {"value":0}
  valid_padding:bool = compare.eq(reserved,nought)
  branch valid_padding resolve(wo,ws0,wp,wop,wmask,wx,wy) malformed(wo,ws0,wp,wop)
block resolve(or_out:mutbuf,sr:mutbuf,pr:u64,opr:u64,mr:u64,xr:u64,yr:u64)
  mnode:u64 = call(sr,mr) {"function":"public_node"}
  xnode:u64 = call(sr,xr) {"function":"public_node"}
  ynode:u64 = call(sr,yr) {"function":"public_node"}
  zr:u64 = const() {"value":0}
  mr_ok:bool = compare.ne(mnode,zr)
  xr_ok:bool = compare.ne(xnode,zr)
  yr_ok:bool = compare.ne(ynode,zr)
  first_two:bool = bool.and(mr_ok,xr_ok)
  all_nodes:bool = bool.and(first_two,yr_ok)
  branch all_nodes execute(or_out,sr,pr,opr,mnode,xnode,ynode) finish(or_out,sr,pr,opr,zr)
block execute(oe:mutbuf,se:mutbuf,pe:u64,ope:u64,me:u64,xe:u64,ye:u64)
  made:u64 = call(se,me,xe,ye) {"function":"where_self_tensor"}
  jump finish(oe,se,pe,ope,made)
block malformed(ob:mutbuf,sb:mutbuf,pb:u64,opb:u64)
  invalid_format:u64 = const() {"value":7}
  drop:u64 = call(sb,invalid_format) {"function":"terror"}
  zb:u64 = const() {"value":0}
  jump finish(ob,sb,pb,opb,zb)
block finish(of:mutbuf,sf:mutbuf,pf:u64,opf:u64,mf:u64)
  prior_results:u64 = const() {"value":8}
  nf:u64 = call(sf,prior_results) {"function":"getw"}
  written:u64 = call(sf,of,pf,opf,nf,mf) {"function":"shape_command_finish"}
  ret(written)
end
