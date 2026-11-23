block g4(gs4:mutbuf,gx4:u64,gd4:u64)
  fone:u64 = const() {"value":1}
  fzero:u64 = const() {"value":0}
  ffour:u64 = const() {"value":4}
  flim:u64 = const() {"value":262144}
  f8:u64 = const() {"value":8}
  f96:u64 = const() {"value":96}
  f104:u64 = const() {"value":104}
  f112:u64 = const() {"value":112}
  f120:u64 = const() {"value":120}
  f128:u64 = const() {"value":128}
  f136:u64 = const() {"value":136}
  f144:u64 = const() {"value":144}
  f152:u64 = const() {"value":152}
  np4:u64 = int.add.wrap(gd4,f8)
  nd0:u64 = int.add.wrap(gd4,f96)
  nd1:u64 = int.add.wrap(gd4,f104)
  nd2:u64 = int.add.wrap(gd4,f112)
  nd3:u64 = int.add.wrap(gd4,f120)
  sp4a:u64 = int.add.wrap(gd4,f128)
  sp4b:u64 = int.add.wrap(gd4,f136)
  sp4c:u64 = int.add.wrap(gd4,f144)
  sp4d:u64 = int.add.wrap(gd4,f152)
  n4:u64 = call(gs4,np4) {"function":"getw"}
  fa:u64 = call(gs4,nd0) {"function":"getw"}
  fb:u64 = call(gs4,nd1) {"function":"getw"}
  fc4:u64 = call(gs4,nd2) {"function":"getw"}
  fd4:u64 = call(gs4,nd3) {"function":"getw"}
  fsa:u64 = call(gs4,sp4a) {"function":"getw"}
  fsb:u64 = call(gs4,sp4b) {"function":"getw"}
  fsc:u64 = call(gs4,sp4c) {"function":"getw"}
  fsd:u64 = call(gs4,sp4d) {"function":"getw"}
  tag4:u64 = call(gs4,gx4) {"function":"tensor_dtype"}
  fak:bool = compare.le(fa,flim)
  fbk:bool = compare.le(fb,flim)
  fck:bool = compare.le(fc4,flim)
  fdk:bool = compare.le(fd4,flim)
  fab:bool = bool.and(fak,fbk)
  fcd:bool = bool.and(fck,fdk)
  fdim:bool = bool.and(fab,fcd)
  fbool:bool = compare.eq(tag4,fone)
  fall:bool = bool.and(fdim,fbool)
  branch fall g4layout(gs4,gx4,n4,fa,fb,fc4,fd4,fsa,fsb,fsc,fsd,flim,fone,fzero,ffour) invalid(gs4)
block g4layout(l4s:mutbuf,l4x:u64,l4n:u64,l4a:u64,l4b:u64,l4c:u64,l4d:u64,l4sa:u64,l4sb:u64,l4sc:u64,l4sd:u64,l4cap:u64,l4o:u64,l4z:u64,l4four:u64)
  za:bool = compare.eq(l4a,l4z)
  zb:bool = compare.eq(l4b,l4z)
  zc:bool = compare.eq(l4c,l4z)
  zd:bool = compare.eq(l4d,l4z)
  zab:bool = bool.or(za,zb)
  zcd:bool = bool.or(zc,zd)
  zany:bool = bool.or(zab,zcd)
  mb:u64 = select(zb,l4o,l4b)
  mc4:u64 = select(zc,l4o,l4c)
  md4:u64 = select(zd,l4o,l4d)
  mid_stride:u64 = int.mul.wrap(mc4,md4)
  outer_st:u64 = int.mul.wrap(mb,mid_stride)
  ok4a:bool = compare.eq(l4sa,outer_st)
  ok4b:bool = compare.eq(l4sb,mid_stride)
  ok4c:bool = compare.eq(l4sc,md4)
  ok4d:bool = compare.eq(l4sd,l4o)
  ok4ab:bool = bool.and(ok4a,ok4b)
  ok4cd:bool = bool.and(ok4c,ok4d)
  ok4:bool = bool.and(ok4ab,ok4cd)
  branch ok4 g4volume(l4s,l4x,l4n,l4a,l4b,l4c,l4d,l4cap,l4o,l4z,l4four,zany) invalid(l4s)
block g4volume(v4s:mutbuf,v4x:u64,v4n:u64,v4a:u64,v4b:u64,v4c:u64,v4d:u64,v4cap:u64,v4o:u64,v4z:u64,v4f:u64,v4e:bool)
  branch v4e g4exact(v4s,v4x,v4n,v4d,v4cap,v4o,v4z,v4f,v4z) g4positive(v4s,v4x,v4n,v4a,v4b,v4c,v4d,v4cap,v4o,v4f)
block g4positive(u4s:mutbuf,u4x:u64,u4n:u64,u4a:u64,u4b:u64,u4c:u64,u4d:u64,u4cap:u64,u4o:u64,u4f:u64)
  block_size:u64 = int.mul.wrap(u4c,u4d)
  tail_size:u64 = int.mul.wrap(u4b,block_size)
  tail_fits:bool = compare.le(tail_size,u4cap)
  branch tail_fits g4multiply(u4s,u4x,u4n,u4a,u4d,u4cap,u4o,u4f,block_size,tail_size) capacity(u4s)
block g4multiply(m4s:mutbuf,m4x:u64,m4n:u64,m4a:u64,m4d:u64,m4cap:u64,m4o:u64,m4f:u64,m4blk:u64,m4tail:u64)
  n_total:u64 = int.mul.wrap(m4a,m4tail)
  jump g4exact(m4s,m4x,m4n,m4d,m4cap,m4o,n_total,m4f,m4blk)
block g4exact(e4s:mutbuf,e4x:u64,e4n:u64,e4d:u64,e4cap:u64,e4o:u64,e4sum:u64,e4f:u64,e4blk:u64)
  exact4:bool = compare.eq(e4n,e4sum)
  branch exact4 limit(e4s,e4x,e4n,e4d,e4cap,e4o,e4o,e4f,e4blk) invalid(e4s)
