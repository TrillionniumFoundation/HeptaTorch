fn nz_command(r:readbuf,o:mutbuf,s:mutbuf,a:u64,op:u64,p:u64) -> (u64) effects(read,write,trap) entry(wire)
block wire(r:readbuf,o:mutbuf,s:mutbuf,a:u64,op:u64,p:u64)
  w8:u64 = const() {"value":8}
  w16:u64 = const() {"value":16}
  w56:u64 = const() {"value":56}
  w0:u64 = const() {"value":0}
  xa:u64 = int.add.wrap(a,w8)
  pa:u64 = int.add.wrap(a,w16)
  pe:u64 = int.add.wrap(a,w56)
  xt:u64 = call(r,xa) {"function":"getr"}
  jump padding(r,o,s,op,p,xt,pa,pe,w0,w8)
block padding(pr:readbuf,po:mutbuf,ps:mutbuf,pop:u64,pp:u64,px:u64,pa0:u64,pend:u64,pz:u64,pstep:u64)
  more:bool = compare.le(pa0,pend)
  branch more pad_word(pr,po,ps,pop,pp,px,pa0,pend,pz,pstep) resolve_x(po,ps,pop,pp,px,pz)
block pad_word(qr:readbuf,qo:mutbuf,qs:mutbuf,qop:u64,qp:u64,qx:u64,qa:u64,qend:u64,qz:u64,qstep:u64)
  raw:u64 = call(qr,qa) {"function":"getr"}
  zero:bool = compare.eq(raw,qz)
  next:u64 = int.add.wrap(qa,qstep)
  branch zero padding(qr,qo,qs,qop,qp,qx,next,qend,qz,qstep) reject_pad(qo,qs,qop,qp)
block reject_pad(bo:mutbuf,bs:mutbuf,bop:u64,bp:u64)
  b7:u64 = const() {"value":7}
  failure:u64 = call(bs,b7) {"function":"nz_error"}
  jump finish(bo,bs,bop,bp,failure)
block resolve_x(xo:mutbuf,xs:mutbuf,xop:u64,xp:u64,xtok:u64,x0:u64)
  xn:u64 = call(xs,xtok) {"function":"public_node"}
  xpass:bool = compare.ne(xn,x0)
  branch xpass execute(xo,xs,xop,xp,xn) finish(xo,xs,xop,xp,x0)
block execute(eo:mutbuf,es:mutbuf,eop:u64,ep:u64,ex:u64)
  made:u64 = call(es,ex) {"function":"nz_tensor"}
  jump finish(eo,es,eop,ep,made)
block finish(fo:mutbuf,fs:mutbuf,fop:u64,fp:u64,made_node:u64)
  off:u64 = const() {"value":8}
  prior:u64 = call(fs,off) {"function":"getw"}
  written:u64 = call(fs,fo,fp,fop,prior,made_node) {"function":"shape_command_finish"}
  ret(written)
end
