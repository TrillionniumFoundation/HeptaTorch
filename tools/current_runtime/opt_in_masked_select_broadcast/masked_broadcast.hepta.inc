fn ms_bshape(s:mutbuf,xd:u64,md:u64,xr:u64,mr:u64) -> (u64) effects(read,write,trap) entry(bstart)
block bstart(s:mutbuf,xd:u64,md:u64,xr:u64,mr:u64)
  z:u64 = const() {"value":0}
  one:u64 = const() {"value":1}
  eight:u64 = const() {"value":8}
  dims:u64 = const() {"value":96}
  mnp:u64 = int.add.wrap(md,eight)
  mn:u64 = call(s,mnp) {"function":"getw"}
  mok:u64 = call(s,md,mr,mn) {"function":"ms_geometry"}
  mask_len:u64 = const() {"value":262144}
  mcap:bool = compare.le(mn,mask_len)
  mshape_ok:bool = compare.ne(mok,z)
  mf:bool = bool.and(mcap,mshape_ok)
  xdim:u64 = int.add.wrap(xd,dims)
  mdim:u64 = int.add.wrap(md,dims)
  branch mf baxis(s,xdim,mdim,xr,mr,z,one,eight) bfail(z)
block baxis(as:mutbuf,ax:u64,am:u64,ar:u64,an:u64,az:u64,ao:u64,ae:u64)
  more:bool = compare.gt(an,az)
  branch more bdim(as,ax,am,ar,an,az,ao,ae) bfail(ao)
block bdim(ds:mutbuf,dx:u64,dm:u64,dr:u64,dn:u64,dz:u64,do:u64,de:u64)
  xi:u64 = int.sub.wrap(dr,do)
  mi:u64 = int.sub.wrap(dn,do)
  xo:u64 = int.mul.wrap(xi,de)
  mo:u64 = int.mul.wrap(mi,de)
  xp:u64 = int.add.wrap(dx,xo)
  mp:u64 = int.add.wrap(dm,mo)
  xe:u64 = call(ds,xp) {"function":"getw"}
  me:u64 = call(ds,mp) {"function":"getw"}
  same:bool = compare.eq(xe,me)
  single:bool = compare.eq(me,do)
  fits:bool = bool.or(same,single)
  branch fits baxis(ds,dx,dm,xi,mi,dz,do,de) bfail(dz)
block bfail(v:u64)
  ret(v)
end
fn ms_bindex(s:mutbuf,x:u64,m:u64,i:u64,imap_mode:u64) -> (u64) effects(read,write,trap) entry(istart)
block istart(s:mutbuf,x:u64,m:u64,i:u64,imap_mode:u64)
  z:u64 = const() {"value":0}
  direct:bool = compare.eq(imap_mode,z)
  branch direct iresult(i) iprep(s,x,m,i,z)
block iprep(ps:mutbuf,px:u64,pm:u64,pi:u64,pz:u64)
  po:u64 = const() {"value":1}
  pe:u64 = const() {"value":8}
  pd:u64 = const() {"value":96}
  stop:u64 = const() {"value":262145}
  xd:u64 = call(ps,px) {"function":"dptr"}
  md:u64 = call(ps,pm) {"function":"dptr"}
  xr:u64 = call(ps,xd) {"function":"getw"}
  mr:u64 = call(ps,md) {"function":"getw"}
  xbase:u64 = int.add.wrap(xd,pd)
  mbase:u64 = int.add.wrap(md,pd)
  nc:u64 = int.add.wrap(md,pe)
  mn:u64 = call(ps,nc) {"function":"getw"}
  jump iaxis(ps,xbase,mbase,xr,mr,pi,po,pz,mn,stop)
block iaxis(as:mutbuf,ax:u64,am:u64,ar:u64,an:u64,aq:u64,astr:u64,aj:u64,alim:u64,abad:u64)
  az:u64 = const() {"value":0}
  more:bool = compare.gt(an,az)
  branch more idims(as,ax,am,ar,an,aq,astr,aj,alim,abad,az) ibounds(aj,alim,abad)
block idims(ds:mutbuf,dx:u64,dm:u64,dr:u64,dn:u64,dq:u64,dstr:u64,dj:u64,dlim:u64,dbad:u64,dz:u64)
  one:u64 = const() {"value":1}
  eight:u64 = const() {"value":8}
  xi:u64 = int.sub.wrap(dr,one)
  mi:u64 = int.sub.wrap(dn,one)
  xo:u64 = int.mul.wrap(xi,eight)
  mo:u64 = int.mul.wrap(mi,eight)
  xp:u64 = int.add.wrap(dx,xo)
  mp:u64 = int.add.wrap(dm,mo)
  xe:u64 = call(ds,xp) {"function":"getw"}
  me:u64 = call(ds,mp) {"function":"getw"}
  nonzero:bool = compare.gt(xe,dz)
  branch nonzero ifold(ds,dx,dm,xi,mi,dq,dstr,dj,dlim,dbad,dz,one,xe,me) iresult(dbad)
block ifold(fs:mutbuf,fx:u64,fm:u64,fxi:u64,fmi:u64,fq:u64,fstr:u64,fj:u64,flim:u64,fbad:u64,fz:u64,fo:u64,fxe:u64,fme:u64)
  ia_single:bool = compare.eq(fme,fo)
  logpos:u64 = int.rem.checked(fq,fxe)
  remain:u64 = int.div.checked(fq,fxe)
  idx:u64 = select(ia_single,fz,logpos)
  inc:u64 = int.mul.wrap(idx,fstr)
  nextj:u64 = int.add.wrap(fj,inc)
  nextstr:u64 = int.mul.wrap(fstr,fme)
  jump iaxis(fs,fx,fm,fxi,fmi,remain,nextstr,nextj,flim,fbad)
block ibounds(j:u64,lim:u64,bad:u64)
  fits:bool = compare.lt(j,lim)
  out:u64 = select(fits,j,bad)
  ret(out)
block iresult(rv:u64)
  ret(rv)
end
