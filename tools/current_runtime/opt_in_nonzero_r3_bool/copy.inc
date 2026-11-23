block spill(ss:mutbuf,sx:u64,sn:u64,sc:u64,sk:u64,so:u64,si:u64,sj:u64,si2:u64,stag:u64,scols:u64,splane:u64)
  sth:u64 = const() {"value":3}
  wide:bool = compare.eq(scols,sth)
  branch wide coords3(ss,sx,sn,sc,sk,so,si,sj,si2,stag,scols,splane) coords2(ss,sx,sn,sc,sk,so,si,sj,si2,stag,scols,splane)
block coords2(ds2:mutbuf,dx2:u64,dn2:u64,dc2:u64,dk2:u64,do2:u64,di2:u64,dj2:u64,de2:u64,dt2:u64,dcols2:u64,dp2:u64)
  ar:u64 = int.div.checked(di2,dc2)
  ac:u64 = int.rem.checked(di2,dc2)
  dz:u64 = const() {"value":0}
  no_third:bool = compare.ne(dz,dz)
  jump pair(ds2,dx2,dn2,dc2,dk2,do2,dj2,de2,dt2,dcols2,dp2,ar,ac,dz,no_third)
block coords3(ds3:mutbuf,dx3:u64,dn3:u64,dc3:u64,dk3:u64,do3:u64,di3:u64,dj3:u64,de3:u64,dt3:u64,dcols3:u64,dp3:u64)
  dr:u64 = int.div.checked(di3,dp3)
  dl:u64 = int.rem.checked(di3,dp3)
  dm:u64 = int.div.checked(dl,dc3)
  dbc:u64 = int.rem.checked(dl,dc3)
  dthree:u64 = const() {"value":3}
  threecol:bool = compare.eq(dcols3,dthree)
  jump pair(ds3,dx3,dn3,dc3,dk3,do3,dj3,de3,dt3,dcols3,dp3,dr,dm,dbc,threecol)
block pair(pws:mutbuf,pwx:u64,pwn:u64,pwc:u64,pwk:u64,pwo:u64,pwj:u64,pwnext:u64,pwt:u64,pcols:u64,pplane:u64,pr:u64,pcv:u64,pxv:u64,pthird:bool)
  addrj:u64 = int.add.wrap(pwj,pwk)
  addr0:u64 = call(pws,pwo,pwj) {"function":"taddr"}
  addr1:u64 = call(pws,pwo,addrj) {"function":"taddr"}
  sw:u64 = const() {"value":8}
  sone:u64 = const() {"value":1}
  ack0:u64 = call(pws,addr0,sw,pr) {"function":"typed_store"}
  ack1:u64 = call(pws,addr1,sw,pcv) {"function":"typed_store"}
  sj2:u64 = int.add.wrap(pwj,sone)
  branch pthird third(pws,pwx,pwn,pwc,pwk,pwo,pwnext,sj2,pwt,pcols,pplane,addrj,pxv,sw) loop(pws,pwx,pwn,pwc,pwk,pwo,pwnext,sj2,pwt,pcols,pplane)
block third(ths:mutbuf,thx:u64,thn:u64,thc:u64,thk:u64,tho:u64,thi:u64,thj:u64,tht:u64,thcols:u64,thplane:u64,thaddr:u64,thv:u64,thw:u64)
  j3:u64 = int.add.wrap(thaddr,thk)
  addr2:u64 = call(ths,tho,j3) {"function":"taddr"}
  ack2:u64 = call(ths,addr2,thw,thv) {"function":"typed_store"}
  jump loop(ths,thx,thn,thc,thk,tho,thi,thj,tht,thcols,thplane)
