block pick34(ks4:mutbuf,kx4:u64,kn4:u64,kc4:u64,kk4:u64,ko4:u64,ki4:u64,kj4:u64,ke4:u64,kt4:u64,kcols4:u64,kp4:u64)
  kth:u64 = const() {"value":3}
  k3:bool = compare.eq(kcols4,kth)
  branch k3 coords3(ks4,kx4,kn4,kc4,kk4,ko4,ki4,kj4,ke4,kt4,kcols4,kp4) coords4(ks4,kx4,kn4,kc4,kk4,ko4,ki4,kj4,ke4,kt4,kcols4,kp4)
block coords4(ds4:mutbuf,dx4:u64,dn4:u64,dc4:u64,dk4:u64,do4:u64,di4:u64,dj4:u64,de4:u64,dt4:u64,dcols4:u64,dp4:u64)
  ixdesc:u64 = call(ds4,dx4) {"function":"dptr"}
  e104:u64 = const() {"value":104}
  idx1:u64 = int.add.wrap(ixdesc,e104)
  axmiddle:u64 = call(ds4,idx1) {"function":"getw"}
  tz4:u64 = const() {"value":0}
  good_mid:bool = compare.ne(axmiddle,tz4)
  four4:u64 = const() {"value":4}
  right_c:bool = compare.eq(dcols4,four4)
  both4:bool = bool.and(good_mid,right_c)
  branch both4 c4write(ds4,dx4,dn4,dc4,dk4,do4,di4,dj4,de4,dt4,dcols4,dp4,axmiddle,right_c) invalid(ds4)
block c4write(c4s:mutbuf,c4x:u64,c4n:u64,c4inner:u64,c4k:u64,c4out:u64,c4i:u64,c4j:u64,c4next:u64,c4tag:u64,c4cols:u64,c4plane:u64,c4mid:u64,c4yes:bool)
  hi4:u64 = int.div.checked(c4i,c4plane)
  lo4:u64 = int.rem.checked(c4i,c4plane)
  outax:u64 = int.div.checked(hi4,c4mid)
  midax:u64 = int.rem.checked(hi4,c4mid)
  nextax:u64 = int.div.checked(lo4,c4inner)
  endax:u64 = int.rem.checked(lo4,c4inner)
  three4:u64 = const() {"value":3}
  word4:u64 = const() {"value":8}
  column4:u64 = int.mul.wrap(c4k,three4)
  target4:u64 = int.add.wrap(c4j,column4)
  addr4:u64 = call(c4s,c4out,target4) {"function":"taddr"}
  store4:u64 = call(c4s,addr4,word4,endax) {"function":"typed_store"}
  jump pair(c4s,c4x,c4n,c4inner,c4k,c4out,c4j,c4next,c4tag,c4cols,c4plane,outax,midax,nextax,c4yes)
