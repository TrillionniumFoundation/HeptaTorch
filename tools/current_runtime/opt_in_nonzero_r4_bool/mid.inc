A=b.parse_source(a)
names=['nz_rank_route','nz_rank2'];edited={}
def change_once(v,x,y):
 assert v.count(x)==1,x
 return v.replace(x,y,1)
v=A['definitions']['nz_rank_route'].decode()
v=change_once(v,'''  shared:bool = bool.or(second,third_rank)
  branch shared r2(s,x,d) r1(s,x)''','''  rfour:u64 = const() {"value":4}
  fourth_rank:bool = compare.eq(r,rfour)
  low_shared:bool = bool.or(second,third_rank)
  shared:bool = bool.or(low_shared,fourth_rank)
  branch shared r2(s,x,d) r1(s,x)''')
edited['nz_rank_route']=v.encode()
v=A['definitions']['nz_rank2'].decode()
v=change_once(v,'  branch ir3 g3(is0,ix0,id0) g2(is0,ix0,id0)','''  branch ir3 g3(is0,ix0,id0) rk24(is0,ix0,id0,rk0)
block rk24(uks:mutbuf,ukx:u64,ukd:u64,ukr:u64)
  ik4:u64 = const() {"value":4}
  ir4:bool = compare.eq(ukr,ik4)
  branch ir4 g4(uks,ukx,ukd) g2(uks,ukx,ukd)''')
v=change_once(v,'block limit(',Path(__file__).with_name('geom4.inc').read_text()+'block limit(')
v=change_once(v,'  maxk:u64 = select(isthree,max3,max2)','''  max_old:u64 = select(isthree,max3,max2)
  ac4:u64 = const() {"value":4}
  max4:u64 = const() {"value":65536}
  isfour:bool = compare.eq(alcols,ac4)
  maxk:u64 = select(isfour,max4,max_old)''')
v=change_once(v,'''  sth:u64 = const() {"value":3}
  wide:bool = compare.eq(scols,sth)
  branch wide coords3(ss,sx,sn,sc,sk,so,si,sj,si2,stag,scols,splane) coords2(ss,sx,sn,sc,sk,so,si,sj,si2,stag,scols,splane)
''','''  stwo:u64 = const() {"value":2}
  twoonly:bool = compare.eq(scols,stwo)
  branch twoonly coords2(ss,sx,sn,sc,sk,so,si,sj,si2,stag,scols,splane) pick34(ss,sx,sn,sc,sk,so,si,sj,si2,stag,scols,splane)
'''+Path(__file__).with_name('copy4.inc').read_text())
edited['nz_rank2']=v.encode()
before=A['definitions']['nz_rank2']
assert before.split(b'block g2(',1)[1].split(b'block limit(',1)[0]==edited['nz_rank2'].split(b'block g2(',1)[1].split(b'block g4(',1)[0]
assert before.split(b'block coords2(',1)[1]==edited['nz_rank2'].split(b'block coords2(',1)[1]
assert edited['nz_rank2'].count(b'{"function":"typed_store"}')==4
assert edited['nz_rank2'].count(b'{"function":"nz_truth"}')==2
def change(v,inverse=False):
 for n in names:
  x,y=A['definitions'][n],edited[n]
  if inverse:x,y=y,x
  assert v.count(x)==1,n
  v=v.replace(x,y,1)
 return v
c=change(a);u=change(rows[p]);assert change(c,True)==a and change(u,True)==rows[p]
C=b.parse_source(c);U=b.parse_source(u,unit=True)
assert C['definitions'].keys()==A['definitions'].keys()
assert {f for f in A['definitions'] if A['definitions'][f]!=C['definitions'][f]}==set(names)
assert C['signatures']==A['signatures']
