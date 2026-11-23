"""Matched clean controls for post-refusal owner consumption."""
from pathlib import Path
import struct,hashlib,json,sys,zlib
H,W,E,O=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p).hexdigest()
sys.path.insert(0,str(H));from test_runtime_candidate import factory,load_wire
W=load_wire(W);assert 'torch' not in sys.modules
O.mkdir();rows=[]
for name in ('sparse','zero'):
 pair=[]
 for mode in ('clean','reject'):
  dst=O/f'{name}-{mode}';s=W.Session(E,directory=dst)
  def run(*args):
   q=W.packet(args);st,r=W.parse(s.exchange(q));assert not st and len(r)==len(args);return r
  def obj(t,shape,words):
   st,r=W.parse(s.exchange(b''.join(factory(t,shape,words))));assert st==0 and len(r)==1
   return r[0]['value']
  def sticky():
   db=s.owner()['bytes'];h=sha(db);p=dst/(h+'.zlib')
   if not p.exists():p.write_bytes(zlib.compress(db,6))
   return h
  def fp(x):return struct.unpack('<I',struct.pack('<f',x))[0]
  try:
   a=obj(0,[513,1],[0x3f000000+13*512*r for r in range(513)])
   b=obj(0,[1,512],[fp(13*i/2**24) for i in range(512)])
   t=run((2,1,a,b))[0]['value']
   n=513*512; m=obj(1,[513,512],[int(name=='sparse' and i in (0,n-1)) for i in range(n)])
   aa=obj(0,[4],[0x80000000,0x7f812345,0,0xffc23456]);mm=obj(1,[4],[1,1,0,1])
   start=sticky()
   if mode=='reject':
    st,rr=W.parse(s.exchange(W.packet([(130,t,m)])));assert st==3 and rr==[] and sticky()==start
   end=sticky()
   z,q,r=run((130,aa,mm),(13,W.reference(1)),(98,W.reference(1),0,3))
   assert r['payload']==struct.pack('<3I',0x80000000,0x7f812345,0xffc23456)
   md=W.meta(q);assert md['version']==0 and md['shape'][0]==3
   pair.append(dict(start_owner=start,post_check_owner=end,result_owner=sticky(),ids=[z['value'],md['storage_id'],md['version_id']],
      raw_words_sha256=sha(r['payload'])))
  finally:s.finish()
 assert pair[0]==pair[1],(name,pair)
 rows.append(dict(mask=name,clean_vs_reject=pair))
(O/'mirror.json').write_text(json.dumps(dict(elf_sha256=sha(E.read_bytes()),cases=rows),sort_keys=True,indent=2)+'\n')
print(json.dumps(rows))
