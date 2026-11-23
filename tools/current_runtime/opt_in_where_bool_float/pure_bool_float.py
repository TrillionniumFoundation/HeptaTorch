"""Offline wire/owner readback; no runtime, protocol, producer or Torch imports."""
from pathlib import Path
import itertools,math,hashlib,json,struct,sys,zlib
B=139264
MASK=(1<<63)-1
sha=lambda b:hashlib.sha256(b).hexdigest()
def q(raw,n=1,at=0):return struct.unpack_from('<'+'Q'*n,raw,at)
def frames(p,magic):
 data=p.read_bytes();o=0
 while o<len(data):
  assert data[o:o+8]==magic
  n,=q(data,1,o+8);assert 32<=n<=1<<20 and o+16+n<=len(data)
  yield data[o+16:o+16+n];o+=16+n
 assert o==len(data)
def req(x):
 n,ln,z=q(x,3,8);assert x[:8]==b'HTORCH03' and ln==len(x) and z==0
 return [q(x,8,32+64*j) for j in range(n)]
def resp(x):
 v,n,s=q(x,3,8);assert x[:8]==b'HTORSP03' and v==3
 if s:assert n==0 and len(x)==32
 a=[];k=32
 for _ in range(n):
  op,err,t,ln=q(x,4,k);assert err==0;k+=32
  assert k+ln<=len(x);a.append((op,t,x[k:k+ln]));k+=ln
 assert k==len(x);return s,a
class Owner:
 def __init__(self,x):
  assert x[64:72]==b'HTSINT02' and x[B:B+8]==b'HTOSIN02'
  self.b=x;self.slots,self.pool=q(x,2,B+16)
  assert 4<=self.slots<=65536 and self.slots&(self.slots-1)==0
  self.start=B+144+552*self.slots
  assert len(x)==self.start+self.pool
 def row(self,kind,tok):
  slot=tok&0xffffff
  assert 0<slot<=self.slots and (tok>>24)&255==kind
  at=B+128+((kind-1)*self.slots+slot-1)*64
  r=q(self.b,8,at);assert r[:2]==(tok>>32,1)
  return at,r
 def node(self,tok):
  i=self.row(1,tok)[1][6]
  assert 0<i<q(self.b,1,8)[0]
  page,=q(self.b,1,4096+(i>>8)*8);assert page>>63
  at=self.start+(page&MASK)+(i&255)*672
  d=q(self.b,84,at);assert d[24]==tok and d[25]==d[30]==1
  return d
 def storage(self,tok):
  _,r=self.row(2,tok)
  pos=r[2]+self.start
  assert self.start<=pos<=pos+r[3]<=len(self.b)
  return pos,r[3],self.b[pos:pos+r[3]]
WIDTH={0:4,1:1,2:8,4:8,5:2,6:2}
ON={0:struct.pack('<f',1.0),4:struct.pack('<d',1.0),5:struct.pack('<e',1.0),6:struct.pack('<f',1.0)[2:]}
FORM={0:(8,23),4:(11,52),5:(5,10),6:(8,7)}
def special(tag,word):
 e,m=FORM[tag]; sign='-' if word>>(e+m) else '+'
 mantissa=word&((1<<m)-1); exponent=(word>>m)&((1<<e)-1)
 if exponent==0:return sign+('subnorm' if mantissa else 'zero')
 if exponent==(1<<e)-1:
  if not mantissa:return sign+'inf'
  return sign+('qnan' if mantissa>>(m-1) else 'snan')
 return sign+'finite'
ELFS={'baseline':'7737917d5fff95de844231a7c0c422c96b7834dbfa2e8862bb3ab4b67a37072d','candidate':'d666b7dc7068c685dec89794ecb740aeb5d8da74d2062c5e30e63a7c0695935e'}
FIX='c3497395c57386a285a7103383fb302ce9c73764267edd63d8973936007dc6c9'
def inspect(path,fixture):
 p=Path(path); r=json.loads((p/'report.json').read_bytes()); assert r['no_torch'] and not (p/'error.json').exists()
 raw_fixture=fixture.read_bytes(); assert sha(raw_fixture)==r['fixture_sha256']==FIX
 rows={x['label']:x for x in json.loads(raw_fixture)['rows']}
 tr=json.loads((p/'transport.json').read_bytes()); assert tr['exit_code']==0 and tr['responses_before_stdin_eof']
 assert tr['executable_sha256']==r['elf']==ELFS[r['mode']]
 raw_q=list(frames(p/'requests.frames.bin',b'HTOWNR01'))
 qs=list(map(req,raw_q)); rs=list(map(resp,frames(p/'responses.frames.bin',b'HTOWNS01')))
 assert len(qs)==len(rs)==tr['frames']==r['native_frames']
 for qq,(st,rr) in zip(qs,rs):
  if not st: assert [a[0] for a in qq]==[a[0] for a in rr]
 def snapshot(rec):
  b=zlib.decompress((p/'owners'/(rec['sha256']+'.zlib')).read_bytes())
  assert sha(b)==rec['sha256'] and len(b)==rec['length'];return b
 def read(frame,h,n,blob):
  i=frame-1
  assert qs[i]==[(13,h,0,0,0,0,0,0),(98,h,0,n,0,0,0,0)]
  status,data=rs[i]
  assert status==0 and data[0][:2]==(13,h) and data[1]==(98,n,blob)
  return q(data[0][2],16)
 def node(o,h,tag,n,shape,blob):
  a=o.node(h);k=o.row(1,h)[1]; st,v=a[27:29]
  assert a[72]==tag and a[83]==0 and a[4:6]==(0,0) and a[29]==0 and a[1]==n
  rank=len(shape);assert a[0]==rank and a[12:12+rank]==tuple(shape)
  assert k[2:4]==(st,v) and k[4:6]==(0,n*WIDTH[tag])
  assert o.storage(st)[1:]==(len(blob),blob)
  assert a[2]==(1<<63)+o.row(2,st)[1][2] # empty roots may reuse no-data sentinel address
  return a,st,v,o.row(3,v)[1][2]
 def argument_dims(f):
  return [f['mask_shape'],f['left_shape'],f['right_shape']]
 def coordinate_shape(shapes):
  rank=max(map(len,shapes))
  assert rank<=4
  padded=[[1]*(rank-len(sh))+sh for sh in shapes]
  result=[]
  for xyz in zip(*padded):
   nonunit=set(xyz)-{1}
   if len(nonunit)>1:raise ValueError(('incompatible axis',xyz))
   result.append(next(iter(nonunit),1))
  return result
 def argument_tags(f):
  x=f['left_dtype'];y=f['right_dtype']
  assert f['dtype'] in ON and {x,y}=={1,f['dtype']}
  return [1,x,y]
 def bit_arguments(f):
  mask=bytes(f['mask']); tags=argument_tags(f); xw,yw=map(WIDTH.get,tags[1:])
  left=[v.to_bytes(xw,'little') for v in f['left']]
  right=[v.to_bytes(yw,'little') for v in f['right']]
  shapes=argument_dims(f)
  shape=coordinate_shape(shapes);assert shape==f['shape']
  coords=lambda shape:itertools.product(*(range(x) for x in shape))
  map3=[dict(zip(coords(s),values,strict=True)) for s,values in zip(shapes,(mask,left,right))]
  def coord(shape,i):
   if not shape:return ()
   return tuple(0 if d==1 else j for d,j in zip(shape,i[-len(shape):]))
  parts=[]; tag_numeric=f['dtype']; count={1:0,tag_numeric:0}; bool_counts=[0,0]; float_seen=set()
  for i in coords(shape):
   flag=map3[0][coord(shapes[0],i)]
   assert flag in (0,1)
   selected=1 if flag else 2
   val=map3[selected][coord(shapes[selected],i)]
   tag=tags[selected];count[tag]+=1
   if tag==1:
    assert val in (b'\x00',b'\x01')
    bool_counts[val[0]]+=1
    val=ON[tag_numeric] if val[0] else b'\x00'*WIDTH[tag_numeric]
   else:
    float_seen.add(int.from_bytes(val,'little'))
   parts.append(val)
  blobs=[mask,b''.join(left),b''.join(right),b''.join(parts)]
  assert tuple(b.hex() for b in blobs)==tuple(f[a+'_hex'] for a in ('mask','left','right','result'))
  assert f['width']==WIDTH[tag_numeric]
  return math.prod(shape),tag_numeric,blobs,count,bool_counts,float_seen
 cnt=dict(cases=0,elements=0,exact_rollbacks=0,baseline_pair_refusals=0,frames=len(qs),scalar=0,empty=0,
          same_input=0,same_frame=0,inputs_first=0,result_first=0,positive_offset_refusals=0,rank0=0,rank1=0,rank2=0,rank3=0,rank4=0,empty_last=0,empty_prefix=0,capacity_refusal=0,selected_bool=0,selected_float=0,bool_false=0,bool_true=0,bool_left=0,float_left=0,scalar_both=0,only_left_scalar=0,only_right_scalar=0,raw_float_words={str(t):[] for t in ON},float_special_classes={str(t):[] for t in ON},dtype_cases={str(t):0 for t in ON})
 seen=set()
 for e in r['rows']:
  kind=e['kind']; label=e['label']
  if kind=='refusal':
   i=e['frame']-1
   assert rs[i]==(e['expected'],[]) and e['expected']==e['status']
   before=snapshot(e['pre']); after=snapshot(e['post'])
   assert before==after and e['pre']['ptr']==e['post']['ptr']
   assert sha(raw_q[i])==e['request_sha256']; cnt['exact_rollbacks']+=1
   c=qs[i]
   if label.startswith('original-'):
    f=rows[label[9:]]; n,k,bs,count,bool_counts,float_seen=bit_arguments(f); (op,hm,hx,hy,*reserved),=c
    assert op==129 and reserved==[0]*4 and e['status']==7
    for h,b,t,lens,sp in zip((hm,hx,hy),bs,argument_tags(f),(len(f['mask']),len(f['left']),len(f['right'])),argument_dims(f)):node(Owner(before),h,t,lens,sp,b)
    cnt['baseline_pair_refusals']+=1
    assert f['label'] not in seen;seen.add(f['label'])
   if label.startswith('pad-'):
    (command,)=c;assert command[0]==129 and sum(command[4:])==1 and command[int(label[-1])]==1
   if label.startswith('stale-'):
    (command,)=c; dead=command[int(label[-1])]
    o=Owner(before);slot=dead&0xffffff; rr=q(before,8,B+128+(slot-1)*64)
    assert slot<=o.slots and (rr[0]!=dead>>32 or rr[1]!=1) and e['status']==1
   if label.startswith('view-'):
    (command,)=c;at=2 if label.endswith('-left') else 3
    v=Owner(before);h=command[at];d=v.node(h);tr=v.row(1,h)[1]
    t=1 if label.startswith('view-bool-') else 0
    assert command[0]==129 and d[83]==1 and d[29]==tr[4]==WIDTH[t] and d[72]==t
    assert d[2]==(1<<63)+v.row(2,d[27])[1][2]+tr[4]
    cnt['positive_offset_refusals']+=1
   if label in ('capacity-over-limit','equal-count-incompatible-values'):
    o=Owner(before); sh=[]; tags=[]
    for h in c[0][1:4]:
     od=o.node(h);sh.append(list(od[12:12+od[0]]));tags.append(od[72])
    assert c[0][0]==129 and tags==([1,1,5] if label=='capacity-over-limit' else [1,1,0])
    if label=='capacity-over-limit':
     assert sh==[[],[513,1],[1,512]] and math.prod(coordinate_shape(sh))==262656 and e['status']==3
     cnt['capacity_refusal']+=1
    else:
     assert sh==[[],[4,2],[2,4]]
     try:coordinate_shape(sh)
     except ValueError:pass
     else:raise AssertionError(('accepted incompatible',sh))
   for prefix,types in {'f32-i64':(0,2),'f16-i64':(5,2),'f16-bf16':(5,6),'f32-f64':(0,4)}.items():
    if label.startswith(prefix):
     o=Owner(before);t=tuple(o.node(h)[72] for h in c[0][2:4])
     assert t==(types if label.endswith('-forward') else types[::-1])
   if label=='numeric-scalar-condition':
    d=Owner(before).node(c[0][1]);assert d[:2]==(0,1) and d[72]==0
   if label=='requires-grad-value':assert Owner(before).node(c[0][3])[4]==1
   if label=='end-of-frame-unknown':assert len(c)==2 and c[0][0]==129 and c[1][0]==65500
   continue
  f=rows[label];n,k,bs,count,bool_counts,float_seen=bit_arguments(f)
  if kind=='same-frame':
   i=e['frame']-1
   assert tuple(qs[i][0])==(129,*e['inputs'],0,0,0,0) and len(qs[i])==3
   assert qs[i][1]==(13,0xfe000001,0,0,0,0,0,0) and qs[i][2]==(98,0xfe000001,0,n,0,0,0,0)
   assert rs[i][0]==0 and rs[i][1][0]==(129,e['result'],b'') and rs[i][1][1][:2]==(13,e['result'])
   assert rs[i][1][2]==(98,n,bs[3]);cnt['same_frame']+=1;continue
  assert kind=='positive' and label not in seen;seen.add(label)
  hm,hx,hy,hr=e['handles']
  assert hx!=hy and hr not in {hm,hx,hy}
  i=e['call']-1
  assert qs[i]==[(129,hm,hx,hy,0,0,0,0)] and rs[i]==(0,[(129,hr,b'')])
  a=Owner(snapshot(e['pre']));b=Owner(snapshot(e['post']))
  sources=[]
  for col,(h,blob,tag,sz,sp) in enumerate(zip((hm,hx,hy),bs,argument_tags(f),(len(f["mask"]),len(f["left"]),len(f["right"])),argument_dims(f))):
   da,sid,vid,tick=node(a,h,tag,sz,sp,blob)
   db,*identity=node(b,h,tag,sz,sp,blob)
   assert da==db and identity==[sid,vid,tick]
   assert a.row(1,h)[1]==b.row(1,h)[1] and a.row(2,sid)[1]==b.row(2,sid)[1] and a.row(3,vid)[1]==b.row(3,vid)[1]
   ma=read(e['before_reads'][col],h,sz,blob);mb=read(e['after_reads'][col],h,sz,blob)
   assert ma==mb and mb[11:16]==(sid,vid,tick,0,len(blob))
   sources.append((sid,vid))
  dr,os,ov,tick=node(b,hr,k,n,f['shape'],bs[3])
  assert tick==0 and os not in {v[0] for v in sources} and ov not in {v[1] for v in sources}
  rm=read(e['read_result'],hr,n,bs[3]);rank=len(f['shape'])
  assert rm[:3]==(rank,n,0) and rm[3:3+rank]==tuple(f['shape']) and rm[7:7+rank]==tuple(f['strides'])
  assert rm[11:16]==(os,ov,0,0,len(bs[3]))
  z=e['weak_frame']-1;assert qs[z][0][:2]==(42,hr) and rs[z]==(0,[(42,e['weak'],b'')])
  z=e['expired']-1;assert qs[z][0][:2]==(43,e['weak']) and rs[z]==(6,[])
  if e['order']=='inputs-first':
   assert read(e['survivor'],hr,n,bs[3])==rm; cnt['inputs_first']+=1
  else:
   assert read(e['survivor'],hx,len(f['left']),bs[1])==read(e['before_reads'][1],hx,len(f['left']),bs[1]);cnt['result_first']+=1
  cnt['cases']+=1;cnt['elements']+=n;cnt['rank'+str(rank)]+=1;cnt['empty_last']+=bool(f['shape']) and f['shape'][-1]==0;cnt['empty_prefix']+=0 in f['shape'][:-1];cnt['scalar']+=rank==0;cnt['empty']+=n==0;cnt['same_input']+=hx==hy
  cnt['selected_bool']+=count[1];cnt['selected_float']+=count[k]
  cnt['bool_false']+=bool_counts[0];cnt['bool_true']+=bool_counts[1]
  tag_key=str(k)
  cnt['dtype_cases'][tag_key]+=1
  cnt['raw_float_words'][tag_key]=sorted(set(cnt['raw_float_words'][tag_key])|{hex(w) for w in float_seen})
  cnt['float_special_classes'][tag_key]=sorted(set(cnt['float_special_classes'][tag_key])|{special(k,w) for w in float_seen})
  cnt['bool_left' if f['left_dtype']==1 else 'float_left']+=1
  ln=len(f['left_shape']);rn=len(f['right_shape'])
  cnt['scalar_both']+=not ln and not rn
  cnt['only_left_scalar']+=not ln and bool(rn)
  cnt['only_right_scalar']+=bool(ln) and not rn
 assert seen==set(rows)
 if r['mode']=='candidate':
  must={z+s for z in ('+','-') for s in ('zero','subnorm','inf','qnan','snan')}
  assert all(must<=set(c) for c in cnt['float_special_classes'].values())
  assert cnt['bool_false'] and cnt['bool_true'] and cnt['selected_float']
 return dict(**cnt,request_stream=sha((p/'requests.frames.bin').read_bytes()),response_stream=sha((p/'responses.frames.bin').read_bytes()))
if __name__=='__main__':
 fixture=Path(sys.argv[1]);print(json.dumps({Path(p).name:inspect(p,fixture) for p in sys.argv[2:]},indent=2,sort_keys=True))
