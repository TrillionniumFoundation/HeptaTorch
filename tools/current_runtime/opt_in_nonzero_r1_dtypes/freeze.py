"""Detached stdlib-only bit-word oracle; no native runtime or Torch."""
from pathlib import Path
import hashlib, json, struct, sys, zlib

CAP = 262144
sha = lambda b: hashlib.sha256(b).hexdigest()
TYPES = {
 'bool': (1,1,0,0,0),
 'int64': (2,8,0,0,0),
 'float32': (0,4,23,8,0x3f800000),
 'float64': (4,8,52,11,0x3ff0000000000000),
 'float16': (5,2,10,5,0x3c00),
 'bfloat16': (6,2,7,8,0x3f80),
}

def patterns(dtype):
 _, width, mant, exp, one = TYPES[dtype]
 if dtype=='bool': return [0,1,0,1,1,0]
 s=1<<(8*width-1)
 if dtype=='int64': return [0,s,0,s-1,s+1,1,0xffffffffffffffff,0,1<<32, (1<<63)+17, 0]
 q=1<<(mant-1)
 e=((1<<exp)-1)<<mant
 return [0,s,1,s|1, (1<<mant)-1,s|((1<<mant)-1), 1<<mant,s|(1<<mant),
         0,one,s|one,e-1,s|(e-1),e,s|e,e|q|3,s|e|q|3,e|1,s|e|1,
         e|q,s|e|q,e|(q-1),s|e|(q-1),s,0]

def source(dtype, size, recipe):
 _,width,mant,exp,one=TYPES[dtype]
 kind=recipe['kind']
 if kind=='raw':
  vals=recipe['words']
  assert len(vals)==size
 elif kind=='repeat':
  v=patterns(dtype); vals=(v[i%len(v)] for i in range(size))
 elif kind=='point':
  p=set(recipe['points'])
  assert len(p)==len(recipe['points']) and all(0<=i<size for i in p)
  vals=(int(i in p)*(one if exp else 1) for i in range(size))
 elif kind=='uniform':
  word=recipe['word']; vals=(word for i in range(size))
 else: raise AssertionError(kind)
 return b''.join(x.to_bytes(width,'little') for x in vals)

def make(dtype,label,n,recipe,admitted=True):
 tag,width,mant,exp,one=TYPES[dtype]
 inp=source(dtype,n,recipe)
 sm=(1<<(width*8-1))-1
 # Integers include their sign bit. Floating classifications are pure IEEE words,
 # not Python float casts (which can quiet sNaNs or lose reduced precision).
 hits=[i for i in range(n) if int.from_bytes(inp[i*width:(i+1)*width],'little') & (sm if exp else 2*sm+1)]
 out=b''.join(struct.pack('<q',i) for i in hits)
 k=len(hits)
 r=dict(dtype='int64',shape=[k,1],strides=[1,k],raw_sha256=sha(out),offset=0,
        storage_bytes=8*k,is_contiguous=True,view=False,version=0,requires_grad=False,
        head=hits[:12],tail=hits[-12:])
 if len(out)<=1024:r['raw_hex']=out.hex()
 assert (n<=CAP)==admitted
 return dict(label=label,input_dtype=dtype,input_tag=tag,input_shape=[n],recipe=recipe,
             input_raw_sha256=sha(inp),expected_torch=r,admitted=admitted,
             old_native_status_if_ordinary_setup_proven=(0 if dtype=='bool' else 7) if admitted else None)

raw=lambda words:dict(kind='raw',words=words)
one=lambda word:dict(kind='uniform',word=word)
point=lambda *i:dict(kind='point',points=i)
rows=[]
for d,(tag,w,m,e,v) in TYPES.items():
 p=patterns(d)
 # Same absolute geometry means varied K is caused by exact raw bits, not shape.
 for name,n,r,ok in [
   ('empty',0,raw([]),True),('all-zero-words',8,raw([0,1<<(8*w-1)]*4 if e else [0]*8),True),
   ('one-true',1,raw([p[1] if not e else p[2]]),True),
   ('signed-bit-matrix',len(p),raw(p),True),('irregular-grain',32769,dict(kind='repeat'),True),
   ('cap-K0',CAP,one(0),True),('cap-dense',CAP,one(v if e else 1),True),
   ('cap-sparse',CAP,point(0,32767,32768,131071,131072,CAP-1),True),
   ('overcap-K0',CAP+1,one(0),False),('overcap-K2',CAP+1,point(0,CAP),False),
 ]:
  rows.append(make(d,d+'-'+name,n,r,ok))
archive=[
 ('float32','archived-f32-truth-control',345, [0,0x80000000,0x7fc12345,0xbf800000,1]),
 ('int64','archived-i64-truth-control',348,[0,1<<63,1,(1<<63)-1,0])
]
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
D=Path(sys.argv[2]); R=json.loads((D/'report.json').read_bytes())
assert R['elf']=='95aadc064a5a16a188988b699b81dda287443b0e7172fa163dde15db3857c6c7'
Q=list(frames(D/'requests.frames.bin',b'HTOWNR01'))
RSP=list(frames(D/'responses.frames.bin',b'HTOWNS01'))
assert len(Q)==len(RSP)==370
for d,name,f,p in archive:
 old_label={'float32':'f32-truth-control','int64':'i64-truth-control'}[d]
 old,=[x for x in R['rows'] if x.get('label')==old_label]
 assert old['frame']==f and old['status']==7 and old['pre']==old['post']
 row=make(d,name,len(p),raw(p))
 oq=Q[f-1];orr=RSP[f-1]
 cmds=req(oq); assert len(cmds)==1 and cmds[0][0]==135 and not any(cmds[0][2:])
 assert sha(oq)==old['q_sha256'] and resp(orr)==(7,[])
 h=cmds[0][1]
 fq=Q[f-2];fr=RSP[f-2]
 fs,recs=resp(fr); assert len(req(fq))==1 and req(fq)[0][0]==97 and fs==0 and len(recs)==1 and recs[0][:2]==(97,h)
 before=old['pre']; owner_raw=zlib.decompress((D/'owners'/(before['sha256']+'.zlib')).read_bytes())
 assert len(owner_raw)==before['length'] and sha(owner_raw)==before['sha256']
 own=Owner(owner_raw);desc=own.node(h);stor,ver=desc[27:29]
 payload=source(d,len(p),raw(p));tag=TYPES[d][0]
 assert desc[:2]==(1,len(p)) and desc[12]==len(p) and desc[16]==1 and desc[72]==tag and desc[29]==desc[83]==0 and desc[4:6]==(0,0)
 assert own.row(1,h)[1][2:6]==(stor,ver,0,len(payload)) and own.storage(stor)[1:]==(len(payload),payload) and own.row(3,ver)[1][2]==0
 assert fq.endswith(payload)
 row['archive_verified_context']=dict(factory_q_sha256=sha(fq),factory_r_sha256=sha(fr),op135_r_sha256=sha(orr),tensor=h,storage=stor,version=ver,raw_sha256=sha(payload))
 row['archived']=dict(old, preceding_factory_at=f-1, require_exact_constructor_payload=True,
     require_exact_live_authority=True, require_request_response_artifact_hashes_and_ordinal=True,
     never_retire_by_request_hash=True)
 rows.append(row)
ad=[x for x in rows if x['admitted']]
summ=dict(all=len(rows),admitted=len(ad),new_admitted=sum(x['input_dtype']!='bool' for x in ad),
 retained_bool=sum(x['input_dtype']=='bool' for x in ad),above_cap_theoretical=len(rows)-len(ad),
 cap_admitted=sum(x['input_shape'][0]==CAP for x in ad),archived=len(archive),
 admitted_words=sum(x['expected_torch']['shape'][0] for x in ad),
 all_words=sum(x['expected_torch']['shape'][0] for x in rows))
out=dict(tier='DETACHED_FROZEN_STDLIB_ONLY',baseline_git='cb83670b272a6a9816b12c4ac8b2ef6f5473a087',
 baseline_tree='2a76d48da8ddd23a45df734c6020491e28fed88a',combined='42cadd9e3f1c7aea5ea391ef32b020efecbc972030a9df3c44a1a689c9ff9f91',
 elf=R['elf'],wire=135,wheel='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
 torch_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 archive_stream_hashes={name:sha((D/name).read_bytes()) for name in ('report.json','requests.frames.bin','responses.frames.bin')},
 summary=summ,cases=rows,
 scope='Exactly rank1 owned row-major ordinary CPU zero-offset non-grad and dtype tags 0/1/2/4/5/6; rank0/rank2/private3/view/out/as_tuple/default untouched. Preserve Bool canonical 0/1 check.',
 proof_plan=[
   'Future official CPU threads1, pinned wheel, integer storage reinterpretation when preparing floats. Test all raw initial Tensor bits before/after, including signed sNaNs and denormals. Do not use Python float conversion or FTZ: if official differs freeze mismatch and narrow before implementation.',
   'Future physical exact fresh ordinary I64 result [K,1], strides [1,K], Version0, offset0, raw 8*K, independently verify owner and read-back of complete input/output. Empty identities compare owner tokens; no data read for truly empty and K0 has no output payload. Both drop orders/weak, exact rollback and seeded scratch.',
   'Do not replay historical status-7 by hash alone; reverify original precursor before contextual retirement, plus exact raw input owner and readback in a new session. All other retained frames unchanged.',
   'At-cap width<=2 fits raw wire factory. Width4 at CAP exactly 1MiB before framing and width8 exceeds it. First try a verified ordinary rank1 constant construction supported by existing ops (zero/one, then try exact sparse formula with independently verified setup); a rank2 result is not equivalent. If setup unavailable record each blocked cap case, never modify fixtures, skip rather than use Cat, unsafe fabricated owner or silent substitution. Current axis cap prevents public >CAP factory: capture setup refusal, not op135 overcap.'])
b=(json.dumps(out,sort_keys=True,separators=(',',':'))+'\n').encode()
with Path(sys.argv[1]).open('xb') as f:f.write(b)
print(len(b),sha(b),summ)
