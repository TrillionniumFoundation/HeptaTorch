"""Detached stdlib-only same-dtype logical-family broadcast preregistration."""
from pathlib import Path
import hashlib,itertools,json,math,struct,sys,zlib
D=Path(__file__).parent;sys.path.insert(0,str(D))
from read_old import Owner,frames,req,resp,sha
W={0:4,1:1,2:8,4:8,5:2,6:2}
OPS={131:'and',133:'or',134:'xor'}
CAP=262144
P=Path('/tmp/hepta-logical-family-prereg')
old=(P/'preregister.json').read_bytes()
assert sha(old)=='299e9a2fc982c4a50d0b023bd0378f7dd6b1c7aaf24a9ea7bd60627173b7614e'
typed={}
for r in json.loads(old)['strict_rows']:
 if r['label'].endswith('-raw-left'):
  x=r['left']; typed[x['dtype']]=list(zip(x['words'],x['raw_classes'],strict=True))
assert set(typed)==set(W)
def ss(shape):return [math.prod(shape[i+1:]) for i in range(len(shape))]
def blob(tag,words):return b''.join(v.to_bytes(W[tag],'little') for v in words)
def tensor(tag,shape,words):
 assert math.prod(shape)==len(words)
 return dict(dtype=tag,shape=shape,strides=ss(shape) if all(shape) else [math.prod(max(1,n) for n in shape[i+1:]) for i in range(len(shape))],words=words,raw_hex=blob(tag,words).hex())
def truth(tag,raw):
 if tag in (1,2):return int(raw!=0)
 e,m={0:(8,23),4:(11,52),5:(5,10),6:(8,7)}[tag]
 return int(bool((raw>>m)&((1<<e)-1) or raw&((1<<m)-1)))
def decide(op,a,b):
 return [a*b,max(a,b),(a+b)%2][{131:0,133:1,134:2}[op]]
def coord(shape):return itertools.product(*(range(n) for n in shape))
def shape_join(a,b):
 rank=max(len(a),len(b));out=[]
 for x,y in zip([1]*(rank-len(a))+a,[1]*(rank-len(b))+b,strict=True):
  if x==y: out.append(x)
  elif x==1:out.append(y)
  elif y==1:out.append(x)
  else:return None
 return out
def selected(t,ij):
 s=t['shape'];j=ij[len(ij)-len(s):]
 return tuple(0 if n==1 else i for n,i in zip(s,j,strict=True))
def second_index(source,out,flat):
 idx=0;stride=1
 for j,dim in enumerate(reversed(out)):
  flat,q=divmod(flat,dim)
  if j<len(source):
   n=source[-1-j]
   idx+=stride*(q if n!=1 else 0);stride*=n
 assert not flat and 0<=idx<math.prod(source)
 return idx
def predict(x,y,op):
 a,b=x['shape'],y['shape']; shape=shape_join(a,b)
 assert shape is not None and 0<math.prod(shape)<=CAP and len(shape)<=4
 tag=x['dtype'];assert tag==y['dtype'];out=[]
 tx,ty=({k:truth(tag,w) for k,w in zip(coord(t['shape']),t['words'],strict=True)} for t in (x,y))
 plain=[list(t.values()) for t in (tx,ty)]
 for j,k in enumerate(coord(shape)):
  l,r=tx[selected(x,k)],ty[selected(y,k)]
  assert (l,r)==tuple(v[second_index(t['shape'],shape,j)] for t,v in zip((x,y),plain,strict=True))
  out.append(decide(op,l,r))
 return dict(dtype=1,shape=shape,strides=ss(shape),raw_hex=bytes(out).hex(),numel=len(out))
geometry=[
 ('true-scalar-vector',[],[5]), ('true-scalar-rank4',[],[1,2,1,3]), ('rank1-one',[1],[4]),
 ('equal-numel-rank-shift',[1,3],[3]), ('right-aligned-vector',[3],[2,1,3]),
 ('single-expand-middle',[2,1,3],[2,4,3]), ('crossed-rank2',[2,1],[1,3]),
 ('crossed-rank3',[2,1,3],[1,4,1]),('crossed-rank4',[1,2,1,4],[3,1,5,1]),
 ('rank2-rank4',[2,1],[3,1,2,5]),('leading-unit',[1,1,1,2],[2]),
]
rows=[]
for tag,bank in typed.items():
 for i,(name,a,b) in enumerate(geometry):
  for rev in (False,True):
   shx,shy=(b,a) if rev else (a,b)
   # Set dispersed zeros while rotating signed special raw words, no numeric float conversion.
   nil=[v for v,c in bank if not truth(tag,v)]
   take=lambda n,h:[(nil[(k+h)%len(nil)] if (k+h)%4==0 else bank[(13*k+h)%len(bank)][0]) for k in range(math.prod(n))]
   x=tensor(tag,shx,take(shx,2*i+int(rev))); y=tensor(tag,shy,take(shy,2*i+3+int(rev)))
   expected={str(op):predict(x,y,op) for op in OPS}
   aout,bout,cout=[bytes.fromhex(expected[str(i)]['raw_hex']) for i in OPS]
   assert all(o==(a^c) and not (a&c) for a,o,c in zip(aout,bout,cout,strict=True))
   rows.append(dict(label=f'dtype{tag}-{name}-{int(rev)}',left=x,right=y,expect=expected))
# Exact original actual witnesses, including the three refusals that must stay refusals.
orig=Path('/tmp/hepta-logical-and-prereg/candidate')
family=P/'candidate'
expect_frames={
 'and':(orig,'db360c3e0b071ff3993d331d61276bbdf6437d3afecb2619b9a402e73080713f','456fb911c1b01d4f780cb8c55aa032c420141754218a09685c552cafb3ae3feb',
          {2419:131,2424:131,2429:131,2439:131}),
 'or-xor':(family,'59cad47e38954c9b85ddef249f77f6505a51533264e1f9337dc756689e456e2d','bf0f296efc2097f913c9831ad6dc42ed37672be7d93e60c37c59494b66bf9f68',{4825:133,4830:133,4835:133,4845:133,4914:134,4919:134,4924:134,4934:134})}
arch=[]
rawdir=D/'exact-old';rawdir.mkdir(exist_ok=True,mode=0o700)
for name,(target,hq,hr,wanted) in expect_frames.items():
 rq=target/'requests.frames.bin';rp=target/'responses.frames.bin'
 assert sha(rq.read_bytes())==hq and sha(rp.read_bytes())==hr
 q=list(frames(rq,b'HTOWNR01'));a=list(frames(rp,b'HTOWNS01'));p=json.loads((target/'report.json').read_bytes())
 for e in p['rows']:
  j=e.get('frame')
  if not(j in wanted and e.get('kind')=='refusal'):continue
  q0,a0=q[j-1],a[j-1];c,=req(q0)
  assert c[0]==wanted[j] and c[3:]==(0,)*5 and resp(a0)==(7,[]) and e['pre']==e['post']
  assert sha(q0)==e['request_sha256']
  meta=e['pre'];comp=(target/'owners'/(meta['sha256']+'.zlib')).read_bytes()
  raw=zlib.decompress(comp);assert len(raw)==meta['length'] and sha(raw)==meta['sha256']
  o=Owner(raw); x=[]
  for h in c[1:3]:
   d=o.node(h);t=d[72];sh=list(d[12:12+d[0]]);st=list(d[16:16+d[0]]);blob0=o.storage(d[27])[2]
   assert len(blob0)==d[1]*W[t] and d[4]==d[5]==d[29]==d[83]==0 and st==ss(sh)
   x.append(tensor(t,sh,[int.from_bytes(blob0[i:i+W[t]],'little') for i in range(0,len(blob0),W[t])]))
  assert x[0]['dtype']==x[1]['dtype'] and x[0]['shape']!=x[1]['shape']
  admissible='incompatible' not in e['label']
  assert (shape_join(x[0]['shape'],x[1]['shape']) is not None)==admissible
  for ext,v in (('request.bin',q0),('response.bin',a0),('owner.zlib',comp)):(rawdir/f'{j}.{ext}').write_bytes(v)
  arch.append(dict(origin=name,frame=j,label=e['label'],op=c[0],before_status=7,after_status=(0 if admissible else 7),
   owner_exact_rollback=True,request_sha256=sha(q0),response_sha256=sha(a0),owner_sha256=sha(raw),owner_compressed_sha256=sha(comp),
   operands=x,prediction=predict(*x,c[0]) if admissible else None))
assert len(arch)==12 and sum(w['after_status']==0 for w in arch)==9
cap=[]
for shx,shy in (([512,1],[1,512]),([513,1],[1,512])):
 x=tensor(1,shx,[int(i%7==2) for i in range(shx[0])]);y=tensor(1,shy,[int(i%5 in (0,3)) for i in range(shy[1])])
 n=math.prod(shape_join(shx,shy)); t={str(op):predict(x,y,op) for op in OPS} if n<=CAP else {}
 cap.append(dict(left=x,right=y,output_shape=shape_join(shx,shy),output_count=n,predicted_status=0 if t else 3,
    result_digests={k:{'raw_sha256':sha(bytes.fromhex(v['raw_hex'])),'ones':sum(bytes.fromhex(v['raw_hex'])),
                    'first':v['raw_hex'][:16],'last':v['raw_hex'][-16:]} for k,v in t.items()}))
guards=[
 dict(label='compatible-empty',left=[0,1,4],right=[1,3,1],upstream=[0,3,4],native_status=7),
 dict(label='incompatible-zero-not-unit',left=[0,2],right=[3,2],upstream=None,native_status=7),
 dict(label='equal-count-wrong-shape',left=[2,3],right=[3,2],upstream=None,native_status=7),
 dict(label='rank-mismatch-conflict',left=[2,3],right=[2],upstream=None,native_status=7),
]
out=dict(stage='stdlib-only-frozen',parent='b545f2a50540390f6fa86300418f83fddff41c96',tree='1b234b74507ab0c4ad2dbc1cd86427a9e2cae4f9',
 source='e37dd3c347cb6143a53d283c39c9e2ff637640d568c8347095117c43b9ab6144',ops=OPS,
 policy='same public dtype, ordinary canonical nonempty ranks0–4, no views/grad or default changes',
 rows=rows,exact_archived_refusals=arch,capacity=cap,geometry_refusals=guards,retained=['op132 unsupported','all original equal shape cases','owner/cursor/weak/late rollback','views, mixed dtype, empty, grad, stale, wire padding'])
raw=(json.dumps(out,sort_keys=True,separators=(',',':'))+'\n').encode()
(D/'preregister.json').write_bytes(raw)
print('pairs',len(rows),'ops',3*len(rows),'elements/operation',sum(r['expect']['131']['numel'] for r in rows),'archive',len(arch),'wire changes',9,'json bytes',len(raw),sha(raw))
assert sum(p.stat().st_size for p in D.rglob('*') if p.is_file())<=500*1024
