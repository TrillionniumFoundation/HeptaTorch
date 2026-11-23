"""Detached logical_and preregistration; stdlib only, no runtime or Torch."""
import hashlib,itertools,json,math,struct
from pathlib import Path
D=Path(__file__).parent
FMT={0:(32,8,23),4:(64,11,52),5:(16,5,10),6:(16,8,7)}
TAGS={0:'f32',1:'bool',2:'i64',4:'f64',5:'f16',6:'bf16'}
W={0:4,1:1,2:8,4:8,5:2,6:2}
def strata(tag):
 if tag==1:return [(0,'false'),(1,'true')]
 if tag==2:return [(0,'zero'),(1,'one'),((1<<63)-1,'max'),(1<<63,'min'),((1<<64)-1,'minus-one')]
 n,e,m=FMT[tag];sign=1<<(n-1);ef=((1<<e)-1)<<m
 u=((0,'zero'),(1,'smallest-subnormal'),((1<<m)-1,'largest-subnormal'),(1<<m,'min-normal'),
    (((1<<(e-1))-1)<<m,'one'),(ef-1,'max-finite'),(ef,'inf'),
    (ef+1,'snan-low'),(ef+(1<<(m-1))-1,'snan-wide'),(ef+(1<<(m-1)),'qnan'),(ef+(1<<m)-1,'qnan-wide'))
 return [(base+ss,label+('+' if not ss else '-')) for ss in (0,sign) for base,label in u]
def coordinate_truth(w,tag):
 # Explicit IEEE fields, independent of the old HIA sign-strip helper.
 if tag in FMT:
  _,e,m=FMT[tag];exp=(w//(1<<m))%(1<<e); fraction=w%(1<<m)
  return exp!=0 or fraction!=0
 return w!=0
def second_oracle(shape,aa,bb,t):
 # Independent raw tuple decode and count/coordinate check, never materialize NaNs.
 cb=bytes.fromhex(aa);db=bytes.fromhex(bb)
 assert len(cb)==len(db)==math.prod(shape)*W[t]
 def nz(raw):
  v=int.from_bytes(raw,'little')
  mask=(1<<(8*len(raw)-(t in FMT)))-1
  return bool(v&mask)
 pairs=list(zip(struct.iter_unpack({1:'<B',2:'<H',4:'<I',8:'<Q'}[W[t]],cb),
                struct.iter_unpack({1:'<B',2:'<H',4:'<I',8:'<Q'}[W[t]],db),strict=True))
 coords=list(itertools.product(*(range(i) for i in shape)))
 assert len(pairs)==len(coords)
 return bytes(nz(x[0].to_bytes(W[t],'little')) and nz(y[0].to_bytes(W[t],'little')) for x,y in pairs).hex()
def pack(words,w):return b''.join(int(x).to_bytes(w,'little') for x in words).hex()
def stride(shape):return [math.prod(max(1,v) for v in shape[i+1:]) for i in range(len(shape))]
def tensor(t,shape,values):
 assert math.prod(shape)==len(values)
 return dict(dtype=t,shape=shape,strides=stride(shape),words=[v for v,_ in values],raw_hex=pack([v for v,_ in values],W[t]),raw_classes=[k for _,k in values])
def add(label,t,shape,a,b,relationship='distinct_handles'):
 x,y=tensor(t,shape,a),tensor(t,shape,b)
 vals=[int(coordinate_truth(av,t) and coordinate_truth(bv,t)) for av,bv in zip(x['words'],y['words'],strict=True)]
 out=tensor(1,shape,[(v,str(v)) for v in vals])
 assert out['raw_hex']==second_oracle(shape,x['raw_hex'],y['raw_hex'],t)
 return dict(label=label,left=x,right=y,expected=out,owner_relationship=relationship)
rows=[]
for t,name in TAGS.items():
 a=strata(t);by=[[x for x in a if int(coordinate_truth(x[0],t))==i] for i in (0,1)]
 # Complete truth table in every rank, including shape[] and canonical singleton dimensions.
 for shape in ([],[9],[2,4],[1,2,1],[1,2,1,4]):
  n=math.prod(shape)
  for l,r in itertools.product((0,1),repeat=2):
   x=[by[l][k%len(by[l])] for k in range(n)];y=[by[r][(k+1)%len(by[r])] for k in range(n)]
   rows.append(add(f'{name}-rank{len(shape)}-{l}{r}',t,shape,x,y))
 # Every bit pattern selected as both left and right; reversed masks make both false and true outcomes observable.
 n=len(a); b=[by[i%2][i%len(by[i%2])] for i in range(n)]
 rows.extend((add(f'{name}-raw-left',t,[n],a,b),add(f'{name}-raw-right',t,[n],b,a),
              add(f'{name}-same-owner',t,[n],a,a,'same_handle')))
 for n in (7,8,9):
  x=[a[(i+2)%len(a)] for i in range(n)];y=[a[(-i-1)%len(a)] for i in range(n)]
  rows.append(add(f'{name}-boundary-{n}',t,[n],x,y))
assert len(rows)==156 and len({r['label'] for r in rows})==156
controls=[
 ('empty-rank1',[0],[0],1,1,'HIA-only scope, upstream succeeds'),
 ('empty-rank3',[2,0,3],[2,0,3],0,0,'HIA-only scope, upstream succeeds'),
 ('broadcast-scalar',[],[2],1,1,'HIA-only scope, upstream succeeds'),
 ('broadcast-singleton',[2,1],[2,3],4,4,'HIA-only scope, upstream succeeds'),
 ('equal-count-incompatible',[2,3],[3,2],2,2,'upstream shape error'),
 ('mixed-dtype',[2],[2],1,0,'HIA-only scope, upstream succeeds'),
 ('same-numel-different-rank',[1,2],[2],5,5,'HIA-only scope, upstream succeeds'),
]
extra=[
 ('transposed or positive-offset child',7,'real constructed tensor; parent owner and metadata remain exact'),
 ('noncanonical singleton-stride view',7,'do not use contiguous bit as canonical-stride proof'),
 ('requires_grad floating input',7,'explicit narrow owner policy; upstream supports reading it'),
 ('stale or wrong-kind handle',1,'record actual public resolver baseline; not a promised universal external status'),
 ('reserved wire padding',7,'all five unused args zero; do not mutate operand state on refusal'),
 ('over-cap preflight or forced allocation pressure',3,'retain exact existing service policy; test rollback on exact status'),
 ('unassigned candidate / independent adjacent opcode',None,'explicit predecessor and opcode collision; do not claim assigned code'),
 ('result followed by dependent read in same frame',None,'use ephemeral result reference; exactly one op response and declared tensor handle'),
 ('later unknown opcode in same frame',None,'initially valid first op fully rolled back; capture original baseline and owner snapshot'),
 ('drop both input owners then read result',None,'fresh output storage/version 0, no parent/weak alias'),
 ('drop result then read both inputs',None,'exact original raw words and versions survive'),
 ('weak result expiry',None,'observe exact before/after owner semantics/native transaction'),
]
bundle=dict(stage='preregistered-stdlib-only-not-measured',reference_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 op='aten.logical_and.default',wire_opcode=None,
 native_scope='CPU ordinary live canonical same public dtype equal shape nonempty ranks 0 through 4, no view, no grad, fresh canonical Bool',
 owner_contract=dict(inputs='compare each owner storage token, version token+counter, shape, stride, bytes, offset and flags before/after',
 output='distinct owner/storage/version from each input, canonical metadata, version 0, arg element width unchanged; tag1 width1',
 same_handle='x==y explicitly legal with same underlying input owner; result still independent',
 outgoing_tuple='exactly one result tensor handle; the condition-only where dynamic tensor tuple is a distinct excluded operation'),
 rows=rows,scope_controls=[dict(label=k,left_shape=x,right_shape=y,tags=[a,b],reason=r) for k,x,y,a,b,r in controls],
 physical_contrasts=[dict(label=k,planned_status=v,detail=n) for k,v,n in extra])
raw=(json.dumps(bundle,separators=(',',':'),sort_keys=True)+'\n').encode()
assert len(raw)<450*1024
(D/'preregister.json').write_bytes(raw)
print('strict',len(rows),'raw output words',sum(len(r['expected']['words']) for r in rows),'scope controls',len(controls),'physical contrasts',len(extra),'fixture bytes',len(raw),hashlib.sha256(raw).hexdigest())
