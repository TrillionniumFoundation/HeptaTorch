"""Detached stdlib-only ordinary F32 inference fixture."""
from fractions import Fraction
import hashlib,json,struct,sys
from pathlib import Path
sha=lambda b:hashlib.sha256(b).hexdigest()
def f32(x):
 x=Fraction(x); raw=struct.pack('<f',float(x))
 assert Fraction(struct.unpack('<f',raw)[0])==x,(x,raw.hex())
 return raw
def matrix(rows):
 x=[[Fraction(v) for v in line] for line in rows]
 assert x and x[0] and all(len(r)==len(x[0]) for r in x)
 return x
def mm(left,right):
 assert len(left[0])==len(right)
 out=[]
 for a in left:
  row=[]
  for b in zip(*right):
   acc=Fraction(0)
   for av,bv in zip(a,b,strict=True):
    p=av*bv;f32(p);acc+=p;f32(acc)
   row.append(acc)
  out.append(row)
 return out
def bias(m,b):
 assert len(m[0])==len(b)
 out=[[x+y for x,y in zip(r,b,strict=True)] for r in m]
 for row in out:
  for x in row:f32(x)
 return out
def relu(m):return [[max(Fraction(0),x) for x in row] for row in m]
def encode(values,shape):
 if len(shape)==2:
  assert [len(values),len(values[0])]==shape
  values=[w for r in values for w in r]
 assert len(values)==__import__('math').prod(shape)
 raw=b''.join(map(f32,values));vals=[str(x) for x in values]
 return dict(dtype='float32',wire_tag=0,shape=shape,strides=[shape[-1],1] if len(shape)==2 else [1],version=0,
     requires_grad=False,offset=0,storage_bytes=len(raw),contiguous=True,view=False,raw_f32le_hex=raw.hex(),
     logical_sha256=sha(raw),physical_sha256=sha(raw),retained_unchanged=True,
     values_exact_rational_row_major=vals)
def case(label,x,w1,b1,w2,b2):
 x,w1,w2=map(matrix,(x,w1,w2));b1=list(map(Fraction,b1));b2=list(map(Fraction,b2))
 a=mm(x,w1);p=bias(a,b1);h=relu(p);v=mm(h,w2);y=bias(v,b2)
 d=lambda a:[len(a),len(a[0])]
 inps={n:encode(m,d(m) if n not in ('b1','b2') else [len(m)])
       for n,m in [('x',x),('w1',w1),('b1',b1),('w2',w2),('b2',b2)]}
 route=[('mm1',[32,'x','w1'],a),('add1',[2,1,'mm1','b1'],p),
        ('relu1',[3,5,'add1'],h),('mm2',[32,'relu1','w2'],v),
        ('prediction',[2,1,'mm2','b2'],y)]
 stages=[dict(name=n,command=wire,expected=encode(m,d(m)),new_distinct_tensor_storage_version_authority=True) for n,wire,m in route]
 before=[w for row in p for w in row];after=[w for row in h for w in row]
 assert any(v<0 for v in before) and any(v>0 for v in before) and before!=after
 assert [len(y),len(y[0])]==[len(x),len(b2)]
 return dict(label=label,input_geometry=[len(x),len(x[0]),len(w2),len(b2)],inputs=inps,stages=stages)
specs=[
 ('rectangular-batch-mixed-signs',[[2,-1,3],[-1,4,0]],[[1,-2,0,1],[2,1,-1,0],[-1,2,1,-2]],[1,-1,2,3],
   [[-1,2],[1,-1],[2,-3],[-2,1]],[-3,4]),
 ('singleton-batch-dyadic',[['1/2','-3/2',2,'-1/4']],[[2,-1,0],['1/2',2,-1],[-1,1,2],[1,-2,4]],
   ['-1/4','1/2',-2],[[2,-1,'1/2'],[-2,'1/2',1],[1,2,-1]],['1/4',-1,'-3/2']),
 ('three-batch-hidden-zero-row',[[0,0],[1,2],[-2,1]],[[1,-1,2],[2,1,-1]],[-1,0,-2],
   [[2],[-1],['1/2']],['-1/4']),
 ('hidden-width-one-broadcast-two-outputs',[[2,-1],[-2,1],[1,2],[0,0]],[[2],[1]],[-2],
   [[-1,2]],[3,-1]),
]
cases=[case(*x) for x in specs]
code,elf,dest=map(Path,sys.argv[1:4]);assert len(sys.argv)==4
src=code.read_bytes();assert sha(src)=='f8f924a82763a28d8ad12adbd371f25458ed86d1fbdee2406bb93912847afb4b'
assert sha(elf.read_bytes())=='5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d'
assert b'call(v750,v765,v762,forward_left45,forward_right45) {"function":"matmul_forward_point"}' in src
assert b'branch check_yes original(check_s,check_y,check_k) b0(check_s,check_y,check_k,check_a,check_b)' in src
assert b'v6:u64 = bit.or(forward_a,forward_a)' in src and b'v9:u64 = bit.or(forward_b,forward_b)' in src
assert cases[0]['stages'][-1]['expected']['values_exact_rational_row_major']==['9','-14','-10','17']
summary=dict(cases=len(cases),stages=sum(len(c['stages']) for c in cases),
 input_words=sum(v['storage_bytes']//4 for c in cases for v in c['inputs'].values()),
 output_words=sum(s['expected']['storage_bytes']//4 for c in cases for s in c['stages']))
r=dict(tier='DETACHED_STD_LIB_PREDICTIONS_ONLY',candidate='HeptaTorch #71 exact opt-in current800 coexistence image',
 source_sha256=sha(src),native_elf_sha256=sha(elf.read_bytes()),
 locked_torch_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 locked_torch_wheel='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
 summary=summary,cases=cases,notes=[
  'No external tensor library used; exact rational predictions with every multiply and partial accumulation representable in binary32.',
  'Future qualifier keep all five HIA operations chained in one owned session by live handles. Python only submits op97 input literals, feeds returned handles into 32/2/3, reads 13/98 and drops 41. Do not inject host-computed expected intermediates or use Torch as native backend.',
  'Require complete logical and independent physical raw slices, fresh distinct Tensor/Storage/Version authorities per stage and unchanged prior stages and inputs. Version0 is a prereg hypothesis for these rank2×rank2 operations, not a measurement. Separate PyTorch process only as oracle, compare bytes and metadata.',
  'Source checked matmul no-grad frontend receives explicit left and right nodes, preserving the old AD route. Matrix products use rank2 × rank2 only. Main typed-ReLU775, reduction, unsupported and held tasks excluded.',
  'At most these four toy numerics; not NN frontend support, general inference parity, gradients, training, performance or default-runtime qualification.'
 ])
raw=(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode()
with dest.open('xb') as f:f.write(raw)
print(len(raw),sha(raw),summary)
