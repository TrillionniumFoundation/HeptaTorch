"""Stdlib-only exact frozen two-layer F32 ReLU training oracle."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math,struct,sys
sha=lambda b:hashlib.sha256(b).hexdigest()
def f32(v):
 b=struct.pack('<f',float(v));assert F(struct.unpack('<f',b)[0])==v
 return b
def dot(a,b):
 s=F()
 for x,y in zip(a,b,strict=True):
  p=x*y;f32(p);s+=p;f32(s)
 return s
def mm(a,b):return [[dot(row,col) for col in zip(*b)] for row in a]
def tr(a):return list(map(list,zip(*a)))
def arr(*rows):return [list(map(F,r)) for r in rows]
def raw(shape,vals,requires=False,version=0):
 b=b''.join(map(f32,vals));assert len(b)==4*math.prod(shape)
 return dict(dtype='float32',tag=0,shape=shape,strides=[shape[1],1] if shape else [],
  version=version,requires_grad=requires,offset=0,storage_bytes=len(b),view=False,contiguous=True,
  logical_sha256=sha(b),physical_sha256=sha(b),raw_f32le_hex=b.hex(),exact=[str(x) for x in vals])
def tt(m,g=False,ver=0):return raw([len(m),len(m[0])],[x for r in m for x in r],g,ver)
X=arr((1,2),(-1,1));Y=arr((1,),(0,));W1=arr((2,-1),(1,1));W2=arr((1,),(-1,))
lr=F(1,32)
def forward(a,b):
 z=mm(X,a);h=[[max(F(),j) for j in row] for row in z];p=mm(h,b)
 e=[[u[0]-v[0]] for u,v in zip(p,Y,strict=True)]
 sq=[[r[0]**2] for r in e];loss=sum((r[0] for r in sq),F())
 for x in (z,h,p,e,sq):tt(x)
 f32(loss)
 return z,h,p,e,sq,loss
old=forward(W1,W2);Z,H,P,E,Q,L=old
# Independent reverse-mode analytical Jacobian. The strict positivity mask is explicit.
delta=[[2*v[0]] for v in E]
g2=mm(tr(H),delta); dh=mm(delta,tr(W2))
dz=[[a if z>0 else F() for a,z in zip(drow,zrow,strict=True)] for drow,zrow in zip(dh,Z,strict=True)]
g1=mm(tr(X),dz)
assert g1==arr((4,-8),(8,-4)) and g2==arr((16,),(-4,))
sign=lambda m:[[v>0 for v in row] for row in m]
# Independent finite differences: along a single parameter with the other fixed,
# and no mask changes, the model and loss are respectively affine and quadratic.
eps=F(1,16)
for side,A,G in ((0,W1,g1),(1,W2,g2)):
 for i,row in enumerate(A):
  for j in range(len(row)):
   low=[r.copy() for r in A];high=[r.copy() for r in A]
   low[i][j]-=eps;high[i][j]+=eps
   fa=forward(low,W2) if side==0 else forward(W1,low)
   fb=forward(high,W2) if side==0 else forward(W1,high)
   assert sign(fa[0])==sign(fb[0])==sign(Z) and (fb[-1]-fa[-1])/(2*eps)==G[i][j]
def update(a,g):return [[x-lr*d for x,d in zip(r,s,strict=True)] for r,s in zip(a,g,strict=True)]
U1=update(W1,g1);U2=update(W2,g2)
new=forward(U1,U2)
assert sign(new[0])==sign(Z) and min(abs(q) for z in (Z,new[0]) for r in z for q in r)>=1
assert L==8 and new[-1]==F(12625,4096) and U1==arr((F(15,8),-F(3,4)),(F(3,4),F(9,8))) and U2==arr((F(1,2),),(-F(7,8),))
assert f32(lr).hex()=='0000003d'
def graph(v,suffix=''):
 z,h,p,e,q,loss=v;k=lambda n:n+suffix
 return [dict(name=k(n),command=cmd,expected=tt(value,True) if n!='loss' else raw([], [value],True))
   for n,cmd,value in [('preactivation',[32,'x','w1'],z),('hidden',[3,5,k('preactivation')],h),
      ('prediction',[32,k('hidden'),'w2'],p),('error',[2,2,k('prediction'),'target'],e),
      ('squared',[2,3,k('error'),k('error')],q),('loss',[4,7,k('squared')],loss)]]
source,elf,repo,out=map(Path,sys.argv[1:])
sr=source.read_bytes();im=elf.read_bytes()
assert sha(sr)=='f8f924a82763a28d8ad12adbd371f25458ed86d1fbdee2406bb93912847afb4b'
assert sha(im)=='5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d'
import re
anchors={}
for name,markers in {
 'matmul':(b'matmul_forward_point',b'record'),
 'back_matmul_point':(b'backward_getw',),
 'back_unary':(b'v42:u64 = const() {"value":5}',b'{"function":"ltf"}',b'{"function":"tget"}'),
 'sgd':(b'{"function":"tgrad"}',b'{"function":"bump_version"}'),
 'reduce':(b'{"function":"record"}',),
}.items():
 text=re.search(rb'^fn '+name.encode()+rb'\(.*?^end\n',sr,re.M|re.S)[0]
 assert all(m in text for m in markers);anchors[name]=sha(text)
paths=['tests/forward_nograd/native.py','tests/owned_tensor_review/candidate-03-review/executed-sources/review_dynamic.py']
tracked={rel:sha((repo/rel).read_bytes()) for rel in paths}
o=dict(tier='DETACHED_ORDINARY_STD_LIB_PREDICTION',parent_git_prefix='ead53ace',source_sha256=sha(sr),native_elf_sha256=sha(im),
 source_body_hashes=anchors,component_test_hashes=tracked,
 torch_wheel='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
 torch_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 inputs={'x':tt(X),'target':tt(Y),'w1':tt(W1,True),'w2':tt(W2,True)},
 before_update=graph(old),backward=dict(command=[7,'loss'],gradient_read_ops=[[11,'w1'],[11,'w2']],
   gradient_w1=tt(g1),gradient_w2=tt(g2),one_backward_only=True),
 updates=[dict(command=[9,k,0x3d000000],expected=tt(w,True,1),same_tensor_storage_version_authority=True)
   for k,w in [('w1',U1),('w2',U2)]],
 after_update=graph(new,'_fresh'),analysis=dict(old_loss=str(L),new_loss=str(new[-1]),exact_epsilon=str(eps),
   analytic_and_coordinate_finite_differences_match=True,min_absolute_relu_preactivation='1',
   sign_mask=sign(Z),learning_rate=str(lr)),
 future_proof_scope=[
  'Both parameter leaves start Version0; same weight/Storage/Version identities across their one SGD each (exact 0→1). Gradients produced once from the old scalar loss, independently check both raw, widths and owned Storage. Preserve X and target exactly. No optional op91 necessary.',
  'One ordinary owned session: only four external literal roots, all intermediates native and passed as live handles; retire old tape in reverse after single backward and gradient read, then perform two individual op9 on the two leaves. No second backward or reuse of old tape.',
  'Fresh post-update native graph only recomputes the same scalar objective; exact bytes and metadata at each stage, with the expected positive zero at masked ReLU coordinate. Pre- and post-ReLU signed branches are away from zero; freeze on any signed-zero discrepancy.',
  'Static source paths and earlier component tests only demonstrate availability. This fixture is not Torch/native measured or composite qualification, no optimizer groups, bias, general training, default, faults or unrelated scope.'
 ])
b=(json.dumps(o,sort_keys=True,separators=(',',':'))+'\n').encode()
with out.open('xb') as f:f.write(b)
print(len(b),sha(b),o['analysis'])
