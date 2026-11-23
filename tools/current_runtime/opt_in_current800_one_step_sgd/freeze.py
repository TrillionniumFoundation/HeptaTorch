"""Detached stdlib-only one-step ordinary F32 linear SGD predictions."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math,struct,sys
sha=lambda b:hashlib.sha256(b).hexdigest()
def bits(x):
 x=F(x);r=struct.pack('<f',float(x))
 assert F(struct.unpack('<f',r)[0])==x
 return r
def tensor(shape,values,grad=False,version=0):
 n=math.prod(shape);assert len(values)==n
 b=b''.join(map(bits,values))
 return dict(dtype='float32',input_tag=0,shape=shape,strides=[] if len(shape)==0 else [shape[-1],1],
             requires_grad=grad,storage_offset=0,storage_bytes=4*n,view=False,contiguous=True,version=version,
             values=list(map(str,values)),raw_f32le_hex=b.hex(),logical_sha256=sha(b),physical_sha256=sha(b))
X=((F(1),F(0)),(F(0),F(2))); T=(F(3),F(0)); W=(F(1),F(1)); LR=F(1,8)
def layers(w):
 y=[]
 for row in X:
  tmp=F(0)
  for a,b in zip(row,w,strict=True):
   term=a*b;bits(term);tmp+=term;bits(tmp)
  y.append(tmp)
 e=[a-b for a,b in zip(y,T,strict=True)]
 q=[a*a for a in e];s=sum(q,F(0))
 for v in (*y,*e,*q,s):bits(v)
 return y,e,q,s
pred,err,sq,loss=layers(W)
gradient=[sum((2*X[i][j]*err[i] for i in range(len(X))),F(0)) for j in range(len(W))]
# Independent symmetric difference is exact for this quadratic (not a float estimate).
for j,g in enumerate(gradient):
 minus=list(W);plus=list(W);step=F(1,8)
 minus[j]-=step;plus[j]+=step
 assert (layers(plus)[3]-layers(minus)[3])/(2*step)==g
updated=[w-LR*g for w,g in zip(W,gradient,strict=True)]
post=layers(updated)
assert [pred,err,sq,loss]==[[1,2],[-2,2],[4,4],8]
assert gradient==[-4,8] and updated==[F(3,2),0] and post==([F(3,2),0],[-F(3,2),0],[F(9,4),0],F(9,4))
assert bits(LR).hex()=='0000003e'
inputs={'x':tensor([2,2],[v for row in X for v in row]),
        'weight':tensor([2,1],list(W),True),
        'target':tensor([2,1],list(T))}
def forward(data,suffix=''):
 pred,err,sq,loss=data;k=lambda key:key+suffix
 return [
  dict(name=k('prediction'),command=[32,'x','weight'],expected=tensor([2,1],pred,True)),
  dict(name=k('error'),command=[2,2,k('prediction'),'target'],expected=tensor([2,1],err,True)),
  dict(name=k('squared'),command=[2,3,k('error'),k('error')],expected=tensor([2,1],sq,True)),
  dict(name=k('loss'),command=[4,7,k('squared')],expected=tensor([], [loss],True)),
 ]
a,b,out,*anchors=map(Path,sys.argv[1:])
assert len(anchors)==4
combined=sha(a.read_bytes()); elf=sha(b.read_bytes())
assert combined=='f8f924a82763a28d8ad12adbd371f25458ed86d1fbdee2406bb93912847afb4b'
assert elf=='5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d'
s=a.read_bytes()
assert b'{"function":"matmul_forward_point"}' in s and b'{"function":"sgd"}' in s
assert b'call(v0,v1,v2) {"function":"backward_seed"}' in s
assert b'call(v120,v121,v125) {"function":"tgrad"}' in s and b'call(v126,v127) {"function":"bump_version"}' in s
for f,marks in zip(anchors,[
 (b'n.one(29,result,seed)',b'gradient=n.one(11,base)'),
 (b'return self.command(9,t,bits(lr))',b'return self.command(29,t,seed)'),
 (b"old.sgd(x,F(1,4))",b"check_values(rows[rx],flatten(r['updated']),'owned-SGD')"),
 (b'gradient',b'(91,parameter)'),
],strict=True):
 content=f.read_bytes();assert all(word in content for word in marks),(f,marks)
result=dict(tier='DETACHED_STD_LIB_PREDICTIONS_ONLY',candidate='HeptaTorch #71 exact opt-in current800 application image',
 parent_git='e7e02a54618bdd111982e447189f810d9164731c',parent_tree='0d5c068662971108f08466558abda82e5fd1c71d',
 source_sha256=combined,native_elf_sha256=elf,
 locked_torch_git='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 locked_torch_wheel='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
 anchors=[dict(path_hint='tests/'+str(x).split('/tests/',1)[-1],sha256=sha(x.read_bytes())) for x in anchors],
 input_tensors=inputs,before_update=forward((pred,err,sq,loss)),backward=dict(command=[7,'loss'],
  mathematical_seed='1',gradient_wire_command=[11,'weight'],optional_gradient_handle=[91,'weight'],
  expected_gradient=tensor([2,1],gradient),version_of_trainable_leaf_remains=0),
 update=dict(command=[9,'weight',0x3e000000],learning_rate='1/8',weight_before=inputs['weight'],
  weight_after=tensor([2,1],updated,True,1),same_tensor_storage_version_authority=True,grad_bytes_unchanged=True),
 after_update=forward(post,'_fresh'),
 numerical_relations=dict(old_loss='8',new_loss='9/4',decrease='23/4',gradient_is_exact_analytical=True,
    gradient_verified_via_exact_quadratic_central_difference=True),
 notes=[
  'Future locked PyTorch oracle: requires_grad only on W, ordinary eager mm/sub/mul/sum, one scalar loss backward and one native-equivalent no-state, no-momentum SGD step. Stop and preserve any raw/metadata mismatch; this file is not a measurement.',
  'Future HIA check: one owned session, 3 literal roots only; chain live results, op13/98 for forward/leaf raw and owner snapshots, op11 actual raw G. Optional op91 is permitted only with fresh known legitimate gradient owner and released extra handle before update, no ABA or invalid experiment.',
  'Retire only ordinary old forward graph handles in reverse topological order after first backward and gradient read, retaining exact input/target/parameter. Apply in-place SGD op9 once to the same legitimate leaf. Fresh recompute with unchanged X and target, then release fresh graph and roots normally; no reused or retained old backward tape and no second backward.',
  'Input and target bit-identical, scalar loss and step bits exact, model weight Tensor/Storage/Version identity preserved and counter 0→1, orthogonal gradient storage. The expected forward-stage Version0 and gradient metadata remain hypotheses pending exact locked observation.',
  'Single toy application shape only; no optimizer-group API, higher order, full training parity, faults, pressure, invalid ownership, current-default or held code.'
 ])
raw=(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n').encode()
with out.open('xb') as f:f.write(raw)
print(len(raw),sha(raw),result['numerical_relations'])
