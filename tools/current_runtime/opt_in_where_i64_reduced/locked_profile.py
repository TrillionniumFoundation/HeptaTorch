"""Independent measured preflight; never imports HIA or the expectation generator."""
import hashlib,itertools,json,math,struct,sys
from pathlib import Path
H,LOCK,INSTALL,FIX,REPORT=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
assert sha(LOCK.read_bytes())=='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
raw=FIX.read_bytes()
fsha=sha(raw)
assert fsha=='57b06e065d8ac10bf92f8d4133d1f525be91cd2f864bbc4d068bd8260a044502'
pred=json.loads(raw);assert pred['status']=='PREREGISTERED_NOT_MEASURED'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
torch,oracle=load_oracle(LOCK,INSTALL)
assert oracle['torch_git_commit']=='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e'
assert oracle['wheel_sha256']=='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3'
torch.set_num_threads(1)
DT={1:torch.bool,2:torch.int64,5:torch.float16,6:torch.bfloat16}
def make(o):
 w=o['words'];tag=o['tag'];shape=o['shape']
 if tag in (1,2) or not w:return torch.tensor(w,dtype=DT[tag]).reshape(shape)
 t=torch.frombuffer(bytearray(struct.pack(f'<{len(w)}H',*w)),dtype=torch.int16)
 return t.view(DT[tag]).reshape(shape)
def bytes_of(a):return bytes(a.contiguous().reshape(-1).view(torch.uint8).tolist())
def pack(w):return struct.pack(f'<{len(w)}H',*w)
def fold(a,width):
 cut=max(0,a.bit_length()-width)
 if not cut:return a
 base=a>>cut; bits=a&((1<<cut)-1); midpoint=1<<(cut-1)
 return (base+(bits>midpoint or bits==midpoint and base&1!=0))<<cut
def val(n,t,first):
 sign=0x8000 if n<0 else 0
 a=abs(n)
 if first:a=fold(a,first)
 p,bias,inf=(11,15,0x7c00) if t==5 else (8,127,0x7f80)
 a=fold(a,p)
 if not a:return sign
 e=a.bit_length()-1; code=((e+bias)<<(p-1))
 return sign|(inf if code>=inf else code|((a<<(p-1))>>e)&((1<<(p-1))-1))
def coords(s):return itertools.product(*(range(d) for d in s))
def flat(o,idx):
 ix=0;ds=o['shape']
 for size,j in zip(ds,idx[len(idx)-len(ds):],strict=True):
  ix=ix*size+(0 if size==1 else j)
 return ix
def klass(t,bits):
 e,f=(5,10) if t==5 else (8,7)
 sign='-' if bits&0x8000 else '+'
 power,payload=divmod(bits&0x7fff,1<<f)
 if power==(1<<e)-1:return sign+('inf' if not payload else 'qnan' if payload>>(f-1) else 'snan')
 if power==0:return sign+('subnormal' if payload else 'zero')
 return sign+'normal'
def record_diff(where,label,a,e,source,mismatches):
 for i,(observed,expected) in enumerate(zip(a,e,strict=True)):
  if observed==expected:continue
  mismatches[where]+=1
  if len(events)<100:events.append(dict(kind=where,label=label,index=i,observed=observed,expected=expected,source=source[i]))
rows=[];events=[];summary={}
for t in (5,6):
 receipt=dict(rows=0,empty=0,direct_rows=0,direct_words=0,where_words=0,selected_int=0,selected_reduced=0,
              counterfactual_direct=0,counterfactual_f64=0,selected_reduced_classes={},mismatches={'direct':0,'where_frozen':0,'counterfactual_account':0,'selected_identity':0})
 for r in (r for r in pred['rows'] if r['dtype']==t):
  args=r['operands']; m,x,y=ts=[make(o) for o in args]
  assert all(a.device.type=='cpu' and a.is_contiguous() and not a.requires_grad for a in ts)
  snap=list(map(bytes_of,ts));ver=[a._version for a in ts]
  iidx=next(i for i,o in enumerate(args) if o['tag']==2)
  intarg=args[iidx]; ip=ts[iidx]
  cast=ip.to(DT[t])
  assert cast.dtype==DT[t] and list(cast.shape)==intarg['shape']
  exact_direct=bytes_of(cast); direct_words=list(v[0] for v in struct.iter_unpack('<H',exact_direct)); ints=intarg['words']
  model=[val(n,t,24) for n in ints]
  record_diff('direct',r['label'],direct_words,model,ints,receipt['mismatches'])
  result=torch.ops.aten.where.self(m,x,y)
  assert result.dtype==DT[t] and list(result.shape)==r['shape'] and result._version==0 and not result.requires_grad
  assert result.untyped_storage()._cdata not in {a.untyped_storage()._cdata for a in ts}
  assert snap==list(map(bytes_of,ts)) and ver==[a._version for a in ts]
  wraw=bytes_of(result); actual=list(v[0] for v in struct.iter_unpack('<H',wraw))
  expected=list(v[0] for v in struct.iter_unpack('<H',bytes.fromhex(r['expected_hex'])))
  counter_d=counter_f=0; source=[]; selected_int=selected_reduced=0
  for ijk,w in zip(coords(r['shape']),actual,strict=True):
   i=1 if args[0]['words'][flat(args[0],ijk)] else 2
   k=flat(args[i],ijk); n=args[i]['words'][k]
   source.append([args[i]['tag'],n])
   if i==iidx:
    selected_int+=1
    counter_d+=w!=val(n,t,None);counter_f+=w!=val(n,t,53)
    record_diff('selected_identity',r['label'],[w],[direct_words[k]],[n],receipt['mismatches'])
   else:
    selected_reduced+=1
    cls=klass(t,n);receipt['selected_reduced_classes'][cls]=receipt['selected_reduced_classes'].get(cls,0)+1
    record_diff('selected_identity',r['label'],[w],[n],[n],receipt['mismatches'])
  record_diff('where_frozen',r['label'],actual,expected,source,receipt['mismatches'])
  record_diff('counterfactual_account',r['label'],[counter_d,counter_f],[r['direct_disagreements'],r['f64_disagreements']],[None,None],receipt['mismatches'])
  assert selected_int==r['selected_int'] and selected_reduced==r['selected_reduced']
  for k,n in [('rows',1),('empty',not actual),('direct_rows',1),('direct_words',len(direct_words)),('where_words',len(actual)),('selected_int',selected_int),
              ('selected_reduced',selected_reduced),('counterfactual_direct',counter_d),('counterfactual_f64',counter_f)]:
   receipt[k]+=n
  rows.append(dict(label=r['label'],where_sha256=sha(wraw),direct_sha256=sha(exact_direct),direct_words=len(direct_words),where_words=len(actual),
                   counterfactual_direct=counter_d,counterfactual_f64=counter_f))
 assert {s+c for s in ('+','-') for c in ('zero','subnormal','inf','qnan','snan')}<=set(receipt['selected_reduced_classes'])
 summary[str(t)]=receipt
out=dict(status='MATCH' if not events else 'MISMATCH',fixture_sha256=fsha,summary=summary,oracle=oracle,first_differences=events,rows=rows,
        rank0_controls_not_measured=len(pred['hypothesis']))
with REPORT.open('x') as o:json.dump(out,o,sort_keys=True,indent=2);o.write('\n')
print(json.dumps(dict(status=out['status'],summary=summary,first_differences=events[:8]),sort_keys=True))
raise SystemExit(bool(events))
