"""Real locked Torch preflight against immutable independent preregistration."""
import hashlib,itertools,json,math,struct,sys
from pathlib import Path
H,L,I,F,O=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
assert sha(L.read_bytes())=='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
assert sha(F.read_bytes())=='81aad2d489f77adbba131bf703456ca5e7fb04ad9876197376307bc128d3cc19'
p=json.loads(F.read_bytes()); assert p['kind']=='UNMEASURED_FROZEN_PREDICTIONS_ONLY'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);t.set_num_threads(1)
assert t.get_num_threads()==1 and pin['torch_git_commit']=='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e' and pin['wheel_sha256']=='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3'
LOW=-(1<<63); HIGH=(1<<63)-1
PAT=(LOW,LOW+1,LOW+0xffffffff,LOW+(1<<40),-(1<<62),-(1<<53)-1,-(1<<32)-1,-(1<<32),-1,
 0,1,0x7fffffff,0x80000000,0xffffffff,1<<32,(1<<53)+1,(1<<62)+0x5a,0x5555555555555555,HIGH-1,HIGH)
def iv(a):
 n=a['value_count'];s=a['source'];seed=s['seed']
 if s['kind']=='signed_pattern':return [PAT[(i+seed)%len(PAT)] for i in range(n)]
 assert s['kind']=='i64_add_cartesian'
 nr,nc=a['value_shape']; bases=(LOW,-(1<<60),1<<61,HIGH-(1<<24))
 return [bases[row&3]+(-1 if row&3==1 else 1)*(8192*row+seed)+13*col for row in range(nr) for col in range(nc)]
def bv(a):
 n=a['mask_count'];c=a['mask'];k=c['kind']
 if k=='literal':return bytes(c['bits'])
 if k=='strided':return bytes(int(i%c['period'] in c['remainders']) for i in range(n))
 if k=='edges':return bytes(int(i in (0,n-1)) for i in range(n))
 assert k in ('all','none')
 return bytes([int(k=='all')])*n
def inode(shape,raw,dtype):
 if not raw:return t.empty(shape,dtype=dtype)
 return t.frombuffer(bytearray(raw),dtype=dtype).reshape(shape).clone()
def raw(x):return bytes(x.detach().contiguous().reshape(-1).view(t.uint8).tolist())
def md(x):return dict(dtype=str(x.dtype).removeprefix('torch.'),shape=list(x.shape),strides=list(x.stride()),offset=x.storage_offset(),
 version=x._version,storage_bytes=x.untyped_storage().nbytes(),requires_grad=x.requires_grad,is_view=x._is_view(),device=str(x.device),count=x.numel())
def call(x,m):return t.ops.aten.masked_select.default(x,m)
cases=[];controls=[];mismatches=[]
bpath=O/'selected-i64le.bin'
with bpath.open('xb') as f:
 for a in p['cases']:
  x0=struct.pack('<%dq'%a['value_count'],*iv(a));m0=bv(a)
  assert sha(x0)==a['pre_source_sha256'] and sha(m0)==a['pre_mask_sha256']
  x=inode(a['value_shape'],x0,t.int64); m=inode(a['mask_shape'],m0,t.bool)
  bound=(x,m);observe=lambda:[dict(meta=md(z),raw_sha256=sha(raw(z))) for z in bound]
  before=observe(); assert [d['raw_sha256'] for d in before]==list(map(sha,(x0,m0)))
  assert all(d['meta']['version']==d['meta']['offset']==0 and not d['meta']['is_view'] and not d['meta']['requires_grad'] for d in before)
  try:y=call(x,m)
  except RuntimeError as err:
   mismatches.append(dict(label=a['label'],runtime_error=str(err)));continue
  got=raw(y);mout=md(y);e=a['expected'];after=observe()
  out=dict(label=a['label'],metadata=mout,offset=f.tell(),length=len(got),sha256=sha(got),input_before=before,input_after=after)
  f.write(got);cases.append(out)
  want=e['metadata']; basic=(all(mout[k]==want[k] for k in want) and not mout['is_view'] and not mout['requires_grad'] and mout['device']=='cpu' and before==after)
  byte_ok=(sha(got)==e['torch_raw_sha256'] and len(got)==e['torch_raw_bytes'] and ('raw_i64le_hex' not in e or e['raw_i64le_hex']==got.hex()))
  # Fresh nonempty storage is independently observable without treating it as public Tensor identity.
  separated=mout['count']==0 or all(y.untyped_storage().data_ptr()!=z.untyped_storage().data_ptr() for z in bound if z.numel())
  out['nonempty_buffers_separate']=separated if mout['count'] else 'EMPTY_POINTER_SENTINEL_NOT_IDENTITY'
  # Local mutate-and-read on one representative fresh result, after original raw capture.
  if a['label']=='equal-all-signed':
   y[0]=123456789;out['separation_mutation_check']=observe()==before and y._version==1
   separated=separated and out['separation_mutation_check']
  if not (basic and byte_ok and separated):mismatches.append(dict(expected=e,actual=out))
for g in p['controls']:
 label=g['label'];r=dict(label=label,new_candidate_hypothesis=g['new'])
 if 'torch_accept' not in g:
  r['status']='NO_ANALOGOUS_WIRE_API';controls.append(r);continue
 xs=g.get('value_shape',[2,3]);ms=g.get('mask_shape',[2,3])
 xn=math.prod(xs);mn=math.prod(ms)
 x=inode(xs,struct.pack('<%dq'%xn,*range(xn)),t.int64)
 m=inode(ms,bytes(i%2 for i in range(mn)),t.bool)
 if label=='nonbool-value-used-as-mask':m=m.to(t.int64)
 if label=='other-value-F64-remains-unadmitted':x=x.to(t.float64)
 if label=='actual-I64-value-offset-view':
  x=inode([7],struct.pack('<7q',*range(7)),t.int64)[1:].reshape(xs);assert x._is_view() and x.storage_offset()==1
 if label=='actual-Bool-mask-offset-view':
  m=inode([7],bytes(i%2 for i in range(7)),t.bool)[1:].reshape(ms);assert m._is_view() and m.storage_offset()==1
 try:
  z=call(x,m);r.update(status='ACCEPT',metadata=md(z),raw_sha256=sha(raw(z)));accepted=True
 except RuntimeError as err:r.update(status='REJECT_RUNTIME_ERROR',error=str(err));accepted=False
 if accepted!=g['torch_accept']:mismatches.append(r)
 controls.append(r)
rep=dict(status='MATCH' if not mismatches else 'MISMATCH',threads=t.get_num_threads(),oracle=pin,fixture_sha256=sha(F.read_bytes()),
 rows=cases,controls=controls,mismatches=mismatches,summary=dict(cases=len(cases),words=sum(r['length']//8 for r in cases),bytes=bpath.stat().st_size,
 raw_sha256=sha(bpath.read_bytes()),candidate_supported_words=sum(r['length']//8 for r in cases if not r['label'].startswith('overcap')),
 overcap=[dict(label=r['label'],selected=r['length']//8) for r in cases if r['label'].startswith('overcap')],
 guards={k:sum(r['status']==k for r in controls) for k in {r['status'] for r in controls}}))
with (O/'receipt.json').open('x') as fd:json.dump(rep,fd,sort_keys=True,indent=2);fd.write('\n')
print(rep['status'],rep['summary'],rep['mismatches'][:1])
raise SystemExit(bool(mismatches))
