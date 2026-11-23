"""Real locked Torch preflight against immutable independent preregistration."""
import hashlib,itertools,json,math,struct,sys
from pathlib import Path
H,L,I,F,O=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
assert sha(L.read_bytes())=='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
assert sha(F.read_bytes())=='f1be3964d390da1450e3a418e8e3dbd17fd3f1f6353cb63046b13d73133b409e'
p=json.loads(F.read_bytes()); assert p['kind']=='FROZEN_PREDICTION_ONLY'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);t.set_num_threads(1)
assert t.get_num_threads()==1 and pin['torch_git_commit']=='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e' and pin['wheel_sha256']=='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3'
CAP=262144
T={
'F64':(4,8,(0,0x8000000000000000,1,0x8000000000000001,0x000fffffffffffff,0x800fffffffffffff,0x0010000000000000,
 0x8010000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000001abc,0xfff8000000002abc,
 0x7ff0000000001357,0xfff0000000002468,0x3ff0000000000000,0xbff0000000000000,0x7fefffffffffffff,0xffefffffffffffff)),
'F16':(5,2,(0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x7c00,0xfc00,0x7e35,0xfe57,0x7c35,0xfc57,0x3c00,0xbc00,0x7bff,0xfbff)),
'BF16':(6,2,(0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x7f80,0xff80,0x7fc5,0xffd7,0x7f85,0xff97,0x3f80,0xbf80,0x7f7f,0xff7f)),
'Bool':(1,1,(0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1))}
def points(s):return itertools.product(*(range(k) for k in s))
def mask(i,c,n):
 v=c['kind']
 if v=='literal':return c['bits'][i]
 if v=='strided':return int(i%c['period'] in c['keep'])
 if v=='edges':return int(i in (0,n-1))
 assert v in ('none','all');return int(v=='all')
def word(i,shape,c,d):
 _,w,p=T[d];v=c['kind'];seed=c['seed']
 if v=='pattern':return p[(i+seed)%len(p)].to_bytes(w,'little')
 if v=='archived_zeros': assert d=='F64';return bytes(w)
 assert v=='finite_f64_exact_add' and d=='F64'
 a,b=divmod(i,shape[1])
 # Both terms and their sum are exact signed integers as binary64.
 return struct.pack('<d',float(((-1 if a&1 else 1)*(2**41+8192*a+seed))+13*b))
DT={'F64':t.float64,'F16':t.float16,'BF16':t.bfloat16,'Bool':t.bool,'I64':t.int64}
def raw(x):return bytes(x.detach().contiguous().reshape(-1).view(t.uint8).tolist())
def make(s,w,d):
 if not w:return t.empty(s,dtype=DT[d])
 v=t.frombuffer(bytearray(w),dtype=t.uint8)
 return v.view(DT[d]).reshape(s).clone()
def md(x):
 return dict(dtype=str(x.dtype).removeprefix('torch.'),device=str(x.device),shape=list(x.shape),strides=list(x.stride()),version=x._version,
   offset=x.storage_offset(),storage_bytes=x.untyped_storage().nbytes(),requires_grad=x.requires_grad,is_view=x._is_view(),words=x.numel())
def observe(x):return dict(md=md(x),raw_sha256=sha(raw(x)))
def call(x,y):return t.ops.aten.masked_select.default(x,y)
rows=[]; guard=[]; mismatch=[]
def report():
 s={z:sum(r['md']['words'] for r in rows if r['token']==z) for z in T}
 body=(O/'selected.bin').read_bytes()
 counts={z:sum(x['status']==z for x in guard) for z in set(x['status'] for x in guard)}
 q=dict(status='MATCH' if not mismatch else 'MISMATCH',oracle=pin,threads=t.get_num_threads(),prereg_sha256=sha(F.read_bytes()),
 rows=rows,guards=guard,mismatch=mismatch,summary=dict(cases=len(rows),words=sum(s.values()),by_dtype=s,raw_bytes=len(body),raw_sha256=sha(body),guards=counts,
 cap=[dict(label=r['label'],k=r['md']['words']) for r in rows if r['value_count']==262144],
 overcap=[dict(label=r['label'],k=r['md']['words']) for r in rows if r['value_count']>262144]),
 historical=p['historical'])
 with (O/'receipt.json').open('x') as f:json.dump(q,f,sort_keys=True,indent=2);f.write('\n')
 print(q['status'],q['summary'],q['mismatch'][:1])
with (O/'selected.bin').open('xb') as f:
 for c in p['cases']:
  xshape,mshape=c['value_shape'],c['mask_shape'];n=c['value_count'];nm=c['mask_count'];d=c['dtype']
  v=b''.join(word(i,xshape,c['source'],d) for i in range(n))
  m=bytes(mask(i,c['mask'],nm) for i in range(nm))
  assert sha(v)==c['value_sha256'] and sha(m)==c['mask_sha256']
  x=make(xshape,v,d);y=make(mshape,m,'Bool')
  b1,b2=observe(x),observe(y)
  assert [b1['raw_sha256'],b2['raw_sha256']]==list(map(sha,(v,m))) and all(a['md']['version']==a['md']['offset']==0 and not a['md']['is_view'] for a in (b1,b2))
  try:z=call(x,y);val=raw(z);zmd=md(z)
  except RuntimeError as err:
   mismatch.append(dict(label=c['label'],error=str(err)));break
  e=c['expected'];k=e['words']; fpos=f.tell();f.write(val)
  item=dict(label=c['label'],token=d,md=zmd,file_offset=fpos,byte_len=len(val),raw_sha256=sha(val),value_count=n,
   input_sha256=[b1['raw_sha256'],b2['raw_sha256']],scope_hypothesis=c['new_hypothesized_status'])
  rows.append(item)
  copied=(len(val)==c['byte_width']*k and sha(val)==e['raw_sha256'] and ('raw_le_hex' not in e or val.hex()==e['raw_le_hex']))
  good=(zmd['dtype']==str(DT[d]).removeprefix('torch.') and zmd['device']=='cpu' and not zmd['requires_grad'] and not zmd['is_view']
    and all(zmd[key]==e[key] for key in ['shape','strides','version','offset','storage_bytes','words']))
  same=(observe(x)==b1 and observe(y)==b2)
  fresh=(not k or all(z.untyped_storage().data_ptr()!=a.untyped_storage().data_ptr() for a in (x,y) if a.numel()))
  item.update(unchanged_source=same,nonempty_distinct_buffers=fresh if k else 'no_empty_pointer_identity_claim')
  if d=='Bool' and c['label']=='Bool-equal-all-raw-signed':
   assert {0,1}==set(val) and len(val)==n
   item['false_selected_bytes']=val.count(0)
  if c['label']=='F64-historical-value-accept':
   old=p['historical']['other-value-F64-remains-unadmitted']
   assert old['old_operand_raw']==[v.hex(),m.hex()] and k==4 and val==bytes(32)
   item['historical_old_frame']=old['frame']
  if not (good and copied and same and fresh):
   mismatch.append(dict(label=c['label'],observed=item,expected=e,raw_ok=copied));break
if mismatch:
 report();raise SystemExit(1)
for g in p['guards']:
 l=g['label']; item=dict(label=l,status='NO_ANALOGOUS_WIRE_API')
 if 'torch_accept' not in g:guard.append(item);continue
 xs=g.get('value_shape',[2,3]);ms=g.get('mask_shape',[2,3]); xname=g['dtype'];mname=g.get('mask','Bool')
 row=lambda d,i: (struct.pack('<q',i) if d=='I64' else word(i,[1],dict(kind='pattern',seed=0),d))
 x=make(xs,b''.join(row(xname,i) for i in range(math.prod(xs))),xname)
 y=make(ms,b''.join(row(mname,i%2) for i in range(math.prod(ms))),mname)
 if l=='actual-offset-value-view':
  x=make([7],b''.join(row(xname,i) for i in range(7)),xname)[1:].reshape(xs);assert x._is_view() and x.storage_offset()==1
 if l=='actual-offset-mask-view':
  y=make([7],b''.join(row(mname,i) for i in range(7)),mname)[1:].reshape(ms);assert y._is_view() and y.storage_offset()==1
 if l=='actual-grad-value':x.requires_grad_(True);assert x.requires_grad
 if l=='integer-mask-still-forbidden':
  x=make([2,3],struct.pack('<6q',*range(6)),xname);y=make([1,3],struct.pack('<3q',1,0,1),mname)
  h=p['historical']['nonbool-value-used-as-mask']
  assert h['old_operand_raw']==[raw(x).hex(),raw(y).hex()]
  item['historical_old_frame']=h['frame']
 try:
  result=call(x,y); item.update(status='ACCEPT',md=md(result),raw_sha256=sha(raw(result)));ok=True
 except RuntimeError as err:
  item.update(status='REJECT_RUNTIME_ERROR',error=str(err));ok=False
 guard.append(item)
 if ok!=g['torch_accept']:mismatch.append(item);break
report();raise SystemExit(bool(mismatch))
