"""Real locked Torch preflight against immutable independent preregistration."""
import hashlib,itertools,json,math,struct,sys
from pathlib import Path
H,L,I,F,O=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
assert sha(L.read_bytes())=='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
assert sha(F.read_bytes())=='d7b8af35d5288fd870c998e0564c3da7c6efcaf88c5723e438cd6e4999493cd1'
p=json.loads(F.read_bytes()); assert p['kind']=='PREDICTION_ONLY'
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
T['F32']=(0,4,(0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f800000,0xff800000,
 0x7fc12345,0xffc23456,0x7f812345,0xff823456,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff))
T['I64']=(2,8,tuple(x&((1<<64)-1) for x in (-(1<<63),-(1<<63)+1,-(1<<62),-(1<<53)-1,-(1<<32)-1,-1,0,1,
 0x7fffffff,0x80000000,0xffffffff,1<<32,(1<<53)+1,(1<<62)+0x5a,0x5555555555555555,(1<<63)-2,(1<<63)-1)))
bytag={v[0]:d for d,v in T.items()}
def points(shape):
 return itertools.product(*(range(n) for n in shape))
def merge(*shapes):
 r=max(map(len,shapes));a,b=[(1,)*(r-len(s))+tuple(s) for s in shapes]
 out=[]
 for y,z in zip(a,b,strict=True): 
  if y==z or z==1:out.append(y)
  elif y==1:out.append(z)
  else:return None
 return tuple(out)
def key(c,shape):
 return tuple(0 if d==1 else k for k,d in zip(c[len(c)-len(shape):],shape,strict=True))
def words(d,shape,seed):
 _,w,p=T[d]; return b''.join(p[(i+seed)%len(p)].to_bytes(w,'little') for i in range(math.prod(shape)))
def flags(m,style):
 count=math.prod(m)
 if style=='all':return b'\1'*count
 if style=='none':return bytes(count)
 if style=='first':return bytes([1])+bytes(count-1)
 if style=='order':return bytes(int((i*7+2)%11 in (0,1,2,3,7,9)) for i in range(count))
 raise AssertionError(style)
DT={'F32':t.float32,'I64':t.int64,'F64':t.float64,'F16':t.float16,'BF16':t.bfloat16,'Bool':t.bool}
def raw(x):return bytes(x.detach().contiguous().reshape(-1).view(t.uint8).tolist())
def new(shape,b,dt):
 if not b:return t.empty(shape,dtype=DT[dt])
 return t.frombuffer(bytearray(b),dtype=t.uint8).view(DT[dt]).reshape(shape).clone()
def md(x):
 return dict(dtype=str(x.dtype).removeprefix('torch.'),device=str(x.device),shape=list(x.shape),strides=list(x.stride()),
  words=x.numel(),offset=x.storage_offset(),version=x._version,storage_bytes=x.untyped_storage().nbytes(),is_view=x._is_view(),requires_grad=x.requires_grad)
def check(x):return md(x),sha(raw(x))
def call(x,m):return t.ops.aten.masked_select.default(x,m)
body=O/'selected.bin';rows=[];mismatches=[];guards=[]
with body.open('xb') as fd:
 for x in p['cases']:
  d=x['dtype'];xs=x['value_shape'];ms=x['mask_shape'];e=x['expected'];rec=x['recipe']
  if 'operands' in rec:b,c=map(bytes.fromhex,rec['operands'])
  else:b=words(d,xs,rec['seed']);c=flags(ms,rec['mask_style'])
  assert [sha(b),sha(c)]==[x['value_sha256'],x['mask_sha256']]
  u=new(xs,b,d);v=new(ms,c,'Bool');pre=[check(u),check(v)]
  assert [pre[0][1],pre[1][1]]==[sha(b),sha(c)] and all(z[0]['offset']==z[0]['version']==0 and not z[0]['is_view'] for z in pre)
  broadcast_shape=list(t.broadcast_shapes(u.shape,v.shape)); assert broadcast_shape==x['merged_shape']
  try:o=call(u,v);mb=md(o);payload=raw(o)
  except RuntimeError as er:
   mismatches.append(dict(label=x['label'],error=str(er)));break
  r=dict(label=x['label'],merged_shape=broadcast_shape,merged_count=x['merged_count'],metadata=mb,at=fd.tell(),length=len(payload),raw_sha256=sha(payload),exact_inputs=[sha(b),sha(c)])
  fd.write(payload);same=[check(u),check(v)]==pre
  good=all(mb[k]==e[k] for k in ('words','shape','strides','offset','version','storage_bytes')) and mb['dtype']==str(DT[d]).removeprefix('torch.') and mb['device']=='cpu' and not mb['is_view'] and not mb['requires_grad']
  fresh=not mb['words'] or all(o.untyped_storage().data_ptr()!=z.untyped_storage().data_ptr() for z in (u,v) if z.numel())
  equal=sha(payload)==e['raw_sha256'] and ('raw_le_hex' not in e or payload.hex()==e['raw_le_hex'])
  r.update(unchanged_inputs=same,distinct_nonempty_storage=fresh if mb['words'] else 'NO_EMPTY_POINTER_IDENTITY')
  if 'operands' in rec:r['held']=dict(stream=rec['corpus'],frame=rec['frame'],old_request_sha256=rec['request_sha256'],old_owner_sha256=rec['identical_owner_sha256'])
  rows.append(r)
  if not (same and good and fresh and equal):mismatches.append(dict(want=e,actual=r));break
if not mismatches:
 for g in p['guards']:
  l=g['label']; x=g.get('value',[2,3]);m=g.get('mask',[2,3]); r=dict(label=l,status='NO_TORCH_EVALUATION')
  if 'torch' not in g:
   if l not in ('nonbool-mask-stays-forbidden','actual-view-mask','actual-grad-f32'):
    guards.append(r);continue
  cases=math.prod(x);mask_count=math.prod(m)
  u=new(x,words('F32',x,0),'F32');v=new(m,flags(m,'order'),'Bool');accept=g.get('torch',True)
  if l=='nonbool-mask-stays-forbidden':v=v.to(t.int64);accept=False
  if l=='actual-view-mask':v=new([7],flags([7],'order'),'Bool')[1:].reshape(m);assert v._is_view() and v.storage_offset()==1
  if l=='actual-grad-f32':u.requires_grad_(True)
  try:y=call(u,v);r.update(status='ACCEPT',metadata=md(y),raw_sha256=sha(raw(y)));actual=True
  except RuntimeError as er:r.update(status='REJECT_RUNTIME_ERROR',error=str(er));actual=False
  guards.append(r)
  if actual!=accept:mismatches.append(r);break
p0=body.read_bytes()
s=dict(cases=len(rows),selected_words=sum(x['metadata']['words'] for x in rows),raw_sha256=sha(p0),raw_bytes=len(p0),
 historical=sum('held' in x for x in rows),zeros=sum(x['merged_count']==0 for x in rows),
 exact_cap=[dict(label=x['label'],k=x['metadata']['words']) for x in rows if x['merged_count']==262144],
 overcap=[dict(label=x['label'],k=x['metadata']['words']) for x in rows if x['merged_count']>262144],
 control={q:sum(x['status']==q for x in guards) for q in {x['status'] for x in guards}})
result=dict(status='MATCH' if not mismatches else 'MISMATCH',summary=s,oracle=pin,threads=t.get_num_threads(),
 prereg_sha256=sha(F.read_bytes()),rows=rows,guards=guards,mismatches=mismatches)
with (O/'receipt.json').open('x') as a:json.dump(result,a,sort_keys=True,indent=2);a.write('\n')
print(result['status'],s,mismatches[:1])
raise SystemExit(bool(mismatches))
