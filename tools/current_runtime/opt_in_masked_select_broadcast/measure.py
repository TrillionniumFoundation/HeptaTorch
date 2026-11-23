"""Real locked Torch preflight against immutable independent preregistration."""
import hashlib,itertools,json,math,struct,sys
from pathlib import Path
H,L,I,F,O=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
assert sha(L.read_bytes())=='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
assert sha(F.read_bytes())=='4f782478f42b9c3639ba3bfed921a8a932c8bb3dff2772096f00e312ae916f18'
p=json.loads(F.read_bytes()); assert p['kind']=='FROZEN_PREDICTION_ONLY'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);t.set_num_threads(1)
assert t.get_num_threads()==1 and pin['torch_git_commit']=='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e' and pin['wheel_sha256']=='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3'
P=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
 0x7f800000,0xff800000,0x7fc12345,0xffc23456,0x7f812345,0xff823456,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff]
def xmake(shape,w):
 if not w:return t.empty(shape,dtype=t.float32)
 return t.frombuffer(bytearray(struct.pack(f'<{len(w)}I',*w)),dtype=t.int32).view(t.float32).reshape(shape).clone()
def maskmake(shape,m):
 if not m:return t.empty(shape,dtype=t.bool)
 return t.frombuffer(bytearray(m),dtype=t.uint8).view(t.bool).reshape(shape).clone()
def raw(x):
 return bytes(x.detach().contiguous().reshape(-1).view(t.uint8).tolist())
def meta(x):
 return dict(dtype=str(x.dtype).removeprefix('torch.'),device=str(x.device),shape=list(x.shape),strides=list(x.stride()),numel=x.numel(),
  version=x._version,storage_offset_elements=x.storage_offset(),storage_bytes=x.untyped_storage().nbytes(),requires_grad=x.requires_grad,is_view=x._is_view())
def result(x,m):return t.ops.aten.masked_select.default(x,m)
def source(a):
 n=a['value_count']; kind=a['source']['kind']; seed=a['source']['seed']
 return [(i if kind=='archive-exact' else (P[(i+seed)%len(P)] if kind=='special' else 0x3f000000+13*i+seed)) for i in range(n)]
rows=[];guards=[];mismatches=[]
body=O/'raw.bin'
with body.open('xb') as f:
 for a in p['cases']:
  v=source(a); e=a['expected']; label=a['label']
  if 'mask_formula' in a:
   cfg=a['mask_formula']; assert cfg['kind']=='one-at-flat-index'
   flags=bytearray(a['mask_count']);flags[cfg['index']]=1
  else:flags=bytes.fromhex(a['mask_raw_u8_hex'])
  x=xmake(a['x_shape'],v);m=maskmake(a['mask_shape'],flags)
  inputs=lambda:[(meta(z),sha(raw(z))) for z in (x,m)]
  pre=inputs(); assert all(not md['is_view'] and md['storage_offset_elements']==md['version']==0 for md,_ in pre)
  assert [h for _,h in pre]==[sha(struct.pack(f'<{len(v)}I',*v)),sha(flags)]
  try:y=result(x,m);b=raw(y);md=meta(y);post=inputs()
  except RuntimeError as err:
   mismatches.append(dict(label=label,raised=str(err)));continue
  out=dict(label=label,meta=md,raw_sha256=sha(b),offset=f.tell(),length=len(b),input_before=pre,input_after=post)
  f.write(b);rows.append(out)
  exact=(md['shape']==e['shape'] and md['strides']==e['strides'] and md['dtype']==e['dtype'] and
   md['numel']==a['mathematical_selected'] and md['version']==e['version'] and md['storage_offset_elements']==e['offset'] and
   md['storage_bytes']==len(b)==4*md['numel'] and md['device']=='cpu' and not md['is_view'] and not md['requires_grad']
   and (e.get('raw_u32le_hex',b.hex())==b.hex()) and sha(b)==e['raw_u32le_sha256'] and pre==post)
  if not exact:mismatches.append(dict(label=label,expected=e,actual=out))
for g in p['guards']:
 l=g['label']; row=dict(label=l,predicted_hta_status=g['status'])
 if l in ('stale-mask','nonzero-padding','valid-broadcast-then-unknown'):
  row['torch']='NO_ANALOGOUS_WIRE_API';guards.append(row);continue
 xs=g.get('x',[2,3]);ms=g.get('mask',[2,3]);nx=math.prod(xs);nm=math.prod(ms)
 x=xmake(xs,[0x3f000000+i for i in range(nx)]);m=maskmake(ms,[i%2 for i in range(nm)])
 if l=='dtype-i64-mask':m=m.to(t.int64)
 if l=='actual-view-mask':
  m=maskmake([7],[i%2 for i in range(7)])[1:].reshape(ms);assert m._is_view() and m.storage_offset()==1
 if l=='actual-grad-value':x.requires_grad_(True);assert x.requires_grad
 expect=g.get('torch_accept', l in ('actual-view-mask','actual-grad-value'))
 try:
  y=result(x,m); row.update(torch='ACCEPT',meta=meta(y),raw_sha256=sha(raw(y))); ok=True
 except RuntimeError as err:
  row.update(torch='REJECT_RUNTIME_ERROR',error=str(err)); ok=False
 if expect!=ok:mismatches.append(row)
 guards.append(row)
archive=[r for r in rows if r['label']=='archived-broadcast'][0]
assert p['old_historical']['old_status']==7
report=dict(status='MATCH' if not mismatches else 'MISMATCH',threads=t.get_num_threads(),oracle=pin,
 prereg_sha256=sha(F.read_bytes()),archived_old_witness=p['old_historical'],archived_torch=archive,rows=rows,controls=guards,mismatches=mismatches,
 summary=dict(cases=len(rows),words=sum(r['meta']['numel'] for r in rows),floor=5_700_000_000,
 raw_sha256=sha(body.read_bytes()),raw_bytes=body.stat().st_size,counts={k:sum(g['torch']==k for g in guards) for k in set(g['torch'] for g in guards)}))
with (O/'receipt.json').open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
print(report['status'],report['summary'],report['mismatches'][:1])
raise SystemExit(bool(mismatches))
