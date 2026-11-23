"""Real locked Torch preflight against immutable independent preregistration."""
import hashlib,itertools,json,math,struct,sys
from pathlib import Path
H,L,I,F,O=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
assert sha(L.read_bytes())=='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
assert sha(F.read_bytes())=='6fdb5598642cc07dfa53571d4324b37e5b7ca27a400f4dacadd10dd419195c74'
p=json.loads(F.read_bytes()); assert p['kind']=='PRE_REGISTERED_PREDICTIONS_ONLY'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);t.set_num_threads(1)
assert t.get_num_threads()==1 and pin['torch_git_commit']=='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e' and pin['wheel_sha256']=='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3'
P=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
 0x7f800000,0xff800000,0x7fc12345,0xffc23456,0x7f812345,0xff823456,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff]
def words(name,n,seed):
 return [P[(i+seed)%len(P)] if name=='special' else 0x3f000000+(13*i+seed)%(1<<23) for i in range(n)]
def flags(name,n):
 return [dict(all=True,none=False,even=(i%2==0),odd=(i%2==1),edge=i in (0,n-1),interior=i%6 in (1,4,5),late=i==n-1)[name] for i in range(n)]
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
rows=[];mismatches=[]
body=O/'selected-u32le.bin'
with body.open('xb') as f:
 for a in p['positive']:
  s=a['input_shape'];n=a['count'];r=a['recipe'];e=a['expected'];v=words(r['source'],n,r['seed']);b=flags(r['mask'],n)
  x=xmake(s,v);m=maskmake(s,b)
  before=[(meta(z),sha(raw(z))) for z in (x,m)]
  assert all(not z[0]['is_view'] and z[0]['storage_offset_elements']==0 for z in before)
  assert before[0][1]==sha(struct.pack(f'<{len(v)}I',*v)) and before[1][1]==sha(bytes(b))
  y=result(x,m); actual=raw(y); q=meta(y); offset=f.tell()
  f.write(actual)
  after=[(meta(z),sha(raw(z))) for z in (x,m)]
  if not (all(q[k]==e[k] for k in ('shape','strides','dtype','numel')) and sha(actual)==e['raw_u32le_sha256'] and before==after):
   mismatches.append(dict(label=a['label'],expected=e,observed=q,sha=sha(actual),payload_unchanged=before==after))
  if 'raw_u32_le_hex' in e:assert actual.hex()==e['raw_u32_le_hex']
  rows.append(dict(label=a['label'],meta=q,raw_offset=offset,raw_byte_count=len(actual),raw_sha256=sha(actual),
     input_meta_unchanged=before==after,input_metadata=[z[0] for z in before],input_raw_sha256=[z[1] for z in before]))
guards=[]
for g in p['guards']:
 l=g['label']; r=dict(label=l,hypothesized_hia_status=g['hypothesized_status'])
 if l in ('stale-input-or-mask-resolved-before-access','reserved-word-nonzero','good-compaction-followed-by-unknown-command'):
  r['torch']='NO_ANALOGOUS_WIRE_API';guards.append(r);continue
 s=g.get('value_shape',[2,3]); ms=g.get('mask_shape',s)
 n=math.prod(s);k=math.prod(ms)
 x=xmake(s,words('ordinal',n,0));m=maskmake(ms,flags(g.get('mask_recipe','edge'),k))
 if l=='wrong-public-mask-dtype':m=m.to(t.int64)
 if l=='valid-ordinary-plus-positive-offset-view':x=xmake([7],words('ordinal',7,0))[1:].reshape(2,3);assert x.storage_offset()==1 and x._is_view()
 if l=='actual-f32-requires-grad-input':x.requires_grad_(True);assert x.requires_grad
 try:
  y=result(x,m);b=raw(y);r.update(torch='ACCEPT',result=meta(y),raw_sha256=sha(b))
  if 'mathematically_selected' in g:assert y.numel()==g['mathematically_selected']
 except RuntimeError as ex:r.update(torch='REJECT_RUNTIME_ERROR',error=str(ex))
 guards.append(r)
summary=dict(rows=len(rows),input_words=p['summary']['input_words'],selected_words=sum(a['meta']['numel'] for a in rows),
 zero_output_rows=sum(a['meta']['numel']==0 for a in rows),versions=sorted({a['meta']['version'] for a in rows}),
 offsets_elements=sorted({a['meta']['storage_offset_elements'] for a in rows}),guard_counts={k:sum(g['torch']==k for g in guards) for k in sorted({g['torch'] for g in guards})},
 raw_bytes=body.stat().st_size,raw_sha256=sha(body.read_bytes()))
report=dict(status='MATCH' if not mismatches else 'MISMATCH',oracle=pin,threads=t.get_num_threads(),cpu_capability=t.backends.cpu.get_cpu_capability(),summary=summary,
 fixture_sha256=sha(F.read_bytes()),rows=rows,guards=guards,mismatches=mismatches,
 fresh_vs_out='Only aten.masked_select.default executed. Tensor._version observed as an integer counter. Exact out semantics and aliases are not inferred or measured. No storage pointer used as public identity.')
(O/'receipt.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],summary=summary,guards=[(g['label'],g['torch']) for g in guards],mismatches=mismatches[:3]),sort_keys=True))
raise SystemExit(bool(mismatches))
