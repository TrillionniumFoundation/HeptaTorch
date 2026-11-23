import hashlib,json
from pathlib import Path
D=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(n):return json.loads((D/n).read_bytes())
p=load('exact-wide-old.json');o=load('oracle-wide.json');s=load('suite.json')
assert sha((D/'exact-wide-old.json').read_bytes())==o['exact_archive_sha256']
base={f['label']:f for f in s['rows']}
ref={x['label']:x for x in o['exact_old']}
W={0:4,1:1,4:8}
def pack(words,width):return b''.join(x.to_bytes(width,'little') for x in words)
def tensor(a):
 t,sh,ws=a;return dict(dtype=t,shape=sh,words=ws,byte_hex=pack(ws,W[t]).hex())
rows=[];arch=[]
for e in p['witnesses']:
 x=e['new_prediction']; dims,ss,words=x['expected'];b=pack(words,4);args=list(map(tensor,x['args']))
 dup=e['exact_operands_duplicate_frame']
 label=f'exact-old-dimensioned-f32-f64-{dup}' if dup else x['label']
 if dup:
  f=base[label]
  assert f['shape']==dims and f['strides']==ss and f['result_hex']==b.hex()
  for v,a in zip(('mask','left','right'),args,strict=True):
   assert (f[v+'_dtype'],f[v+'_shape'],f[v+'_hex'])==(a['dtype'],a['shape'],a['byte_hex'])
 else:
  z=ref[label]
  assert z['shape']==dims and z['strides']==ss and z['observed_words']==words and z['result_sha256']==sha(b)
  assert z['owner_sha256']==e['owner_sha256'] and z['input_sha256']==e['request_sha256']
  f=dict(label=label,shape=dims,strides=ss,torch_strides=ss,dtype=0,width=4,result_hex=b.hex())
  for v,a in zip(('mask','left','right'),args,strict=True):
   f[v]=a['words']; f[v+'_shape']=a['shape']; f[v+'_dtype']=a['dtype']; f[v+'_hex']=a['byte_hex']
  rows.append(f)
 arch.append(dict(**e,operands=args,matched_positive_prediction=label))
assert len(rows)==len(ref)==2 and len(arch)==4
(D/'suite-wide.json').write_text(json.dumps(dict(rows=rows,controls=[],archived=arch),sort_keys=True,separators=(',',':'))+'\n')
