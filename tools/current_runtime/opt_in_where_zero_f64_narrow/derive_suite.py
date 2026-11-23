import hashlib,json
from pathlib import Path
D=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def pin(name,h):
 b=(D/name).read_bytes();assert sha(b)==h;return json.loads(b)
p=pin('preregister.json','4908b0833ea9e6e677ff1acb12c5b330743909677ee5eb881f69bc32ad52b124')
o=pin('oracle-receipt.json','4da120373abe06586c02d60473694ffd9a6b72d3de36155c9db3962b77c2f324')
old=pin('exact-old.json','c6baaa7da4361ad5a15230260bae90347ff2ab89676820411232c5fcd2041fa3')
archive=json.loads((D/'oracle-archived.json').read_bytes()); assert archive['exact_archive_sha256']==sha((D/'exact-old.json').read_bytes())
assert o['oracle']==archive['oracle'] and o['cpu']==archive['cpu'] and o['threads']==archive['threads']==1
W={0:4,1:1,2:8,4:8,5:2,6:2}
def pack(words,width):return b''.join(w.to_bytes(width,'little') for w in words)
def tensor(v):
 dtype,shape,words=v
 return dict(dtype=dtype,shape=shape,words=words,byte_hex=pack(words,W[dtype]).hex())
ref={x['label']:x for x in o['finite_and_infinity']+o['nan_hypothesis']+archive['exact_old']}
r=[]
for f in p['finite_and_infinity']+p['nan_unmeasured_hypothesis']+[x['new_prediction'] for x in old['witnesses']]:
 t=ref[f['label']];shape,stride,words=f['expected']; b=pack(words,4)
 assert t['shape']==shape and t['strides']==stride and t['observed_words']==words and t['result_sha256']==sha(b)
 v=dict(label=f['label'],dtype=0,width=4,shape=shape,strides=stride,torch_strides=stride,result_hex=b.hex())
 for k,c in zip(('mask','left','right'),map(tensor,f['args']),strict=True):
  v[k]=c['words'];v[k+'_hex']=c['byte_hex'];v[k+'_shape']=c['shape'];v[k+'_dtype']=c['dtype']
 r.append(v)
controls=[dict(label=x['label'],**dict(zip(('mask','x','y'),map(tensor,x['args']),strict=True)),status=7) for x in p['policy_guards']]
archived=[]
for x in old['witnesses']:
 f=x['new_prediction']
 assert ref[f['label']]['owner_sha256']==x['owner_sha256'] and ref[f['label']]['input_sha256']==x['request_sha256']
 archived.append(dict(**x,operands=list(map(tensor,f['args'])),matched_positive_prediction=f['label']))
assert len(r)==len(ref)==262 and len(controls)==8 and len(archived)==8
(D/'suite.json').write_text(json.dumps(dict(rows=r,controls=controls,archived=archived,
 oracle_archived_sha256=sha((D/'oracle-archived.json').read_bytes())),sort_keys=True,separators=(',',':'))+'\n')
print('scheduled',len(r),'strict',sum(len(b['result_hex'])//8 for b in r),'elements and',len(controls),'policy')
