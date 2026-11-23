"""Pure physical-suite build from exact frozen oracle and historical inputs."""
import hashlib,json,sys
from pathlib import Path
D=Path(__file__).parent
p=(D/'preregister.json').read_bytes(); o=(D/'oracle-receipt.json').read_bytes()
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(p)=='788ab1647671ba2b0d9540572e9096407a261089b91c5b3ad8cefdf0e12be8c9'
assert sha(o)=='6e876743d54dc50795934c01ad575c949f29473baecb98fbeecc657824bcb657'
p,o=map(json.loads,(p,o))
W={0:4,1:1,2:8,4:8,5:2,6:2}
def tensor(x):
    t,s,w=x;b=b''.join(x.to_bytes(W[t],'little') for x in w)
    return dict(dtype=t,shape=s,words=w,byte_hex=b.hex())
ref={r['label']:r for r in o['result']}
rows=[]
for f in p['positive']:
    label=f['label']; dims,st,words=f['expected'];blob=b''.join(x.to_bytes(4,'little') for x in words)
    z=ref[label];assert z['shape']==dims and z['strides']==st and z['result_sha256']==sha(blob)
    v=dict(label=label,dtype=0,width=4,shape=dims,strides=st,torch_strides=st,result_hex=blob.hex())
    for k,a in zip(('mask','left','right'),map(tensor,f['args'])):
        v[k]=a['words'];v[k+'_hex']=a['byte_hex'];v[k+'_shape']=a['shape'];v[k+'_dtype']=a['dtype']
    rows.append(v)
C=[dict(label=x['label'],**dict(zip(('mask','x','y'),map(tensor,x['args']))),status=7)
    for x in p['negatives']+[x for x in p['archived'] if 'proposed_native_status' in x]]
A=[dict(**x,operands=list(map(tensor,x['args']))) for x in p['archived']]
assert len(rows)==438 and len(C)==24 and len(A)==6
(D/'suite.json').write_text(json.dumps(dict(rows=rows,controls=C,archived=A),sort_keys=True,separators=(',',':'))+'\n')
