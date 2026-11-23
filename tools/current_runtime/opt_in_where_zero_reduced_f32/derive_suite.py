"""Pure physical-suite build from exact prereg, locked oracle, and archived inputs."""
import hashlib,json
from pathlib import Path
D=Path(__file__).parent
p=(D/'preregister.json').read_bytes(); o=(D/'oracle-receipt.json').read_bytes()
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(p)=='c55640895603176220c30e6b206ea2a6a50178f0651805cc9e577f02e0d51174'
assert sha(o)=='a360a5009d94eff34a5e6e5efaff6d4c798ffb81eeb4cbc77e6b35fe4e413ca6'
p,o=map(json.loads,(p,o))
W={0:4,1:1,2:8,4:8,5:2,6:2}
def tensor(x):
    t,s,w=x;b=b''.join(x.to_bytes(W[t],'little') for x in w)
    return dict(dtype=t,shape=s,words=w,byte_hex=b.hex())
ref={r['label']:r for r in o['where']}
rows=[]
for f in p['positive']:
    label=f['label']; dims,st,words=f['expected'];blob=b''.join(x.to_bytes(4,'little') for x in words)
    z=ref[label];assert z['shape']==dims and z['stride']==st and z['raw_sha256']==sha(blob) and z['matched']
    v=dict(label=label,dtype=0,width=4,shape=dims,strides=st,torch_strides=st,result_hex=blob.hex())
    for k,a in zip(('mask','left','right'),map(tensor,f['args'])):
        v[k]=a['words'];v[k+'_hex']=a['byte_hex'];v[k+'_shape']=a['shape'];v[k+'_dtype']=a['dtype']
    rows.append(v)
C=[dict(label=x['label'] if 'origin' not in x else x['origin']+':'+x['label'],
        **dict(zip(('mask','x','y'),map(tensor,x['args']))),status=7)
    for x in p['policy']+[x for x in p['archived'] if 'retain_policy_status' in x]]
A=[dict(**x,operands=list(map(tensor,x['args']))) for x in p['archived']]
assert len(rows)==100 and len(C)==38 and len(A)==18
(D/'suite.json').write_text(json.dumps(dict(rows=rows,controls=C,archived=A),sort_keys=True,separators=(',',':'))+'\n')
