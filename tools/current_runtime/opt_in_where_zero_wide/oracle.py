"""Official locked Torch, separate preregistered true-rank0 profile verification."""
import hashlib,json,sys
from pathlib import Path
H,L,I,D=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
p=(D/'expected.json').read_bytes();h=(D/'historical.json').read_bytes()
assert sha(p)=='862909cf158b91d4908444d41ce36bfa41958631fe13934232b3b7f795df4139'
assert sha(h)=='53c9606a956b11afccbe49a6026dac18756ff51e70e94156d25cd5508b9c413d'
p,h=map(json.loads,(p,h))
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);t.set_num_threads(1)
DT={0:(t.int32,t.float32),1:(t.uint8,t.bool),2:(t.int64,t.int64),4:(t.int64,t.float64),
    5:(t.int16,t.float16),6:(t.int16,t.bfloat16)}
def make(row):
    it,ft=DT[row['dtype']]
    blob=bytes.fromhex(row['byte_hex'])
    return t.frombuffer(bytearray(blob),dtype=it).view(ft).reshape(row['shape']) if blob else t.empty(row['shape'],dtype=ft)
def raw(x):
    return bytes(x.contiguous().reshape(-1).view(t.uint8).tolist())
def args(row):
    a=[make(row[k]) for k in ('mask','x','y')]
    assert all(z.device.type=='cpu' and z.is_contiguous() for z in a)
    assert [list(z.shape) for z in a]==[row[k]['shape'] for k in ('mask','x','y')]
    return a
R=[];cas={};controls=[]
for row in p['positive']+[u['expected_if_admitted'] for u in h['retiring_candidates']]:
    ins=args(row); mask,x,y=ins
    scalar,float64=(x,y) if x.ndim==0 else (y,x)
    assert scalar.ndim==0 and scalar.numel()==1 and float64.ndim>0 and float64.dtype==t.float64
    prev=[(raw(z),z._version) for z in ins]
    sx=row['x'] if x.ndim==0 else row['y']
    c=scalar.to(t.float64)
    assert c.ndim==0 and c.dtype==t.float64
    raw_c=raw(c).hex();key=(sx['dtype'],sx['words'][0])
    assert cas.setdefault(key,raw_c)==raw_c
    ans=t.ops.aten.where.self(*ins); b=raw(ans)
    assert ans.dtype==t.float64 and list(ans.shape)==row['result']['shape']
    assert ans._version==0 and not ans.requires_grad
    assert ans.untyped_storage()._cdata not in {z.untyped_storage()._cdata for z in ins}
    assert [(raw(z),z._version) for z in ins]==prev
    assert b.hex()==row['result']['byte_hex'],(row['label'],raw_c,b.hex(),row['result']['byte_hex'])
    # For the high-discrimination cases, verify scalar direct bytes and original F64 bytes independently.
    if row['label'].startswith('mixed-mask-'):
        f0=prev[1 if x.ndim else 2][0]
        assert len(f0)==float64.numel()*8 and mask.shape==(2,1,1)
        first_scalar=x.ndim==0
        assert mask.flatten().tolist()==[first_scalar,not first_scalar]
        assert b[:len(f0)]==bytes.fromhex(raw_c)*float64.numel()
        assert b[len(f0):]==f0
    R.append(dict(label=row['label'],kind='historical' if row['label'].startswith('historical-') else 'prereg',
                  words=ans.numel(),shape=list(ans.shape),stride=list(ans.stride()),direct_scalar_hex=raw_c,result_sha256=sha(b)))
for row in p['policy_negative']+h['retained_policy_candidates']:
    ins=args(row)
    y=t.ops.aten.where.self(*ins)
    controls.append(dict(label=row['label'],torch_dtype=str(y.dtype),torch_shape=list(y.shape),
                         proposed_native_policy=7,torch_is_semantic_control_only=True))
assert len(R)==184 and len(controls)==32 and len(cas)==72
output=dict(oracle=pin,threads=t.get_num_threads(),cpu_capability=t.backends.cpu.get_cpu_capability(),
            prereg_sha256=sha((D/'expected.json').read_bytes()),historical_sha256=sha((D/'historical.json').read_bytes()),
            where=R,distinct_direct=[dict(tag=tag,word=word,f64_hex=w) for (tag,word),w in sorted(cas.items())],
            out_of_scope_semantic_control=controls)
(D/'oracle-receipt.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
print('where',len(R),'words',sum(x['words'] for x in R),'unique direct',len(cas),'semantic controls',len(controls))
