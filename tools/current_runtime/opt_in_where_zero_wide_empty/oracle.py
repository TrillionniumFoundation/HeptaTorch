"""Physical locked Torch semantic preflight for frozen empty rank0/F64 cases only."""
import collections,hashlib,json,sys
from pathlib import Path
H,L,I,D=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
p=(D/'preregister.json').read_bytes()
assert sha(p)=='368c483a6f871c946974520e2ac6418d5989705d27d7a65855c09104fa9e6853'
p=json.loads(p)
P=Path('/tmp/hepta-where-zero-wide-prereg')
known_b=(P/'expected.json').read_bytes()
assert sha(known_b)=='862909cf158b91d4908444d41ce36bfa41958631fe13934232b3b7f795df4139'
K={a['label']:a for a in json.loads(known_b)['positive']}
assert sha((P/'oracle-receipt.json').read_bytes())==p['prior_oracle_receipt_sha256']
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);t.set_num_threads(1)
DTS={0:(4,t.int32,t.float32),1:(1,t.uint8,t.bool),2:(8,t.int64,t.int64),4:(8,t.int64,t.float64),5:(2,t.int16,t.float16),6:(2,t.int16,t.bfloat16)}

def make(tag,shape,words):
    w,it,dtype=DTS[tag]
    blob=b''.join(v.to_bytes(w,'little') for v in words)
    return t.frombuffer(bytearray(blob),dtype=it).view(dtype).reshape(shape) if blob else t.empty(shape,dtype=dtype)
def raw(x):
    return bytes(x.contiguous().reshape(-1).view(t.uint8).tolist())
def state(x):
    return list(x.shape),list(x.stride()),raw(x),x._version
def inputs(a):
    xt=a['left_dtype'];yt=a['right_dtype'];r=[make(1,a['mask_shape'],a['mask_words'])]
    r+=[make(dtype,a[key+'_shape'],a['f64_words'] if dtype==4 else [a['scalar_word']])
        for key,dtype in [('left',xt),('right',yt)]]
    return r
def verify(a,label,pred):
    old=[state(z) for z in a]
    assert all(z.is_contiguous() and z.device.type=='cpu' for z in a)
    assert sum(z.ndim==0 for z in a[1:])==1 and len({a[1].dtype,a[2].dtype})==2
    o=t.ops.aten.where.self(*a)
    shape=list(o.shape);stride=list(o.stride())
    assert o.dtype==t.float64 and o.device.type=='cpu' and o.numel()==0 and not raw(o)
    assert o.storage_offset()==0 and not o.requires_grad and o._version==0
    assert o.untyped_storage()._cdata not in {z.untyped_storage()._cdata for z in a}
    assert [state(z) for z in a]==old
    assert (shape,stride)==(pred['shape'],pred['strides']),(label,shape,stride,pred,[j[:2] for j in old])
    return dict(label=label,shape=shape,strides=stride,input_strides=[j[1] for j in old],contiguous_branch=pred['contiguous_allocation'],
                explicit_zero_stride_dim_gt_one=any(s==0 and n>1 for s,n in zip(stride,shape)))

new=[verify(inputs(z),z['label'],z['prediction']) for z in p['empty']]
old=[]
for z in p['known']:
    row=K[z['label']]
    a=[make(r['dtype'],r['shape'],r['words']) for r in (row['mask'],row['x'],row['y'])]
    old.append(verify(a,z['label'],z['prediction']))
reject=[]
for z in p['shape_errors']:
    a=inputs(z); prev=[state(j) for j in a]
    try:t.ops.aten.where.self(*a)
    except RuntimeError as e:
        message=str(e)
        assert 'must match the size of tensor' in message and [state(j) for j in a]==prev,(z['label'],message)
        reject.append(dict(label=z['label'],shapes=[x[0] for x in prev],torch_error=message))
    else:raise AssertionError(('accepted incompatible',z['label']))
assert len(new)==128 and len(old)==16 and len(reject)==48
counts=dict(new=len(new),old=len(old),incompatible=len(reject),canonical_new=sum(r['contiguous_branch'] for r in new),
 canonical_old=sum(r['contiguous_branch'] for r in old),explicit_zero_stride_size_gt_one=sum(r['explicit_zero_stride_dim_gt_one'] for r in new))
out=dict(oracle=pin,threads=t.get_num_threads(),cpu_capability=t.backends.cpu.get_cpu_capability(),
 prereg_sha256=sha((D/'preregister.json').read_bytes()),counts=counts,results=new,old_witnesses=old,shape_refusals=reject,
 scope='Official semantic metadata only, no HIA execution or native qualification.')
(D/'oracle-receipt.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(counts)
