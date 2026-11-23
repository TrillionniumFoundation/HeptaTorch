"""Measure frozen genuine rank0 reduced/F32 against exact locked CPU Torch."""
import hashlib,json,math,sys
from pathlib import Path
H,L,I,D=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
freeze=(D/'preregister.json').read_bytes()
assert sha(freeze)=='c55640895603176220c30e6b206ea2a6a50178f0651805cc9e577f02e0d51174'
p=json.loads(freeze)
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);t.set_num_threads(1)
DTS={0:(4,t.int32,t.float32),1:(1,t.uint8,t.bool),2:(8,t.int64,t.int64),4:(8,t.int64,t.float64),
     5:(2,t.int16,t.float16),6:(2,t.int16,t.bfloat16)}
def make(z):
    tag,sh,w=z; n,s,d=DTS[tag]
    data=b''.join(v.to_bytes(n,'little') for v in w)
    return t.frombuffer(bytearray(data),dtype=s).view(d).reshape(sh) if data else t.empty(sh,dtype=d)
def raw(x):return bytes(x.contiguous().reshape(-1).view(t.uint8).tolist())
def words(x):
    b=raw(x);return [int.from_bytes(b[i:i+4],'little') for i in range(0,len(b),4)]
def plain(x):return list(x.shape),list(x.stride()),x._version,x.storage_offset(),raw(x)
def canon(x):return [math.prod(max(1,d) for d in x[i+1:]) for i in range(len(x))]
direct=[];case=[];neg=[];direct_diffs=[];diffs=[]
for row in p['reference_direct_profiles']:
    z=make(row['input']);prev=plain(z)
    value=z.to(t.float32); observed=words(value)
    assert list(value.shape)==row['input'][1] and value.dtype==t.float32 and prev==plain(z)
    predicted=row['expected_f32_hypothesis']
    assert len(observed)==len(predicted)
    rec=dict(label=row['label'],tag=row['input'][0],shape=list(z.shape),input_raw=row['input'][2],
             predicted=predicted,observed=observed)
    mismatch=[i for i,(x,y) in enumerate(zip(observed,predicted,strict=True)) if x!=y]
    if mismatch:direct_diffs.append(dict(**rec,word_indices=mismatch))
    direct.append(dict(**rec,matched=not mismatch))
for row in p['positive']:
    ts=list(map(make,row['args'])); prev=list(map(plain,ts))
    assert all(x.device.type=='cpu' and x.is_contiguous() for x in ts) and all(s[1]==canon(s[0]) for s in prev)
    m,x,y=ts
    scalar,f=(x,y) if x.ndim==0 else (y,x)
    assert scalar.ndim==0 and scalar.numel()==1 and f.ndim>0 and f.dtype==t.float32
    o=t.ops.aten.where.self(*ts); d=scalar.to(t.float32)
    sh,st,pred=row['expected'];obs=words(o)
    assert o.dtype==t.float32 and list(o.shape)==sh and list(o.stride())==st and not o.requires_grad
    assert o._version==o.storage_offset()==0 and list(map(plain,ts))==prev
    assert o.untyped_storage()._cdata not in {z.untyped_storage()._cdata for z in ts}
    if row['label'].startswith('mixed-mask-'):
        s,f_observed=o.unbind(0)
        assert raw(f_observed)==raw(f) and raw(s)==raw(d)*f.numel()
    mm=[i for i,(a,b) in enumerate(zip(obs,pred,strict=True)) if a!=b]
    if mm:diffs.append(dict(label=row['label'],indices=mm,actual_selected_direct=words(d),
                            differences=[dict(at=i,observed=obs[i],predicted=pred[i]) for i in mm]))
    case.append(dict(label=row['label'],shape=sh,stride=st,numel=len(obs),raw_sha256=sha(raw(o)),
                     observed_words=obs,actual_direct_scalar=words(d),matched=not mm))
for r in p['policy']+[a for a in p['archived'] if 'retain_policy_status' in a]:
    inputs=list(map(make,r['args']));before=list(map(plain,inputs))
    try:o=t.ops.aten.where.self(*inputs)
    except RuntimeError as e:
        assert r['label'].startswith('incompatible') and 'must match the size of tensor' in str(e)
        neg.append(dict(label=r['label'],status='shape_error',message=str(e)))
    else:
        assert not r['label'].startswith('incompatible')
        neg.append(dict(label=r['label'],status='outside_native_scope_only',shape=list(o.shape),stride=list(o.stride()),dtype=str(o.dtype)))
    assert list(map(plain,inputs))==before
assert len(direct)==108 and len(case)==100 and len(neg)==38
out=dict(oracle=pin,threads=t.get_num_threads(),cpu_capability=t.backends.cpu.get_cpu_capability(),
 prereg_sha256=sha(freeze),direct=direct,where=case,semantic_controls=neg,
 direct_prediction_mismatches=direct_diffs,where_prediction_mismatches=diffs)
(D/'oracle-receipt.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print('where',len(case),'words',sum(r['numel'] for r in case),'direct',len(direct),'direct words',sum(len(x['observed']) for x in direct),'inactive',len(neg))
print('hypothesis discrepancies: direct',len(direct_diffs),'where',len(diffs))
