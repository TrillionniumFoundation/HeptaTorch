"""Official locked Torch check against already frozen mathematical expectations."""
import hashlib,json,sys
from pathlib import Path
H,L,I,expected,out=map(Path,sys.argv[1:])
assert hashlib.sha256((H/'qualify_runtime_torch.py').read_bytes()).hexdigest()=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
digest=lambda v: hashlib.sha256(v).hexdigest()
blob=expected.read_bytes()
assert digest(blob)=='f141705d3a987ec01e08a031bec72a384706f0fd1ec2e59c0ba5167141db7aba'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
torch,pin=load_oracle(L,I)
torch.set_num_threads(1)
D={0:(torch.int32,torch.float32),1:(torch.uint8,torch.bool),2:(torch.int64,torch.int64)}
def make(row):
    d,t=D[row['dtype']]
    b=bytes.fromhex(row['byte_hex'])
    return torch.frombuffer(bytearray(b),dtype=d).view(t).reshape(row['shape']) if b else torch.empty(row['shape'],dtype=t)
def raw(t):
    return bytes(t.contiguous().reshape(-1).view(torch.uint8).tolist())
def u32(t):
    b=raw(t)
    return [int.from_bytes(b[i:i+4],'little') for i in range(0,len(b),4)]
fixture=json.loads(blob)
results=[]
for row in fixture['positives']:
    ts=[make(row[k]) for k in ('mask','x','y')]
    initial=[(raw(t),t._version) for t in ts]
    assert all(t.device.type=='cpu' and t.is_contiguous() for t in ts)
    r=torch.ops.aten.where.self(*ts)
    rb=raw(r)
    assert r.dtype==torch.float32 and list(r.shape)==row['result']['shape'] and r.device.type=='cpu'
    assert rb.hex()==row['result']['byte_hex'],(row['label'],[i for i,(a,b) in enumerate(zip(u32(r),row['result']['words'])) if a!=b])
    assert r._version==0 and not r.requires_grad
    assert r.untyped_storage()._cdata not in {t.untyped_storage()._cdata for t in ts}
    assert [(raw(t),t._version) for t in ts]==initial
    results.append(dict(label=row['label'],shape=list(r.shape),stride=list(r.stride()),words=r.numel(),
                        result_sha256=digest(rb),exact_math=True))
trs=fixture['direct_double_rounding_traps']
# Independently test every drafted selected I64 word when converted by Torch.
base=make(fixture['positives'][0]['x'])
baseline_direct=u32(base.to(torch.float32))
assert baseline_direct==fixture['positives'][0]['result']['words'][::2]
ints=torch.tensor([r['i64'] for r in trs],dtype=torch.int64)
direct=u32(ints.to(torch.float32))
via=u32(ints.to(torch.float64).to(torch.float32))
assert direct==[r['direct_f32'] for r in trs]
assert all(d!=v for d,v in zip(direct,via)),(direct,via)
controls=[]
for row in fixture['rank0_native_policy_denials']:
    r=torch.ops.aten.where.self(*(make(row[k]) for k in ('mask','x','y')))
    controls.append(dict(label=row['label'],torch_dtype=str(r.dtype),torch_shape=list(r.shape),
                         intentionally_outside_native_scope=True))
out.write_text(json.dumps(dict(oracle=pin, torch_threads=torch.get_num_threads(), cpu_capability=torch.backends.cpu.get_cpu_capability(),
 expectation_sha256=digest(blob), results=results, direct_vector_words=len(baseline_direct), double_rounding_disconfirmed=len(trs),
 trap_words=[dict(i64=t['i64'],direct_f32=d,via_f64=v) for t,d,v in zip(trs,direct,via)],rank0_observed_not_qualified=controls),indent=2,sort_keys=True)+'\n')
print('where cases',len(results),'words',sum(r['words'] for r in results),'direct conversion',len(baseline_direct),'direct vs via-F64 mismatches',len(trs))
