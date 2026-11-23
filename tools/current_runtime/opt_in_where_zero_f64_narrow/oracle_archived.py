"""Locked physical CPU Torch preflight for frozen genuine scalar F64→F32."""
import hashlib,json,math,sys
from pathlib import Path
H,L,I,D=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
freeze=(D/'preregister.json').read_bytes()
assert sha(freeze)=='4908b0833ea9e6e677ff1acb12c5b330743909677ee5eb881f69bc32ad52b124'
p=json.loads(freeze)
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);t.set_num_threads(1)
DTS={0:(4,t.int32,t.float32),1:(1,t.uint8,t.bool),4:(8,t.int64,t.float64)}
def make(z):
    tag,sh,w=z;n,raw,dtype=DTS[tag]
    b=b''.join(v.to_bytes(n,'little') for v in w)
    return t.frombuffer(bytearray(b),dtype=raw).view(dtype).reshape(sh) if b else t.empty(sh,dtype=dtype)
def raw(x):return bytes(x.contiguous().reshape(-1).view(t.uint8).tolist())
def u32(x):
    b=raw(x)
    return [int.from_bytes(b[i:i+4],'little') for i in range(0,len(b),4)]
def state(x):return list(x.shape),list(x.stride()),x._version,x.storage_offset(),raw(x)
def canonical(shape):return [math.prod(max(1,d) for d in shape[i+1:]) for i in range(len(shape))]
finite=[];nans=[];policy=[];per_direct={};diff=[];native_diffs=[]
# Load only pure definitions, not generation or paths to any live source.
z={}
chunk=(D/'preregister.py').read_text();begin=chunk.index('F64MASK=');end=chunk.index('neighbors=')
exec(compile('import itertools,math\n'+chunk[begin:end],'frozen_math','exec'),z)
narrow=z['narrow']

frozen=(D/'exact-old.json').read_bytes()
assert sha(frozen)=='c6baaa7da4361ad5a15230260bae90347ff2ab89676820411232c5fcd2041fa3'
rows=[]
for witness in json.loads(frozen)['witnesses']:
    case=witness['new_prediction'];ts=list(map(make,case['args']))
    prev=list(map(state,ts)); m,left,right=ts
    assert all(x[1]==y for x,y in zip(prev,witness['input_strides'],strict=True))
    s,= [x for x in (left,right) if x.dtype==t.float64]
    assert s.ndim==0 and s.numel()==1
    direct,=u32(s.to(t.float32))
    result=t.ops.aten.where.self(*ts); words=u32(result)
    shape,strides,expected=case['expected']
    assert result.dtype==t.float32 and list(result.shape)==shape and list(result.stride())==strides and words==expected and list(map(state,ts))==prev
    assert result._version==result.storage_offset()==0 and result.untyped_storage()._cdata not in {v.untyped_storage()._cdata for v in ts}
    assert direct==narrow(witness['f64_word'],'x86-payload-hypothesis')
    rows.append(dict(label=case['label'],shape=shape,strides=strides,observed_words=words,
       result_sha256=sha(raw(result)),direct=direct,owner_sha256=witness['owner_sha256'],input_sha256=witness['request_sha256']))
out=dict(oracle=pin,threads=t.get_num_threads(),cpu=t.backends.cpu.get_cpu_capability(),exact_archive_sha256=sha(frozen),exact_old=rows)
assert len(rows)==8
(D/'oracle-archived.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('verified direct/canonical where exact archived',len(rows),sum(len(x['observed_words']) for x in rows),'words')
