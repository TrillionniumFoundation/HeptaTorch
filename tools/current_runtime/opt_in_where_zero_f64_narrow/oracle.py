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
for case in p['finite_and_infinity']+p['nan_unmeasured_hypothesis']:
    m,x,y=ts=list(map(make,case['args']))
    old=list(map(state,ts))
    assert all(v.device.type=='cpu' and v.is_contiguous() for v in ts)
    assert all(x[1]==canonical(x[0]) for x in old)
    s,f=(x,y) if x.ndim==0 else (y,x)
    assert s.ndim==0 and s.numel()==1 and s.dtype==t.float64 and f.ndim>0 and f.dtype==t.float32
    inword=int.from_bytes(raw(s),'little')
    d=s.to(t.float32);word,=u32(d);assert d.ndim==0
    prior=per_direct.setdefault(inword,word);assert prior==word
    result=t.ops.aten.where.self(*ts);a=u32(result);shape,stride,expected=case['expected']
    assert result.dtype==t.float32 and list(result.shape)==shape and list(result.stride())==stride
    assert result._version==result.storage_offset()==0 and not result.requires_grad and list(map(state,ts))==old
    assert result.untyped_storage()._cdata not in {v.untyped_storage()._cdata for v in ts}
    if m.ndim==3:
        scalar_result,val_result=result.unbind(0)
        assert raw(scalar_result)==raw(d)*f.numel() and raw(val_result)==raw(f)
    is_nan='f64_word' in case
    expected_cast=narrow(inword,'x86-payload-hypothesis')
    current=narrow(inword,'local-hia-warning')
    delta=[i for i,(j,k) in enumerate(zip(a,expected,strict=True)) if j!=k]
    status=dict(label=case['label'],input_raw_f64=inword,observed_direct=word,predicted_direct=expected_cast,current_narrow64_reference=current,
                shape=shape,strides=stride,word_count=len(a),observed_words=a,result_sha256=sha(raw(result)),hypothesis_mismatch=delta)
    if delta or word!=expected_cast:diff.append(status)
    if is_nan:
        nans.append(status)
        old_payload=case['current_hia_canonical'][2]
        delta=[i for i,(j,k) in enumerate(zip(a,old_payload,strict=True)) if j!=k]
        if delta:native_diffs.append(dict(label=case['label'],input_raw_f64=inword,locked_word=word,current_narrow64_reference=current,differing_words=len(delta)))
    else:finite.append(status)
for case in p['policy_guards']+[x for x in p['archived'] if 'already_changed_or_still_excluded' in x]:
    ts=list(map(make,case['args']));prev=list(map(state,ts))
    try:o=t.ops.aten.where.self(*ts)
    except RuntimeError as e:
        assert case['label'].startswith('incompatible') and 'must match the size of tensor' in str(e)
        policy.append(dict(label=case['label'],semantic_class='shape_error',message=str(e)))
    else:
        assert not case['label'].startswith('incompatible')
        policy.append(dict(label=case['label'],semantic_class='outside_proposed_native_only',dtype=str(o.dtype),shape=list(o.shape),strides=list(o.stride())))
    assert list(map(state,ts))==prev
assert len(finite)==210 and len(nans)==44 and len(policy)==12
assert sum(x['word_count'] for x in finite)==4100
out=dict(oracle=pin,threads=t.get_num_threads(),cpu=t.backends.cpu.get_cpu_capability(),preregister_sha256=sha(freeze),
         finite_and_infinity=finite,nan_hypothesis=nans,policy_semantics=policy,
         unique_direct_f64=[dict(f64_raw=k,f32_raw=v) for k,v in sorted(per_direct.items())],frozen_hypothesis_discrepancies=diff,
         nan_discrepancies_vs_unchanged_hia_reference=native_diffs)
(D/'oracle-receipt.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('finite',len(finite),sum(x['word_count'] for x in finite),'nan',len(nans),sum(x['word_count'] for x in nans),
      'distinct direct',len(per_direct),'semantic',len(policy),'hypothesis mismatches',len(diff),
      'vs narrow64 NaN',len(native_diffs),sum(k['differing_words'] for k in native_diffs))
