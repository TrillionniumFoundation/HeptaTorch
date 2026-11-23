"""Official CPU torch only; compare first-frozen true rank-0 corpus."""
import hashlib,json,sys,math,collections
from pathlib import Path
H,L,I,D=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
blob=(D/'preregister.json').read_bytes();assert sha(blob)=='788ab1647671ba2b0d9540572e9096407a261089b91c5b3ad8cefdf0e12be8c9'
p=json.loads(blob);sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);t.set_num_threads(1)
DTS={0:(4,t.int32,t.float32),1:(1,t.uint8,t.bool),2:(8,t.int64,t.int64),
     4:(8,t.int64,t.float64),5:(2,t.int16,t.float16),6:(2,t.int16,t.bfloat16)}
def make(arg):
    tag,sh,words=arg;w,it,ft=DTS[tag]
    b=b''.join(x.to_bytes(w,'little') for x in words)
    return t.frombuffer(bytearray(b),dtype=it).view(ft).reshape(sh) if b else t.empty(sh,dtype=ft)
def raw(x):
    return bytes(x.contiguous().reshape(-1).view(t.uint8).tolist())
def state(z): return (list(z.shape),list(z.stride()),z._version,raw(z))
def f32words(z):
    b=raw(z);return [int.from_bytes(b[i:i+4],'little') for i in range(0,len(b),4)]
def canon(sh):return [math.prod(max(1,d) for d in sh[i+1:]) for i in range(len(sh))]
result=[];scalar={};sem=[]
for r in p['positive']:
    m,x,y=a=list(map(make,r['args'])); old=list(map(state,a))
    assert all(n.device.type=='cpu' and n.is_contiguous() for n in a)
    assert all(st[1]==canon(st[0]) for st in old)
    i,f=(x,y) if x.dtype==t.int64 else (y,x)
    assert i.ndim==0 and list(i.shape)==[] and i.numel()==1 and f.ndim>0 and f.dtype==t.float32
    direct=i.to(t.float32);assert direct.ndim==0
    b,=f32words(direct);key=i.item();assert scalar.setdefault(key,b)==b
    o=t.ops.aten.where.self(*a)
    sh,st,words=r['expected']
    assert o.dtype==t.float32 and list(o.shape)==sh and list(o.stride())==st and o.numel()>0
    assert o.storage_offset()==0 and not o.requires_grad and o._version==0
    assert o.untyped_storage()._cdata not in {z.untyped_storage()._cdata for z in a}
    assert list(map(state,a))==old
    assert f32words(o)==words,(r['label'],f32words(o),words)
    if r['label'].startswith('ties-'):
        left, right=o.unbind(0)
        assert raw(left)==raw(direct)*f.numel() and raw(right)==raw(f)
    result.append(dict(label=r['label'],kind='historical' if r['label'].startswith('historical-') else 'fresh',shape=sh,strides=st,
                       words=len(words),direct_f32=b,result_sha256=sha(raw(o)),input_strides=[x[1] for x in old]))
traps=[]
for z in p['traps']:
    i=z['i64'];x=t.tensor(i,dtype=t.int64);assert x.ndim==0
    a,=f32words(x.to(t.float32));b,=f32words(x.to(t.float64).to(t.float32))
    assert a==z['direct_f32']==scalar[i] and b!=a,(i,a,b)
    traps.append(dict(i64=i,direct_f32=a,via_f64=b))
for row in p['negatives']+[v for v in p['archived'] if 'proposed_native_status' in v]:
    a=list(map(make,row['args'])); prior=list(map(state,a)); label=row['label']
    try:
        result0=t.ops.aten.where.self(*a)
    except RuntimeError as e:
        assert label.startswith('incompatible') and 'must match the size of tensor' in str(e)
        sem.append(dict(label=label,torch_error=str(e),proposed_native_status=7))
    else:
        assert not label.startswith('incompatible')
        sem.append(dict(label=label,torch_dtype=str(result0.dtype),torch_shape=list(result0.shape),torch_strides=list(result0.stride()),proposed_native_status=7))
    assert list(map(state,a))==prior
assert len(result)==438 and len(scalar)==216 and len(traps)==20 and len(sem)==24
out=dict(oracle=pin,threads=t.get_num_threads(),cpu_capability=t.backends.cpu.get_cpu_capability(),prereg=sha(blob),
    result=result,distinct_direct=[dict(i64=k,f32=v) for k,v in sorted(scalar.items())],double_rounding_contrasts=traps,
    semantic_control_only=sem)
(D/'oracle-receipt.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print('cases',len(result),'words',sum(q['words'] for q in result),'distinct scalar',len(scalar),'traps',len(traps),'controls',len(sem))
