"""Independent official CPU Torch verification of exact supplemental replay witnesses."""
import hashlib,json,sys
from pathlib import Path
H,L,I,S,O=map(Path,sys.argv[1:])
sha=lambda v:hashlib.sha256(v).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
assert sha(S.read_bytes())=='179d57f33c9a388c51e4924510f5134f17976260a4874ebd202a2dbb2f7264a0'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);t.set_num_threads(1)
T={0:(t.int32,t.float32),1:(t.uint8,t.bool),2:(t.int64,t.int64)}
def make(tag,shape,blob):
    ity,dtype=T[tag]
    return t.frombuffer(bytearray(bytes.fromhex(blob)),dtype=ity).view(dtype).reshape(shape) if blob else t.empty(shape,dtype=dtype)
def raw(x):
    return bytes(x.contiguous().reshape(-1).view(t.uint8).tolist())
result=[]
for row in json.loads(S.read_bytes())['rows']:
    operands=[make(row[k+'_dtype'],row[k+'_shape'],row[k+'_hex']) for k in ('mask','left','right')]
    initial=[(raw(x),x._version) for x in operands]
    output=t.ops.aten.where.self(*operands)
    blob=raw(output)
    assert output.dtype==t.float32 and list(output.shape)==row['shape'] and list(output.stride())==row['strides']
    assert output._version==0 and not output.requires_grad
    assert output.untyped_storage()._cdata not in {x.untyped_storage()._cdata for x in operands}
    assert [(raw(x),x._version) for x in operands]==initial
    assert blob.hex()==row['result_hex'],row['label']
    result.append(dict(label=row['label'],words=output.numel(),sha256=sha(blob)))
O.write_text(json.dumps(dict(oracle=pin,threads=t.get_num_threads(),cpu_capability=t.backends.cpu.get_cpu_capability(),
 suite_sha256=sha(S.read_bytes()),results=result),sort_keys=True,indent=2)+'\n')
print(len(result),sum(r['words'] for r in result))
