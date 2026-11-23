"""Official locked CPU Torch only. Native execution is deliberately absent."""
from pathlib import Path
import hashlib, json, struct, sys
H,L,I,out=map(Path,sys.argv[1:])
assert hashlib.sha256((H/'qualify_runtime_torch.py').read_bytes()).hexdigest()=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
torch,oracle=load_oracle(L,I)
words=[0,0x8000,1,0x8001,0x7f80,0xff80,0x7fc1,0x7fa1,0xffc3,0xffa3,0x3f80,0xbf80,0x3f81,0xb801,0x0080,0x8080,0x007f,0x807f,0x7f7f,0xff7f,0x4000,0x4401,0x3555,0x481b]
rows=[]
for label,root_shape,shape,strides,n in [('rank2-permutation',[3,2],[2,3],[1,2],6),('rank3-permutation',[2,3,4],[3,4,2],[4,1,12],24),('rank2-row-major',[6],[2,3],[3,1],6)]:
 seq=words[:n] if label!='rank2-row-major' else words[6:12]
 raw=struct.pack('<'+'H'*n,*seq)
 root=torch.frombuffer(bytearray(raw),dtype=torch.int16).view(torch.bfloat16).reshape(root_shape)
 version=root._version
 child=torch.as_strided(root,shape,strides,0)
 sibling=torch.as_strided(root,shape,strides,0)
 def bytes_of(t):return bytes(t.view(torch.int16).contiguous().view(torch.uint8).reshape(-1).tolist())
 assert root._version==child._version==sibling._version==version
 assert child is not root and child is not sibling
 assert root.untyped_storage().data_ptr()==child.untyped_storage().data_ptr()==sibling.untyped_storage().data_ptr() and child.storage_offset()==0
 assert bytes_of(root)==raw and bytes_of(sibling)==bytes_of(child)
 rows.append(dict(label=label,root_shape=root_shape,shape=shape,strides=strides,root_u16=seq,raw_root=raw.hex(),raw_view=bytes_of(child).hex(),version_delta=0))
out.write_text(json.dumps(dict(rows=rows,oracle=oracle),sort_keys=True,indent=2)+'\n')
