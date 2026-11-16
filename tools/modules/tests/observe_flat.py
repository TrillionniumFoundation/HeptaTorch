"""Disposable stdlib-only ABI observer; never imports a compiler or linker."""
import ctypes,hashlib,json,mmap,resource,sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(15,15));resource.setrlimit(resource.RLIMIT_AS,(256<<20,256<<20))
req=json.loads(Path(sys.argv[1]).read_text());raw=Path(req['image']).read_bytes();data=Path(req['request']).read_bytes()
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(raw)==req['image_sha256'] and sha(data)==req['request_sha256']
capacity=req['capacity'];wn=req['workspace'];fill=req.get('workspace_fill',90)
assert 0<=capacity<=8<<20 and 0<=wn<=16<<20 and 0<=len(data)<=8<<20
libc=ctypes.CDLL(None);libc.mprotect.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int];libc.mprotect.restype=ctypes.c_int
code=mmap.mmap(-1,len(raw),prot=mmap.PROT_READ|mmap.PROT_WRITE);code.write(raw)
code_at=ctypes.addressof(ctypes.c_char.from_buffer(code));assert libc.mprotect(code_at,len(raw),mmap.PROT_READ|mmap.PROT_EXEC)==0
source=ctypes.create_string_buffer(data,max(1,len(data)))
G=4096
output=mmap.mmap(-1,capacity+2*G);output[:]=bytes([0x81])*G+bytes([0xa5])*capacity+bytes([0x27])*G
workspace=mmap.mmap(-1,wn+2*G);workspace[:]=bytes([0x62])*G+bytes([fill])*wn+bytes([0x35])*G
op=ctypes.addressof(ctypes.c_char.from_buffer(output))+G;wp=ctypes.addressof(ctypes.c_char.from_buffer(workspace))+G
fn=ctypes.CFUNCTYPE(ctypes.c_uint64,ctypes.c_void_p,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_uint64)(code_at)
result=int(fn(ctypes.addressof(source),len(data),op,capacity,wp,wn))
observed=bytes(output[G:G+capacity]);produced=observed[:result] if result<=capacity else None
report={'result':result,'output_unchanged':observed==bytes([0xa5])*capacity,'workspace_unchanged':workspace[G:G+wn]==bytes([fill])*wn,'source_unchanged':source.raw[:len(data)]==data,'output_guards_unchanged':output[:G]==bytes([0x81])*G and output[G+capacity:]==bytes([0x27])*G,'workspace_guards_unchanged':workspace[:G]==bytes([0x62])*G and workspace[G+wn:]==bytes([0x35])*G,'produced_sha256':None if produced is None else sha(produced),'forbidden_modules':[n for n in sys.modules if n=='hepta' or n.startswith('hepta.') or n=='tools' or n.startswith('tools.')]}
print(json.dumps(report))
