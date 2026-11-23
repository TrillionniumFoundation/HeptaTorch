"""Pinned physical Torch F32 multiply-backward precedence census."""
from pathlib import Path
import argparse,json,struct
from types import ModuleType
from hashlib import sha256
BASE='476365aae838542bb2fee39a1bd0a8b5785e75f4'
def sha(b):return sha256(b).hexdigest()
def oracle(q,loader,lock,install):
    raw=q.read_bytes()
    assert __import__('hashlib').sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==BASE
    m=ModuleType('prior_observer');m.__file__=str(q)
    exec(compile(raw,str(q),'exec'),m.__dict__)
    return m.oracle_parts(loader).load_oracle(lock,install)
def census(torch):
    w=[0,0x80000000,1,0x80000001,0x3f800000,0xbf800000,0x40000000,0xc0000000,0x7f800000,0xff800000,
       0x7fc01234,0xffc02345,0x7f804567,0xff805678]
    c=[(a,b) for a in w for b in w]
    def ten(v,req=False):return torch.frombuffer(bytearray(struct.pack(f'<{len(v)}I',*v)),dtype=torch.float32).requires_grad_(req)
    rows=[]
    for n in (1,7,8,9,15,16,17,31,32,33):
      for i in range(len(c)):
        p=[c[(i+j)%len(c)] for j in range(n)]
        g,o=zip(*p)
        results=[]
        for flip in (False,True):
          x=ten([0x40000000]*n,True);y=ten(o)
          z=(y*x) if flip else (x*y)
          r=torch.autograd.grad(z,x,ten(g))[0].view(torch.int32).numpy().tobytes()
          results.append(list(struct.unpack(f'<{n}I',r)))
        rows.append([n,i,*results])
    return {'words':w,'rows':rows}
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('observer','loader','lock','install','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); t,pin=oracle(a.observer,a.loader,a.lock,a.install);r=census(t);r['oracle']=pin
    a.output.write_text(json.dumps(r,separators=(',',':'))+'\n')
