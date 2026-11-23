"""Physically compare rank-outside-profile behavior across the exact two images."""
from pathlib import Path
from types import ModuleType
from hashlib import sha1,sha256
import argparse,json,struct
BASE='9074399826154287c2a97bd7172d6d86562eb5dfe1f5ab641ac33d6a03fb59de'
NEXT='1f0c6efc64e96fe25106acee7bf77a2ef86dd3d5b129b55bb773f5e5ff46e768'
HELPER='2e25fabb4af5fa719138071bab3a7e1ac10c82f7'
def ensure(ok):
    if not ok:raise ValueError('scope control unexpectedly changed')
def load(f):
    b=f.read_bytes()
    ensure(sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==HELPER)
    m=ModuleType('predecessor_observer'); m.__file__=str(f);exec(compile(b,m.__file__,'exec'),m.__dict__)
    return m
def run(a):
    m=load(a.observer);o=m.oracle_parts(a.loader);w=o.load_wire(a.wire);a.output.mkdir()
    observations=[]
    for name,p,h in (('prior',a.prior,BASE),('candidate',a.candidate,NEXT)):
        ensure(sha256(p.read_bytes()).hexdigest()==h)
        s=w.Session(p,directory=a.output/name)
        rs=[]
        try:
            def make(shape,words,grad=False):
                k,r=w.parse(s.exchange(o.factory(0,shape,tuple(words),requires=int(grad))))
                ensure(k==0 and len(r)==1 and r[0]['opcode']==97)
                return r[0]['value']
            def op(*v):return s.run([v])[0]
            for reverse in (False,True):
                x=make((1,4),[0]*4,True)
                z=[0,0]
                order=(1,0) if reverse else (0,1)
                for i in order:
                    z[i]=op(18,x)['value']
                root=op(34,0,w.Payload(struct.pack('<3Q',2,*z)))['value']
                seed=make((2,4),[0x7fc01234,0xffc02345,0x7f800000,0xff805678,0,0xffc02345,0xff800000,0x7f804567])
                ensure(op(58,root,seed,0)['value']==0)
                r=op(40,x,0,4)
                ws=list(struct.unpack('<4I',r['payload']))
                ensure(ws==[0x7fc00000]*4)
                rs.append({'creation':order,'leaf_read_frame':s.frames,'leaf_words':ws})
                for y in (root,seed,*z,x):op(41,y)
                ensure(op(53)['value']==1)
        finally:
            s.finish()
        observations.append({'role':name,'rows':rs,'frames':s.frames,'elf_sha256':h})
    d=a.output
    for f in ('requests.frames.bin','responses.frames.bin'):
        ensure((d/'prior'/f).read_bytes()==(d/'candidate'/f).read_bytes())
    (d/'scope.json').write_text(json.dumps(observations,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('prior','candidate','observer','loader','wire','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
