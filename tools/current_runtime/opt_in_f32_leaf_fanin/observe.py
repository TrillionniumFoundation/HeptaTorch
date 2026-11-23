"""Pinned live fan-in diagnostics: fixed cat operand order, reversed actual arm arrival."""
from hashlib import sha256,sha1
from types import ModuleType
from pathlib import Path
import argparse,json,struct
ABI='1f0c6efc64e96fe25106acee7bf77a2ef86dd3d5b129b55bb773f5e5ff46e768'
OBSERVER='2e25fabb4af5fa719138071bab3a7e1ac10c82f7'
GRID=(0,0x80000000,1,0x80000001,0x3f800000,0xbf800000,0x7f800000,0xff800000,
      0x7fc01234,0xffc02345,0x7f804567,0xff805678)
LENS=(1,7,8,9,15,16,17,31,32,33)
def sha(b):return sha256(b).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def write(p,data):p.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')

def experiment(a):
    expect=a.stack_observer.read_bytes()
    require(sha1(b'blob '+str(len(expect)).encode()+b'\0'+expect).hexdigest()==OBSERVER,'wrong previous loader')
    o=ModuleType('mul_observer');o.__file__=str(a.stack_observer)
    exec(compile(expect,o.__file__,'exec'),o.__dict__)
    source=o.oracle_parts(a.loader)
    t,pin=source.load_oracle(a.lock,a.install);w=source.load_wire(a.protocol)
    require(sha(a.native.read_bytes())==ABI,'wrong exact existing executable')
    a.output.mkdir()
    def tensor(x,grad=False):
        return t.frombuffer(bytearray(struct.pack(f'<{len(x)}I',*x)),dtype=t.float32).requires_grad_(grad)
    def words(x):
        n=x.numel()
        return list(struct.unpack(f'<{n}I',x.detach().view(t.int32).numpy().tobytes()))
    def torch_graph(arms,creation,hooked):
        leaf=tensor([0x3f800000]*len(arms[0]),True)
        report=[];r={}
        def hook(label):
            def record(g):
                report.append([label,words(g)])
            return record
        for tag in creation:
            x=leaf.clone();r[tag]=x
            if hooked:x.register_hook(hook(tag))
        y=t.cat(tuple(r[i] for i in range(len(arms))))
        seed=tensor([n for arm in arms for n in arm])
        y.backward(seed)
        result=words(leaf.grad)
        if hooked:
            require([row[0] for row in report]==list(reversed(creation)),('Torch actual arrival',creation,report))
            require(all(vals==list(arms[tag]) for tag,vals in report),'Torch identity-edge payload unexpectedly changed')
        return result,report

    def native_graph(arms,creation,s):
        def op(*tuples):
            return s.run([tuples])[0]
        def make(xs,req):
            status,rows=w.parse(s.exchange(source.factory(0,(len(xs),),xs,requires=int(req))))
            require(status==0 and len(rows)==1 and rows[0]['opcode']==97,'native factory')
            return rows[0]['value']
        leaf=make([0x3f800000]*len(arms[0]),True);r={}
        row={'case_start_frame':s.frames,'retained_frames':{}}
        for tag in creation:
            r[tag]=op(18,leaf)['value'];op(93,r[tag])
        config=w.Payload(struct.pack(f'<{len(r)+1}Q',len(r),*(r[i] for i in range(len(r)))))
        root=op(34,0,config)['value']
        seed=make([n for arm in arms for n in arm],False)
        bw=op(58,root,seed,0)
        require(bw['value']==0,'native backward nested status')
        row['backward_frame']=s.frames
        for tag in range(len(arms)):
            retained=op(91,r[tag])['value'];require(retained!=0,'missing retained')
            result=op(39,retained,0,len(arms[tag]))
            require(result['value']==len(arms[tag]),'native retained extent')
            vals=list(struct.unpack(f'<{len(arms[tag])}I',result['payload']))
            row['retained_frames'][str(tag)]=[s.frames,vals]
            op(41,retained)
        result=op(40,leaf,0,len(arms[0]));require(result['value']==len(arms[0]),'native leaf extent')
        row['leaf_frame']=s.frames
        row['leaf']=list(struct.unpack(f'<{len(arms[0])}I',result['payload']))
        for h in (root,seed,*r.values(),leaf):op(41,h)
        require(op(53)['value']==1,'owner audit')
        return row

    pairs=[(g,o) for g in GRID for o in GRID]
    trials=[]
    # Finite associativity witness: same cat slots / seed, changed order of actual edge arrivals only.
    trials.extend([('order-witness',(0,1,2),([0xcb800000],[0x4b800000],[0x3f800000])),
                   ('order-witness',(2,0,1),([0xcb800000],[0x4b800000],[0x3f800000]))])
    for n in LENS:
        for o2 in (0,20,37,56,81,101,112,128,132,142):
            ss=tuple(zip(*(pairs[(o2+i)%len(pairs)] for i in range(n))))
            for c in ((0,1),(1,0)): trials.append((f'{n}/{o2}',c,ss))
    # Long independent full Cartesian stripe in both actual orders.
    trials.extend([('full-grid',c,tuple(zip(*pairs))) for c in ((0,1),(1,0))])
    real=w.Session(a.native,directory=a.output/'native')
    results=[]
    try:
        with t.autograd.set_multithreading_enabled(False):
            for label,c,arms in trials:
                got,trace=torch_graph(arms,c,True)
                same,_=torch_graph(arms,c,False)
                require(got==same,'read-only hooks changed the observed Torch result')
                row={'label':label,'creation':c,'cat_arms':arms,'torch_arrival':trace,'torch_leaf':got}
                try:row['native']=native_graph(arms,c,real)
                except BaseException as e:
                    row['native_failure']=repr(e)
                    results.append(row)
                    break
                results.append(row)
    finally:
        real.finish()
    write(a.output/'observations.json',{'native_sha256':ABI,'oracle':pin,'cases':results,'native_frames':real.frames,
       'total_requested':len(trials)})
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('stack-observer','loader','lock','install','protocol','native','output'):
        p.add_argument('--'+n,type=Path,required=True)
    experiment(p.parse_args())
