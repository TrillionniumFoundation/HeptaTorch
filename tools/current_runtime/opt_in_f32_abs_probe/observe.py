"""Raw CPU AVX2 PyTorch observations and physical wire captures for contiguous F32 abs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import traceback

OLD='5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72'
NEW='f9408443b278cd4c5d2131845ae31b3ad0e03469ca5e6e3c69c031e31694fe46'
ORACLE_LOADER='21a42c23b68a5f1f8b8651c5b94e465d3033567a5bab722e3f3ad1cf9ca55ce2'
SOURCE_TRANSFORM='a495bc95fb84046a1f5ce6aeb43baa6f410f007c7d690cf84d39c428bef0ba1e'

def sha(raw): return hashlib.sha256(raw).hexdigest()
def dump(path,data): path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
def check(ok,msg):
    if not ok: raise AssertionError(msg)
def read_words(words):
    return struct.pack('<'+'I'*len(words),*words)
def oracle_parts(path):
    from types import ModuleType
    def load(f,name,pin):
        raw=f.read_bytes()
        check(sha(raw)==pin,('wrong loader source',f.name))
        m=ModuleType(name)
        m.__file__=str(f)
        exec(compile(raw,str(f),'exec'),m.__dict__)
        return m
    sys.modules['prepare']=load(path.with_name('prepare.py'),'prepare',SOURCE_TRANSFORM)
    return load(path,'exact_abs_oracle',ORACLE_LOADER)

def experiment(a):
    o=oracle_parts(a.loader)
    torch, pin=o.load_oracle(a.lock,a.install)
    w=o.load_wire(a.protocol)
    check(sha(a.prior.read_bytes())==OLD and sha(a.candidate.read_bytes())==NEW,'wrong claimed binaries')
    a.raw.mkdir()
    results=[]
    def execute(name, body, prior=False):
        s=w.Session(a.prior if prior else a.candidate,directory=a.raw/name)
        case={'name':name,'variant':'prior' if prior else 'candidate','steps':[],'failures':[]}
        d=a.raw/name
        def expect(actual,wanted,what):
            if actual!=wanted: case['failures'].append({'label':what,'native':actual,'expected':wanted})
        def command(request,status=0,ops=None):
            got,rows=w.parse(s.exchange(request)); n=s.frames
            if ops is None:
                ops=[] if status else [struct.unpack_from('<Q',request,32)[0]]
            case['steps'].append({'frame':n,'status':status,'opcodes':ops})
            check((got,[r['opcode'] for r in rows])==(status,ops),('command',n,got,ops,status))
            return (rows[0] if len(rows)==1 else rows)
        def op(*words,status=0): return command(w.packet([words]),status)
        def create(shape, words, req=False):
            return command(o.factory(0,shape,words,requires=int(req)))['value']
        def read(handle,expected,grad=False):
            data=expected.detach().reshape(-1).view(torch.int32).numpy().tobytes()
            row=op(40 if grad else 39,handle,0,expected.numel())
            frame=s.frames
            (d/f'oracle-{frame:04d}.bin').write_bytes(data)
            case['steps'][-1].update(payload_sha256=sha(data),payload_size=len(data),value=expected.numel())
            expect(row['value'],expected.numel(),f'extent frame {frame}')
            if row['payload']!=data:
                from itertools import islice
                diff=[(i,hex(v[0]),hex(v[1])) for i,v in enumerate(zip(struct.unpack('<'+'I'*(len(row['payload'])//4),row['payload']),struct.unpack('<'+'I'*(len(data)//4),data))) if v[0]!=v[1]]
                case['failures'].append({'label':f'raw frame {frame}', 'mismatched_words':list(islice(diff,24)),
                                         'native_sha256':sha(row['payload']),'expected_sha256':sha(data)})
        def meta(handle,expected):
            m=w.meta(op(13,handle))
            wants={'rank':expected.ndim,'numel':expected.numel(),'requires_grad':int(expected.requires_grad),
                   'shape':[*expected.shape,*([0]*(4-expected.ndim))],
                   'strides':[*expected.stride(),*([0]*(4-expected.ndim))],
                   'storage_offset':expected.storage_offset(),'version':expected._version}
            case['steps'][-1]['metadata']=wants
            for k,v in wants.items(): expect(m[k],v,f'metadata frame {s.frames} {k}')
            return m
        def physical(before, suffix, run_error=True):
            raw=before['bytes']
            (d/f'owner-{suffix}.bin').write_bytes(raw)
            val={k:v for k,v in before.items() if k!='bytes'}
            val.update(sha256=sha(raw))
            return val
        def refusal(request,status,tag,inner=False):
            bef=s.owner(); befval=physical(bef,tag+'-before')
            if inner:
                row=command(request,0,[58])
                case['steps'][-1]['value']=status
                expect(row['value'],status,'inner backward error '+tag)
            else:
                command(request,status)
            aft=s.owner(); aftval=physical(aft,tag+'-after')
            case['steps'][-1]['owner_equal']={'name':tag,'before':befval,'after':aftval}
            expect(aftval,befval,'rollback '+tag)
        try:
            body(s,case,command,op,create,read,meta,refusal,expect)
        except BaseException as err:
            case['failures'].append({'exception':repr(err),'traceback':traceback.format_exc()})
        finally:
            try:
                s.finish()
            except BaseException as err:
                case['failures'].append({'close_error':repr(err)})
            case.update(frames=s.frames,exit=s.exit_code,pid=s.pid)
            dump(d/'checks.json',case)
            results.append(case)
    def tensor(shape,words,req=False):
        t=(torch.frombuffer(bytearray(read_words(words)),dtype=torch.float32).reshape(shape)
           if words else torch.empty(shape,dtype=torch.float32))
        return t.requires_grad_(req)
    def common(b,shape,words,seed=None,leaf=True,release=True):
        s,c,command,op,create,read,meta,refusal,expect=b
        x=tensor(shape,words,seed is not None)
        root=create(shape,words,seed is not None)
        orig=x
        if not leaf:
            two=tensor(shape,(0x40000000,)*len(words))
            h2=create(shape,(0x40000000,)*len(words))
            x=x*two
            h=op(2,3,root,h2)['value']
        else:
            h=root
        y=x.abs(); hx=meta(h,x)
        out=op(3,7,h)['value']
        yo=meta(out,y)
        expect(out!=h and hx['storage_id']!=yo['storage_id'] and hx['version_id']!=yo['version_id'],True,'fresh owner and distinct storage/version')
        read(out,y); read(h,x)
        if seed is not None:
            grad=tensor(shape,seed); sh=create(shape,seed)
            y.backward(grad,retain_graph=True)
            expect(op(58,out,sh,1)['value'],0,'retained backward success'); c['steps'][-1]['value']=0; read(root,orig.grad,True)
            y.backward(grad,retain_graph=False)
            expect(op(58,out,sh,0)['value'],0,'default backward success'); c['steps'][-1]['value']=0; read(root,orig.grad,True)
            if release:
                try: y.backward(grad)
                except RuntimeError as err:
                    check('second time' in str(err),err)
                    c['torch_release_refusal']=str(err)
                else: raise AssertionError('expected actual Torch release refusal')
                refusal(w.packet([(58,out,sh,0)]),10,'released',inner=True)
            op(41,sh)
        op(41,out);op(41,h)
        if not leaf: op(41,h2);op(41,root)
        expect(op(53)['value'],1,'owner audit')

    finite=(0xc0200000,0x40600000,0xbf800000,0x3ecccccd,0xc15d0000,0x41200000)
    seed=(0x3f800000,0xc0600000,0x40000000,0xbf000000,0x3e800000,0x40400000)
    signed=(0,0x80000000,0x3f800000,0xbf800000,1,0x80000001,0x00800000,0x80800000)
    signed_seed=(0xbf800000,0x3f800000,0x80000000,0,0x3f800000,0x3f800000,0xbf800000,0xbf800000)
    special=(0x7f800000,0xff800000,0x7fc12345,0xffc12345,0x7f812345,0xff812345,0,0x80000000)
    special_seed=(0x3f800000,0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7f845678,0x7fc45678,0xffc45678)
    cases=(('finite-dense',(2,3),finite,seed,True),('finite-chain',(2,3),finite,seed,False),
           ('signed-zero-and-subnormal',(8,),signed,signed_seed,True),
           ('inf-and-nan',(8,),special,special_seed,True),
           ('scalar',(),(0xc0200000,),(0x40000000,),True),
           ('empty',(0,3),(),(),True),('forward-only',(8,),special,None,True))
    for name,shape,words,sv,leaf in cases:
        execute(name,lambda *b: common(b,shape,words,sv,leaf))
    def prior(s,c,command,op,create,read,meta,refusal,expect):
        for req in (False,True):
            h=create((2,3),finite,req)
            refusal(w.packet([(3,7,h)]),4,'prior-'+str(req))
            op(41,h)
        expect(op(53)['value'],1,'owner audit')
    execute('original-refusal',prior,True)
    def stale_saved(s,c,command,op,create,read,meta,refusal,expect):
        x=tensor((2,3),finite,True); h=create((2,3),finite,True)
        y=x.abs(); out=op(3,7,h)['value']
        sh=create((2,3),seed); g=tensor((2,3),seed)
        try:
            with torch.no_grad(): x.add_(1)
            y.backward(g)
        except RuntimeError as err:
            check('modified by an inplace operation' in str(err),err)
            c['torch_saved_version_refusal']=str(err)
        else: raise AssertionError('actual Torch saved version refusal expected')
        op(95,h,w.bits(1))
        meta(h,x);read(h,x)
        refusal(w.packet([(58,out,sh,0)]),6,'saved-version',inner=True)
        for t in (out,h,sh): op(41,t)
        expect(op(53)['value'],1,'owner audit')
    execute('saved-version',stale_saved)
    def transaction(s,c,command,op,create,read,meta,refusal,expect):
        x=tensor((2,3),finite,True)
        raw=o.factory(0,(2,3),finite,suffix=((3,7,w.reference(1)),),requires=1)
        a,b=command(raw,0,[97,3]); h,out=a['value'],b['value']
        meta(out,x.abs());read(out,x.abs())
        refusal(w.packet([(3,7,h),(125,h)]),4,'suffix')
        for t in (out,h):op(41,t)
        expect(op(53)['value'],1,'owner audit')
    execute('same-frame-and-rollback',transaction)
    def oom(s,c,command,op,create,read,meta,refusal,expect):
        words=(finite*11000)[:65536]
        x=tensor((len(words),),words,True); h=create((len(words),),words,True)
        request=w.packet([(3,7,h)])
        bef=s.owner(); original=resource.prlimit(s.pid,resource.RLIMIT_AS)
        page=os.sysconf('SC_PAGE_SIZE')
        vmem=int(Path(f'/proc/{s.pid}/statm').read_bytes().split()[0])*page
        ceiling=vmem+((bef['length']+16+page-1)//page)*page+65536
        resource.prlimit(s.pid,resource.RLIMIT_AS,(ceiling,original[1]))
        try:
            c['rlimit_as']={'mapping_bytes':vmem,'original':original,'limited':resource.prlimit(s.pid,resource.RLIMIT_AS)}
            refusal(request,3,'genuine-oom')
        finally:
            resource.prlimit(s.pid,resource.RLIMIT_AS,original)
            c['rlimit_as']['restored']=resource.prlimit(s.pid,resource.RLIMIT_AS)
        a=command(request)['value']; meta(a,x.abs()); read(a,x.abs())
        for t in (a,h):op(41,t)
        expect(op(53)['value'],1,'owner audit')
    execute('real-address-space-refusal-retry',oom)
    summary={'status':'PASS' if not any(r['failures'] for r in results) else 'FAIL',
             'torch':pin,'prior':OLD,'candidate':NEW,'cases':[{'case':r['name'],'frames':r['frames'],'failures':r['failures']} for r in results]}
    dump(a.raw/'summary.json',summary)
    print(json.dumps(summary,sort_keys=True))
    return summary['status']=='PASS'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('loader','lock','install','protocol','prior','candidate','raw'):
        p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(0 if experiment(p.parse_args()) else 1)
