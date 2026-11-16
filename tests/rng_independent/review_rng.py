"""Targeted native RNG review independent of the original packet oracle."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import struct
import subprocess

from flat_probe import observe

MASK=(1<<64)-1
SHA=lambda b:hashlib.sha256(b).hexdigest()
PINS={'1a1daa8390af5d79e257973f0e4ce43ca759265e75d917ce4cf1c6cff0cac657':
      'bc0f60e23b2b3839f62cfb91c1f68f937a7c4dc1fde2c1f84c89400ea2295eab',
      'cffc7c2f6574bf08a184e7d4f84e41a38ce0fb9be2f645bf9c332c93c2621eac':
      'c1de63f5f046bfb20aa4d4f65cbffa6217a38a36c0dc63b05f08c7ddc7612032'}


def temper(x):
    x ^= x>>11;x ^= (x<<7)&0x9d2c5680;x ^= (x<<15)&0xefc60000;x ^= x>>18
    return x&0xffffffff


def untemper(value):
    original=value
    x=value
    for _ in range(3): x=value^(x>>18)
    value=x
    for _ in range(3): x=value^((x<<15)&0xefc60000)
    value=x
    for _ in range(5): x=value^((x<<7)&0x9d2c5680)
    value=x
    for _ in range(3): x=value^(x>>11)
    x &= 0xffffffff
    assert temper(x)==original
    return x


def canonical(state):
    out=bytearray(5056)
    seed,left,seeded,next_word=struct.unpack_from('<QIIQ',state)
    nxt=next_word&0xffffffff
    struct.pack_into('<QIIQ',out,0,seed,left,1,nxt)
    for i in range(624):struct.pack_into('<Q',out,24+8*i,struct.unpack_from('<Q',state,24+8*i)[0]&0xffffffff)
    dv=struct.unpack_from('<I',state,5040)[0]
    fv=state[5052]
    if dv:out[5024:5032]=state[5024:5032];struct.pack_into('<I',out,5040,1)
    if fv:out[5048:5052]=state[5048:5052];out[5052]=1
    return bytes(out)


def make_state(outputs=(),nxt=1,caches=True):
    out=bytearray(5056)
    struct.pack_into('<QIIQ',out,0,0xfedcba9876543210,625-nxt,1,nxt)
    for i in range(624):struct.pack_into('<Q',out,24+8*i,(0x6c078965*(i+1)+0x12345678)&0xffffffff)
    for i,value in enumerate(outputs):
        slot=nxt+i
        assert slot<624
        encoded=untemper(value);assert temper(encoded)==value
        struct.pack_into('<Q',out,24+8*slot,encoded)
    if caches:
        struct.pack_into('<Q',out,5024,0xfff0000000000123)
        struct.pack_into('<I',out,5040,1)
        struct.pack_into('<I',out,5048,0x7f800123);out[5052]=1
    return bytes(out)


def packet(commands,state=None,seed=0):
    head=b'HTRNG001'+struct.pack('<IIQQ',int(state is not None),len(commands),seed,5056 if state is not None else 0)
    return head+(state or b'')+b''.join(struct.pack('<IIQQQ',op,count,a&MASK,b&MASK,0) for op,count,a,b in commands)


def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    image=a.native.read_bytes();source=a.source.read_bytes();native=a.native.resolve()
    assert SHA(image) in PINS and PINS[SHA(image)]==SHA(source)
    cpp=Path(__file__).with_name('state_reference.cpp');binary=a.output/'state-reference'
    compiler=shutil.which('g++');assert compiler
    built=subprocess.run([compiler,'-std=c++17','-O2',str(cpp),'-o',str(binary)],capture_output=True)
    assert built.returncode==0 and not built.stderr
    rows=[]

    def raw_reference(state,count):
        canonical_state=canonical(state)
        words=struct.unpack_from('<624Q',canonical_state,24)
        nxt=struct.unpack_from('<Q',canonical_state,16)[0]
        req=struct.pack('<626Q',*words,nxt or 624,count)
        run=subprocess.run([str(binary.resolve())],input=req,capture_output=True,timeout=30,env={'PATH':''})
        assert run.returncode==0 and not run.stderr and len(run.stdout)==8*(count+625)
        values=struct.unpack('<'+'Q'*(count+625),run.stdout)
        next_state=bytearray(canonical_state)
        new_index=values[-1] if count else nxt
        struct.pack_into('<I',next_state,8,625-new_index if new_index else 1)
        struct.pack_into('<Q',next_state,16,new_index)
        struct.pack_into('<624Q',next_state,24,*values[-625:-1])
        return list(values[:count]),bytes(next_state)

    def expected(state,commands):
        count=sum(n*(2 if op in (1,3) or op==4 and hi-lo>=1<<28 else 1) for op,n,lo,hi in commands)
        raw,after=raw_reference(state,count);index=0;values=[]
        for op,n,low,high in commands:
            assert op!=7
            for _ in range(n):
                width=2 if op in (1,3) or op==4 and high-low>=1<<28 else 1
                value=raw[index] if width==1 else (raw[index]<<32)|raw[index+1]
                index+=width
                if op in (2,5,6):
                    unit=(value&0xffffff)/(1<<24)
                    if op==2:value=struct.unpack('<I',struct.pack('<f',unit))[0]
                    elif op==5:
                        value=struct.unpack('<H',struct.pack('<e',unit))[0]
                        if value==0x3c00:value=0
                    else:
                        bits=struct.unpack('<I',struct.pack('<f',unit))[0]
                        upper,lower=divmod(bits,1<<16)
                        value=upper+(lower>0x8000 or lower==0x8000 and upper&1)
                        if value==0x3f80:value=0
                elif op==3:value=struct.unpack('<Q',struct.pack('<d',(value&((1<<53)-1))/(1<<53)))[0]
                elif op==4:value=(value%(high-low)+low)&MASK
                if op!=8:values.append(value)
        assert index==count
        return values,after

    def invoke(name,request):
        run=subprocess.run([str(native)],input=request,capture_output=True,timeout=60,env={'PATH':''})
        (a.output/(name+'.request.bin')).write_bytes(request)
        (a.output/(name+'.response.bin')).write_bytes(run.stdout)
        assert run.returncode==0 and not run.stderr
        return run.stdout

    def check(name,state,commands,exact_values=None):
        wanted,after=expected(state,commands)
        if exact_values is not None:assert wanted==exact_values
        request=packet(commands,state);response=invoke(name,request)
        expected_bytes=struct.pack('<3Q',0,len(wanted),5056)+after+struct.pack('<'+'Q'*len(wanted),*wanted)
        assert response==expected_bytes,name
        rows.append({'case':name,'status':'PASS','values':len(wanted),'state_bytes':5056,
                     'request_sha256':SHA(request),'response_sha256':SHA(response)})

    half=[0,1,1023,1024,0xffefff,0xfff000,0xfff001,0xffffff,0xffffffff]
    check('half-round-up-clamp-boundary',make_state(half),[(5,len(half),0,0)],
          [0,1,0x3ff,0x400,0x3bff,0,0,0,0])
    bf=[0,1,0x800000,0xff7fff,0xff8000,0xff8001,0xffffff,0xffffffff]
    check('bf16-round-up-clamp-boundary',make_state(bf),[(6,len(bf),0,0)],
          [0,0x3380,0x3f00,0x3f7f,0,0,0,0])
    d=[0,0,0,1,1,0,0xffffffff,0xffffffff,0x12345678,0x9abcdef0]
    check('double-low53-high-word-first',make_state(d),[(3,len(d)//2,0,0)])
    raw=[0,0xffffffff,0x80000000,0x12345678,0xdeadbeef,1,0x87654321,0xffffffff,0xffffffff,9,10,11]
    mixed=[(4,1,-113,-113+(1<<28)-1),(4,1,-113,-113+(1<<28)),
           (4,1,-113,-113+(1<<28)+1),(0,1,0,0),(4,1,-(1<<63),(1<<63)-1),(1,1,0,0)]
    check('signed-threshold-consumption-and-following-words',make_state(raw),mixed)
    state=make_state([0xdeadbeef],623)
    check('raw64-pair-across-refresh',state,[(1,2,0,0),(0,3,0,0)])
    dirty=bytearray(make_state())
    struct.pack_into('<I',dirty,12,0x80000001)
    struct.pack_into('<Q',dirty,16,0xdeadbeef00000001)
    for i in range(624):struct.pack_into('<I',dirty,28+8*i,0x12340000+i)
    for start,end in [(5016,5024),(5032,5040),(5044,5048),(5053,5056)]:dirty[start:end]=bytes([0xa7])*(end-start)
    check('cache-signaling-payload-and-full-state-truncation',bytes(dirty),[])
    dead=bytearray(dirty);struct.pack_into('<I',dead,5040,0);dead[5052]=0
    check('invalid-caches-and-padding-canonicalized',bytes(dead),[])
    empty=[(op,0,-7,9) if op==4 else (op,0,0,0) for op in range(9) if op!=7]
    check('zero-count-all-operations-preserve-caches',make_state(),empty)
    original=make_state();reseed=[(7,0,MASK,0),(0,5,0,0)]
    left=invoke('reseed-invalidates-both-caches',packet(reseed,original))
    right=invoke('reseed-fresh-counterpart',packet([(0,5,0,0)],seed=MASK))
    assert left==right
    rows.append({'case':'reseed-invalidates-both-caches','status':'PASS','values':5,'state_bytes':5056,'response_sha256':SHA(left)})
    check('mixed-exact-one-million-word-budget',make_state(),[(8,999998,0,0),(3,1,0,0)])

    good=packet([],make_state());bad=[]
    def change(name,offset,fmt,value,status=1,request=good):
        b=bytearray(request);struct.pack_into(fmt,b,offset,value);bad.append((name,bytes(b),status))
    change('state-size-u64-overflow',24,'<Q',MASK)
    change('command-count-u32-overflow',12,'<I',0xffffffff)
    change('restore-seed-must-be-zero',16,'<Q',1)
    change('negative-signed-left',32+8,'<I',0xffffffff,2)
    change('safe-but-noncanonical-next',32+8,'<I',1,2)
    change('negative-normal-cache-flag',32+5040,'<I',0xffffffff,2)
    change('unsupported-mode',8,'<I',2)
    change('command-reserved',32+24,'<Q',1,request=packet([(0,1,0,0)]))
    change('count-u32-max',32+4,'<I',0xffffffff,3,request=packet([(0,1,0,0)]))
    bad.append(('late-work-overflow-with-double',packet([(8,999999,0,0),(3,1,0,0)],make_state()),3))
    bad.append(('late-work-overflow-with-threshold-range',packet([(8,999999,0,0),(4,1,0,1<<28)],make_state()),3))
    bad.append(('late-invalid-reseed',packet([(0,5,0,0),(7,0,1,1)],make_state()),1))
    for name,request,status in bad:
        observed,out,workspace=observe(image,request,8192,5056)
        assert observed['signal'] is None and observed['returned_u32']==24,name
        assert out[:24]==struct.pack('<3Q',status,0,0) and out[24:]==bytes([0xa5])*(len(out)-24),name
        assert observed['workspace_unchanged'] and observed['input_unchanged'],name
        assert invoke(name,request)==out[:24],name
        rows.append({'case':name,'status':'PASS','native_status':status,'failure_workspace_unchanged':True,
                     'request_sha256':SHA(request),'flat_observation':observed})
    for name,request,work_cap,out_cap,wanted in [
        ('workspace-one-byte-short',packet([]),5055,5080,1),
        ('output-one-value-short',packet([(0,1,0,0)]),5056,5087,3),
        ('input-header-one-byte-short',packet([])[:31],5056,5080,1)]:
        observed,out,workspace=observe(image,request,out_cap,work_cap)
        assert observed['signal'] is None and observed['returned_u32']==24
        assert out[:24]==struct.pack('<3Q',wanted,0,0) and observed['workspace_unchanged']
        assert out[24:]==bytes([0xa5])*(len(out)-24)
        rows.append({'case':name,'status':'PASS','failure_workspace_unchanged':True,'flat_observation':observed})
    assert len(rows)==25 and len({r['case'] for r in rows})==25
    assert native.read_bytes()==image and a.source.read_bytes()==source
    report={'schema':'heptatorch.rng-independent-state-and-admission-review.v1','status':'PASS',
            'native_sha256':SHA(image),'source_sha256':SHA(source),'script_sha256':SHA(Path(__file__).read_bytes()),
            'cpp_reference_source_sha256':SHA(cpp.read_bytes()),'cpp_reference_image_sha256':SHA(binary.read_bytes()),
            'cpp_compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
            'cases':len(rows),'rows':rows,'state_cache_consumption_implemented':False,
            'normal_distribution_implemented':False,'all_pytorch_rng_qualified':False,
            'whole_pytorch_rewrite_complete':False}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
