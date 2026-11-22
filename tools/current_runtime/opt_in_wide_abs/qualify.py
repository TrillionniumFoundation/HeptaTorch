"""Focused opt-in I64/F64 CPU abs differential, not the canonical current gate."""
import argparse
from importlib import metadata, util
import json
import math
import os
import platform
from pathlib import Path
import resource
import struct
import sys
from prepare import BASE_ELF, CANDIDATE_ELF, sha

CANDIDATE = CANDIDATE_ELF
LOCK = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
WIRE = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
TRANSPORT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
S = 1 << 63
I64 = (0,1,2,S-1,S,S+1,2*S-1,2*S-2,2**32,2*S-2**32)
F64 = (0,S,1,S|1,0x000fffffffffffff,0x800fffffffffffff,0x0010000000000000,
       0x8010000000000000,0x3ff0000000000000,0xbff0000000000000,
       0x7fefffffffffffff,0xffefffffffffffff,0x7ff0000000000000,0xfff0000000000000,
       0x7ff0000000000001,0xfff0000000000001,0x7ff8000000000001,0xfff8000000000001)

def need(condition, message):
    if not condition:
        raise ValueError(message)

def load_oracle(lock, report):
    raw = lock.read_bytes()
    need(sha(raw)==LOCK and sys.implementation.name=='cpython' and sys.version_info[:2]==(3,12)
         and sys.platform=='linux' and platform.machine()=='x86_64' and sys.byteorder=='little',
         'wrong oracle lock or interpreter')
    pin = json.loads(raw)['oracle']
    wheel = pin['wheel']
    d = metadata.distribution('torch')
    need(d.version==pin['version'] and sha(d.read_text('METADATA').encode())==wheel['metadata_sha256'],
         'wrong installed metadata')
    installs = [row for row in json.loads(report.read_bytes())['install']
                if row.get('metadata',{}).get('name','').lower()=='torch']
    need(len(installs)==1 and installs[0]['requested'] and installs[0]['metadata']['version']==pin['version'],
         'wrong oracle install record')
    for x in (json.loads(d.read_text('direct_url.json')), installs[0]['download_info']):
        need(x['url']==wheel['url'] and x['archive_info']['hashes']['sha256']==wheel['sha256'], 'wrong oracle artifact identity')
    need('torch' not in sys.modules,'Torch imported before dispatch pin')
    os.environ['ATEN_CPU_CAPABILITY']='avx2'
    import torch
    need(torch.__version__==pin['version'] and torch.version.git_version==pin['git_commit']
         and Path(torch.__file__).resolve()==Path(d.locate_file('torch/__init__.py')).resolve()
         and {k:getattr(torch.version,k,None) for k in pin['backend_versions']}==pin['backend_versions']
         and torch.backends.cpu.get_cpu_capability()=='AVX2','wrong loaded CPU oracle')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return torch, {'version':pin['version'],'commit':pin['git_commit'],'wheel_sha256':wheel['sha256'],'cpu_capability':'AVX2','threads':1}

def load_wire(path):
    need(sha(path.read_bytes())==WIRE and sha(path.with_name('owner_client.py').read_bytes())==TRANSPORT,'wrong wire source')
    spec=util.spec_from_file_location('bound_abs_wire',path)
    v=util.module_from_spec(spec);spec.loader.exec_module(v)
    return v

def factory(dtype,shape,words, suffix=(), requires=0):
    data = b''.join(w.to_bytes(8 if dtype in (2,4) else 4,'little') for w in words)
    count=1+len(suffix); offset=32+64*count
    config=struct.pack('<8Q',len(shape),requires,*shape,*([0]*(4-len(shape))),offset+64,len(data))
    return (b'HTORCH03'+struct.pack('<3Q',count,offset+64+len(data),0)
            +struct.pack('<8Q',97,dtype,offset,0,0,0,0,0)
            +b''.join(struct.pack('<8Q',*op,*([0]*(8-len(op)))) for op in suffix)+config+data)

def run(wire, baseline, candidate, torch):
    need(__debug__ and sha(baseline.read_bytes())==BASE_ELF and sha(candidate.read_bytes())==CANDIDATE,
         'wrong baseline/candidate or optimized interpreter')
    b=wire.Session(baseline); n=wire.Session(candidate)
    tests=[]; controls=[]; input_refusals=[]
    try:
        def call(s, packet): return wire.parse(s.exchange(packet))

        def one(s,*op):
            status, records=call(s,wire.packet([op]))
            need(status==0 and [r['opcode'] for r in records]==[op[0]],('failed command',op,status))
            return records[0]

        def create(s,tag,shape,words):
            status, records=call(s,factory(tag,shape,words))
            need(status==0 and [r['opcode'] for r in records]==[97],('failed factory',tag,shape,status))
            return records[0]['value']

        def inspect(s, h):
            m=wire.meta(one(s,13,h))
            tag,width,_,_=struct.unpack('<4Q',one(s,96,h,0)['payload'])
            data=one(s,98,h,0,m['numel'])
            need(data['value']==m['numel'],'bad read extent')
            return m,tag,width,data['payload']

        def reference(tag, shape, words, view):
            dtype=torch.int64 if tag==2 else torch.float64
            x=torch.frombuffer(bytearray(struct.pack(f'<{len(words)}Q',*words)),dtype=dtype).reshape(shape) if words else torch.empty(shape,dtype=dtype)
            return x if view is None else x.as_strided(*view)

        def expected(x):
            r=torch.abs(x)
            return r,r.reshape(-1).view(torch.int64).numpy().tobytes()

        def verify(s,src,out,ref,raw,tag,before):
            m,d,w,v=inspect(s,out)
            need((d,w,v)==(tag,8,raw),('values',tag,tuple(ref.shape)))
            need((m['rank'],tuple(m['shape'][:ref.ndim]),tuple(m['strides'][:ref.ndim]),m['storage_offset'],m['requires_grad'])
                 ==(ref.ndim,tuple(ref.shape),ref.stride(),ref.storage_offset(),0),
                 ('layout',tag,tuple(ref.shape),ref.stride(),m['strides']))
            need(out!=src and m['storage_id']!=before[0]['storage_id'] and m['version_id']!=before[0]['version_id']
                 and inspect(s,src)==before,'fresh output and immutable input required')

        def add_case(tag,words,shape,label,view=None):
            bs=create(b,tag,shape,words); ns=create(n,tag,shape,words)
            bb=bs; nb=ns
            if view:
                dims,strides,offset=view; rank=len(dims)
                cfg=struct.pack('<10Q',rank,*dims,*([0]*(4-rank)),*strides,*([0]*(4-rank)),offset)
                rc_b, rows_b = call(b,wire.packet([(111,bs,wire.Payload(cfg),80)]))
                rc_n, rows_n = call(n,wire.packet([(111,ns,wire.Payload(cfg),80)]))
                need(rc_b==rc_n,('changed view status',label,rc_b,rc_n))
                if rc_n:
                    input_refusals.append([tag,label,rc_n])
                    one(b,41,bb);one(n,41,nb)
                    return
                bs=rows_b[0]['value'];ns=rows_n[0]['value']
            ref, raw=expected(reference(tag,shape,words,view))
            before=inspect(n,ns)
            old=call(b,wire.packet([(3,7,bs)]))
            need(old==(7,[]),('old refusal not preserved on baseline',old))
            r=one(n,3,7,ns)['value']
            verify(n,ns,r,ref,raw,tag,before)
            original_neg=one(b,3,6,bs)['value'];new_neg=one(n,3,6,ns)['value']
            bm,bd,bw,bv=inspect(b,original_neg);nm,nd,nw,nv=inspect(n,new_neg)
            for key in ('shape','rank','strides','numel','storage_offset','requires_grad'):
                need(bm[key]==nm[key],('neg metadata changed',key))
            need((bd,bw,bv)==(nd,nw,nv),'neg values changed')
            tests.append({'dtype':tag,'label':label,'values':ref.numel(),'torch_raw_sha256':sha(raw),'baseline_op7_status':7})
            one(b,41,original_neg);one(n,41,new_neg);one(n,41,r)
            if view: one(b,41,bs);one(n,41,ns)
            one(b,41,bb);one(n,41,nb)

        pinned_historical_probes = (
            (2, (1,S), 'wide_neg:i64-protocol-controls:unary7'),
            (4, (0x3ff0000000000000,S), 'wide_neg:successor-unsupported-F64-neg:unary7'),
        )
        for tag,words,label in pinned_historical_probes:
            add_case(tag,words,(2,),label)

        oom_attempts=[]
        for tag,words in ((2,I64),(4,F64)):
            recipes=[('dense',words,(len(words),),None),('zero_dim_empty',(),(0,3),None),
                     *(('scalar_'+str(i),(w,),(),None) for i,w in enumerate(words))]
            sweep = tuple((sign | exp<<52 | fraction) for sign in (0,S) for exp in range(2048)
                          for fraction in (0,1,0x8000000000000,0xfffffffffffff)) if tag==4 else tuple(
                          (sign*(1<<bit)+off) % (2*S) for sign in (-1,1) for bit in range(64) for off in (-2,-1,0,1,2))
            recipes += [('exponents_or_integer_boundaries',sweep,(len(sweep),),None)]
            vec=(words*10)[:80]
            recipes += [(name,vec,(80,),view) for name,view in (
                ('transpose',((2,3),(1,2),0)),('holes',((2,3),(7,2),1)),('broadcast',((2,3),(0,1),0)),
                ('overlap',((2,3),(1,1),0)),('offset_dense',((2,3),(3,1),3)),
                ('channels_last',((2,3,2,2),(12,1,6,3),1)),('empty_view',((0,3),(42,1),2)))]
            for name,vec,size,view in recipes: add_case(tag,vec,size,name,view)

            current=create(n,tag,(2,),words[:2]); initial=inspect(n,current)
            for invalid in (0,1):
                need(call(n,wire.packet([(3,7,invalid)]))==(1,[]),'invalid handle')
                controls.append([tag,'invalid_handle',invalid,1])
            stale=create(n,tag,(1,),words[:1]);one(n,41,stale)
            need(call(n,wire.packet([(3,7,stale)]))==(1,[]),'stale handle')
            controls.append([tag,'stale_handle',1])
            owner=n.owner()
            need(call(n,wire.packet([(3,7,current),(125,current)]))==(4,[])
                 and n.owner()==owner and inspect(n,current)==initial,'rollback')
            controls.append([tag,'invalid_suffix_rollback',4])
            rs=call(n,factory(tag,(2,),words[:2],requires=1))
            need(rs==(5,[]),'requires-grad envelope not refused');controls.append([tag,'requires_grad_factory',5])
            # FE selects the output-ordinal-zero result within the same transaction.
            status,rows=call(n,factory(tag,(2,),words[:2],suffix=((3,7,wire.reference(1)),)))
            need(status==0 and [r['opcode'] for r in rows]==[97,3],('same frame',status))
            src,out=[r['value'] for r in rows]; ref,raw=expected(reference(tag,(2,),words[:2],None))
            verify(n,src,out,ref,raw,tag,inspect(n,src))
            controls.append([tag,'same_frame',0]);one(n,41,out);one(n,41,src);one(n,41,current)
            large=(words*65536)[:65536]
            src=create(n,tag,(len(large),),large)
            read_before=inspect(n,src)
            ref,raw=expected(reference(tag,(len(large),),large,None))
            owner=n.owner()
            probe=wire.packet([(3,7,src)])
            page=os.sysconf('SC_PAGE_SIZE')
            address_limit=resource.prlimit(n.pid,resource.RLIMIT_AS)
            vmem=int(Path(f'/proc/{n.pid}/statm').read_bytes().split()[0])*page
            ceiling=vmem+((owner['length']+16+page-1)//page)*page+65536
            need(address_limit[1]==resource.RLIM_INFINITY or ceiling<=address_limit[1],'RLIMIT_AS hard limit')
            held=[]
            try:
                resource.prlimit(n.pid,resource.RLIMIT_AS,(ceiling,address_limit[1]))
                need(resource.prlimit(n.pid,resource.RLIMIT_AS)==(ceiling,address_limit[1]),'RLIMIT_AS not applied')
                for attempt in range(1,33):
                    owner_at=n.owner()
                    status,rows=call(n,probe)
                    if status:
                        need(status==3 and not rows and n.owner()==owner_at,'wrong OOM or owner rollback')
                        break
                    need([r['opcode'] for r in rows]==[3],'wrong bounded success response')
                    held.append(rows[0]['value'])
                else:
                    raise ValueError('no real allocation refusal')
            finally:
                resource.prlimit(n.pid,resource.RLIMIT_AS,address_limit)
            need(resource.prlimit(n.pid,resource.RLIMIT_AS)==address_limit,'RLIMIT_AS not restored')
            status,rows=call(n,probe)
            need(status==0 and [r['opcode'] for r in rows]==[3],'identical-probe OOM recovery')
            out=rows[0]['value'];verify(n,src,out,ref,raw,tag,read_before)
            oom_attempts.append({'dtype':tag,'attempt':attempt,'same_packet_sha256':sha(probe),
                                 'values':ref.numel(),'result_sha256':sha(raw)})
            for handle in [*held,out,src]:one(n,41,handle)

        for tag,w in ((0,0x3f800000),):
            ob=create(b,tag,(1,),(w,));on=create(n,tag,(1,),(w,))
            original=call(b,wire.packet([(3,7,ob)]))
            need(original==(4,[]) and call(n,wire.packet([(3,7,on)]))==original,'wrong dtype boundary')
            controls.append([tag,'other_dtype',4]);one(b,41,ob);one(n,41,on)
        need(one(b,53)['value']==one(n,53)['value']==1,'final owner audit')
    finally:
        b.finish();n.finish()
    return {'cases':len(tests),'value_comparisons':sum(row['values'] for row in tests),
            'baseline_frames':b.frames,'candidate_frames':n.frames,'cases_detail':tests,
            'baseline_matching_view_refusals':input_refusals,'controls':controls,'oom_recoveries':oom_attempts,
            'oom_retry_value_comparisons':sum(row['values'] for row in oom_attempts),
            'historical_refusal_successors':[row for row in tests if row['label'].startswith('wide_neg:')]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('baseline','candidate','protocol','oracle-lock','install-report'):
        p.add_argument('--'+arg,required=True,type=Path)
    a=p.parse_args()
    torch, oracle=load_oracle(a.oracle_lock,a.install_report)
    res=run(load_wire(a.protocol),a.baseline.resolve(),a.candidate.resolve(),torch)
    print(json.dumps({'status':'PASS_SCOPED_OPT_IN_WIDE_ABS','canonical_runtime_qualification':False,
                      'baseline_image_sha256':BASE_ELF,'candidate_image_sha256':CANDIDATE,'oracle':oracle,**res},
                     indent=2,sort_keys=True))

if __name__=='__main__':
    main()
