"""Replay the exact opt-in observers with only explicit ELF rebinding."""
import argparse
from pathlib import Path
import struct
import sys
import tempfile
import types

from prepare_dual import CANDIDATE_NATIVE, bound, insist, serialize, sha

BASE_NATIVE = '083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'
LOGICAL_NATIVE = '0cd10504717c5e5b01114455c55030e63a22b62bb1ba43f188355d399d6c158a'
ABS_NATIVE = '2adc665f1b042145725f8f92ded85e3f7f051f293180d704287847b16a95e005'
PINS = {
    'test_component.py': '97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c',
    'test_runtime_candidate.py': '40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226',
    'qualify_runtime_torch.py': 'b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd',
    'qualify_runtime_layout_torch.py': 'db5c1ac04464b9436a4259fc09f4af9e746c454afe15c2a6231c7bc4da3d57e7',
    'prepare.py': 'a495bc95fb84046a1f5ce6aeb43baa6f410f007c7d690cf84d39c428bef0ba1e',
    'qualify.py': '21a42c23b68a5f1f8b8651c5b94e465d3033567a5bab722e3f3ad1cf9ca55ce2',
}

def load_all(logical, abs_path):
    modules = {}
    # Fixed sources are authenticated before only two narrowly declared image-binding edits.
    for path in (*(logical/name for name in list(PINS)[:4]), abs_path/'prepare.py', abs_path/'qualify.py'):
        name = path.stem
        insist(name not in sys.modules, 'unexpected prior observer module: '+name)
        raw = bound(path,PINS[path.name])
        if path.name=='test_runtime_candidate.py':
            old = ('NATIVE_SHA256 = "'+LOGICAL_NATIVE+'"').encode()
            new = ('NATIVE_SHA256 = "'+CANDIDATE_NATIVE+'"').encode()
            insist(raw.count(old)==1, 'logical binding changed')
            raw=raw.replace(old,new,1)
        if path.name=='qualify.py':
            old=b'CANDIDATE = CANDIDATE_ELF\n'
            new=('CANDIDATE = "'+CANDIDATE_NATIVE+'"\n').encode()
            insist(raw.count(old)==1, 'abs binding changed')
            raw=raw.replace(old,new,1)
        m=types.ModuleType(name)
        m.__file__=str(path)
        exec(compile(raw,str(path),'exec'),m.__dict__)
        sys.modules[name]=m
        modules[name]=m
    return modules

def mixed_checks(wire, baseline, candidate, torch, factory, bool_factory):
    results = []
    failures = []
    with_baseline = wire.Session(baseline)
    try:
        status, rows = wire.parse(with_baseline.exchange(b''.join(bool_factory(1, (1,), (1,)))))
        insist(status==0 and [row['opcode'] for row in rows]==[97], 'baseline Bool factory')
        before = with_baseline.owner()
        status, refusal = wire.parse(with_baseline.exchange(wire.packet([(3,7,rows[0]['value'])])))
        insist(status==7 and not refusal and with_baseline.owner()==before, 'baseline Bool abs refusal')
    finally:
        with_baseline.finish()
    s = wire.Session(candidate)
    try:
        def req(packet):
            return wire.parse(s.exchange(packet))
        def one(*op):
            return s.run([op])[0]
        def inspect(handle):
            meta=wire.meta(one(13,handle))
            tag,width,_,_=struct.unpack('<4Q', one(96,handle,0)['payload'])
            values=one(98,handle,0,meta['numel'])
            insist(values['value']==meta['numel'], 'wrong typed read extent')
            return meta,tag,width,values['payload']
        recipes = [
            (2, (0,), ()),
            (2, (), (1<<63,)),
            (2, (2,3), (0,1,2**64-1,1<<63,2**63-1,2**63+1)),
            (4, (0,3), ()),
            (4, (), (1<<63,)),
            (4, (2,4), (0,1<<63,1,2**63+1,0xfff8000000000421,
                         0xfff0000000000111,0xfff0000000000000,0xbff0000000000000)),
        ]
        for tag,shape,words in recipes:
            dtype = torch.int64 if tag==2 else torch.float64
            buf=struct.pack(f'<{len(words)}Q',*words)
            x=torch.frombuffer(bytearray(buf),dtype=dtype).reshape(shape) if words else torch.empty(shape,dtype=dtype)
            a=torch.abs(x)
            z=torch.logical_not(a)
            request=factory(tag,shape,words,suffix=((3,7,wire.reference(1)),(127,wire.reference(2))))
            status,rows=req(request)
            insist(status==0 and [row['opcode'] for row in rows]==[97,3,127], 'cross-operator same-frame failed')
            input_,abs_,not_ = handles=[row['value'] for row in rows]
            observed=[inspect(handle) for handle in handles]
            a_raw=a.reshape(-1).view(torch.int64).numpy().tobytes()
            z_raw=bytes(z.reshape(-1).tolist())
            insist([(tag,8,buf),(tag,8,a_raw),(1,1,z_raw)]==[row[1:] for row in observed],
                   'wrong mixed result values')
            for obs,expected in zip(observed,(x,a,z),strict=True):
                meta=obs[0]
                insist((meta['rank'],tuple(meta['shape'][:len(shape)]),
                        tuple(meta['strides'][:len(shape)]),meta['storage_offset'],meta['requires_grad'])
                        ==(len(shape),shape,expected.stride(),expected.storage_offset(),0),
                        'wrong mixed result metadata')
            insist(len(set(handles))==3 and all(len({row[0][key] for row in observed})==3 for key in ('storage_id','version_id')),
                   'mixed aliases or identity re-use')
            before=s.owner()
            reversed_=wire.packet([(127,input_),(3,7,wire.reference(1))])
            status,rows=req(reversed_)
            insist(status==7 and not rows and s.owner()==before and inspect(input_)==observed[0],
                   ('reverse order must refuse unsupported Bool abs and roll back exactly',tag,shape,status,rows,s.owner()==before,inspect(input_)==observed[0]))
            for handle in reversed(handles):
                one(41,handle)
            results.append({'dtype':tag,'shape':list(shape),'values_per_operator':len(words),
                            'packet_sha256':sha(request),'abs_sha256':sha(a_raw),'logical_not_sha256':sha(z_raw)})
            failures.append({'dtype':tag,'shape':list(shape),'unsupported_bool_abs_status':status,
                             'owner_snapshot_preserved':True,'packet_sha256':sha(reversed_)})
        insist(one(53)['value']==1,'owner audit failed')
    finally:
        s.finish()
    return {'mixed_successes':results,'mixed_rollbacks':failures,'native_frames':s.frames,
            'baseline_bool_abs_status':7,'baseline_native_frames':with_baseline.frames}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('logical-dir','abs-dir','protocol','oracle-lock','install-report','binary','baseline','output'):
        p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args()
    insist(__debug__ and sys.dont_write_bytecode and sys.flags.ignore_environment and sys.flags.no_user_site,
           'run with assertions and -s -E -B')
    with tempfile.TemporaryDirectory(prefix='dual-observer-cache-') as cache:
        previous=sys.pycache_prefix
        sys.pycache_prefix=cache
        try:
            qualify(a)
        finally:
            sys.pycache_prefix=previous

def qualify(a):
    baseline=a.baseline.resolve()
    binary=a.binary.resolve()
    bound(baseline, BASE_NATIVE)
    bound(binary, CANDIDATE_NATIVE)
    observers=load_all(a.logical_dir, a.abs_dir)
    abs_obs=observers['qualify']
    logic=observers['qualify_runtime_torch']
    layout=observers['qualify_runtime_layout_torch']
    logistics=observers['test_runtime_candidate']
    # The abs oracle configures fixed AVX2 before importing torch; independently
    # apply the logical reader's original lock and metadata checks to that module.
    torch,abs_oracle=abs_obs.load_oracle(a.oracle_lock,a.install_report)
    logical_torch,logic_oracle=logic.load_oracle(a.oracle_lock,a.install_report)
    insist(torch is logical_torch,'oracles did not share the same Torch module')
    wire=logistics.load_wire(a.protocol)
    abs_wire=abs_obs.load_wire(a.protocol)
    logical_controls=logistics.run_native(wire,binary,baseline)
    logical_values=logic.run(wire,binary,torch)
    logical_layout=layout.run(wire,binary,torch)
    abs_result=abs_obs.run(abs_wire,baseline,binary,torch)
    mixed=mixed_checks(wire,baseline,binary,torch,abs_obs.factory,logistics.factory)
    insist(logical_layout['failed']==0 and not abs_result['baseline_matching_view_refusals'],
           'a scope test did not complete')
    bound(binary,CANDIDATE_NATIVE)
    bound(baseline,BASE_NATIVE)
    receipt={'status':'PASS_SCOPED_DUAL_ABS_LOGICAL_NOT','native_image_sha256':CANDIDATE_NATIVE,
             'baseline_image_sha256':BASE_NATIVE,'canonical_runtime_qualification':False,
             'full_candidate_suite_qualification':False, 'whole_pytorch_rewrite_complete':False,
             'observer_sources_sha256':PINS,'abs_oracle':abs_oracle,'logical_oracle':logic_oracle,
             'logical_controls':logical_controls,'logical_values':logical_values,
             'logical_layout':logical_layout,'abs':abs_result,'mixed':mixed}
    with a.output.open('xb') as f:
        f.write(serialize(receipt))

if __name__=='__main__':
    main()
