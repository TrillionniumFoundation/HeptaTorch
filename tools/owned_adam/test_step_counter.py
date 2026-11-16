"""Exact actual-float32 step boundaries, Versions, and None-before-admission."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,struct
from fractions import Fraction

SHA=lambda data:hashlib.sha256(data).hexdigest()
Q=lambda words:struct.pack('<'+'Q'*len(words),*words)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','protocol','build-report','output'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    snapshots={p.resolve():p.read_bytes() for p in (Path(__file__),a.binary,a.source,a.protocol,a.build_report)}
    build=json.loads(snapshots[a.build_report.resolve()])
    assert SHA(snapshots[a.binary.resolve()])==build['native_sha256'] and SHA(snapshots[a.source.resolve()])==build['source_sha256']
    assert SHA(snapshots[a.protocol.resolve()])=='5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec=importlib.util.spec_from_file_location('adam_step_wire',a.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    s=wire.Session(a.binary,directory=a.output/'session',address_limit=64<<20)
    report={'schema':'heptatorch.actual-float32-Adam-step-boundaries.v1','status':'RUNNING',
        'native_sha256':build['native_sha256'],'source_sha256':build['source_sha256'],
        'native_executed':True,'Torch_executed_here':False,'rows':[],'rollback_bytes':0,
        'inputs_sha256':{str(p):SHA(raw) for p,raw in snapshots.items()}}
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    p=s.run([wire.tensor((),[1],1)])[0]['value'];g=s.run([wire.tensor((),[2])])[0]['value']
    s.run([(58,p,g,0)])
    opt=s.run([(71,1,wire.Payload(Q([wire.bits(.125)]+[0]*7)),wire.Payload(Q([1,1,0]+[0]*8+[p])))])[0]['value']
    s.run([(72,opt)]);step=s.run([(77,opt,0,0,1)])[0]['value']
    step_version=1;updates=1
    def metadata(t):return wire.meta(s.run([(13,t)])[0])
    # Expected words are exact IEEE-754 additions in these explicitly stated
    # representable/tie cases, independent of the target's implementation.
    for name,before,after in [('positive-zero',0,0x3f800000),('negative-zero',0x80000000,0x3f800000),
        ('fraction-rounds-to-integer',0x3f7fffff,0x40000000),
        ('counting-stagnation-at-2pow24',0x4b800000,0x4b800000),
        ('largest-binary32-below-2pow32',0x4f7fffff,0x4f7fffff)]:
        s.run([(12,step,0,before)]);step_version+=1
        s.run([(72,opt)]);step_version+=1;updates+=1
        raw=s.run([(10,step)])[0]['payload'];assert raw==struct.pack('<I',after)
        assert metadata(step)['version']==step_version and metadata(p)['version']==updates
        assert s.run([(10,p)])[0]['payload']==struct.pack('<f',float(Fraction(1)-Fraction(updates,8)))
        report['rows'].append({'case':name,'status':'PASS','external_word':before,
            'incremented_word':after,'step_Version':step_version,'parameter_updates':updates});save()
    for name,before in [('result-zero',0xbf800000),('result-fraction',0xbf000000),
                       ('2pow32',0x4f800000),('positive-infinity',0x7f800000),
                       ('negative-infinity',0xff800000),('NaN',0x7fc00001)]:
        s.run([(12,step,0,before)]);step_version+=1
        report['rollback_bytes']+=s.error([(72,opt)],7)
        assert s.run([(10,step)])[0]['payload']==struct.pack('<I',before)
        assert metadata(step)['version']==step_version and metadata(p)['version']==updates
        report['rows'].append({'case':name,'status':'PASS','external_word':before,'rejection':7,
            'step_Version':step_version,'all_owner_bytes_unchanged_after_refusal':True});save()
    # The invalid externally stored step is not even consumed when grad=None.
    s.run([(73,opt,1),(72,opt)])
    assert s.run([(59,p)])[0]['value']==0
    assert s.run([(10,step)])[0]['payload']==struct.pack('<I',0x7fc00001)
    assert metadata(step)['version']==step_version and metadata(p)['version']==updates
    s.run([(53,)]);s.finish()
    for path,raw in snapshots.items():assert path.read_bytes()==raw,path
    (a.output/'executed-test.py').write_bytes(snapshots[Path(__file__).resolve()])
    report.update(status='PASS',cases=len(report['rows']),frames=s.frames,
        None_skip_precedes_step_domain_validation=True,all_responses_before_stdin_eof=True)
    save();print(json.dumps({k:report[k] for k in ('status','cases','frames','rollback_bytes')}))

if __name__=='__main__':main()
