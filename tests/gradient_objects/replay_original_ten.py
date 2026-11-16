"""Run the immutable public-gradient cases with only a physical-layout observer adapter.

The original case, input, tensor/snapshot comparison and mutation-fixture methods
are inherited byte-for-byte. Only read-only descriptor addressing is selected
from the previously frozen portable materialization observer.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, traceback

BASE_SHA='8a9b56c837e9c7b3af47ed401a91e932fb3359a4c109157df738e5aa63e64f85'
LAYOUT_SHA='67b8ec4e141e2837db1e7694dfd7684978ab79a0ca61989be63ad61b491462da'

def sha(data):return hashlib.sha256(data).hexdigest()
def load(name,path,pin):
    assert sha(path.read_bytes())==pin
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','controls','observer','base-replay','layout-adapter','observations','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--expected-source-sha256',required=True)
    p.add_argument('--state-format',default='HTGRAD02')
    p.add_argument('--backing-format',default='HTOGRAD2')
    p.add_argument('--descriptor-bytes',type=int,default=576)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources={path.resolve():path.read_bytes() for path in (a.source,a.binary,a.base_replay,a.layout_adapter,a.controls,a.observer,a.observations)}
    assert sha(a.binary.read_bytes())==a.expected_sha256 and sha(a.source.read_bytes())==a.expected_source_sha256
    base=load('incoming_original_ten',a.base_replay,BASE_SHA)
    layout=load('incoming_layout_observer',a.layout_adapter,LAYOUT_SHA)
    codec=base.load('incoming_controls',a.controls,base.CONTROLS_SHA)
    observer=base.load('incoming_owner_observer',a.observer,base.OBSERVER_SHA)
    raw=a.observations.read_bytes();assert sha(raw)==base.OBSERVATION_SHA
    ref=json.loads(raw)
    assert ref['status']=='OBSERVED' and ref['cases']==10 and ref['probe_errors']==0 and ref['probe_sha256']==base.PROBE_SHA
    checks=ref['oracle_provenance']['checks'];assert len(checks)==11 and all(v is True for v in checks.values())
    Layout=layout.replay_class(base)
    class Replay(base.Replay):
        def __init__(self,c,b):super().__init__(c,b);self.args=a
        raw_node=Layout.raw_node
    assert all(getattr(Replay,'case'+str(i)) is getattr(base.Replay,'case'+str(i)) for i in range(1,11))
    assert all(getattr(Replay,n) is getattr(base.Replay,n) for n in ('eq','identity','tensor','snapshot','event','mutate_literal'))
    report={'schema':'heptatorch.original-gradient-ten-layout-adapter.v1','status':'RUNNING',
        'native_sha256':a.expected_sha256,'source_sha256':a.expected_source_sha256,
        'reference_sha256':sha(raw),'probe_sha256':base.PROBE_SHA,'runner_sha256':sha(Path(__file__).read_bytes()),
        'base_runner_sha256':BASE_SHA,'layout_observer_sha256':LAYOUT_SHA,
        'layout':{'state_format':a.state_format,'backing_format':a.backing_format,'descriptor_bytes':a.descriptor_bytes},
        'case_methods_identical':True,'all_numerical_identity_version_assertions_identical':True,
        'comparison_policy':'Original ten cases and all exact assertions unchanged; only read-only physical descriptor addressing adapted.',
        'oracle_provenance':ref['oracle_provenance'],'cases':[],
        'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    for i,row in enumerate(ref['rows'],1):
        session=observer.OwnerSession(a.binary.resolve(),directory=out/f'case-{i:02d}',address_limit=64<<20)
        c=codec.Check(session);replay=Replay(c,codec);item={'case':row['case'],'status':'RUNNING'}
        try:
            getattr(replay,f'case{i}')(row['observation']);c.audit()
            item['status']='PARTIAL' if replay.open else 'QUALIFIED_EXPOSED_SCOPE'
        except Exception as error:item.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        finally:
            try:item['transport']=session.finish()
            except Exception as error:item.update(status='FAIL',transport_error=repr(error))
            item.update(frames=session.frames,exact_comparisons=c.comparisons,rollback_bytes=c.rollback_bytes,
                frame_records=c.frames,events=replay.events,open_actions=replay.open,literal_mutation_fixtures=replay.fixtures)
            report['cases'].append(item)
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report.update(reference_groups=10,executed_groups=10,
        qualified_groups=sum(r['status']=='QUALIFIED_EXPOSED_SCOPE' for r in report['cases']),
        partial_groups=sum(r['status']=='PARTIAL' for r in report['cases']),failed_groups=sum(r['status']=='FAIL' for r in report['cases']),
        frames=sum(r['frames'] for r in report['cases']),exact_comparisons=sum(r['exact_comparisons'] for r in report['cases']),
        rollback_bytes=sum(r['rollback_bytes'] for r in report['cases']))
    report['status']='FAIL' if report['failed_groups'] else 'QUALIFIED_WITH_EXPLICIT_OPEN_BOUNDARIES'
    assert all(path.read_bytes()==data for path,data in sources.items())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','oracle_provenance')},indent=2))
    return int(bool(report['failed_groups']))

if __name__=='__main__':raise SystemExit(main())
