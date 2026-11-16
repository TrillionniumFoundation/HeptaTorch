"""Replay fixed genuine Torch observations of Bernoulli gradient ownership.

Only frozen framing and independent metadata observers are imported. All
Tensor mutations, copies, aliases, gradients and RNG operations execute natively.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, struct, traceback

PROBE='f43ba1d1cc220c76279e3c8f96fb1358ce421e9f309d209bcdb5a5dbc554f718'
BASE='8a9b56c837e9c7b3af47ed401a91e932fb3359a4c109157df738e5aa63e64f85'
LAYOUT='67b8ec4e141e2837db1e7694dfd7684978ab79a0ca61989be63ad61b491462da'
CASE_NAMES=(
    'contiguous-parameter-contiguous-seed', 'contiguous-parameter-transpose-seed',
    'transpose-parameter-transpose-seed', 'transpose-parameter-contiguous-seed',
    'non-dense-parameter-transpose-seed', 'empty-contiguous-parameter-transpose-seed',
    'empty-transpose-parameter-contiguous-seed', 'special-word-contiguous-seed',
)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(name,path,pin):
    assert sha(path.read_bytes())==pin
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','controls','observer','base-replay','layout-adapter','observations','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-sha256','expected-source-sha256','observations-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--state-format',default='HTGAD001');parser.add_argument('--backing-format',default='HTOGAD01')
    parser.add_argument('--descriptor-bytes',type=int,default=576)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources={path.resolve():path.read_bytes() for path in (Path(__file__),a.binary,a.source,a.controls,a.observer,a.base_replay,a.layout_adapter,a.observations)}
    assert sha(a.binary.read_bytes())==a.expected_sha256 and sha(a.source.read_bytes())==a.expected_source_sha256
    raw=a.observations.read_bytes();assert sha(raw)==a.observations_sha256
    reference=json.loads(raw)
    assert reference['status']=='OBSERVED' and reference['cases']==8 and reference['probe_errors']==0
    assert reference['probe_sha256']==PROBE and tuple(reference['case_plan'])==CASE_NAMES
    assert len(reference['oracle_provenance']['checks'])==11 and all(x is True for x in reference['oracle_provenance']['checks'].values())
    base=load('bernoulli_grad_original_observer',a.base_replay,BASE)
    layout=load('bernoulli_grad_layout_observer',a.layout_adapter,LAYOUT)
    codec=base.load('bernoulli_grad_controls',a.controls,base.CONTROLS_SHA)
    observer=base.load('bernoulli_grad_transport',a.observer,base.OBSERVER_SHA)
    Parent=layout.replay_class(base)
    class Replay(Parent):
        def tensor(self,t,ref):
            assert 'dtype' not in ref and 'device' not in ref
            return super().tensor(t,dict(ref,dtype='torch.float32',device='cpu'))
        def snap(self,ref,p,output,held):
            row=self.snapshot({'parameters':{'p':ref['parameter']},'gradients':{'p':ref['gradient']},
                'held_Tensors':ref['held_Tensors']},{'p':p},held)
            row['output']=self.tensor(output,ref['output']);self.snapshots.append(row);return row
        def event_one(self,ref,p,output,held,action):
            before=self.snap(ref['before'],p,output,held)
            self.eq(ref['outcome'],'SUCCEEDED','actual action outcome')
            action()
            after=self.snap(ref['after'],p,output,held)
            self.events.append({'action':ref['action'],'before':before,'after':after})
        def new_case(self,name):
            empty=name.startswith('empty-')
            transpose_parameter=name.startswith('transpose-') or name.startswith('empty-transpose-')
            if empty:
                p=self.c.new([3,0] if transpose_parameter else [0,3],[])
            elif name.startswith('non-dense-'):
                backing=self.c.new([2,6],[.5]*12)
                p=self.c.make(69,backing,self.blob([2,2,3,0,0,6,2,0,0,0]),80)
                self.c.one(41,backing)
            else:
                p=self.c.new([3,2] if transpose_parameter else [2,3],[.5]*6)
            if transpose_parameter:
                original=p;p=self.c.make(6,original,0,1);self.c.one(41,original)
            self.c.one(90,p,1)
            transpose_seed=name.endswith('transpose-seed')
            if empty:
                seed=self.c.new([3,0] if transpose_seed else [0,3],[])
            elif name=='special-word-contiguous-seed':
                words=(2147483648,2139095040,4286578688,2143289345,1065353216,1)
                seed=self.c.make(1,2,0,2,3,0,0,codec.Blob(struct.pack('<6I',*words)))
            else:
                seed=self.c.new([3,2] if transpose_seed else [2,3],[-0.,2.,3.,4.,5.,6.])
            if transpose_seed:
                original=seed;seed=self.c.make(6,original,0,1);self.c.one(41,original)
            return p,seed
        def state(self,g,expected,label):
            data=self.c.one(63,g)
            self.eq(data['value'],5056,label+' state length')
            self.eq(len(data['payload']),5056,label+' payload length')
            self.eq(data['payload'].hex(),expected,label+' complete Generator state')
        def observe(self,name,ref):
            p,seed=self.new_case(name);g=self.c.make(60,12345)
            self.state(g,ref['generator_state_before_forward_hex'],'before forward')
            output=self.c.make(82,p,g,0)
            self.state(g,ref['generator_state_after_forward_hex'],'after forward')
            held={'seed':seed};events=ref['events'];self.eq(len(events),8,'fixed action count')
            self.event_one(events[0],p,output,held,lambda:self.back(output,seed))
            held['old_gradient']=self.c.make(91,p)
            held['old_alias']=self.c.make(19,held['old_gradient'])
            oldweak=self.c.make(42,held['old_gradient'])
            self.event_one(events[1],p,output,held,lambda:self.c.one(95,seed,codec.bits(.5)))
            self.event_one(events[2],p,output,held,lambda:self.c.one(95,held['old_gradient'],codec.bits(1.)))
            self.event_one(events[3],p,output,held,lambda:self.c.one(92,p,0))
            self.event_one(events[4],p,output,held,lambda:self.c.one(95,held['old_alias'],codec.bits(2.)))
            self.event_one(events[5],p,output,held,lambda:self.back(output,seed))
            held['new_gradient']=self.c.make(91,p);newweak=self.c.make(42,held['new_gradient'])
            self.event_one(events[6],p,output,held,lambda:self.c.one(95,held['old_alias'],codec.bits(4.)))
            self.c.expiry(oldweak,ref['old_storage_expired_before_drops'])
            self.c.one(41,held.pop('old_gradient'))
            self.c.expiry(oldweak,ref['old_alias_only']['storage_expired'])
            self.snap(ref['old_alias_only']['snapshot'],p,output,held)
            self.c.one(41,held.pop('old_alias'))
            self.c.expiry(oldweak,ref['old_storage_expired_after_final_alias_drop'])
            self.event_one(events[7],p,output,held,lambda:self.c.one(92,p,0))
            self.c.expiry(newweak,ref['new_storage_expired_before_public_drop'])
            self.c.one(41,held.pop('new_gradient'))
            self.c.expiry(newweak,ref['new_storage_expired_after_public_drop'])
            self.snap(ref['final'],p,output,held)
            self.state(g,ref['generator_state_after_all_backward_hex'],'after backward and mutation')
            self.c.run([(44,oldweak),(44,newweak),(66,g)])
            self.c.audit()
    report={'schema':'heptatorch.bernoulli-gradient-layout-native-replay.v1','status':'RUNNING',
        'source_sha256':a.expected_source_sha256,'native_sha256':a.expected_sha256,
        'reference_sha256':a.observations_sha256,'probe_sha256':PROBE,
        'test_source_hashes':{str(p):sha(data) for p,data in sources.items() if p.suffix=='.py'},
        'oracle_provenance':reference['oracle_provenance'],'comparison':'exact words/layout/Version/opaque identities/weak lifetime/Generator bytes; no tolerance or Version reset',
        'mutation_implementation':'real native command95 scalar add; no literal overwrite fixture',
        'native_executed':True,'pinned_Torch_executed_during_replay':False,
        'rows':[],'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for index,(name,row) in enumerate(zip(CASE_NAMES,reference['rows'],strict=True)):
        assert row['case']==name and row['status']=='OBSERVED'
        session=observer.OwnerSession(a.binary.resolve(),directory=out/f'case-{index+1:02d}',address_limit=64<<20)
        check=codec.Check(session);replay=Replay(check,codec,a);item={'case':name,'status':'RUNNING'}
        try:
            replay.observe(name,row['observation'])
            item['status']='FAIL' if replay.mismatches else 'QUALIFIED_EXPOSED_SCOPE'
        except Exception as error:item.update(status='FAIL',exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        finally:
            try:item['transport']=session.finish()
            except Exception as error:item.update(status='FAIL',transport_error=repr(error))
            item.update(frames=session.frames,exact_comparisons=check.comparisons,rollback_bytes=check.rollback_bytes,
                frame_records=check.frames,events=replay.events,snapshots=replay.snapshots,mismatches=replay.mismatches)
            report['rows'].append(item);save()
    report.update(cases=len(report['rows']),qualified=sum(r['status']=='QUALIFIED_EXPOSED_SCOPE' for r in report['rows']),
        failed=sum(r['status']=='FAIL' for r in report['rows']),frames=sum(r['frames'] for r in report['rows']),
        exact_comparisons=sum(r['exact_comparisons'] for r in report['rows']),
        rollback_bytes=sum(r['rollback_bytes'] for r in report['rows']),
        mismatch_fields=sum(len(r['mismatches']) for r in report['rows']))
    assert all(p.read_bytes()==data for p,data in sources.items())
    report['status']='FAIL' if report['failed'] else 'QUALIFIED_EXPOSED_SCOPE';save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','oracle_provenance','test_source_hashes')},indent=2))
    return int(report['failed']!=0)

if __name__=='__main__':raise SystemExit(main())
