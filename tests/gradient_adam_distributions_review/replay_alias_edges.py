"""Replay the twelve independently observed alias/mutation/edge-order variants.

Expected values and relations come from the immutable hosted result. Fixed
literal low-level mutation fixtures are explicitly distinguished from add_.
No implementation, authoring or numerical target helper is imported.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, traceback

REFERENCE_SHA='4c9fce4425e401281b2f6d6a19b9bd6dedec250baa144a9cbc956a452d72b43a'
PROBE_SHA='cc0392888f0db9f7026c3f24e0787845b60e43b5e18d8c63cdcb8c720d51acd5'
BASE_SHA='8a9b56c837e9c7b3af47ed401a91e932fb3359a4c109157df738e5aa63e64f85'
LAYOUT_SHA='67b8ec4e141e2837db1e7694dfd7684978ab79a0ca61989be63ad61b491462da'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(name,path,pin):
    assert sha(path.read_bytes())==pin
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def replay_class(base,layout):
    Parent=layout.replay_class(base)
    class Replay(Parent):
        def tensor(self,t,reference):
            # The new probe asserts these constants in info(); it omits them
            # from each repeated snapshot. No observed field is replaced.
            assert 'dtype' not in reference and 'device' not in reference
            return super().tensor(t,dict(reference,dtype='torch.float32',device='cpu'))
        def snap(self,reference,p,held):
            value=self.snapshot({'parameters':{'p':reference['parameter']},
                'gradients':{'p':reference['gradient']},'held_Tensors':reference['held_Tensors']},{'p':p},held)
            self.snapshots.append(value);return value
        def event_one(self,row,p,held,action):
            before=self.snap(row['before'],p,held)
            self.eq(row['outcome'],'SUCCEEDED','actual action succeeds')
            action();after=self.snap(row['after'],p,held)
            self.events.append({'action':row['action'],'before':before,'after':after})
        def drop_keys(self,held,keys):
            ids={held.pop(k) for k in keys}
            for t in ids-set(held.values()):self.c.one(41,t)
        def shared_view(self,variant,r):
            p=self.c.new([2],[1.,-2.],True);out=self.c.make(17,p,2,1,2,0,0)
            held={'seed':self.c.new([1,2],[-0.,2.])};ev=r['events']
            self.event_one(ev[0],p,held,lambda:self.back(out,held['seed']))
            held['gradient']=self.c.make(91,p);held['alias']=self.c.make(19,held['gradient'])
            weak=self.c.make(42,held['seed'])
            order=('seed','gradient') if variant=='seed-first' else ('gradient','seed')
            self.event_one(ev[1],p,held,lambda:self.mutate_literal(held[order[0]],[.5,2.5]))
            self.event_one(ev[2],p,held,lambda:self.mutate_literal(held[order[1]],[1.,3.]))
            self.event_one(ev[3],p,held,lambda:self.c.one(92,p,0))
            self.event_one(ev[4],p,held,lambda:self.mutate_literal(held['alias'],[2.,4.]))
            self.c.expiry(weak,r['storage_expired_before_drops'])
            self.drop_keys(held,('seed','gradient'))
            self.c.expiry(weak,r['alias_only']['storage_expired']);self.snap(r['alias_only']['snapshot'],p,held)
            self.drop_keys(held,('alias',));self.c.expiry(weak,r['storage_expired_after_final_alias_drop']);self.c.one(44,weak)
        def equal_values(self,variant,r):
            p=self.c.new([2],[1.,-2.],True);out=self.c.make(17,p,2,1,2,0,0)
            held={'seed':self.c.new([1,2],[1.,2.])};self.back(out,held['seed'])
            held['gradient']=self.c.make(91,p);held['other']=self.c.make(18 if variant=='clone' else 55,held['gradient'])
            weak=self.c.make(42,held['gradient']);self.snap(r['initial'],p,held);ev=r['events']
            self.event_one(ev[0],p,held,lambda:self.mutate_literal(held['other'],[2.,3.]))
            self.event_one(ev[1],p,held,lambda:self.mutate_literal(held['gradient'],[3.,4.] if variant=='clone' else [4.,5.]))
            self.event_one(ev[2],p,held,lambda:self.c.one(92,p,0))
            self.event_one(ev[3],p,held,lambda:self.mutate_literal(held['other'],[2.5,3.5] if variant=='clone' else [4.5,5.5]))
            self.drop_keys(held,('seed','gradient'))
            self.c.expiry(weak,r['other_only']['storage_expired']);self.snap(r['other_only']['snapshot'],p,held)
            self.drop_keys(held,('other',));self.c.expiry(weak,r['storage_expired_after_all_drops']);self.c.one(44,weak)
        def edge_order(self,variant,r,shared):
            p=self.c.new([1],[1.],True);held={}
            target=self.constant_mul(p,1.) if shared else p
            if shared:held['shared']=target
            def cat():
                joined=self.c.make(34,0,self.blob([2,target,target]));result=self.sum(joined);self.c.one(41,joined);return result
            def mul():
                term=self.constant_mul(target,16777216.);result=self.sum(term);self.c.one(41,term);return result
            if variant=='right-created-last':left=cat();right=mul()
            else:right=mul();left=cat()
            held.update(cat_branch=left,mul_branch=right);out=self.c.make(2,1,left,right)
            self.event_one(r['events'][0],p,held,lambda:self.back(out))
            g=self.c.make(91,p);self.tensor(g,r['parameter_gradient'])
        def clear_recreate(self,variant,r):
            p=self.c.new([2],[1.,-2.],True)
            out=self.c.make(17,p,2,1,2,0,0) if variant=='view' else self.c.make(18,p)
            held={'seed':self.c.new([1,2] if variant=='view' else [2],[1.,2.])};ev=r['events']
            self.event_one(ev[0],p,held,lambda:self.back(out,held['seed']))
            held['old_gradient']=self.c.make(91,p);held['old_alias']=self.c.make(19,held['old_gradient'])
            weak=self.c.make(42,held['old_gradient'])
            self.event_one(ev[1],p,held,lambda:self.c.one(92,p,0))
            self.event_one(ev[2],p,held,lambda:self.mutate_literal(held['seed'],[1.5,2.5]))
            self.event_one(ev[3],p,held,lambda:self.back(out,held['seed']))
            held['new_gradient']=self.c.make(91,p)
            self.event_one(ev[4],p,held,lambda:self.mutate_literal(held['old_alias'],[2.5,3.5] if variant=='view' else [2.,3.]))
            self.drop_keys(held,('old_gradient','old_alias','seed'))
            self.c.expiry(weak,r['new_gradient_only']['storage_expired']);self.snap(r['new_gradient_only']['snapshot'],p,held)
            self.event_one(ev[5],p,held,lambda:self.c.one(92,p,0))
            self.drop_keys(held,('new_gradient',));self.c.expiry(weak,r['storage_expired_after_all_gradient_and_seed_drops']);self.c.one(44,weak)
        def transpose_alias(self,variant,r):
            p=self.c.new([2,2],[1.,-2.,.5,2.],True);out=self.c.make(6,p,0,1)
            held={'seed_base':self.c.new([2,2],[1.,2.,3.,4.])}
            held['seed']=self.c.make(6,held['seed_base'],0,1) if variant=='matching' else held['seed_base']
            ev=r['events'];self.event_one(ev[0],p,held,lambda:self.back(out,held['seed']))
            held['gradient']=self.c.make(91,p);weak=self.c.make(42,held['gradient'])
            self.event_one(ev[1],p,held,lambda:self.mutate_literal(held['seed'],[1.5,3.5,2.5,4.5] if variant=='matching' else [1.5,2.5,3.5,4.5]))
            self.event_one(ev[2],p,held,lambda:self.mutate_literal(held['gradient'],[2.5,3.5,4.5,5.5] if variant=='matching' else [2.,4.,3.,5.]))
            self.event_one(ev[3],p,held,lambda:self.c.one(92,p,0))
            self.drop_keys(held,('seed','seed_base'))
            self.c.expiry(weak,r['gradient_only']['storage_expired']);self.snap(r['gradient_only']['snapshot'],p,held)
            self.drop_keys(held,('gradient',));self.c.expiry(weak,r['storage_expired_after_final_gradient_drop']);self.c.one(44,weak)
        def observe_alias(self,family,variant,r):
            if family==0:self.shared_view(variant,r)
            elif family==1:self.equal_values(variant,r)
            elif family in (2,3):self.edge_order(variant,r,family==3)
            elif family==4:self.clear_recreate(variant,r)
            else:self.transpose_alias(variant,r)
            self.c.audit()
    return Replay

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','controls','observer','base-replay','layout-adapter','observations','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True);p.add_argument('--expected-source-sha256',required=True)
    p.add_argument('--state-format',default='HTGRAD02');p.add_argument('--backing-format',default='HTOGRAD2')
    p.add_argument('--descriptor-bytes',type=int,default=576)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources={path.resolve():path.read_bytes() for path in (a.source,a.binary,a.base_replay,a.layout_adapter,a.controls,a.observer,a.observations)}
    assert sha(a.binary.read_bytes())==a.expected_sha256 and sha(a.source.read_bytes())==a.expected_source_sha256
    raw=a.observations.read_bytes();assert sha(raw)==REFERENCE_SHA
    ref=json.loads(raw);assert ref['status']=='OBSERVED' and ref['cases']==6 and ref['variants']==12 and ref['probe_errors']==0
    assert ref['probe_sha256']==PROBE_SHA
    checks=ref['oracle_provenance']['checks'];assert len(checks)==11 and all(v is True for v in checks.values())
    base=load('alias_original_ten',a.base_replay,BASE_SHA);layout=load('alias_layout_observer',a.layout_adapter,LAYOUT_SHA)
    codec=base.load('alias_controls',a.controls,base.CONTROLS_SHA);observer=base.load('alias_owner_observer',a.observer,base.OBSERVER_SHA)
    Replay=replay_class(base,layout)
    report={'schema':'heptatorch.gradient-alias-edge-native-replay.v1','status':'RUNNING',
        'native_sha256':a.expected_sha256,'source_sha256':a.expected_source_sha256,
        'reference_sha256':sha(raw),'probe_sha256':PROBE_SHA,'runner_sha256':sha(Path(__file__).read_bytes()),
        'base_runner_sha256':BASE_SHA,'layout_observer_sha256':LAYOUT_SHA,
        'layout':{'state_format':a.state_format,'backing_format':a.backing_format,'descriptor_bytes':a.descriptor_bytes},
        'comparison_policy':'Exact words, layouts, Versions under mutation, opaque Tensor/Storage relations and Storage Weak expiry. No tolerance or Version reset.',
        'mutation_boundary':'Fixed dyadic literal writes are native low-level fixtures with an independently versioned temporary data alias, one real source write and real alias drop. They are not an implementation or qualification of Tensor.add_. Literal arrays are fixed in this runner, not copied from expected snapshots.',
        'oracle_provenance':ref['oracle_provenance'],'records':[],
        'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    for family,row in enumerate(ref['rows']):
        for j,v in enumerate(row['observations']):
            session=observer.OwnerSession(a.binary.resolve(),directory=out/f'family-{family+1:02d}-variant-{j+1:02d}',address_limit=64<<20)
            c=codec.Check(session);replay=Replay(c,codec,a);item={'case':row['case'],'variant':v['variant'],'status':'RUNNING'}
            try:
                replay.observe_alias(family,v['variant'],v['observation'])
                item['status']='FAIL' if replay.mismatches else 'QUALIFIED_EXPOSED_SCOPE'
            except Exception as error:item.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
            finally:
                try:item['transport']=session.finish()
                except Exception as error:item.update(status='FAIL',transport_error=repr(error))
                item.update(frames=session.frames,comparisons_attempted=c.comparisons,rollback_bytes=c.rollback_bytes,
                    frame_records=c.frames,events=replay.events,snapshots=replay.snapshots,mismatches=replay.mismatches,
                    literal_mutation_fixtures=replay.fixtures)
                report['records'].append(item);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report.update(reference_families=6,reference_variants=12,executed_variants=12,
        qualified_variants=sum(r['status']=='QUALIFIED_EXPOSED_SCOPE' for r in report['records']),
        failed_variants=sum(r['status']=='FAIL' for r in report['records']),
        frames=sum(r['frames'] for r in report['records']),comparisons_attempted=sum(r['comparisons_attempted'] for r in report['records']),
        rollback_bytes=sum(r['rollback_bytes'] for r in report['records']),mismatch_fields=sum(len(r['mismatches']) for r in report['records']))
    report['status']='FAIL' if report['failed_variants'] else 'QUALIFIED_WITH_LITERAL_MUTATION_BOUNDARY'
    assert all(path.read_bytes()==data for path,data in sources.items())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','oracle_provenance')},indent=2))
    return int(bool(report['failed_variants']))

if __name__=='__main__':raise SystemExit(main())
