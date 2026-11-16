"""Replay the frozen ten actual Torch .grad observations through native HIA.

No authoring or target arithmetic is imported. The fixed recipes serialize
requests; all expected words/metadata/identities/Versions are read from the
independently hosted reference. The child owner is read only to observe stable
Tensor incarnations and the native leaf role, never to mutate target state.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,struct,traceback

OBSERVATION_SHA='4583a53c5e3a16e8fa17af1b3192d8876e6be834591b920fabc1e68d86f49dbb'
PROBE_SHA='e69b849b3521ceae35a835f84b76eca8c472253acd2c6ec20869f7f7bc2e3e62'
CONTROLS_SHA='43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
OBSERVER_SHA='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def load(name,path,pin):
    assert sha(path.read_bytes())==pin,path
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

class Replay:
    def __init__(self,c,b):
        self.c,self.b=c,b;self.tensor_ids={};self.storage_ids={};self.events=[];self.open=[];self.fixtures=[]

    def eq(self,actual,expected,label):
        assert actual==expected,(label,actual,expected)
        self.c.comparisons+=1

    def identity(self,table,reference,native,label):
        if reference in table:self.eq(table[reference],native,label+' stable relation')
        else:
            assert native not in table.values(),(label,'distinct actual identities collided',reference,native)
            table[reference]=native;self.c.comparisons+=1

    def raw_node(self,tid=None,node=None):
        owner=self.c.session.owner();raw=owner['bytes']
        def u(at):
            assert 0<=at<=len(raw)-8
            return struct.unpack_from('<Q',raw,at)[0]
        assert raw[64:72]==b'HTGRAD01' and raw[139264:139272]==b'HTOGRAD1'
        slots=u(139264+16);poolbase=139264+144+552*slots
        if tid is not None:
            self.eq((tid>>24)&255,1,'real Tensor kind')
            slot=tid&0xffffff;assert 0<slot<=slots
            row=139264+128+(slot-1)*64
            assert u(row)==tid>>32 and u(row+8)==1
            node=u(row+48)
        assert node is not None and 0<node<u(8)
        page=u(4096+(node>>8)*8)
        assert page>>63==1
        at=poolbase+(page&((1<<63)-1))+(node&255)*448
        assert u(at+304)==1
        if tid is not None:assert u(at+192)==tid and u(at+200)==1
        return {offset:u(at+offset) for offset in (32,40,192,200,208,240,312,320,360,384,392,400,408)}

    def tensor(self,tid,reference):
        meta=list(self.b.words(self.c.one(13,tid)))
        rank=len(reference['shape'])
        self.eq(len(meta),16,'metadata width')
        for actual,expected,label in (
            (meta[0],rank,'rank'),(meta[1],len(reference['binary32_words']),'numel'),
            (meta[2],int(reference['requires_grad']),'requires_grad'),
            (meta[3:3+rank],reference['shape'],'shape'),(meta[7:7+rank],reference['strides'],'strides'),
            (meta[13],reference['version'],'Version'),(meta[14],reference['storage_offset_bytes'],'offset bytes'),
            (meta[15],reference['storage_bytes'],'Storage size')):
            self.eq(actual,expected,label)
        node=self.raw_node(tid=tid)
        self.eq(node[40]==0,reference['is_leaf'],'native leaf role')
        self.eq(node[40]==0,reference['grad_fn_type'] is None,'grad function absent/present')
        assert reference['dtype']=='torch.float32' and reference['device']=='cpu'
        self.identity(self.tensor_ids,reference['tensor_identity'],node[320],'Tensor incarnation')
        self.identity(self.storage_ids,reference['storage_identity'],meta[11],'Storage identity')
        rec=self.c.one(10,tid);expected=reference['binary32_words']
        self.eq(rec['value'],len(expected),'logical value count')
        got=list(struct.unpack('<'+'I'*len(expected),rec['payload']))
        self.eq(got,expected,'exact binary32 words');self.c.comparisons+=len(expected)
        return {'token':tid,'incarnation':node[320],'metadata':meta,'binary32_words':got,'is_leaf':node[40]==0}

    def snapshot(self,reference,params,held):
        got={'parameters':{},'gradients':{},'held_Tensors':{}}
        self.eq(set(params),set(reference['parameters']),'parameter names')
        self.eq(set(held),set(reference['held_Tensors']),'held names')
        for name,p in params.items():
            got['parameters'][name]=self.tensor(p,reference['parameters'][name])
            old=self.raw_node(tid=p)
            public=False
            if old[384]:public=self.raw_node(node=old[384])[200]==1
            g=self.c.make(91,p)
            info=reference['gradients'][name]
            self.eq(g==0,info is None,'getter None')
            self.eq(self.c.make(59,p),int(info is not None),'presence agrees with binding')
            if info is not None:
                got['gradients'][name]=self.tensor(g,info)
                if not public:self.c.one(41,g)
            else:got['gradients'][name]=None
        for name,tid in held.items():got['held_Tensors'][name]=self.tensor(tid,reference['held_Tensors'][name])
        return got

    def event(self,row,action,params,held=None,status=0,runtime=0):
        held=held or {};before=self.snapshot(row['before'],params,held)
        self.eq(row['outcome'],'RAISED' if status or runtime else 'SUCCEEDED','actual exception category')
        observed=action()
        if runtime:self.eq(observed['value'],runtime,'structured runtime error')
        after=self.snapshot(row['after'],params,held)
        self.events.append({'action':row['action'],'before':before,'after':after,
            'actual_outcome':row['outcome'],'native_header_status':status,'native_runtime_error':runtime,
            'actual_warnings_diagnostic_only':row.get('warnings',[])})

    def blob(self,words):return self.b.Blob(struct.pack('<'+'Q'*len(words),*words))
    def opt(self,parameters):
        return self.c.make(71,0,self.blob([self.b.bits(.125),0,0,0,0,0]),self.blob([1,len(parameters),0,0,0,0,0,0,0,*parameters]))
    def temporary_zero(self,parameters,none):
        opt=self.opt(parameters);result=self.c.one(73,opt,none);self.c.one(78,opt);return result
    def back(self,target,seed=0,retain=0):
        rec=self.c.one(58,target,seed,retain);self.eq(rec['value'],0,'backward success');return rec
    def mutate_literal(self,t,values):
        before=list(self.b.words(self.c.one(54)));meta=list(self.b.words(self.c.one(13,t)))
        alias=self.c.make(55,t);aliasmeta=list(self.b.words(self.c.one(13,alias)))
        self.eq(aliasmeta[11],meta[11],'literal mutation same Storage')
        assert aliasmeta[12]!=meta[12]
        commands=[(12,alias,i,self.b.bits(value)) for i,value in enumerate(values)]
        if values:commands.append((12,t,0,self.b.bits(values[0])))
        commands.append((41,alias));self.c.run(commands)
        after=list(self.b.words(self.c.one(54)));self.eq(after[4:11],before[4:11],'temporary alias leaves no live objects/free-extent change')
        self.c.one(13,alias,status=1);self.c.audit()
        self.fixtures.append({'kind':'fixed literal native mutation fixture; not Tensor.add_ implementation',
            'literal_words':[self.b.bits(v) for v in values],'temporary_Tensor_Version_released':True,
            'Storage_and_Version_live_counts_and_free_bytes_restored':True})

    def case1(self,r):
        p=self.c.new([2],[1.,-2.],True);seed=self.c.new([2],[-0.,2.]);params={'p':p}
        self.snapshot(r['initial'],params,{'seed':seed})
        self.eq([self.c.make(91,p)==0 for _ in range(3)],r['None_reads'],'repeated None getter')
        self.event(r['direct_actions'][0],lambda:self.back(p,seed),params,{'seed':seed})
        g=self.c.make(91,p)
        self.eq(self.c.make(91,p)==g and self.c.make(91,p)==g,r['getter_relationships']['same_Tensor_on_repeated_get'],'repeated getter identity')
        self.eq(g==seed,r['getter_relationships']['same_Tensor_as_seed'],'seed Tensor relation')
        self.eq(self.b.words(self.c.one(13,g))[11]==self.b.words(self.c.one(13,seed))[11],r['getter_relationships']['same_Storage_as_seed'],'seed Storage relation')
        def second():
            s=self.c.new([2],[3.,4.]);self.back(p,s);self.c.one(41,s)
        self.event(r['direct_actions'][1],second,params,{'gradient':g,'seed':seed})
        q=self.c.new([2],[1.,-2.],True);middle=self.c.make(2,3,q,q);root=self.c.make(4,7,middle);self.c.one(41,middle)
        self.event(r['graph_actions'][0],lambda:self.back(root,0,1),{'q':q})
        gq=self.c.make(91,q)
        self.event(r['graph_actions'][1],lambda:self.back(root),{'q':q},{'gradient':gq})
        self.eq(self.c.make(91,q)==gq,r['graph_gradient_same_Tensor'],'graph grad identity')

    def case2(self,r):
        for i,row in enumerate(r['layouts']):
            self.tensor_ids={};self.storage_ids={}
            if i==0:
                base=self.c.new([2,3],[1.,-2.,3.,.5,2.,-1.]);p=self.c.make(6,base,0,1);seed=self.c.new([3,2],[1.,2.,3.,-1.,-2.,-3.])
            else:
                base=self.c.new([4],[99.,1.,99.,-2.]);p=self.c.make(69,base,self.blob([1,2,0,0,0,2,0,0,0,1]),80);seed=self.c.new([2],[2.,-1.])
            self.c.one(90,p,1)
            self.event(row['events'][0],lambda:self.back(p,seed),{'p':p},{'base':base,'seed':seed})
            g=self.c.make(91,p)
            self.event(row['events'][1],lambda:self.back(p,seed),{'p':p},{'base':base,'seed':seed,'gradient':g})
            self.eq(self.c.make(91,p)==g,row['getter_is_held_gradient'],'layout getter identity')
            self.eq(self.b.words(self.c.one(13,g))[11]==self.b.words(self.c.one(13,seed))[11],row['same_Storage_as_seed'],'layout seed Storage relation')

    def case3(self,r):
        p=self.c.new([2],[1.,-2.],True);old=self.c.new([2],[-0.,2.]);new=self.c.new([2],[3.,4.]);params={'p':p};held={'old':old,'new':new}
        wold=self.c.make(42,old);wnew=self.c.make(42,new)
        actions=[lambda:self.c.one(92,p,old),lambda:self.c.one(92,p,new),lambda:self.mutate_literal(old,[.5,2.5]),lambda:self.c.one(92,p,new),lambda:self.c.one(92,p,0)]
        for row,action in zip(r['events'],actions,strict=True):self.event(row,action,params,held)
        self.c.expiry(wold,r['before_external_drop']['old_expired']);self.c.expiry(wnew,r['before_external_drop']['new_expired'])
        self.c.run([(41,old),(41,new)])
        self.c.expiry(wold,r['after_external_drop']['old_expired']);self.c.expiry(wnew,r['after_external_drop']['new_expired'])
        self.tensor(p,r['parameter_after']);self.eq(self.c.make(91,p)==0,r['grad_is_None'],'cleared None')
        self.c.run([(44,wold),(44,wnew)])

    def case4(self,r):
        p=self.c.new([2,3],[1.,-2.,3.,.5,2.,-1.],True);base=self.c.new([3,2],[1.,2.,3.,4.,5.,6.]);g=self.c.make(6,base,0,1)
        params={'p':p};held={'gradient':g,'base':base}
        self.event(r['events'][0],lambda:self.c.one(92,p,g),params,held)
        def backward():
            seed=self.c.new([2,3],[1.]*6);self.back(p,seed);self.c.one(41,seed)
        self.event(r['events'][1],backward,params,held)
        self.event(r['events'][2],lambda:self.temporary_zero([p],0),params,held)
        self.eq(self.c.make(91,p)==g,r['getter_is_assigned_Tensor'],'assigned noncontiguous identity')

    def case5(self,r):
        p=self.c.new([2],[1.,-2.],True);g=self.c.make(19,p);opt=self.opt([p]);params={'p':p};held={'gradient':g}
        def backward():
            seed=self.c.new([2],[2.,-1.]);self.back(p,seed);self.c.one(41,seed)
        actions=[lambda:self.c.one(92,p,g),backward,lambda:self.c.one(72,opt),lambda:self.c.one(73,opt,0),lambda:self.c.one(73,opt,1)]
        for row,action in zip(r['events'],actions,strict=True):self.event(row,action,params,held)
        self.eq(g==p,r['same_Tensor'],'distinct aliased Tensor')
        self.eq(self.b.words(self.c.one(13,g))[11]==self.b.words(self.c.one(13,p))[11],r['same_Storage'],'same aliased Storage')

    def case6(self,r):
        p=self.c.new([2],[1.,-2.],True);g=self.c.new([2],[2.,-1.],True);opt=self.opt([p]);params={'p':p};held={'gradient':g}
        actions=[lambda:self.c.one(92,p,g),lambda:self.c.one(73,opt,0),lambda:self.c.one(73,opt,1),lambda:self.c.one(72,opt)]
        for row,action in zip(r['events'],actions,strict=True):self.event(row,action,params,held)

    def case7(self,r):
        for i,row in enumerate(r['variants']):
            self.tensor_ids={};self.storage_ids={}
            p=self.c.new([2],[1.,-2.],True);base=self.c.new([2],[2.,-1.],True)
            if i==0:
                two=self.c.new([],[2.]);g=self.c.make(2,3,base,two);self.c.one(41,two)
            else:g=self.c.make(5,base,1,2,0,0,0)
            opt=self.opt([p]);params={'p':p};held={'base':base,'gradient':g}
            self.event(row['events'][0],lambda:self.c.one(92,p,g),params,held)
            status=5 if i else 0
            self.event(row['events'][1],lambda:self.c.one(73,opt,0,status=status),params,held,status=status)
            self.event(row['events'][2],lambda:self.c.one(73,opt,1),params,held)

    def case8(self,r):
        p=self.c.new([2],[1.,-2.],True);g=self.c.new([2],[3.,4.]);self.c.one(92,p,g);params={'p':p};held={'valid_gradient':g}
        for i,row in enumerate(r['events']):
            if i in (0,5,6):
                self.open.append({'action':row['action'],'reason':'Python non-Tensor value or dtype/device outside this f32 CPU wire representation','actual_exception':row.get('exception_type')})
                continue
            if i==1:action=lambda:self.c.one(92,p,p,status=2);status=2
            elif i in (2,3,4):
                shape,values=(([1,2],[1.,2.]) if i==2 else ([1],[1.]) if i==3 else ([0],[]))
                invalid=self.c.new(shape,values)
                action=lambda:self.c.one(92,p,invalid,status=2);status=2
            elif i in (7,9):action=lambda:self.c.one(92,p,0);status=0
            else:action=lambda:self.c.one(92,p,g);status=0
            self.event(row,action,params,held,status=status)

    def case9(self,r):
        for i,row in enumerate(r['variants']):
            self.tensor_ids={};self.storage_ids={}
            shape=[] if i==0 else [0,3]
            p=self.c.new(shape,[2.] if i==0 else [],True);seed=self.c.new(shape,[-0.] if i==0 else []);opt=self.opt([p]);params={'p':p}
            self.event(row['events'][0],lambda:self.back(p,seed),params,{'seed':seed})
            g=self.c.make(91,p);weak=self.c.make(42,g);held={'gradient':g,'seed':seed}
            self.event(row['events'][1],lambda:self.c.one(73,opt,0),params,held)
            self.event(row['events'][2],lambda:self.c.one(73,opt,1),params,held)
            self.c.expiry(weak,not row['old_storage_alive_after_clear'])
            self.event(row['events'][3],lambda:self.back(p,seed),params,held)
            newer=self.c.make(91,p)
            self.eq(newer==g,row['new_equals_old']['Tensor'],'new gradient Tensor identity')
            self.eq(self.b.words(self.c.one(13,newer))[11]==self.b.words(self.c.one(13,g))[11],row['new_equals_old']['Storage'],'new gradient Storage identity')
            self.c.one(41,g);self.c.expiry(weak,row['old_storage_expired_after_external_drop']);self.c.one(44,weak)

    def case10(self,r):
        bad=self.c.new([2],[1.,-2.],True);good=self.c.new([2],[3.,4.],True);coefficient=self.c.new([],[2.])
        bad_branch=self.c.make(2,3,bad,coefficient);good_branch=self.c.make(2,3,good,good);total=self.c.make(2,1,bad_branch,good_branch);root=self.c.make(4,7,total)
        self.c.one(41,total);self.c.one(12,coefficient,0,self.b.bits(3.));params={'bad':bad,'good':good}
        self.event(r['events'][0],lambda:self.c.one(58,root,0,0),params,runtime=6)
        g=self.c.make(91,good);weak=self.c.make(42,g)
        self.event(r['events'][1],lambda:self.temporary_zero([bad,good],1),params,{'ready_gradient':g})
        self.c.expiry(weak,not r['external_ready_gradient_survived_clear'])
        self.c.one(41,g);self.c.expiry(weak,r['ready_gradient_Storage_expired_after_external_drop']);self.c.one(44,weak)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','controls','observer','observations','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--expected-sha256',required=True);ap.add_argument('--expected-source-sha256',required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=a.source.read_bytes();binary=a.binary.read_bytes();raw=a.observations.read_bytes()
    assert sha(source)==a.expected_source_sha256 and sha(binary)==a.expected_sha256 and sha(raw)==OBSERVATION_SHA
    ref=json.loads(raw);assert ref['status']=='OBSERVED' and ref['cases']==10 and ref['probe_errors']==0 and ref['probe_sha256']==PROBE_SHA
    checks=ref['oracle_provenance']['checks'];assert len(checks)==11 and all(v is True for v in checks.values())
    codec=load('grad_object_codec',a.controls,CONTROLS_SHA);observer=load('grad_object_observer',a.observer,OBSERVER_SHA)
    custody=out/'executed-sources';custody.mkdir()
    for name,data in [('replay.py',Path(__file__).read_bytes()),('native_release_controls.py',a.controls.read_bytes()),('client.py',a.observer.read_bytes()),('torch-gradient-objects.json',raw)]:
        (custody/name).write_bytes(data)
    report={'schema':'heptatorch.actual-gradient-object-native-replay.v1','status':'RUNNING',
        'native_sha256':sha(binary),'source_sha256':sha(source),'reference_sha256':sha(raw),'probe_sha256':PROBE_SHA,
        'runner_sha256':sha(Path(__file__).read_bytes()),'oracle_provenance':ref['oracle_provenance'],'cases':[],
        'comparison_policy':'Exact binary32 words, layout, None, Versions, stable opaque Tensor/Storage relations, leaf role and Storage Weak expiry; no numerical tolerance.',
        'explicit_open':['Python warning/message rendering and exact grad_fn class names','Python value/wrong dtype/device setter paths outside f32 CPU wire','Higher-order/create_graph and retain_grad on nonleaf parameters','Incoming-gradient steal optimization and its versions outside these fixed recipes','Nonleaf detach with existing downstream AD consumers and cyclic gradient-reference ownership'],
        'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    for i,row in enumerate(ref['rows'],1):
        session=observer.OwnerSession(a.binary.resolve(),directory=out/f'case-{i:02d}',address_limit=64<<20);c=codec.Check(session);replay=Replay(c,codec)
        item={'case':row['case'],'status':'RUNNING'}
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
    report.update(reference_groups=10,executed_groups=len(report['cases']),
        qualified_groups=sum(r['status']=='QUALIFIED_EXPOSED_SCOPE' for r in report['cases']),
        partial_groups=sum(r['status']=='PARTIAL' for r in report['cases']),failed_groups=sum(r['status']=='FAIL' for r in report['cases']),
        frames=sum(r['frames'] for r in report['cases']),exact_comparisons=sum(r['exact_comparisons'] for r in report['cases']),
        rollback_bytes=sum(r['rollback_bytes'] for r in report['cases']))
    report['status']='FAIL' if report['failed_groups'] else 'QUALIFIED_WITH_EXPLICIT_OPEN_BOUNDARIES'
    assert a.source.read_bytes()==source and a.binary.read_bytes()==binary and a.observations.read_bytes()==raw
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','oracle_provenance')},indent=2))
    return int(report['failed_groups']!=0)

if __name__=='__main__':raise SystemExit(main())
