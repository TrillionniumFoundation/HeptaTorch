#!/usr/bin/env python3
"""Independently map frozen actual Adam observations21–24 to native objects.

Only transport/inspection helpers and the unchanged old replay are imported.
No HIA authoring, target numerical implementation, NumPy or Torch is imported.
The original first20 recipes are qualified by running their frozen CLI unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import struct
import traceback

SHA=lambda raw:hashlib.sha256(raw).hexdigest()
REFERENCE='7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160'
PROBE='f69340170dc2e55886bca73a8d37265ba2c95a8c793c7633c7e8a22234cfa386'
LEGACY='ea748d4100b539e0144f032ca301b4886d0f9520c36676c980279e0bbc9bc034'
CONTROLS='43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
CLIENT='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','controls','transport-observer','legacy-replayer','observations','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True);p.add_argument('--expected-source-sha256',required=True)
    a=p.parse_args()
    paths={name:getattr(a,name.replace('-','_')).resolve() for name in
           ('binary','source','controls','transport-observer','legacy-replayer','observations')}
    inputs={path:path.read_bytes() for path in paths.values()};inputs[Path(__file__).resolve()]=Path(__file__).read_bytes()
    expected={'binary':a.expected_sha256,'source':a.expected_source_sha256,'controls':CONTROLS,
              'transport-observer':CLIENT,'legacy-replayer':LEGACY,'observations':REFERENCE}
    for name,path in paths.items():assert SHA(inputs[path])==expected[name],name
    ref=json.loads(inputs[paths['observations']])
    assert ref['status']=='OBSERVED' and ref['cases']==24 and ref['probe_errors']==0 and ref['probe_sha256']==PROBE
    assert len(ref['oracle_provenance']['checks'])==11 and all(ref['oracle_provenance']['checks'].values())
    policy=ref['predeclared_future_native_policy']
    assert policy['Adam_parameter_and_moment_values']['rtol']==4e-5
    assert policy['Adam_parameter_and_moment_values']['atol']==5e-6
    codec=load('adam_independent_codec',paths['controls'])
    transport=load('adam_independent_transport',paths['transport-observer'])
    old=load('adam_unchanged_legacy_replay',paths['legacy-replayer'])
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    archive=out/'executed-sources';archive.mkdir()
    for path,raw in inputs.items():
        if path not in (paths['binary'],paths['source']):(archive/path.name).write_bytes(raw)

    class AdamReplay(old.Replay):
        def __init__(self,c):
            super().__init__(c,codec,'optimizer')
            self.algorithms={};self.versions={};self.numeric_comparisons=[]

        def create(self,names,algorithm):
            cfg=[codec.bits(.125),codec.bits(.5),codec.bits(.75),codec.bits(.01),codec.bits(.25),0,0,0]
            body=[1,len(names),0,*([0]*8),*(self.params[n] for n in names)]
            record=self.c.one(71,algorithm,self.words(cfg),self.words(body));opt=record['value']
            self.eq((opt>>24)&255,8,'Optimizer typed kind')
            self.eq(list(codec.words(record)),[1,len(names),len(set(names)),len(names)-len(set(names))],'Adam creation inventory')
            self.members[opt]=[list(names)];self.algorithms[opt]=algorithm
            return opt

        def tensor(self,tid,info,*,gradient=False,approximate=False):
            if gradient:
                return super().tensor(tid,info,gradient=True)
            meta=list(codec.words(self.c.one(13,tid)))
            self.eq(len(meta),16,'metadata width');rank=len(info['shape'])
            for actual,wanted,label in [(meta[0],rank,'rank'),(meta[1],len(info['binary32_words']),'numel'),
                (meta[2],int(info['requires_grad']),'requires_grad'),(meta[3:3+rank],info['shape'],'shape'),
                (meta[7:7+rank],info['strides'],'strides'),(meta[13],info['version'],'Version counter'),
                (meta[14],info['storage_offset_bytes'],'storage offset bytes'),(meta[15],info['storage_bytes'],'Storage bytes')]:
                self.eq(actual,wanted,label)
            key=('incarnation',self.handle_incarnations[tid]) if tid in self.handle_incarnations else ('public',tid)
            self.identity(self.ref_tensor,info['tensor_identity'],key,'Tensor')
            self.identity(self.ref_storage,info['storage_identity'],meta[11],'Storage')
            if info['tensor_identity'] in self.versions:
                self.eq(meta[12],self.versions[info['tensor_identity']],'same actual Tensor keeps its native Version identity')
            else:self.versions[info['tensor_identity']]=meta[12]
            record=self.c.one(10,tid);wanted=info['binary32_words']
            self.eq(record['value'],len(wanted),'value count');self.eq(len(record['payload']),4*len(wanted),'value bytes')
            actual=list(struct.unpack('<'+'I'*len(wanted),record['payload']))
            if not approximate:
                self.eq(actual,wanted,'exact step or non-optimizer Tensor words')
                self.c.comparisons+=len(wanted)
            else:
                for index,(got_bits,want_bits) in enumerate(zip(actual,wanted)):
                    got=struct.unpack('<f',struct.pack('<I',got_bits))[0]
                    want=struct.unpack('<f',struct.pack('<I',want_bits))[0]
                    tolerance=5e-6+4e-5*abs(want)
                    ok=math.isfinite(got) and math.isfinite(want) and abs(got-want)<=tolerance
                    comparison={'Tensor':tid,'index':index,'actual_word':got_bits,'expected_word':want_bits,
                                'absolute_error':abs(got-want),'allowed_error':tolerance,'exact':got_bits==want_bits,'pass':ok}
                    self.numeric_comparisons.append(comparison)
                    assert ok,('fixed Adam value policy',comparison)
            return {'metadata':meta,'binary32_words':actual,'comparison':'predeclared Adam tolerance' if approximate else 'exact words'}

        def groups(self,opt,expected,params):
            self.eq(len(expected),len(self.members[opt]),'group count')
            lookup={value['tensor_identity']:name for name,value in params.items()};locations={};result=[]
            for group,wanted in enumerate(expected):
                config=wanted['config']
                for flag in ('amsgrad','maximize','foreach','capturable','differentiable','fused'):
                    self.eq(config[flag],False,'fixed CPU Adam mode '+flag)
                self.eq(config['decoupled_weight_decay'],self.algorithms[opt]==2,'algorithm is actual coupled or decoupled decay')
                config_words=[codec.bits(config['lr']),codec.bits(config['betas'][0]),codec.bits(config['betas'][1]),
                              codec.bits(config['eps']),codec.bits(config['weight_decay']),int(config['amsgrad']),int(config['maximize']),0]
                rec=self.c.one(74,opt,group);values=list(codec.words(rec));count=len(wanted['parameters'])
                self.eq(rec['value'],count,'group occurrence count');self.eq(len(values),10+2*count,'Adam group width')
                self.eq(values[:10],[group,count,*config_words],'exact effective binary32 Adam configuration')
                self.eq([lookup[x] for x in wanted['parameters']],self.members[opt][group],'ordered parameter membership')
                for occurrence,refid in enumerate(wanted['parameters']):
                    incarnation,public=values[10+2*occurrence:12+2*occurrence];name=lookup[refid]
                    self.eq(public,self.params[name],'same retained parameter public Tensor')
                    self.bind_incarnation(public,incarnation)
                    self.eq(self.c.make(75,opt,group,occurrence),public,'parameter getter preserves exact public identity')
                    if name in locations:self.eq(incarnation,locations[name][2],'duplicate occurrences share one Tensor identity')
                    else:locations[name]=(group,occurrence,incarnation)
                result.append(values)
            return result,locations

        def state(self,opt,location,expected):
            group,occurrence,incarnation=location;record=self.c.one(76,opt,group,occurrence)
            values=list(codec.words(record));self.eq(len(values),8,'three actual state Tensor inventory width')
            self.eq(values[0],incarnation,'state keyed by Tensor identity')
            present=expected['entry_exists'];self.eq(record['value'],int(present),'lazy state entry existence')
            self.eq(values[1],14 if present else 0,'exact realized state field mask')
            if not present:
                self.eq(expected['fields'],{},'actual reference absent fields');self.eq(values[2:],[0]*6,'no hidden state Tensor')
                for selector in (1,2,3):self.eq(self.c.make(77,opt,group,occurrence,selector),0,'absent getter does not insert state')
                self.eq(list(codec.words(self.c.one(76,opt,group,occurrence))),values,'state remains absent after all getters')
                return {'inventory':values,'fields':{},'present':False}
            self.eq(set(expected['fields']),{'step','exp_avg','exp_avg_sq'},'actual state field set')
            self.eq(len(set(values[2::2])),3,'three distinct Tensor incarnations')
            assert incarnation not in values[2::2]
            fields={};versions=[];storage=[]
            for j,name in enumerate(('step','exp_avg','exp_avg_sq')):
                state_inc,existing=values[2+2*j:4+2*j]
                tid=self.c.make(77,opt,group,occurrence,j+1)
                if existing:self.eq(tid,existing,'getter preserves existing public state token')
                self.eq(self.c.make(77,opt,group,occurrence,j+1),tid,'repeated state getter returns same Tensor')
                self.bind_incarnation(tid,state_inc)
                fields[name]=self.tensor(tid,expected['fields'][name]['tensor'],approximate=name!='step')
                versions.append(fields[name]['metadata'][12]);storage.append(fields[name]['metadata'][11])
                if not existing:self.c.one(41,tid)
            self.eq(len(set(versions)),3,'native distinct state Version identities')
            self.eq(len(set(storage)),3,'native distinct state Storage identities')
            return {'inventory':values,'fields':fields,'present':True}

        def snapshot(self,expected,opt=None,params=None):
            names=self.params if params is None else params
            result={'parameters':{},'gradients':{}};locations={}
            if opt is not None:
                result['groups'],locations=self.groups(opt,expected['groups'],expected['parameters']);result['states']={}
            for name,tid in names.items():
                result['parameters'][name]=self.tensor(tid,expected['parameters'][name],approximate=opt is not None)
                gradient=expected['gradients'][name];presence=self.c.one(59,tid)
                self.eq(presence['payload'],b'','presence has no fabricated gradient Tensor')
                self.eq(presence['value'],int(gradient is not None),'None versus defined gradient')
                if gradient is None:
                    record=self.c.one(11,tid);self.eq(record['value'],len(expected['parameters'][name]['binary32_words']),'None compatibility count')
                    self.eq(record['payload'],bytes(4*record['value']),'None compatibility zeros');result['gradients'][name]=None
                else:result['gradients'][name]=self.tensor(tid,gradient,gradient=True)
                if opt is not None:result['states'][name]=self.state(opt,locations[name],expected['optimizer_state'][name])
            if opt is not None:self.eq(sum(v['present'] for v in result['states'].values()),expected['state_entries'],'unique state entries')
            return result

        def hold_state(self,opt,selector):
            values=list(codec.words(self.c.one(76,opt,0,0)));self.eq(values[1],14,'realized state before external hold')
            tid=self.c.make(77,opt,0,0,selector);self.bind_incarnation(tid,values[2+2*(selector-1)])
            return tid

        def recipe(self,number,obs):
            p=self.p('p',[1.,-2.]);opt=self.create(['p','p'] if number==23 else ['p'],2 if number==22 else 1)
            if number in (21,22):
                functions=[lambda:self.step(opt),lambda:self.backward(p,[0.,0.]),lambda:self.step(opt),
                    lambda:self.clear(opt),lambda:self.step(opt),lambda:self.backward(p,[2.,-1.]),
                    lambda:self.step(opt),lambda:self.clear(opt,False),lambda:self.step(opt)]
                self.actions(obs['actions'],functions,opt)
            elif number==23:
                self.eq(obs['construction']['outcome'],'SUCCEEDED','duplicate construction is accepted')
                self.actions(obs['actions'],[lambda:self.backward(p,[2.,-1.]),lambda:self.step(opt),lambda:self.step(opt)],opt)
                self.extra['warnings']='The fixed Torch duplicate warning is retained in the reference; Python warning emission is not supplied by the native wire.'
            else:
                self.backward(p,[2.,-1.]);self.step(opt)
                step=self.hold_state(opt,1);avg=self.hold_state(opt,2);square=self.hold_state(opt,3)
                weak_step,weak_avg,weak_square=[self.c.make(42,t) for t in (step,avg,square)]
                self.c.one(41,square)
                q=self.p('q',[3.,4.]);product=self.c.make(2,3,q,avg);root=self.c.make(4,7,product)
                self.c.one(41,product)
                before={'step':self.tensor(step,obs['before']['step']),
                        'exp_avg':self.tensor(avg,obs['before']['exp_avg'],approximate=True),
                        'optimizer':self.snapshot(obs['before']['optimizer'],opt,{'p':p})}
                self.c.run([(78,opt),(41,p)])
                for name,weak in [('step',weak_step),('exp_avg',weak_avg),('exp_avg_sq',weak_square)]:
                    self.storage_expiry(weak,obs['after_optimizer_drop'][name+'_expired'])
                self.c.one(13,p,status=1)
                for tid in (step,avg):
                    before_meta=list(codec.words(self.c.one(13,tid)))
                    mutation=self.c.one(95,tid,codec.bits(.5))
                    self.eq((mutation['value'],mutation['payload']),(tid,b''),'native no-grad add returns same actual Tensor')
                    after_meta=list(codec.words(self.c.one(13,tid)))
                    self.eq(after_meta[11:13],before_meta[11:13],'real addition preserves Storage and Version identity')
                    self.eq(after_meta[13],before_meta[13]+1,'one whole Tensor addition increments Version once')
                mutated={'step':self.tensor(step,obs['after_external_state_mutation']['step']),
                         'exp_avg':self.tensor(avg,obs['after_external_state_mutation']['exp_avg'],approximate=True)}
                self.action(obs['backward'],lambda:self.backward(root,error=6),params={'q':q})
                self.c.run([(41,root),(41,step),(41,avg)])
                self.storage_expiry(weak_step,obs['after_all_external_and_graph_drop']['step_expired'])
                self.storage_expiry(weak_avg,obs['after_all_external_and_graph_drop']['exp_avg_expired'])
                self.c.run([(44,weak_step),(44,weak_avg),(44,weak_square)])
                self.extra['state_lifetime']={'before':before,'mutated':mutated,'native_arithmetic_command':95,
                    'target_values_were_not_written_from_reference':True,'actual_weak_expiration_compared':True}
            self.c.audit()
            return {'actions':self.steps,'additional_observations':self.extra,'numeric_comparisons':self.numeric_comparisons,
                    'identities':{'Tensor':self.ref_tensor,'Storage':self.ref_storage,'nativeVersion':self.versions}}

    rows=[]
    for number in range(21,25):
        actual=ref['rows'][number-1];directory=out/('case-'+str(number));session=transport.OwnerSession(paths['binary'],directory=directory,address_limit=64<<20)
        check=codec.Check(session);replay=AdamReplay(check)
        row={'index':number,'case':actual['case'],'status':'FAIL','native_executed':True}
        try:
            row['details']=replay.recipe(number,actual['observation']);session.finish();row['status']='QUALIFIED_EXPOSED_SCOPE'
        except Exception as error:
            row.update(error=repr(error),traceback=traceback.format_exc(),completed_actions=replay.steps,
                       numeric_comparisons=replay.numeric_comparisons,additional_observations=replay.extra)
            if session.poll() is None:os.kill(session.pid,signal.SIGKILL)
            try:session.finish(expected=-signal.SIGKILL)
            except Exception as cleanup:row['cleanup_error']=repr(cleanup)
        row.update(frames=check.frames,exact_comparisons=check.comparisons,rollback_bytes=check.rollback_bytes)
        (directory/'report.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    for path,raw in inputs.items():assert path.read_bytes()==raw,path
    failures=sum(r['status']=='FAIL' for r in rows)
    report={'schema':'heptatorch.independent-native-adam-object-replay.v1','status':'FAIL' if failures else 'QUALIFIED_EXPOSED_SCOPE',
            'native_source_sha256':a.expected_source_sha256,'native_executable_sha256':a.expected_sha256,
            'actual_reference_sha256':REFERENCE,'original_actual_case_count':24,'executed_original_indices':[21,22,23,24],
            'qualified_exposed_scope':4-failures,'failures':failures,'frames':sum(len(r['frames']) for r in rows),
            'exact_comparisons':sum(r['exact_comparisons'] for r in rows),'rollback_bytes':sum(r['rollback_bytes'] for r in rows),
            'first20_test_source_modified':False,'numeric_policy':policy,'oracle_provenance':ref['oracle_provenance'],
            'same_fixed_Torch_executed_in_this_replay':False,'target_numerical_host_runtime':False,
            'public_grad_object_scope_open':True,'native_general_real_step_exponent_open':True,
            'full_optimizer_complete':False,'full_pytorch_rewrite_complete':False,'rows':rows,
            'input_custody_after':True,'inputs':{str(path):SHA(raw) for path,raw in inputs.items()}}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','qualified_exposed_scope','failures','frames','exact_comparisons','rollback_bytes')},indent=2))
    return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
