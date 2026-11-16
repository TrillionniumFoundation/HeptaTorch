"""Observe actual pinned Torch grad presence and optimizer skip semantics.

The reference uses real Torch backward and optimizers only. Its eight recipes
cover initially absent versus produced zero gradients, zero_grad(False), empty
leaves, and optimizer updates after a partially failed backward. No HIA target
is executed and no Torch outcome is invented from native values.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import sys
import traceback
import warnings

SUPPORT_SHA='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
PIN='31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
KINDS=('SGD-momentum','Adam','AdamW')
CASE_NAMES=tuple('absent-then-produced-zero-'+k for k in KINDS)+tuple('partial-backward-then-step-'+k for k in KINDS)+(
    'empty-leaf-produced-gradient-is-present','nonleaf-and-produced-zero-gradient-presence')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def execute(torch):
    def tensor_info(t):
        return {'shape':list(t.shape),'strides':list(t.stride()),'dtype':str(t.dtype),
                'requires_grad':bool(t.requires_grad),'version':int(t._version),
                'values':t.detach().reshape(-1).tolist()}
    def plain(value):
        if isinstance(value,torch.Tensor): return {'tensor':tensor_info(value)}
        if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [plain(v) for v in value]
        if value is None or isinstance(value,(bool,int,float,str)): return value
        return {'type':type(value).__name__,'text':str(value)}
    def gradients(params):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            return {name:None if p.grad is None else tensor_info(p.grad) for name,p in params.items()}
    def snapshot(params,opt=None):
        row={'parameters':{name:tensor_info(p) for name,p in params.items()},'gradients':gradients(params)}
        if opt is not None:
            row['optimizer_state']={name:{'entry_exists':p in opt.state,'fields':plain(opt.state.get(p,{}))} for name,p in params.items()}
        return row
    def optimizer(kind,params):
        if kind=='SGD-momentum':
            config={'lr':.125,'momentum':.5,'dampening':0.,'weight_decay':.25,'nesterov':False,'foreach':False,'fused':False}
            return torch.optim.SGD(list(params.values()),**config),config
        config={'lr':.125,'betas':(.5,.75),'eps':.01,'weight_decay':.25,'amsgrad':False,'foreach':False,'fused':False}
        return (torch.optim.Adam if kind=='Adam' else torch.optim.AdamW)(list(params.values()),**config),config
    def action(label,fn,params,opt=None):
        row={'action':label,'before':snapshot(params,opt)}
        try: fn();row['outcome']='SUCCEEDED'
        except Exception as error: row.update(outcome='RAISED',exception_type=type(error).__name__,message=str(error))
        row['after']=snapshot(params,opt);return row
    def absent_zero(kind):
        p=torch.tensor([1.,-2.,.5],requires_grad=True);params={'p':p};opt,config=optimizer(kind,params);rows=[]
        rows.append(action('step with initially absent gradient',opt.step,params,opt))
        rows.append(action('zero_grad(set_to_none=False) while absent',lambda:opt.zero_grad(set_to_none=False),params,opt))
        rows.append(action('step while gradient remains absent',opt.step,params,opt))
        loss=(p*0.).sum();rows.append(action('backward produces a defined zero gradient',loss.backward,params,opt))
        rows.append(action('step with present zero gradient and weight decay',opt.step,params,opt))
        rows.append(action('zero_grad(set_to_none=False) while present',lambda:opt.zero_grad(set_to_none=False),params,opt))
        rows.append(action('second step with present zero gradient and weight decay',opt.step,params,opt))
        return {'optimizer':kind,'config':config,'actions':rows}
    def partial_step(kind):
        bad=torch.tensor([1.,2.,3.],requires_grad=True);good=torch.tensor([2.,-1.,.5],requires_grad=True)
        params={'bad':bad,'good':good};opt,config=optimizer(kind,params)
        a=bad*bad;b=good*good;loss=(a+b).sum()
        with torch.no_grad(): bad.add_(1.)
        rows=[action('backward fails after newer independent branch',loss.backward,params,opt),
              action('optimizer step skips only absent-gradient parameter',opt.step,params,opt),
              action('zero_grad(False) preserves absent versus present distinction',lambda:opt.zero_grad(set_to_none=False),params,opt),
              action('optimizer step with one absent and one present-zero gradient',opt.step,params,opt)]
        return {'optimizer':kind,'config':config,'graph_creation_order':['bad*bad','good*good','add','sum'],
                'mutation':{'parameter':'bad','operation':'add_','scalar':1.},'actions':rows}
    def empty():
        p=torch.empty(0,dtype=torch.float32,requires_grad=True);params={'p':p};opt,config=optimizer('AdamW',params)
        loss=p.sum();rows=[action('zero_grad(False) while empty gradient absent',lambda:opt.zero_grad(set_to_none=False),params,opt),
                          action('empty backward produces defined empty gradient',loss.backward,params,opt),
                          action('AdamW step processes present empty gradient',opt.step,params,opt),
                          action('zero_grad(False) leaves defined empty gradient present',lambda:opt.zero_grad(set_to_none=False),params,opt)]
        return {'optimizer':'AdamW','config':config,'actions':rows}
    def nonleaf():
        x=torch.tensor([1.,-2.,3.],requires_grad=True);constant=torch.tensor(0.);y=x*constant;loss=y.sum()
        params={'x':x,'constant':constant,'nonleaf':y}
        return {'actions':[action('backward produces zero leaf gradient, nonleaf remains unretained',loss.backward,params)]}
    rows=[]
    with torch.autograd.set_multithreading_enabled(False):
        for name,fn in [(CASE_NAMES[i],lambda k=k:absent_zero(k)) for i,k in enumerate(KINDS)]+[(CASE_NAMES[3+i],lambda k=k:partial_step(k)) for i,k in enumerate(KINDS)]+[(CASE_NAMES[6],empty),(CASE_NAMES[7],nonleaf)]:
            try: rows.append({'case':name,'status':'OBSERVED','observation':fn()})
            except Exception as error: rows.append({'case':name,'status':'PROBE_ERROR','exception_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','lock','repo','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    for name in ('reference','lock','repo','output'): setattr(a,name,getattr(a,name).resolve())
    assert digest(a.reference)==SUPPORT_SHA,'Changed reference support'
    base={'schema':'heptatorch.pinned-torch-gradient-presence-observations.v1','probe_sha256':digest(__file__),
          'support_sha256':SUPPORT_SHA,'lock_sha256':digest(a.lock),'upstream_commit':PIN,'expected_cases':8,'case_names':list(CASE_NAMES),
          'target_execution':False,'native_values_compared':0,'full_rewrite_complete':False,'pytorch_api_denominator':2590,
          'scope':'Real CPU Torch None versus defined zero/empty gradients, optimizer skip and zero_grad(False). No target compatibility is asserted.',
          'engine_mode':'Autograd multithreading disabled; first-order backward; foreach=False and fused=False optimizers.',
          'not_covered':['zero_grad(set_to_none=True) protocol implementation','assignment to .grad','retain_grad on nonleaf','optimizer-state lifetime equivalence with explicit native state tensors']}
    if a.prepare_only: write(a.output,dict(base,status='PREPARED_NOT_RUN',cases=0));return 0
    a.output.parent.mkdir(parents=True,exist_ok=True);provenance=a.output.with_name(a.output.stem+'-provenance.json')
    previous_argv,previous_directory=sys.argv,Path.cwd()
    try:
        os.chdir(a.repo);sys.argv=[str(a.reference),'/dev/null',str(provenance),str(a.lock)]
        try: support=runpy.run_path(str(a.reference),run_name='pinned_gradient_presence_support')
        except SystemExit as error:
            observed=json.loads(provenance.read_text()) if provenance.exists() else {'exit':error.code}
            write(a.output,dict(base,status='NOT_RUN',cases=0,reason='Pinned support did not admit this environment',support_observation=observed));return 2
    finally: sys.argv=previous_argv;os.chdir(previous_directory)
    assert support['PROVENANCE']['status']=='PASS' and len(support['PROVENANCE']['checks'])==11 and all(support['PROVENANCE']['checks'].values())
    rows=execute(support['torch']);assert tuple(r['case'] for r in rows)==CASE_NAMES;errors=sum(r['status']!='OBSERVED' for r in rows)
    report=dict(base,status='PROBE_ERROR' if errors else 'OBSERVED',cases=len(rows),probe_errors=errors,rows=rows,
                oracle_provenance=support['PROVENANCE'],probe_unchanged=digest(__file__)==base['probe_sha256'],reference_unchanged=digest(a.reference)==SUPPORT_SHA)
    write(a.output,report);print(json.dumps({'status':report['status'],'cases':len(rows),'probe_errors':errors}));return 1 if errors else 0


if __name__=='__main__': raise SystemExit(main())
