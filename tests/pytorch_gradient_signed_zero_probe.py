"""Observe exact pinned Torch signed-zero gradient propagation and presence.

The explicit binary32 words are the reference values. Each operation runs in
real Torch only; no HIA target or alternative numerical runtime is invoked.
This is a separate precision boundary from the frozen 16 partial-failure cases.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import sys
import traceback

SUPPORT_SHA='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
PIN='31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CASE_NAMES=('leaf-negative-zero-first','leaf-positive-zero-first','leaf-negative-zero-twice',
            'leaf-positive-zero-then-negative-zero','clone-negative-zero-seed',
            'view-negative-zero-seed','transpose-negative-zero-seed','neg-positive-zero-seed',
            'mul-one-negative-zero-seed','shared-add-negative-zero-seed',
            'sum-negative-zero-seed','gather-negative-zero-with-unused-element','empty-leaf-defined-gradient')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def execute(torch):
    def info(t):
        assert t.dtype==torch.float32
        words=t.detach().contiguous().view(torch.int32).reshape(-1).tolist()
        return {'shape':list(t.shape),'strides':list(t.stride()),'requires_grad':bool(t.requires_grad),
                'version':int(t._version),'values':t.detach().reshape(-1).tolist(),
                'binary32_words':[int(x)&0xffffffff for x in words]}
    def leaf_grad(x): return None if x.grad is None else info(x.grad)
    def build(index):
        shape=[];values=[2.];sequence=[0x80000000]
        if index==1: sequence=[0]
        if index==2: sequence=[0x80000000,0x80000000]
        if index==3: sequence=[0,0x80000000]
        if index==5: shape=[1]
        if index==6: shape=[2,2];values=[2.,3.,4.,5.]
        if index in (10,11): shape=[2];values=[2.,3.]
        if index==12: shape=[0];values=[]
        x=torch.tensor(values,dtype=torch.float32).reshape(shape).requires_grad_(True)
        if index in (0,1,2,3,12): y=x;operation='leaf'
        elif index==4: y=x.clone();operation='clone'
        elif index==5: y=x.view(());operation='view'
        elif index==6: y=x.transpose(0,1);operation='transpose'
        elif index==7: y=-x;operation='neg';sequence=[0]
        elif index==8: y=x*torch.tensor(1.,dtype=torch.float32);operation='mul-one'
        elif index==9: y=x+x;operation='shared-add'
        elif index==10: y=x.sum();operation='sum'
        elif index==11: y=torch.gather(x,0,torch.tensor([0],dtype=torch.int64));operation='gather'
        else: raise AssertionError(index)
        return x,y,sequence,operation
    rows=[]
    with torch.autograd.set_multithreading_enabled(False):
        for index,name in enumerate(CASE_NAMES):
            try:
                x,y,sequence,operation=build(index)
                initial={'input':info(x),'output':info(y),'gradient':leaf_grad(x),'output_has_grad_fn':y.grad_fn is not None}
                calls=[]
                for word in sequence:
                    signed=word if word<0x80000000 else word-(1<<32)
                    seed=torch.full(tuple(y.shape),signed,dtype=torch.int32).view(torch.float32)
                    call={'seed':info(seed),'retain_graph':False,'gradient_before':leaf_grad(x)}
                    try: y.backward(seed,retain_graph=False);call['outcome']='SUCCEEDED'
                    except Exception as error: call.update(outcome='RAISED',exception_type=type(error).__name__,message=str(error))
                    call['gradient_after']=leaf_grad(x);calls.append(call)
                rows.append({'case':name,'status':'OBSERVED','observation':{'operation':operation,'initial':initial,'calls':calls}})
            except Exception as error:
                rows.append({'case':name,'status':'PROBE_ERROR','exception_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','lock','repo','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    for name in ('reference','lock','repo','output'): setattr(a,name,getattr(a,name).resolve())
    assert digest(a.reference)==SUPPORT_SHA
    base={'schema':'heptatorch.pinned-torch-gradient-signed-zero-observations.v1','probe_sha256':digest(__file__),
          'support_sha256':SUPPORT_SHA,'lock_sha256':digest(a.lock),'upstream_commit':PIN,'expected_cases':len(CASE_NAMES),'case_names':list(CASE_NAMES),
          'target_execution':False,'native_values_compared':0,'full_rewrite_complete':False,'pytorch_api_denominator':2590,
          'scope':'Exact CPU binary32 zero-sign propagation through initial/accumulated leaf gradients and selected built-in derivatives.',
          'reference_encoding':'Unsigned 32-bit words are authoritative for +0 versus -0; JSON float equality must not collapse them.',
          'native_compatibility_asserted':False}
    if a.prepare_only: write(a.output,dict(base,status='PREPARED_NOT_RUN',cases=0));return 0
    a.output.parent.mkdir(parents=True,exist_ok=True);provenance=a.output.with_name(a.output.stem+'-provenance.json')
    previous_argv,previous_directory=sys.argv,Path.cwd()
    try:
        os.chdir(a.repo);sys.argv=[str(a.reference),'/dev/null',str(provenance),str(a.lock)]
        try: support=runpy.run_path(str(a.reference),run_name='pinned_gradient_signed_zero_support')
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
