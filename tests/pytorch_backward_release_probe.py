"""Observe exact pinned Torch backward/Saved-release boundaries, without a target.

This is an independent reference probe. All values, exceptions and weak-storage
states come from the verified Torch process. It neither executes native HIA nor
constructs numerical expectations from native output.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import runpy
import sys
import traceback


SUPPORT_SHA='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
PIN='31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
REPEAT_CASES=(
    'add-tensor','clone','view','transpose','cat','stack',
    'mul-shared-input','mul-left-only','mul-empty','relu',
    'index-select','gather','max-pool','avg-pool',
    'scatter-source-grad','scatter-self-only','index-add-source-grad','index-add-self-only',
    'conv2d','matmul-fold-copy',
)
SPECIAL_CASES=(
    'retain-true-true-false-true-sequence',
    'implicit-default-releases-saved',
    'no-saved-input-mutation-and-repeat',
    'saved-mutation-rejection-before-release',
    'failed-upstream-version-check-after-downstream-release',
    'saved-freed-marker-with-public-data-still-alive',
    'leaf-storage-held-by-AccumulateGrad-after-default-backward',
    'nongrad-coefficient-released-by-default-backward',
    'saved-output-weak-retain-then-default',
    'external-unpacked-alias-keeps-storage-but-not-saved-validity',
    'shared-subgraph-second-root-sees-freed-saved',
    'unvisited-independent-node-keeps-its-saved',
    'seed-shape-error-before-engine-preserves-saved',
    'backward-gradient-shape-error-preserves-failing-node-saved',
    'empty-index-select-still-has-saved-tensor-lifetime',
    'leaf-backward-twice-without-grad-function',
)
CASE_NAMES=tuple('repeat-default-'+name for name in REPEAT_CASES)+SPECIAL_CASES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def execute(torch,F):
    rows=[]

    def tensor_info(t):
        return {'shape':list(t.shape),'strides':list(t.stride()),'requires_grad':bool(t.requires_grad),
                'version':int(t._version),'values':t.detach().reshape(-1).tolist()}

    def plain(value):
        if isinstance(value,torch.Tensor): return {'tensor':tensor_info(value)}
        if isinstance(value,(list,tuple)): return [plain(v) for v in value]
        if value is None or isinstance(value,(bool,int,float,str)): return value
        return {'type':type(value).__name__,'text':str(value)}

    def saved(node):
        if node is None: return {}
        result={}
        for name in sorted(n for n in dir(node) if n.startswith('_saved_')):
            try: result[name]={'status':'AVAILABLE','value':plain(getattr(node,name))}
            except Exception as error:
                result[name]={'status':'RAISED','exception_type':type(error).__name__,'message':str(error)}
        return result

    def gradients(leaves):
        return [None if t.grad is None else tensor_info(t.grad) for t in leaves]

    def backward(output,leaves,seed=None,retain='OMITTED'):
        row={'retain_graph_argument':retain,'seed':None if seed is None else tensor_info(seed)}
        try:
            if retain=='OMITTED': output.backward(seed)
            else: output.backward(seed,retain_graph=retain)
            row['outcome']='SUCCEEDED'
        except Exception as error:
            row.update(outcome='RAISED',exception_type=type(error).__name__,message=str(error))
        row['gradients_after_call']=gradients(leaves)
        return row

    class WeakStorage:
        def __init__(self,t): self.token=t.untyped_storage()._weak_ref()
        def expired(self):
            gc.collect()
            return bool(torch.UntypedStorage._expired(self.token))
        def close(self): torch.UntypedStorage._free_weak_ref(self.token)

    def observe(name,fn):
        try: rows.append({'case':name,'status':'OBSERVED','observation':fn()})
        except Exception as error:
            rows.append({'case':name,'status':'PROBE_ERROR','exception_type':type(error).__name__,
                         'message':str(error),'traceback':traceback.format_exc()})

    def basic_x(): return torch.tensor([[1.,-2.,3.],[4.,-.5,2.]],requires_grad=True)

    def builder(kind):
        x=basic_x(); leaves=[x]
        if kind=='add-tensor':
            b=torch.tensor([.5,-1.,2.],requires_grad=True);leaves.append(b);y=x+b
        elif kind=='clone': y=x.clone()
        elif kind=='view': y=x.view(3,2)
        elif kind=='transpose': y=x.transpose(0,1)
        elif kind=='cat': y=torch.cat([x,x],1)
        elif kind=='stack': y=torch.stack([x,x],0)
        elif kind=='mul-shared-input': y=x*x
        elif kind=='mul-left-only':
            b=torch.tensor([.5,-1.,2.]);leaves.append(b);y=x*b
        elif kind=='mul-empty':
            x=torch.empty((0,3),requires_grad=True);leaves=[x];y=x*x
        elif kind=='relu': y=torch.relu(x)
        elif kind=='index-select': y=torch.index_select(x,1,torch.tensor([2,0,2],dtype=torch.int64))
        elif kind=='gather': y=torch.gather(x,1,torch.tensor([[2,0],[1,1]],dtype=torch.int64))
        elif kind in ('max-pool','avg-pool'):
            x=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4).requires_grad_(True);leaves=[x]
            y=F.max_pool2d(x,2) if kind=='max-pool' else F.avg_pool2d(x,2)
        elif kind.startswith('scatter-'):
            b=torch.tensor([[.5,-1.,2.],[3.,-.25,.75]],requires_grad=kind=='scatter-source-grad')
            leaves.append(b);idx=torch.tensor([[2,0,2],[1,1,0]],dtype=torch.int64)
            y=torch.scatter_add(x,1,idx,b)
        elif kind.startswith('index-add-'):
            b=torch.tensor([[.5,-1.],[3.,-.25]],requires_grad=kind=='index-add-source-grad')
            leaves.append(b);y=torch.index_add(x,1,torch.tensor([2,0],dtype=torch.int64),b,alpha=.5)
        elif kind=='conv2d':
            x=torch.arange(9,dtype=torch.float32).reshape(1,1,3,3).requires_grad_(True)
            b=torch.tensor([[[[.5,1.],[-1.,.25]]]],requires_grad=True);leaves=[x,b]
            y=F.conv2d(x,b)
        elif kind=='matmul-fold-copy':
            x=torch.arange(24,dtype=torch.float32).reshape(2,3,4).transpose(0,1)
            b=torch.tensor([[.5,1.],[-1.,.25],[.75,-.5],[2.,.125]],requires_grad=True);leaves=[x,b]
            y=torch.matmul(x,b)
        else: raise AssertionError(kind)
        return leaves,y

    def repeat(kind):
        leaves,y=builder(kind);node=y.grad_fn
        seed=(torch.arange(y.numel(),dtype=torch.float32).reshape(y.shape)+1)/4
        result={'output':tensor_info(y),'grad_fn':type(node).__name__,
                'leaves':[tensor_info(t) for t in leaves],'saved_before':saved(node)}
        result['first']=backward(y,leaves,seed,False);result['saved_after_first']=saved(node)
        result['second']=backward(y,leaves,seed,False);result['saved_after_second']=saved(node)
        return result

    for kind in REPEAT_CASES:
        observe('repeat-default-'+kind,lambda kind=kind:repeat(kind))

    def retain_sequence():
        x=torch.tensor([1.,-2.,3.],requires_grad=True);y=x*x;seed=torch.tensor([.5,-1.,2.]);node=y.grad_fn
        steps=[]
        for keep in (True,True,False,True):
            row=backward(y,[x],seed,keep);row['saved_after']=saved(node);steps.append(row)
        return {'steps':steps}
    observe(SPECIAL_CASES[0],retain_sequence)

    def implicit_default():
        x=torch.tensor([1.,-2.,3.],requires_grad=True);square=x*x;loss=square.sum();node=square.grad_fn
        return {'first':backward(loss,[x]),'saved_after_default':saved(node),
                'second':backward(loss,[x],retain=True)}
    observe(SPECIAL_CASES[1],implicit_default)

    def no_saved_mutation():
        x=torch.tensor([1.,-2.,3.],requires_grad=True);y=x+2.;loss=y.sum()
        first=backward(loss,[x])
        with torch.no_grad(): x.add_(4.)
        second=backward(loss,[x])
        return {'first':first,'second':second,'input_after_mutation':tensor_info(x),'saved':saved(y.grad_fn)}
    observe(SPECIAL_CASES[2],no_saved_mutation)

    def saved_mutation():
        x=torch.tensor([1.,-2.,3.],requires_grad=True);b=torch.tensor([2.,3.,4.]);y=x*b;loss=y.sum();node=y.grad_fn
        before=saved(node)
        b.add_(1.)
        first=backward(loss,[x,b],retain=True);after_first=saved(node)
        second=backward(loss,[x,b],retain=False)
        return {'saved_before':before,'first':first,'saved_after_first':after_first,
                'second':second,'saved_after_second':saved(node)}
    observe(SPECIAL_CASES[3],saved_mutation)

    def partial_release_failure():
        x=torch.tensor([1.,2.,3.],requires_grad=True);middle=x*x;top=middle*middle;loss=top.sum()
        inner,outer=middle.grad_fn,top.grad_fn
        before={'upstream':saved(inner),'downstream':saved(outer)}
        with torch.no_grad(): x.add_(1.)
        call=backward(loss,[x],retain=False)
        return {'before':before,'call':call,'upstream_after_failure':saved(inner),
                'downstream_after_failure':saved(outer),'retry':backward(loss,[x],retain=True),
                'native_transaction_comparison':'Native whole-frame rollback is a distinct contract; do not require these partial Torch side effects.'}
    observe(SPECIAL_CASES[4],partial_release_failure)

    def public_payload():
        x=torch.tensor([1.,2.,3.],requires_grad=True);y=torch.relu(x);node=y.grad_fn;seed=torch.ones_like(y)
        first=backward(y,[x],seed,False)
        return {'first':first,'public_output_after':tensor_info(y),'saved_after':saved(node),
                'second':backward(y,[x],seed,True)}
    observe(SPECIAL_CASES[5],public_payload)

    def leaf_lifetime():
        x=torch.tensor([1.,2.,3.],requires_grad=True);loss=(x+1.).sum();weak=WeakStorage(x)
        try:
            del x
            before=weak.expired();first=backward(loss,[]);after=weak.expired()
            del loss
            return {'expired_after_public_drop':before,'call':first,
                    'expired_after_default_backward':after,'expired_after_root_drop':weak.expired()}
        finally: weak.close()
    observe(SPECIAL_CASES[6],leaf_lifetime)

    def coefficient_lifetime():
        x=torch.tensor([1.,2.,3.],requires_grad=True);b=torch.tensor([2.,3.,4.]);loss=(x*b).sum();weak=WeakStorage(b)
        try:
            del b
            before=weak.expired();first=backward(loss,[x]);after=weak.expired()
            return {'expired_after_public_drop':before,'call':first,'expired_after_default_backward':after}
        finally: weak.close()
    observe(SPECIAL_CASES[7],coefficient_lifetime)

    def output_lifetime():
        x=torch.tensor([-1.,0.,1.],requires_grad=True);y=torch.tanh(x);loss=(y*y).sum();weak=WeakStorage(y)
        try:
            del y
            before=weak.expired();retained=backward(loss,[x],retain=True);after_retained=weak.expired()
            default=backward(loss,[x]);after_default=weak.expired()
            return {'expired_after_public_drop':before,'retained':retained,'expired_after_retained':after_retained,
                    'default':default,'expired_after_default':after_default}
        finally: weak.close()
    observe(SPECIAL_CASES[8],output_lifetime)

    def unpacked_alias():
        x=torch.tensor([-.5,.25,1.],requires_grad=True);y=torch.tanh(x);square=y*y;node=square.grad_fn;loss=square.sum()
        external=node._saved_self.detach();weak=WeakStorage(y)
        try:
            del y,square
            first=backward(loss,[x]);after=saved(node);with_alias=weak.expired();values=tensor_info(external)
            del external
            return {'call':first,'saved_after':after,'expired_with_external_alias':with_alias,
                    'external_values':values,'expired_after_external_alias_drop':weak.expired(),
                    'retry':backward(loss,[x],retain=True)}
        finally: weak.close()
    observe(SPECIAL_CASES[9],unpacked_alias)

    def shared_roots():
        x=torch.tensor([1.,2.,3.],requires_grad=True);y=x*x;left=y.sum();right=(y+1.).sum();node=y.grad_fn
        return {'left':backward(left,[x]),'saved_after_left':saved(node),
                'right':backward(right,[x]),'public_intermediate':tensor_info(y)}
    observe(SPECIAL_CASES[10],shared_roots)

    def unvisited_branch():
        x=torch.tensor([1.,2.,3.],requires_grad=True);a=x*x;b=x*x;left=a.sum();right=b.sum();node=b.grad_fn
        first=backward(left,[x]);unvisited=saved(node);second=backward(right,[x])
        return {'first':first,'unvisited_saved_after_first':unvisited,'second':second,'saved_after_own_backward':saved(node)}
    observe(SPECIAL_CASES[11],unvisited_branch)

    def invalid_seed():
        x=torch.tensor([1.,2.,3.],requires_grad=True);y=x*x;node=y.grad_fn
        bad=backward(y,[x],torch.ones(1,3),False);after=saved(node)
        good=backward(y,[x],torch.ones(3),False)
        return {'invalid_seed_call':bad,'saved_after_invalid_seed':after,'valid_seed_call':good,
                'saved_after_valid_seed':saved(node)}
    observe(SPECIAL_CASES[12],invalid_seed)

    def invalid_gradient_shape():
        x=torch.arange(6,dtype=torch.float32).reshape(2,3).requires_grad_(True)
        b=torch.ones(2,3,requires_grad=True);index=torch.tensor([[0,1],[1,2]],dtype=torch.int64)
        y=torch.scatter_add(x,1,index,b);node=y.grad_fn;seed=torch.ones_like(y)
        first=backward(y,[x,b],seed,False);after=saved(node);second=backward(y,[x,b],seed,False)
        return {'source_shape':list(b.shape),'index_shape':list(index.shape),'first':first,
                'saved_after_failed_apply':after,'second':second,'saved_after_second':saved(node)}
    observe(SPECIAL_CASES[13],invalid_gradient_shape)

    def empty_index():
        x=basic_x();y=torch.index_select(x,1,torch.empty(0,dtype=torch.int64));node=y.grad_fn;seed=torch.empty_like(y)
        first=backward(y,[x],seed,False);after=saved(node);second=backward(y,[x],seed,False)
        return {'first':first,'saved_after_first':after,'second':second,'output':tensor_info(y)}
    observe(SPECIAL_CASES[14],empty_index)

    def leaf_repeat():
        x=torch.tensor(2.,requires_grad=True)
        return {'grad_fn_is_none':x.grad_fn is None,'first':backward(x,[x]),'second':backward(x,[x])}
    observe(SPECIAL_CASES[15],leaf_repeat)

    assert tuple(row['case'] for row in rows)==CASE_NAMES
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    for name in ('reference','lock','repo','output'): setattr(args,name,getattr(args,name).resolve())
    assert digest(args.reference)==SUPPORT_SHA,'Changed reference support'
    base={'schema':'heptatorch.pinned-torch-backward-release-observations.v1',
          'probe_sha256':digest(__file__),'support_sha256':SUPPORT_SHA,'lock_sha256':digest(args.lock),
          'upstream_commit':PIN,'expected_cases':len(CASE_NAMES),'case_names':list(CASE_NAMES),
          'target_execution':False,'native_values_compared':0,'full_rewrite_complete':False,
          'pytorch_api_denominator':2590,'scope':'Observed pinned Torch behavior; native compatibility not asserted by this probe.'}
    if args.prepare_only:
        write(args.output,dict(base,status='PREPARED_NOT_RUN',cases=0));return 0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    provenance=args.output.with_name(args.output.stem+'-provenance.json')
    previous_argv,previous_directory=sys.argv,Path.cwd()
    try:
        os.chdir(args.repo)
        sys.argv=[str(args.reference),'/dev/null',str(provenance),str(args.lock)]
        try: support=runpy.run_path(str(args.reference),run_name='pinned_backward_release_support')
        except SystemExit as error:
            observed=json.loads(provenance.read_text()) if provenance.exists() else {'exit':error.code}
            write(args.output,dict(base,status='NOT_RUN',cases=0,reason='Pinned support did not admit this environment',
                                   support_observation=observed));return 2
    finally:
        sys.argv=previous_argv;os.chdir(previous_directory)
    assert support['PROVENANCE']['status']=='PASS' and len(support['PROVENANCE']['checks'])==11
    assert all(v is True for v in support['PROVENANCE']['checks'].values())
    rows=execute(support['torch'],support['F'])
    errors=sum(row['status']!='OBSERVED' for row in rows)
    report=dict(base,status='PROBE_ERROR' if errors else 'OBSERVED',cases=len(rows),probe_errors=errors,
                rows=rows,oracle_provenance=support['PROVENANCE'],probe_unchanged=digest(__file__)==base['probe_sha256'],
                reference_unchanged=digest(args.reference)==SUPPORT_SHA)
    write(args.output,report)
    print(json.dumps({'status':report['status'],'cases':len(rows),'probe_errors':errors}))
    return 1 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
