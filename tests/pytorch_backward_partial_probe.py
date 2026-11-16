"""Observe pinned CPU autograd scheduling and partial failure side effects.

Every recipe runs once with read-only node hooks and once without hooks. The
probe records genuine leaf gradients, Saved access, errors and actual node
sequence numbers. It never executes a native HIA target or predicts numerical
results from a substitute implementation. Existing 36- and 35-case probes are
unchanged.
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
CASE_NAMES=(
    'chain-upstream-version-error-default',
    'chain-upstream-version-error-retained',
    'independent-good-newer-nonuniform-seed',
    'independent-bad-newer',
    'independent-good-newer-retained-retry',
    'independent-good-newer-preexisting-gradient',
    'shared-leaf-waits-for-failing-branch',
    'shared-leaf-pending-and-independent-leaf-committed',
    'no-Saved-good-branch-repeats-before-version-error',
    'shared-intermediate-diamond-and-independent-leaf',
    'gradient-shape-error-after-good-branch',
    'gradient-shape-error-before-good-branch',
    'released-node-error-after-live-good-branch',
    'shared-leaf-retained-failure-then-good-root',
    'mutated-output-Saved-error-after-good-branch',
    'successful-shared-leaf-readiness-control',
)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def execute(torch):
    def tensor_info(t):
        return {'shape':list(t.shape),'strides':list(t.stride()),
                'requires_grad':bool(t.requires_grad),'version':int(t._version),
                'values':t.detach().reshape(-1).tolist()}

    def plain(value):
        if isinstance(value,torch.Tensor): return {'tensor':tensor_info(value)}
        if isinstance(value,(tuple,list)): return [plain(v) for v in value]
        if value is None or isinstance(value,(bool,int,float,str)): return value
        return {'type':type(value).__name__,'text':str(value)}

    def saved(node):
        result={}
        for name in sorted(n for n in dir(node) if n.startswith('_saved_')):
            try: result[name]={'status':'AVAILABLE','value':plain(getattr(node,name))}
            except Exception as error:
                result[name]={'status':'RAISED','exception_type':type(error).__name__,'message':str(error)}
        return result

    def gradients(leaves):
        return {name:None if tensor.grad is None else tensor_info(tensor.grad)
                for name,tensor in leaves.items()}

    def construct(case):
        named={};leaves={};mutations=[];setup=[]
        def put(name,value): named[name]=value;return value
        def new(name,values,shape=None):
            value=torch.tensor(values,dtype=torch.float32)
            if shape is not None: value=value.reshape(shape)
            value.requires_grad_(True);leaves[name]=value;named[name]=value
            return value
        def ss(name,value): return put(name,value.sum())
        x=new('x',[1.,2.,3.]);seed=None;retain=False;retry_output=None;retry_retain=True
        if case in CASE_NAMES[:2]:
            inner=put('inner',x*x);outer=put('outer',inner*inner);output=ss('loss',outer)
            mutations.append(('x',x,1.));retain=case==CASE_NAMES[1]
        elif case in (CASE_NAMES[2],CASE_NAMES[3],CASE_NAMES[4],CASE_NAMES[5],CASE_NAMES[8]):
            good_leaf=new('g',[2.,-1.,.5])
            if case==CASE_NAMES[5]:
                prime=(good_leaf*2.).sum();prime.backward()
                setup.append({'operation':'prime (g * 2).sum().backward()','gradients_after':gradients(leaves)})
            if case==CASE_NAMES[3]:
                good=put('good',good_leaf*good_leaf);bad=put('bad',x*x)
            else:
                bad=put('bad',x*x)
                good=put('good',good_leaf+good_leaf if case==CASE_NAMES[8] else good_leaf*good_leaf)
            merged=put('merged',bad+good)
            if case==CASE_NAMES[2]:
                output=merged;seed=torch.tensor([.5,-1.,2.],dtype=torch.float32)
            else: output=ss('loss',merged)
            mutations.append(('x',x,1.));retain=case==CASE_NAMES[4]
        elif case in (CASE_NAMES[6],CASE_NAMES[7],CASE_NAMES[13]):
            coefficient=put('coefficient',torch.tensor([2.,3.,4.],dtype=torch.float32))
            bad=put('bad',x*coefficient);good=put('good',x*x)
            if case==CASE_NAMES[13]: retry_output=ss('good_root',good);retry_retain=False;retain=True
            merged=put('merged',bad+good)
            if case==CASE_NAMES[7]:
                g=new('g',[2.,-1.,.5]);independent=put('independent',g*g);merged=put('all_branches',merged+independent)
            output=ss('loss',merged);mutations.append(('coefficient',coefficient,1.))
        elif case==CASE_NAMES[9]:
            g=new('g',[2.,-1.,.5]);shared=put('shared',x*x)
            left=put('left',shared*shared);right=put('right',shared+g)
            output=ss('loss',put('merged',left+right));mutations.append(('x',x,1.))
        elif case in (CASE_NAMES[10],CASE_NAMES[11]):
            # The unused initial x is excluded from the gradient set: each
            # graph below has only the actual source/self/g leaves.
            del leaves['x'],named['x']
            self_tensor=new('self',[0.,1.,2.,3.,4.,5.],(2,3))
            source=new('source',[1.]*6,(2,3));g=new('g',[1.,2.,3.,4.,5.,6.],(2,3))
            index=torch.tensor([[0,1],[1,2]],dtype=torch.int64)
            if case==CASE_NAMES[11]:
                good=put('good',g*g);bad=put('bad',torch.scatter_add(self_tensor,1,index,source))
            else:
                bad=put('bad',torch.scatter_add(self_tensor,1,index,source));good=put('good',g*g)
            output=ss('loss',put('merged',bad+good))
        elif case==CASE_NAMES[12]:
            spent=put('spent',x*x);prior=spent.sum();prior.backward()
            setup.append({'operation':'spent.sum().backward() before constructing the tested root','gradients_after':gradients(leaves),
                          'spent_saved_after':saved(spent.grad_fn)})
            g=new('g',[2.,-1.,.5]);good=put('good',g*g);output=ss('loss',put('merged',spent+good))
        elif case==CASE_NAMES[14]:
            bad=put('bad',torch.relu(x));g=new('g',[2.,-1.,.5]);good=put('good',g*g)
            output=ss('loss',put('merged',bad+good));mutations.append(('bad',bad,1.))
        elif case==CASE_NAMES[15]:
            g=new('g',[2.,-1.,.5]);left=put('left',x*x);right=put('right',x+x);good=put('good',g*g)
            output=ss('loss',put('all_branches',put('merged',left+right)+good))
        else: raise AssertionError(case)
        before_mutation={name:tensor_info(value) for name,value in named.items()}
        with torch.no_grad():
            for _,target,increment in mutations: target.add_(increment)
        recipe={'named_values_before_mutation':before_mutation,
                'mutations':[{'name':name,'operation':'add_','scalar':increment} for name,_,increment in mutations],
                'named_values_after_mutation':{name:tensor_info(value) for name,value in named.items()},
                'setup':setup,'first_retain_graph':retain,'retry_retain_graph':retry_retain}
        return named,leaves,output,retry_output if retry_output is not None else output,seed,retain,retry_retain,recipe

    def run(case,traced):
        named,leaves,output,retry,seed,retain,retry_retain,recipe=construct(case)
        labels={};nodes=[];handles=[];events=[]
        # Keep every node wrapper alive. No pointer or guessed sequence order is
        # serialized; sequence numbers are decimal strings to preserve uint64.
        for name,value in named.items():
            node=value.grad_fn
            if node is not None and node not in labels: labels[node]=name
        for name,value in leaves.items():
            node=torch.autograd.graph.get_gradient_edge(value).node
            if node not in labels: labels[node]='AccumulateGrad('+name+')'
        queue=[output.grad_fn,retry.grad_fn];seen=set()
        while queue:
            node=queue.pop(0)
            if node is None or node in seen: continue
            seen.add(node);nodes.append(node)
            if node not in labels: labels[node]='unnamed-'+str(len(nodes))+'-'+node.name()
            queue.extend(next_node for next_node,_ in node.next_functions if next_node is not None)
        def saved_all(): return {labels[node]:saved(node) for node in nodes if any(n.startswith('_saved_') for n in dir(node))}
        def snapshot(phase,node,grad_inputs=None,grad_outputs=None):
            events.append({'index':len(events),'phase':phase,'node':labels[node],'node_type':node.name(),
                           'sequence_nr_decimal':str(node._sequence_nr()),
                           'grad_inputs':plain(grad_inputs),'grad_outputs':plain(grad_outputs),
                           'leaf_gradients':gradients(leaves),'saved':saved_all()})
        def pre(node):
            def hook(grad_outputs): snapshot('pre',node,grad_outputs=grad_outputs)
            return hook
        def post(node):
            def hook(grad_inputs,grad_outputs): snapshot('post',node,grad_inputs,grad_outputs)
            return hook
        topology=[]
        for node in nodes:
            topology.append({'node':labels[node],'node_type':node.name(),
                             'sequence_nr_decimal':str(node._sequence_nr()),
                             'next_edges':[{'node':None if parent is None else labels[parent],'input_nr':nr}
                                           for parent,nr in node.next_functions]})
            if traced: handles.extend((node.register_prehook(pre(node)),node.register_hook(post(node))))
        initial={'gradients':gradients(leaves),'saved':saved_all()};calls=[]
        try:
            for call_index,(root,keep) in enumerate(((output,retain),(retry,retry_retain))):
                events.clear()
                used_seed=seed if root is output else None
                row={'call_index':call_index,'output_node':labels[root.grad_fn],
                     'retain_graph':keep,'seed':None if used_seed is None else tensor_info(used_seed),
                     'leaf_gradients_before':gradients(leaves),'saved_before':saved_all()}
                try:
                    root.backward(used_seed,retain_graph=keep)
                    row['outcome']='SUCCEEDED'
                except Exception as error:
                    row.update(outcome='RAISED',exception_type=type(error).__name__,message=str(error))
                row.update(leaf_gradients_after=gradients(leaves),saved_after=saved_all(),events=list(events))
                if traced:
                    entered=[e['node'] for e in events if e['phase']=='pre'];finished=[e['node'] for e in events if e['phase']=='post']
                    row.update(actual_entered_order=entered,actual_finished_order=finished,
                               actual_finished_leaf_accumulators=[e['node'] for e in events if e['phase']=='post' and e['node_type']=='torch::autograd::AccumulateGrad'],
                               entered_without_post=[name for name in entered if name not in finished])
                calls.append(row)
        finally:
            for handle in handles: handle.remove()
        return {'read_only_hooks':traced,'recipe':recipe,'node_topology':topology,'initial':initial,'calls':calls}

    def value_signature(value):
        # Node post hooks may change internal gradient storage-stealing choices.
        # Compare observable values/shapes/presence and Saved state, recording
        # all raw versions in both variants without inventing their equality.
        if isinstance(value,list): return [value_signature(v) for v in value]
        if isinstance(value,dict): return {k:value_signature(v) for k,v in value.items() if k!='version'}
        return value

    def signature(trial):
        keys=('outcome','exception_type','message','leaf_gradients_before','leaf_gradients_after','saved_before','saved_after','retain_graph','seed','output_node')
        return value_signature({'initial':trial['initial'],'calls':[{k:r[k] for k in keys if k in r} for r in trial['calls']]})

    rows=[]
    with torch.autograd.set_multithreading_enabled(False):
        for name in CASE_NAMES:
            row={'case':name,'status':'PROBE_ERROR'}
            try:
                traced=run(name,True);plain_run=run(name,False)
                control=signature(traced)==signature(plain_run)
                row.update(status='OBSERVED' if control else 'PROBE_ERROR',
                           observation={'traced':traced,'untraced':plain_run,
                                        'instrumentation_values_shapes_presence_and_saved_control_equal':control})
                if not control: row['message']='Read-only hook and hook-free executions differed in compared observable values or Saved/error state.'
            except Exception as error:
                row.update(exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            rows.append(row)
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
    base={'schema':'heptatorch.pinned-torch-backward-partial-observations.v1',
          'probe_sha256':digest(__file__),'support_sha256':SUPPORT_SHA,'lock_sha256':digest(args.lock),
          'upstream_commit':PIN,'expected_cases':len(CASE_NAMES),'case_names':list(CASE_NAMES),
          'target_execution':False,'native_values_compared':0,'full_rewrite_complete':False,'pytorch_api_denominator':2590,
          'scope':'Actual CPU first-order default/retained backward scheduling and partial failure; no native equivalence is asserted.',
          'engine_mode':'Autograd multithreading explicitly disabled; no reentrant backward, custom Function or numerical hook mutation.',
          'instrumentation':'Each recipe runs with read-only node pre/post hooks and separately without hooks; compare values, shapes, gradient presence, Saved/error state. Raw gradient versions are recorded without equating storage-stealing side effects.',
          'uint64_encoding':'Node sequence numbers are decimal strings; AccumulateGrad UINT64_MAX must not traverse a floating JSON number.'}
    if args.prepare_only:
        write(args.output,dict(base,status='PREPARED_NOT_RUN',cases=0));return 0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    provenance=args.output.with_name(args.output.stem+'-provenance.json')
    previous_argv,previous_directory=sys.argv,Path.cwd()
    try:
        os.chdir(args.repo);sys.argv=[str(args.reference),'/dev/null',str(provenance),str(args.lock)]
        try: support=runpy.run_path(str(args.reference),run_name='pinned_backward_partial_support')
        except SystemExit as error:
            observed=json.loads(provenance.read_text()) if provenance.exists() else {'exit':error.code}
            write(args.output,dict(base,status='NOT_RUN',cases=0,reason='Pinned support did not admit this environment',support_observation=observed));return 2
    finally:
        sys.argv=previous_argv;os.chdir(previous_directory)
    assert support['PROVENANCE']['status']=='PASS' and len(support['PROVENANCE']['checks'])==11
    assert all(v is True for v in support['PROVENANCE']['checks'].values())
    rows=execute(support['torch']);errors=sum(row['status']!='OBSERVED' for row in rows)
    assert tuple(row['case'] for row in rows)==CASE_NAMES
    report=dict(base,status='PROBE_ERROR' if errors else 'OBSERVED',cases=len(rows),probe_errors=errors,rows=rows,
                oracle_provenance=support['PROVENANCE'],probe_unchanged=digest(__file__)==base['probe_sha256'],
                reference_unchanged=digest(args.reference)==SUPPORT_SHA)
    write(args.output,report)
    print(json.dumps({'status':report['status'],'cases':len(rows),'probe_errors':errors}))
    return 1 if errors else 0


if __name__=='__main__': raise SystemExit(main())
