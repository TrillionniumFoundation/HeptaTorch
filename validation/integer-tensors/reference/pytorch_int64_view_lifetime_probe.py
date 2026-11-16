#!/usr/bin/env python3
"""Observe int64 view ultimate-base identity after public locals are removed.

No expected outcomes or native runtime are supplied. Storage weak references
are explicit; Python wrapper weakrefs are never used as Tensor lifetime proof.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback
import warnings

SUPPORT='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
CHECKS={'python_matches','platform_matches','version_backend_matches','git_matches_oracle_build_pin',
        'distribution_version_matches','direct_url_matches','direct_url_archive_sha256_matches',
        'pip_report_url_matches','pip_report_archive_sha256_matches','pip_report_version_matches','pip_report_requested'}
sha=lambda b:hashlib.sha256(b).hexdigest()


def plan():
    return [{'name':form+('-empty' if empty else '-six'),'form':form,'empty':empty,
             'base_shape':[0,3] if empty else [2,3],
             'sequence':['construct first view','construct as_strided view of view','drop base local',
                         'drop first view','second.detach_','second.add_(1)','second.detach()',
                         'drop second view','detached.detach_','drop final detached Tensor']}
            for form in ('as_strided','transpose','view') for empty in (False,True)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('reference','lock','repo','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();assert Path.cwd().resolve()==a.repo.resolve()
    a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists()
    inputs={p.resolve():p.read_bytes() for p in (Path(__file__),a.reference,a.lock)}
    assert sha(inputs[a.reference.resolve()])==SUPPORT and sha(inputs[a.lock.resolve()])==LOCK
    common={'schema':'heptatorch.int64-view-base-lifetime-observations.v1','probe_sha256':sha(inputs[Path(__file__).resolve()]),
            'primary_source_ref':'31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457','expected_cases':6,'case_plan':plan(),
            'native_executed':False,'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590,
            'identity_scope':'Actual TensorImpl/StorageImpl identity strings; _base is read only while an extant view owns it. No Python wrapper weakref identity claim.'}
    if a.prepare_only:
        report={**common,'status':'PREPARED_NOT_RUN','torch_executed':False,'cases':0,'rows':[]}
    else:
        old=sys.argv
        try:
            sys.argv=[str(a.reference),str(a.repo/'reference-only-no-native-execution'),str(a.output.with_suffix('.provenance.json')),str(a.lock)]
            ctx=runpy.run_path(str(a.reference),run_name='int64_view_base_provenance_only')
        finally:sys.argv=old
        t,prov=ctx['torch'],ctx['PROVENANCE'];assert prov['status']=='PASS' and set(prov['checks'])==CHECKS and all(v is True for v in prov['checks'].values())
        def tensor(x):
            if x is None:return None
            raw=bytes(x.detach().contiguous().reshape(-1).view(t.uint8).tolist())
            return {'tensor_identity':str(x._cdata),'storage_identity':str(x.untyped_storage()._cdata),
                    'dtype':str(x.dtype),'element_size':x.element_size(),'shape':list(x.shape),'strides':list(x.stride()),
                    'numel':x.numel(),'storage_bytes':x.untyped_storage().nbytes(),'storage_offset_elements':x.storage_offset(),
                    'version':x._version,'requires_grad':x.requires_grad,'is_leaf':x.is_leaf,'is_view':x._is_view(),
                    'base_tensor_identity':None if x._base is None else str(x._base._cdata),'logical_bytes_hex':raw.hex()}
        def capture(fn):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:value=fn();event={'accepted':True,'error':None,'returned':tensor(value)}
                except Exception as error:value=None;event={'accepted':False,'error':{'type':type(error).__name__,'message':str(error)},'returned':None}
            event['warnings']=[str(w.message) for w in caught]
            return event
        rows=[];errors=[]
        with t.random.fork_rng(devices=[]):
            for index,spec in enumerate(plan()):
                weak=None
                try:
                    t.manual_seed(1729);before=bytes(t.get_rng_state().tolist())
                    base=t.empty((0,3),dtype=t.int64) if spec['empty'] else t.tensor([[-(1<<63),(1<<63)-1,-1],[0,1,3]],dtype=t.int64)
                    root_identity=str(base._cdata);weak=base.untyped_storage()._weak_ref()
                    initial=tensor(base)
                    if spec['form']=='as_strided':first=base.as_strided((2,0,3),(17,101,2),7) if spec['empty'] else base.as_strided((2,2),(3,1),1)
                    elif spec['form']=='transpose':first=base.transpose(0,1)
                    else:first=base.view((3,0) if spec['empty'] else (3,2))
                    second=first.as_strided(tuple(first.shape),tuple(first.stride()),first.storage_offset())
                    row={**spec,'index':index,'initial_base':initial,'root_tensor_identity':root_identity,
                         'first':tensor(first),'second':tensor(second),'second_base_before_drop':tensor(second._base)}
                    del base;gc.collect()
                    row['after_base_local_drop']={'first':tensor(first),'second':tensor(second),'second_base':tensor(second._base),'Storage_expired':bool(t.UntypedStorage._expired(weak))}
                    del first;gc.collect()
                    row['after_first_view_drop']={'second':tensor(second),'second_base':tensor(second._base),'Storage_expired':bool(t.UntypedStorage._expired(weak))}
                    row['view_detach_inplace_before']=tensor(second);row['view_detach_inplace']=capture(lambda:second.detach_());row['view_detach_inplace_after']=tensor(second)
                    row['view_scalar_add']=capture(lambda:second.add_(1));row['after_view_mutation']={'second':tensor(second),'second_base':tensor(second._base)}
                    held=second.detach();row['detached_before_last_view_drop']=tensor(held)
                    del second;gc.collect()
                    row['after_last_view_drop']={'detached':tensor(held),'Storage_expired':bool(t.UntypedStorage._expired(weak))}
                    row['detached_detach_inplace']=capture(lambda:held.detach_());row['detached_after_detach_inplace']=tensor(held)
                    del held;gc.collect();row['after_all_strong_drop']={'Storage_expired':bool(t.UntypedStorage._expired(weak))}
                    after=bytes(t.get_rng_state().tolist());row.update(status='OBSERVED',rng_before_hex=before.hex(),rng_after_hex=after.hex(),rng_unchanged=before==after);rows.append(row)
                except BaseException as error:
                    event={**spec,'index':index,'status':'PROBE_ERROR','error':repr(error),'traceback':traceback.format_exc()};errors.append(event);rows.append(event)
                finally:
                    if weak is not None:t.UntypedStorage._free_weak_ref(weak)
        report={**common,'status':'OBSERVED' if not errors else 'PROBE_ERROR','torch_executed':True,
                'oracle_provenance':prov,'cases':len(rows),'rows':rows,'probe_errors':errors,'probe_error_count':len(errors)}
    assert all(p.read_bytes()==raw for p,raw in inputs.items());report['source_files_unchanged']=True
    report['inputs_sha256']={str(p):sha(raw) for p,raw in inputs.items()}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:report[k] for k in ('status','cases','probe_sha256')}))
    return int(report['status']=='PROBE_ERROR')


if __name__=='__main__':raise SystemExit(main())
