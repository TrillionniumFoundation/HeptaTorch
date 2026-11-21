#!/usr/bin/env python3
"""Observe matmul and softmax F32 forward/backward using the actual independently pinned Torch wheel."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import struct
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from forward_nograd.cases import CASES,CASE_SHA256,raw_input,seed_raw
from numeric_predicates.reference import identity
import pinned_reference_custody as custody

SUPPORT='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'

def tensor(torch,value,folder):
    # Only observe actual bytes. A uint8 view avoids Python calling into an
    # UntypedStorage once per byte for the 64 MiB capacity-boundary result.
    logical=bytes(value.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8).tolist())
    owned=value.untyped_storage()
    byte_view=torch.empty(0,dtype=torch.uint8).set_(owned,0,(owned.nbytes(),),(1,))
    storage=bytes(byte_view.tolist())
    def blob(raw):
        name=hashlib.sha256(raw).hexdigest()+'.bin';path=folder/'blobs'/name
        if not path.exists():path.write_bytes(raw)
        return dict(path='blobs/'+name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return dict(dtype=str(value.dtype),shape=list(value.shape),strides=list(value.stride()),
        numel=value.numel(),element_size=value.element_size(),storage_bytes=owned.nbytes(),
        storage_offset=value.storage_offset(),version=value._version,requires_grad=value.requires_grad,
        is_leaf=value.is_leaf,is_view=value._is_view(),history_present=value.grad_fn is not None,
        retains_grad=value.retains_grad,logical=blob(logical),storage=blob(storage))

def source_files():
    return [Path(__file__).resolve(),ROOT/'tests/forward_nograd/cases.py',ROOT/'tests/forward_nograd/__init__.py',
        ROOT/'tests/numeric_predicates/reference.py',ROOT/'tests/numeric_predicates/cases.py',
        ROOT/'tests/numeric_predicates/__init__.py',ROOT/'tests/pytorch_oracle.py',
        ROOT/'compatibility/pytorch-oracle-lock.json',ROOT/'compatibility/upstream-lock.json',
        ROOT/'tests/pinned_reference_custody.py',Path(sys.executable).resolve()] + custody.source_files(ROOT)

def local_identity(torch):
    expected=json.loads((ROOT/'compatibility/pytorch-oracle-lock.json').read_bytes())['oracle']
    distribution=importlib.metadata.distribution(expected['distribution'])
    checks=dict(python_matches=sys.implementation.name==expected['python']['implementation']
        and list(sys.version_info[:2])==expected['python']['version'],
        platform_matches=dict(sys_platform=sys.platform,machine=platform.machine())==expected['platform'],
        version_backend_matches=str(torch.__version__)==expected['version']
        and {key:getattr(torch.version,key,'MISSING') for key in expected['backend_versions']}==expected['backend_versions'],
        git_matches_oracle_build_pin=torch.version.git_version==expected['git_commit'],
        distribution_version_matches=distribution.version==expected['version'])
    assert all(checks.values()),checks
    return dict(status='EXPLORATORY_NOT_PINNED_WHEEL_QUALIFICATION',checks=checks,expected=expected,
        lock_sha256=LOCK,version=str(torch.__version__),git_version=torch.version.git_version,
        package=identity(torch.__file__),extension=identity(torch._C.__file__),
        metadata_location=str(distribution._path),direct_url_present=distribution.read_text('direct_url.json') is not None,
        full_installation_provenance_verified=False)

def setup(torch,spec):
    raw=raw_input(spec)
    backing=torch.frombuffer(bytearray(raw),dtype=torch.float32).clone() if raw else torch.empty(0,dtype=torch.float32)
    if spec['strides'] is None:
        source=backing.reshape(spec['shape']).clone(memory_format=torch.contiguous_format)
        source.requires_grad_(spec['requires_grad']);base=source
    else:
        backing.requires_grad_(spec['requires_grad'])
        source=backing.as_strided(spec['shape'],spec['strides'],spec['offset']);base=backing
    return source,base

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-unverified-local',action='store_true',help='Exploratory only; never complete CI qualification.')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'blobs').mkdir()
    assert Path.cwd().resolve()==ROOT and not sys.flags.optimize
    context=custody.begin(ROOT,out,__file__,args.allow_unverified_local)
    report=dict(schema='heptatorch.forward-nograd.reference.v1',status='RUNNING',case_sha256=CASE_SHA256,
        expected_cases=len(CASES),rows=[],Torch_executed=False,native_executed=False,compiler_executed=False,
        host_expected_values_computed=False,production_oracle_qualified=False,failures=[],
        actual_process=dict(pid=os.getpid(),argv=sys.orig_argv,python=identity(sys.executable)))
    files=source_files()
    try:
        if not args.allow_unverified_local:files.append(ROOT/'build/pytorch-install.json')
        inputs={str(path):identity(path) for path in files};report['inputs']=inputs
        assert inputs[str(ROOT/'tests/pytorch_oracle.py')]['sha256']==SUPPORT
        assert inputs[str(ROOT/'compatibility/pytorch-oracle-lock.json')]['sha256']==LOCK
        if args.allow_unverified_local:
            import torch
            report['oracle_provenance']=local_identity(torch)
        else:
            old=sys.argv
            try:
                sys.argv=[str(ROOT/'tests/pytorch_oracle.py'),str(out/'NO_NATIVE_EXECUTION'),str(out/'oracle-provenance.json'),str(ROOT/'compatibility/pytorch-oracle-lock.json')]
                ctx=runpy.run_path(str(ROOT/'tests/pytorch_oracle.py'),run_name='forward_nograd_reference_only')
            finally:sys.argv=old
            torch,provenance=ctx['torch'],ctx['PROVENANCE']
            assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(provenance['checks'].values())
            report.update(oracle_provenance=provenance,production_oracle_qualified=True)
        custody.finish_capture(context)
        report['reference_custody']=custody.capture(ROOT,out,__file__,torch,report['oracle_provenance'],args.allow_unverified_local,context)
        torch.set_num_threads(1);torch.manual_seed(1729);torch.backends.mkldnn.enabled=False
        torch.use_deterministic_algorithms(True)
        assert torch.backends.cpu.get_cpu_capability()=='AVX2'
        report.update(Torch_executed=True,cpu_capability='AVX2',torch_threads=1)
        rng=bytes(torch.get_rng_state().tolist())
        for index,spec in enumerate(CASES):
            a,base_a=setup(torch,spec['a']);b=base_b=None
            if spec['b'] is not None:
                b,base_b=(a,base_a) if spec['same_input'] else setup(torch,spec['b'])
            before_a=tensor(torch,a,out);before_b=tensor(torch,b,out) if b is not None else None
            row=dict(index=index,spec=spec,input_a=before_a,input_b=before_b)
            axis=spec['axis'] if spec['axis']<1<<63 else spec['axis']-(1<<64)
            try:
                if spec['operation']=='matmul':result=torch.matmul(a,b)
                elif spec['operation']=='softmax':result=torch.softmax(a,axis)
                else:result=torch.log_softmax(a,axis)
            except (RuntimeError,IndexError) as error:
                row.update(accepted=False,error=dict(type=type(error).__name__,message=str(error)))
            else:
                row.update(accepted=True,result=tensor(torch,result,out),fresh_tensor=result._cdata!=a._cdata,
                    fresh_storage=result.untyped_storage()._cdata!=a.untyped_storage()._cdata)
                if b is not None:
                    row['fresh_tensor'] &= result._cdata!=b._cdata
                    row['fresh_storage'] &= result.untyped_storage()._cdata!=b.untyped_storage()._cdata
                assert row['fresh_tensor'] and row['fresh_storage']
                if result.requires_grad:
                    raw=seed_raw(result.numel())
                    seed=torch.frombuffer(bytearray(raw),dtype=torch.float32).clone().reshape(result.shape) if raw else torch.empty_like(result)
                    result.backward(seed)
                    row['seed']=tensor(torch,seed,out)
                    for label,base in [('a',base_a),('b',base_b)]:
                        if base is not None and base.requires_grad:
                            assert base.grad is not None
                            row['gradient_'+label]=tensor(torch,base.grad,out)
            assert tensor(torch,a,out)==before_a
            if b is not None:assert tensor(torch,b,out)==before_b
            assert bytes(torch.get_rng_state().tolist())==rng
            row.update(inputs_unchanged=True,rng_unchanged=True);report['rows'].append(row)
            (out/('case-%04d.json'%index)).write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
        assert all(identity(path)==inputs[str(path)] for path in files)
        for row in report['rows']:
            for key in ('input_a','input_b','result','seed','gradient_a','gradient_b'):
                value=row.get(key)
                if value is not None:
                    for field in ('logical','storage'):
                        item=value[field];raw=(out/item['path']).read_bytes()
                        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        custody.finish_reference(context)
        report.update(status='OBSERVED_FORWARD_NOGRAD' if report['production_oracle_qualified'] else 'OBSERVED_EXPLORATORY_FORWARD_NOGRAD',
            cases=len(report['rows']),inputs_unchanged=True,rng_unchanged=True,
            all_observed_blobs_unchanged=True,rng_sha256=hashlib.sha256(rng).hexdigest())
    except BaseException as error:
        custody.finish_capture(context)
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:report.get(key) for key in ('status','cases','expected_cases','production_oracle_qualified','error')}),flush=True)
    return 0 if report['status'].startswith('OBSERVED_') else 1

if __name__=='__main__':raise SystemExit(main())
