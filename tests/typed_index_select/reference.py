#!/usr/bin/env python3
"""Observe index_select with real Torch; strict fixed-wheel provenance is default."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import stat
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from typed_index_select.cases import CASES,CASE_SHA256,DTYPES,raw_input
from numeric_predicates.reference import tensor,identity

SUPPORT='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'


def source_files():
    return [Path(__file__).resolve(),ROOT/'tests/typed_index_select/cases.py',ROOT/'tests/typed_index_select/__init__.py',
        ROOT/'tests/numeric_predicates/reference.py',ROOT/'tests/numeric_predicates/cases.py',
        ROOT/'tests/numeric_predicates/__init__.py',ROOT/'tests/pytorch_oracle.py',
        ROOT/'compatibility/pytorch-oracle-lock.json',ROOT/'compatibility/upstream-lock.json',Path(sys.executable).resolve()]


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
    dtype=torch.float32 if spec.get('cast_from_f32') else getattr(torch,DTYPES[spec['dtype']])
    backing=torch.frombuffer(bytearray(raw),dtype=dtype).clone() if raw else torch.empty(0,dtype=dtype)
    if spec.get('legacy_bool'):
        backing=torch.zeros(4,dtype=torch.bool)
    if spec['strides'] is None:
        source=backing.reshape(spec['shape']).clone(memory_format=torch.contiguous_format)
        source.requires_grad_(spec.get('requires_grad',False))
    else:
        source=backing.as_strided(spec['shape'],spec['strides'],spec['offset'])
    if spec.get('cast_from_f32'):
        source=source.to(getattr(torch,DTYPES[spec['dtype']]))
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-unverified-local',action='store_true',help='Exploratory results only; never a production qualification.')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'blobs').mkdir()
    assert Path.cwd().resolve()==ROOT and not sys.flags.optimize
    os.environ['ATEN_CPU_CAPABILITY']='avx2'
    report=dict(schema='heptatorch.typed-index-select.reference.v1',status='RUNNING',
        case_sha256=CASE_SHA256,expected_cases=len(CASES),rows=[],Torch_executed=False,
        native_executed=False,compiler_executed=False,host_expected_values_computed=False,
        production_oracle_qualified=False,failures=[],actual_process=dict(pid=os.getpid(),argv=sys.orig_argv,python=identity(sys.executable)))
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
                ctx=runpy.run_path(str(ROOT/'tests/pytorch_oracle.py'),run_name='index_select_reference_only')
            finally:sys.argv=old
            torch,provenance=ctx['torch'],ctx['PROVENANCE']
            assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(provenance['checks'].values())
            report.update(oracle_provenance=provenance,production_oracle_qualified=True)
        torch.set_num_threads(1)
        torch.manual_seed(1729)
        torch.backends.mkldnn.enabled=False
        torch.use_deterministic_algorithms(True)
        assert os.environ.get('ATEN_CPU_CAPABILITY')=='avx2' and torch.backends.cpu.get_cpu_capability()=='AVX2'
        report.update(Torch_executed=True,cpu_capability='AVX2',torch_threads=1)
        rng=bytes(torch.get_rng_state().tolist())
        for index,spec in enumerate(CASES):
            source=setup(torch,spec['input']);before=tensor(torch,source,out)
            indices=torch.tensor(spec['indices'],dtype=torch.int64)
            row=dict(index=index,spec=spec,input=before)
            try:
                result=torch.index_select(source,spec['axis'],indices)
            except (RuntimeError,IndexError) as error:
                row.update(accepted=False,error=dict(type=type(error).__name__,message=str(error)))
            else:
                row.update(accepted=True,result=tensor(torch,result,out),fresh_tensor=result._cdata!=source._cdata,
                    fresh_storage=result.untyped_storage()._cdata!=source.untyped_storage()._cdata)
                assert row['fresh_tensor'] and row['fresh_storage']
                if spec['input'].get('requires_grad'):
                    result.backward(torch.ones_like(result));row['gradient']=tensor(torch,source.grad,out)
                if spec['input'].get('legacy_bool'):
                    generator=torch.Generator(device='cpu').manual_seed(1729)
                    rng_before=bytes(generator.get_state().tolist())
                    target=torch.zeros(1,dtype=torch.bool)
                    floating=torch.full((2,2),.25,dtype=torch.float32)
                    target.logical_not_();floating.uniform_(0,1,generator=generator)
                    rng_after=bytes(generator.get_state().tolist())
                    row['legacy_prefix']=dict(mutation_result=tensor(torch,target,out),
                        floating_result=tensor(torch,floating,out),rng_before=rng_before.hex(),rng_after=rng_after.hex())
            assert tensor(torch,source,out)==before and bytes(torch.get_rng_state().tolist())==rng
            row.update(input_unchanged=True,rng_unchanged=True)
            report['rows'].append(row)
            (out/('case-%04d.json'%index)).write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
        assert all(identity(path)==inputs[str(path)] for path in files)
        for row in report['rows']:
            values=[row[key] for key in ('input','result','gradient') if key in row]
            if 'legacy_prefix' in row:values.extend(row['legacy_prefix'][key] for key in ('mutation_result','floating_result'))
            for value in values:
                for field in ('logical','storage'):
                    item=value[field];raw=(out/item['path']).read_bytes()
                    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        report.update(status='OBSERVED_TYPED_INDEX_SELECT' if report['production_oracle_qualified'] else 'OBSERVED_EXPLORATORY_INDEX_SELECT',
            cases=len(report['rows']),inputs_unchanged=True,rng_unchanged=True,
            all_observed_blobs_unchanged=True,rng_sha256=hashlib.sha256(rng).hexdigest())
    except BaseException as error:
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','cases','expected_cases','production_oracle_qualified','error')}),flush=True)
    return 0 if report['status'].startswith('OBSERVED_') else 1


if __name__=='__main__':raise SystemExit(main())
