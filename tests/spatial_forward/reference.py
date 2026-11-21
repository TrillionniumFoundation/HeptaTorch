#!/usr/bin/env python3
"""Observe the fixed spatial/reduction cases with the actual pinned Torch CPU wheel."""
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
from spatial_forward.cases import CASES,CASE_SHA256,raw_input,seed_raw
from numeric_predicates.reference import tensor,identity
import pinned_reference_custody as custody


def source_files():
    return [ROOT/'tests/spatial_forward'/name for name in ('__init__.py','cases.py','reference.py')]+[
        ROOT/'tests/pinned_reference_custody.py',ROOT/'tests/numeric_predicates/reference.py',
        ROOT/'tests/numeric_predicates/cases.py',ROOT/'tests/numeric_predicates/__init__.py',
        ROOT/'tests/pytorch_oracle.py',ROOT/'compatibility/pytorch-oracle-lock.json',
        ROOT/'compatibility/upstream-lock.json',ROOT/'compatibility/pytorch-reference-requirements.txt',
        Path(sys.executable).resolve()]+custody.source_files(ROOT)


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
        lock_sha256=custody.LOCK,version=str(torch.__version__),git_version=torch.version.git_version,
        package=identity(torch.__file__),extension=identity(torch._C.__file__),
        metadata_location=str(distribution._path),direct_url_present=distribution.read_text('direct_url.json') is not None,
        full_installation_provenance_verified=False)


def setup(torch,spec):
    if spec is None:return None,None
    raw=raw_input(spec)
    backing=torch.frombuffer(bytearray(raw),dtype=torch.float32).clone() if raw else torch.empty(0,dtype=torch.float32)
    if spec['strides'] is None:
        source=backing.reshape(spec['shape']).clone(memory_format=torch.contiguous_format)
        source.requires_grad_(spec['requires_grad']);base=source
    else:
        backing.requires_grad_(spec['requires_grad'])
        source=backing.as_strided(spec['shape'],spec['strides'],spec['offset']);base=backing
    return source,base


def operate(torch,spec,values):
    op,args=spec['operation'],spec['args'];F=torch.nn.functional
    if op=='conv2d':return F.conv2d(*values,stride=args['stride'],padding=args['padding'],dilation=args['dilation'],groups=args['groups'])
    if op=='pool2d':
        return F.avg_pool2d(values[0],args['kernel'],args['stride'],args['padding'],count_include_pad=args['include']) if args['average'] else F.max_pool2d(values[0],args['kernel'],args['stride'],args['padding'])
    if op=='reduce_axis':
        axis=args['axis'] if args['axis']<1<<63 else args['axis']-(1<<64)
        return values[0].mean(axis,keepdim=args['keep']) if args['mean'] else values[0].sum(axis,keepdim=args['keep'])
    if op=='layernorm':return F.layer_norm(values[0],(values[0].shape[-1],),values[1],values[2],struct.unpack('<f',struct.pack('<I',args['epsilon_bits']))[0])
    if op=='batchnorm_eval':return F.batch_norm(*values,training=False,eps=struct.unpack('<f',struct.pack('<I',args['epsilon_bits']))[0])
    if op=='linear':return F.linear(*values)
    if op=='cross_entropy_probs':return F.cross_entropy(*values)
    raise ValueError(op)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-unverified-local',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'blobs').mkdir()
    assert Path.cwd().resolve()==ROOT and not sys.flags.optimize
    report=dict(schema='heptatorch.spatial-forward.reference.v1',status='RUNNING',case_sha256=CASE_SHA256,
        expected_cases=len(CASES),rows=[],Torch_executed=False,native_executed=False,compiler_executed=False,
        host_expected_values_computed=False,production_oracle_qualified=False,failures=[],
        actual_process=dict(pid=os.getpid(),argv=sys.orig_argv,python=identity(sys.executable)))
    files=source_files();capture=None
    try:
        if not args.allow_unverified_local:files.extend([ROOT/'build/pytorch-install.json',ROOT/'build/pytorch-reference-support-install.json'])
        inputs={str(path):identity(path) for path in files};report['inputs']=inputs
        assert inputs[str(ROOT/'tests/pytorch_oracle.py')]['sha256']==custody.SUPPORT
        assert inputs[str(ROOT/'compatibility/pytorch-oracle-lock.json')]['sha256']==custody.LOCK
        capture=custody.begin(ROOT,out,__file__,args.allow_unverified_local)
        if args.allow_unverified_local:
            import torch
            provenance=local_identity(torch)
        else:
            old=sys.argv
            try:
                sys.argv=[str(ROOT/'tests/pytorch_oracle.py'),str(out/'NO_NATIVE_EXECUTION'),str(out/'oracle-provenance.json'),str(ROOT/'compatibility/pytorch-oracle-lock.json')]
                context=runpy.run_path(str(ROOT/'tests/pytorch_oracle.py'),run_name='spatial_forward_reference_only')
            finally:sys.argv=old
            torch,provenance=context['torch'],context['PROVENANCE']
            assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(provenance['checks'].values())
        custody.finish_capture(capture)
        report.update(oracle_provenance=provenance,production_oracle_qualified=not args.allow_unverified_local)
        report['reference_custody']=custody.capture(ROOT,out,__file__,torch,provenance,args.allow_unverified_local,capture)
        torch.set_num_threads(1);torch.manual_seed(1729);torch.backends.mkldnn.enabled=False
        torch.use_deterministic_algorithms(True)
        assert torch.backends.cpu.get_cpu_capability()=='AVX2'
        report.update(Torch_executed=True,cpu_capability='AVX2',torch_threads=1)
        rng=bytes(torch.get_rng_state().tolist())
        for index,spec in enumerate(CASES):
            pairs=[setup(torch,item) for item in spec['inputs']]
            values=[pair[0] for pair in pairs];bases=[pair[1] for pair in pairs]
            before=[tensor(torch,value,out) if value is not None else None for value in values]
            row=dict(index=index,spec=spec,inputs=before)
            result=operate(torch,spec,values)
            row.update(accepted=True,result=tensor(torch,result,out),
                fresh_tensor=all(value is None or result._cdata!=value._cdata for value in values),
                fresh_storage=all(value is None or result.untyped_storage()._cdata!=value.untyped_storage()._cdata for value in values))
            assert row['fresh_tensor'] and row['fresh_storage']
            if result.requires_grad:
                raw=seed_raw(result.numel());seed=torch.frombuffer(bytearray(raw),dtype=torch.float32).reshape(result.shape).clone(memory_format=torch.contiguous_format) if raw else torch.empty_like(result)
                result.backward(seed);row['seed']=tensor(torch,seed,out)
                row['gradients']=[tensor(torch,base.grad,out) if base is not None and base.requires_grad else None for base in bases]
                assert all(base is None or not base.requires_grad or base.grad is not None for base in bases)
            assert [tensor(torch,value,out) if value is not None else None for value in values]==before
            assert bytes(torch.get_rng_state().tolist())==rng
            row.update(inputs_unchanged=True,rng_unchanged=True);report['rows'].append(row)
            (out/('case-%04d.json'%index)).write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
        custody.finish_reference(capture)
        assert all(identity(path)==inputs[str(path)] for path in files)
        for row in report['rows']:
            values=row['inputs']+[row['result']]+([row['seed']]+row['gradients'] if 'seed' in row else [])
            for value in values:
                if value is not None:
                    for field in ('logical','storage'):
                        item=value[field];raw=(out/item['path']).read_bytes()
                        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        report.update(status='OBSERVED_SPATIAL_FORWARD' if report['production_oracle_qualified'] else 'OBSERVED_EXPLORATORY_SPATIAL_FORWARD',
            cases=len(report['rows']),inputs_unchanged=True,rng_unchanged=True,all_observed_blobs_unchanged=True,
            rng_sha256=hashlib.sha256(rng).hexdigest())
    except BaseException as error:
        if capture is not None:custody.finish_capture(capture)
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:report.get(key) for key in ('status','cases','expected_cases','production_oracle_qualified','error')}),flush=True)
    return 0 if report['status'].startswith('OBSERVED_') else 1


if __name__=='__main__':raise SystemExit(main())
