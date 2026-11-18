#!/usr/bin/env python3
"""Observe ordinary public torch.float64 Tensors on the actual pinned wheel."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import traceback

HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('public_f64_literal_recipes',HERE/'cases.py')
cases=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(cases)
SUPPORT_SHA='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def tensor(torch,value):
    raw=bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
    return dict(dtype=str(value.dtype),shape=list(value.shape),strides=list(value.stride()),
        numel=value.numel(),element_size=value.element_size(),storage_bytes=value.untyped_storage().nbytes(),
        storage_offset=value.storage_offset(),version=value._version,requires_grad=value.requires_grad,
        is_leaf=value.is_leaf,is_view=value._is_view(),
        grad_fn=None if value.grad_fn is None else type(value.grad_fn).__name__,logical_bytes_hex=raw.hex())


def dense(torch,spec):
    raw=cases.raw_input(spec)
    if raw:
        # The final clone makes an ordinary owned Tensor at the target shape,
        # with no view relationship and version zero. No numeric conversion.
        source=torch.frombuffer(bytearray(raw),dtype=torch.float64).reshape(spec['shape']).clone()
    else:source=torch.empty(spec['shape'],dtype=torch.float64)
    assert source.dtype==torch.float64 and not source.requires_grad and not source._is_view()
    assert source.is_leaf and source.grad_fn is None and source._version==0
    assert bytes(source.detach().reshape(-1).view(torch.uint8).tolist())==raw
    return source


def operation(torch,source,spec):
    if spec['operation']=='sum_bool':
        return torch.ops.aten.sum.dim_IntList(source,spec['axes'],keepdim=spec['keepdim'],dtype=torch.bool)
    return getattr(torch.ops.aten,spec['operation']).dims(source,spec['axes'],keepdim=spec['keepdim'])


def observe(torch,spec,source=None):
    if source is None:source=dense(torch,spec)
    before=tensor(torch,source);rng=bytes(torch.get_rng_state().tolist())
    try:
        result=operation(torch,source,spec)
        row=dict(accepted=True,result=tensor(torch,result),error=None,
            fresh_tensor=result._cdata!=source._cdata,
            fresh_storage=result.untyped_storage()._cdata!=source.untyped_storage()._cdata)
        assert row['fresh_tensor'] and row['fresh_storage']
    except (IndexError,RuntimeError) as error:
        row=dict(accepted=False,result=None,error=dict(type=type(error).__name__,message=str(error)))
    assert before==tensor(torch,source) and rng==bytes(torch.get_rng_state().tolist())
    return dict(spec=spec,input=before,input_unchanged=True,rng_unchanged=True,**row)


def observe_boundary(torch,spec):
    dense_spec=cases.recipe('boundary',[cases.Z,cases.NZ,cases.TINY,cases.QNAN],[2,2],None)
    source=dense(torch,dense_spec)
    if spec['kind']=='requires_grad_public_float64':source.requires_grad_(True)
    elif spec['kind']=='as_strided_public_float64':source=source.as_strided([2,2],[1,2],0)
    else:raise AssertionError(spec)
    recipe=dict(dense_spec,operation=spec['operation'],name=spec['name'])
    row=observe(torch,recipe,source)
    row['scope']='OUTSIDE_DENSE_NO_GRAD_NATIVE_EXTENSION'
    row['native_comparison_required_here']=False
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();repo=args.repository.resolve();out=args.output.resolve()
    assert not sys.flags.optimize and Path.cwd().resolve()==repo
    out.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__).resolve(),HERE/'cases.py',repo/'tests/pytorch_oracle.py',
           repo/'compatibility/pytorch-oracle-lock.json',repo/'compatibility/upstream-lock.json',
           repo/'build/pytorch-install.json']
    inputs={str(path):dict(sha256=sha(path.read_bytes()),bytes=path.stat().st_size) for path in paths}
    assert inputs[str(repo/'tests/pytorch_oracle.py')]['sha256']==SUPPORT_SHA
    assert inputs[str(repo/'compatibility/pytorch-oracle-lock.json')]['sha256']==LOCK_SHA
    report=dict(schema='heptatorch.public-f64-dense-logical.reference.v1',status='RUNNING',
        expected_cases=192,expected_allocation_observations=2,expected_boundary_observations=6,
        case_sha256=cases.CASE_SHA256,allocation_sha256=cases.ALLOCATION_SHA256,
        boundary_sha256=cases.BOUNDARY_SHA256,rows=[],allocation_rows=[],boundary_rows=[],
        torch_executed=False,native_executed=False,inputs=inputs,
        wrapped_scalar_used=False,original147_modified=False,full_schema_closure=False)
    code=0
    try:
        old=sys.argv
        try:
            sys.argv=[str(repo/'tests/pytorch_oracle.py'),str(out/'NO_NATIVE_EXECUTION'),
                str(out/'oracle-provenance.json'),str(repo/'compatibility/pytorch-oracle-lock.json')]
            ctx=runpy.run_path(str(repo/'tests/pytorch_oracle.py'),run_name='public_f64_reference_only')
        finally:sys.argv=old
        torch=ctx['torch'];provenance=ctx['PROVENANCE']
        assert provenance['status']=='PASS' and len(provenance['checks'])==11
        assert all(value is True for value in provenance['checks'].values())
        report.update(torch_executed=True,oracle_provenance=provenance)
        rng=bytes(torch.get_rng_state().tolist())
        for index,spec in enumerate(cases.CASES):report['rows'].append(dict(index=index,**observe(torch,spec)))
        for spec in cases.ALLOCATION_CASES:report['allocation_rows'].append(observe(torch,spec))
        for spec in cases.BOUNDARY_OBSERVATIONS:report['boundary_rows'].append(observe_boundary(torch,spec))
        assert rng==bytes(torch.get_rng_state().tolist())
        assert all(sha(path.read_bytes())==inputs[str(path)]['sha256'] for path in paths)
        report.update(status='OBSERVED_PUBLIC_F64_DENSE_REFERENCE',cases=192,
            allocation_observations=2,boundary_observations=6,total_actual_observations=200,
            rng_unchanged=True,inputs_unchanged=True,rng_hex=rng.hex(),rng_sha256=sha(rng),
            source_sha256=inputs[str(Path(__file__).resolve())]['sha256'])
    except BaseException as error:
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),
                                              traceback=traceback.format_exc()));code=1
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','cases','total_actual_observations','error')}))
    return code


if __name__=='__main__':raise SystemExit(main())
