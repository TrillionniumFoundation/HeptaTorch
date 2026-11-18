#!/usr/bin/env python3
"""Observe all logical-reduction fixtures on the repository's pinned CPU wheel."""
import argparse
import hashlib
import json
import runpy
import struct
import sys
import traceback
from pathlib import Path

from cases import CASES, CASE_SHA256, canonical, raw_input

SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
sha = lambda b: hashlib.sha256(b).hexdigest()


def observe(torch, spec):
    def tensor(value):
        raw = bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
        return dict(dtype=str(value.dtype), shape=list(value.shape), strides=list(value.stride()),
                    numel=value.numel(), element_size=value.element_size(),
                    storage_bytes=value.untyped_storage().nbytes(), storage_offset=value.storage_offset(),
                    version=value._version, requires_grad=value.requires_grad, is_leaf=value.is_leaf,
                    is_view=value._is_view(), grad_fn=None if value.grad_fn is None else type(value.grad_fn).__name__,
                    logical_bytes_hex=raw.hex())
    raw = raw_input(spec)
    if spec['dtype'] == 'float64':
        parent = torch.ones(1, dtype=torch.float32, requires_grad=True)
        product = parent * struct.unpack('<d', raw)[0]
        source = product.grad_fn._saved_other
        assert source.dtype == torch.float64 and tuple(source.shape) == ()
        assert bytes(source.detach().reshape(-1).view(torch.uint8).tolist()) == raw
        backing = source
    else:
        backing = (torch.frombuffer(bytearray(raw), dtype=getattr(torch, spec['dtype'])).clone()
                   if raw else torch.empty(0, dtype=getattr(torch, spec['dtype'])))
        backing.requires_grad_(spec['requires_grad'])
        source = backing.as_strided(spec['shape'], spec['strides'], spec['offset'])
    before = tensor(source)
    rng = bytes(torch.get_rng_state().tolist())
    try:
        if spec['operation'] == 'sum_bool':
            result = torch.ops.aten.sum.dim_IntList(source, spec['axes'], keepdim=spec['keepdim'], dtype=torch.bool)
        else:
            result = getattr(torch.ops.aten, spec['operation']).dims(source, spec['axes'], keepdim=spec['keepdim'])
        row = dict(accepted=True, result=tensor(result), error=None,
                   fresh_tensor=result._cdata != source._cdata,
                   fresh_storage=result.untyped_storage()._cdata != source.untyped_storage()._cdata)
        assert row['fresh_tensor'] and row['fresh_storage']
    except (IndexError, RuntimeError) as error:
        row = dict(accepted=False, result=None, error=dict(type=type(error).__name__, message=str(error)))
    assert before == tensor(source)
    assert rng == bytes(torch.get_rng_state().tolist())
    return dict(spec=spec, input=before, input_unchanged=True, rng_unchanged=True, **row)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo=args.repository.resolve();out=args.output.resolve()
    assert Path.cwd().resolve()==repo and not sys.flags.optimize
    out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__).resolve(),Path(__file__).with_name('cases.py').resolve(),
           repo/'tests/pytorch_oracle.py',repo/'compatibility/pytorch-oracle-lock.json',
           repo/'compatibility/upstream-lock.json',repo/'build/pytorch-install.json']
    inputs={str(p):sha(p.read_bytes()) for p in files}
    assert inputs[str(repo/'tests/pytorch_oracle.py')]==SUPPORT_SHA
    assert inputs[str(repo/'compatibility/pytorch-oracle-lock.json')]==LOCK_SHA
    report=dict(schema='heptatorch.logical-reductions.reference.v1',status='RUNNING',case_sha256=CASE_SHA256,
                expected_cases=len(CASES),rows=[],inputs=inputs,torch_executed=False,native_executed=False)
    code=0
    try:
        old=sys.argv
        try:
            sys.argv=[str(repo/'tests/pytorch_oracle.py'),str(out/'NO_NATIVE_EXECUTION'),str(out/'oracle-provenance.json'),
                      str(repo/'compatibility/pytorch-oracle-lock.json')]
            ctx=runpy.run_path(str(repo/'tests/pytorch_oracle.py'),run_name='logical_reference_only')
        finally:sys.argv=old
        torch=ctx['torch'];provenance=ctx['PROVENANCE']
        assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(provenance['checks'].values())
        report.update(torch_executed=True,oracle_provenance=provenance)
        rng=bytes(torch.get_rng_state().tolist())
        for i,spec in enumerate(CASES):
            row=observe(torch,spec);row['index']=i;report['rows'].append(row)
        assert rng==bytes(torch.get_rng_state().tolist())
        assert all(sha(p.read_bytes())==inputs[str(p)] for p in files)
        report.update(status='OBSERVED',cases=len(report['rows']),rng_unchanged=True,inputs_unchanged=True,
                      rng_sha256=sha(rng),rng_hex=rng.hex(),source_sha256=inputs[str(Path(__file__).resolve())])
    except BaseException as error:
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        code=1
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','cases','expected_cases','error')}))
    return code

if __name__=='__main__':raise SystemExit(main())
