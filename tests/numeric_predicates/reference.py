#!/usr/bin/env python3
"""Observe native-classification input recipes with the fixed actual Torch wheel."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import stat
import struct
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from numeric_predicates.cases import CASES, CASE_SHA256, raw_input
sha=lambda raw:hashlib.sha256(raw).hexdigest()
SUPPORT='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'


def identity(path):
    path=Path(path).resolve(strict=True);raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),mode=stat.S_IMODE(path.stat().st_mode),sha256=sha(raw))


def tensor(torch,value,folder):
    logical=bytes(value.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8).tolist())
    storage=bytes(value.untyped_storage())
    def blob(raw):
        name=sha(raw)+'.bin';path=folder/'blobs'/name
        if not path.exists():path.write_bytes(raw)
        return dict(path='blobs/'+name,bytes=len(raw),sha256=sha(raw))
    return dict(dtype=str(value.dtype),shape=list(value.shape),strides=list(value.stride()),
        numel=value.numel(),element_size=value.element_size(),storage_bytes=value.untyped_storage().nbytes(),
        storage_offset=value.storage_offset(),version=value._version,requires_grad=value.requires_grad,
        is_leaf=value.is_leaf,is_view=value._is_view(),history_present=value.grad_fn is not None,
        retains_grad=value.retains_grad,logical=blob(logical),storage=blob(storage))


def setup(torch,spec):
    raw=raw_input(spec)
    if spec['dtype']=='wrapped_float64':
        parent=torch.ones(1,dtype=torch.float32,requires_grad=True)
        product=parent*struct.unpack('<d',raw)[0]
        source=product.grad_fn._saved_other
        assert source.dtype==torch.float64 and tuple(source.shape)==()
        assert bytes(source.reshape(-1).view(torch.uint8).tolist())==raw
        return source,(parent,product,source)
    dtype=getattr(torch,spec['dtype'])
    backing=torch.frombuffer(bytearray(raw),dtype=dtype).clone() if raw else torch.empty(0,dtype=dtype)
    if spec['strides'] is None:
        source=backing.reshape(spec['shape']).clone(memory_format=torch.contiguous_format)
        source.requires_grad_(spec['requires_grad'])
    else:
        backing.requires_grad_(spec['requires_grad'])
        source=backing.as_strided(spec['shape'],spec['strides'],spec['offset'])
    return source,(backing,source)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=args.repository.resolve();out=args.output.resolve()
    assert Path.cwd().resolve()==repo and not sys.flags.optimize
    out.mkdir(parents=True,exist_ok=False);(out/'blobs').mkdir()
    files=[Path(__file__).resolve(),HERE/'__init__.py',HERE/'cases.py',
           repo/'tests/pytorch_oracle.py',repo/'compatibility/pytorch-oracle-lock.json',
           repo/'compatibility/upstream-lock.json',repo/'build/pytorch-install.json',Path(sys.executable).resolve()]
    inputs={str(p):identity(p) for p in files}
    assert inputs[str(repo/'tests/pytorch_oracle.py')]['sha256']==SUPPORT
    assert inputs[str(repo/'compatibility/pytorch-oracle-lock.json')]['sha256']==LOCK
    report=dict(schema='heptatorch.numeric-predicates.reference.v1',status='RUNNING',case_sha256=CASE_SHA256,
        expected_cases=len(CASES),rows=[],inputs=inputs,Torch_executed=False,native_executed=False,
        host_expected_values_computed=False,compiler_executed=False,failures=[],
        actual_process=dict(pid=os.getpid(),argv=list(sys.orig_argv),python=identity(sys.executable)))
    try:
        original=sys.argv
        try:
            sys.argv=[str(repo/'tests/pytorch_oracle.py'),str(out/'NO_NATIVE_EXECUTION'),str(out/'oracle-provenance.json'),str(repo/'compatibility/pytorch-oracle-lock.json')]
            ctx=runpy.run_path(str(repo/'tests/pytorch_oracle.py'),run_name='predicate_reference_only')
        finally:sys.argv=original
        torch,provenance=ctx['torch'],ctx['PROVENANCE']
        assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(provenance['checks'].values())
        report.update(Torch_executed=True,oracle_provenance=provenance)
        rng=bytes(torch.get_rng_state().tolist())
        for index,spec in enumerate(CASES):
            source,retained=setup(torch,spec);before=tensor(torch,source,out)
            result=getattr(torch,spec['operation'])(source)
            actual=tensor(torch,result,out)
            assert source._cdata!=result._cdata and source.untyped_storage()._cdata!=result.untyped_storage()._cdata
            assert tensor(torch,source,out)==before and bytes(torch.get_rng_state().tolist())==rng
            row=dict(index=index,spec=spec,input=before,result=actual,accepted=True,input_unchanged=True,rng_unchanged=True,
                fresh_tensor=True,fresh_storage=True)
            report['rows'].append(row)
            (out/('case-%03d.json'%index)).write_text(json.dumps(row,sort_keys=True,indent=2,allow_nan=False)+'\n')
        assert all(identity(p)==inputs[str(p)] for p in files)
        report.update(status='OBSERVED_NUMERIC_PREDICATES',cases=len(report['rows']),inputs_unchanged=True,rng_unchanged=True,
            rng_sha256=sha(rng),all_observed_blobs_unchanged=True)
        for row in report['rows']:
            for value in (row['input'],row['result']):
                for field in ('logical','storage'):
                    b=value[field];raw=(out/b['path']).read_bytes();assert len(raw)==b['bytes'] and sha(raw)==b['sha256']
    except BaseException as error:
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','cases','expected_cases','error')}),flush=True)
    return 0 if report['status']=='OBSERVED_NUMERIC_PREDICATES' else 1

if __name__=='__main__':raise SystemExit(main())
