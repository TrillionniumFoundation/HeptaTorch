#!/usr/bin/env python3
"""Observe Half negation only through the fixed real Torch CPU wheel."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from half_neg.cases import CASES,CASE_SHA256,DTYPES,raw_input,overwrite_words
from numeric_predicates.reference import tensor,identity

SUPPORT='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
CPU_CAPABILITY='AVX2'
REQUESTED_CPU_CAPABILITY='avx2'
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def setup(torch,spec):
    if 'cast' in spec:
        value=setup(torch,spec['source']).to(getattr(torch,DTYPES[spec['cast']]))
    else:
        raw=raw_input(spec);dtype=getattr(torch,DTYPES[spec['dtype']])
        flat=torch.frombuffer(bytearray(raw),dtype=dtype).clone() if raw else torch.empty(0,dtype=dtype)
        value=(flat.reshape(spec['shape']).clone(memory_format=torch.contiguous_format) if spec['strides'] is None
            else flat.as_strided(spec['shape'],spec['strides'],spec['offset']))
    # These are literal input bits assigned through a public integer dtype view.
    # Both this assignment and native raw-set12 increment the shared version.
    words=overwrite_words(spec)
    if words:
        bits=value.view(torch.uint16);assert value.element_size()==2
        indices=itertools.product(*(range(size) for size in value.shape))
        for coordinates,word in zip(indices,words):bits[coordinates]=word
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'blobs').mkdir()
    assert Path.cwd().resolve()==ROOT and not sys.flags.optimize
    files=[Path(__file__),ROOT/'tests/half_neg/__init__.py',ROOT/'tests/half_neg/cases.py',
        ROOT/'tests/numeric_predicates/reference.py',ROOT/'tests/numeric_predicates/cases.py',ROOT/'tests/numeric_predicates/__init__.py',
        ROOT/'tests/pytorch_oracle.py',ROOT/'compatibility/pytorch-oracle-lock.json',ROOT/'compatibility/upstream-lock.json',
        ROOT/'build/pytorch-install.json',Path(sys.executable).resolve()]
    inputs={str(p.resolve()):identity(p) for p in files}
    assert inputs[str(ROOT/'tests/pytorch_oracle.py')]['sha256']==SUPPORT
    assert inputs[str(ROOT/'compatibility/pytorch-oracle-lock.json')]['sha256']==LOCK
    inherited=os.environ.get('ATEN_CPU_CAPABILITY');os.environ['ATEN_CPU_CAPABILITY']=REQUESTED_CPU_CAPABILITY
    report=dict(schema='heptatorch.half-neg.reference.v1',status='RUNNING',case_sha256=CASE_SHA256,expected_cases=len(CASES),
        rows=[],inputs=inputs,Torch_executed=False,native_executed=False,host_expected_values_computed=False,failures=[],
        actual_process=dict(pid=os.getpid(),argv=list(sys.orig_argv),python=identity(sys.executable)),
        requested_cpu_capability=REQUESTED_CPU_CAPABILITY,cpu_environment=os.environ.get('ATEN_CPU_CAPABILITY'),
        inherited_cpu_capability=inherited)
    try:
        original=sys.argv
        try:
            sys.argv=[str(ROOT/'tests/pytorch_oracle.py'),str(out/'NO_NATIVE_EXECUTION'),str(out/'oracle-provenance.json'),str(ROOT/'compatibility/pytorch-oracle-lock.json')]
            ctx=runpy.run_path(str(ROOT/'tests/pytorch_oracle.py'),run_name='half_neg_reference_only')
        finally:sys.argv=original
        torch,provenance=ctx['torch'],ctx['PROVENANCE']
        assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(provenance['checks'].values())
        capability=torch.backends.cpu.get_cpu_capability();assert capability==CPU_CAPABILITY,capability
        assert torch.get_num_threads()==1
        report.update(Torch_executed=True,oracle_provenance=provenance,cpu_capability=capability,torch_threads=1)
        rng=bytes(torch.get_rng_state().tolist())
        for index,spec in enumerate(CASES):
            source=setup(torch,spec['input']);before=tensor(torch,source,out)
            result=torch.neg(source)
            row=dict(index=index,spec=spec,input=before,result=tensor(torch,result,out),accepted=True,
                fresh_tensor=result._cdata!=source._cdata,
                fresh_storage=result.untyped_storage()._cdata!=source.untyped_storage()._cdata)
            assert row['fresh_tensor'] and row['fresh_storage']
            assert tensor(torch,source,out)==before and bytes(torch.get_rng_state().tolist())==rng
            row.update(input_unchanged=True,rng_unchanged=True)
            report['rows'].append(row)
            (out/('case-%03d.json'%index)).write_text(json.dumps(row,sort_keys=True,indent=2,allow_nan=False)+'\n')
        assert all(identity(p)==inputs[str(p.resolve())] for p in files)
        report.update(status='OBSERVED_HALF_NEG',cases=len(report['rows']),inputs_unchanged=True,rng_unchanged=True,rng_sha256=sha(rng))
    except BaseException as error:
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','cases','expected_cases','error')}),flush=True)
    return 0 if report['status']=='OBSERVED_HALF_NEG' else 1


if __name__=='__main__':raise SystemExit(main())
