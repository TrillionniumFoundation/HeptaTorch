#!/usr/bin/env python3
"""Observe same public F64 arithmetic only through the fixed real Torch CPU wheel."""
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
from same_f64_arithmetic.cases import CASES,CASE_SHA256,DTYPES,OPERATIONS,raw_input,overwrite_words,ORIGINAL_CASE_COUNT,ORIGINAL_CASE_SHA256
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
        width=value.element_size();assert width in (2,4,8)
        bits=value.view({2:torch.int16,4:torch.int32,8:torch.int64}[width])
        indices=itertools.product(*(range(size) for size in value.shape))
        for coordinates,word in zip(indices,words):
            bits[coordinates]=word if word < (1 << (8*width-1)) else word-(1 << (8*width))
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'blobs').mkdir()
    assert Path.cwd().resolve()==ROOT and not sys.flags.optimize
    files=[Path(__file__),ROOT/'tests/same_f64_arithmetic/__init__.py',ROOT/'tests/same_f64_arithmetic/cases.py',
        ROOT/'tests/numeric_predicates/reference.py',ROOT/'tests/numeric_predicates/cases.py',ROOT/'tests/numeric_predicates/__init__.py',
        ROOT/'tests/pytorch_oracle.py',ROOT/'compatibility/pytorch-oracle-lock.json',ROOT/'compatibility/upstream-lock.json',
        ROOT/'build/pytorch-install.json',Path(sys.executable).resolve()]
    inputs={str(p.resolve()):identity(p) for p in files}
    assert inputs[str(ROOT/'tests/pytorch_oracle.py')]['sha256']==SUPPORT
    assert inputs[str(ROOT/'compatibility/pytorch-oracle-lock.json')]['sha256']==LOCK
    inherited_capability=os.environ.get('ATEN_CPU_CAPABILITY')
    os.environ['ATEN_CPU_CAPABILITY']=REQUESTED_CPU_CAPABILITY
    report=dict(schema='heptatorch.same-f64-arithmetic.reference.v1',status='RUNNING',case_sha256=CASE_SHA256,expected_cases=len(CASES),
        original_case_count=ORIGINAL_CASE_COUNT,original_case_sha256=ORIGINAL_CASE_SHA256,original_case_prefix_unchanged=True,
        rows=[],inputs=inputs,Torch_executed=False,native_executed=False,host_expected_values_computed=False,failures=[],
        same_f64_nan_payload_qualified=True,legacy_same_f64_route_unchanged=False,
        actual_process=dict(pid=os.getpid(),argv=list(sys.orig_argv),python=identity(sys.executable)),
        requested_cpu_capability=REQUESTED_CPU_CAPABILITY,cpu_environment=os.environ.get('ATEN_CPU_CAPABILITY'),
        inherited_cpu_capability=inherited_capability)
    try:
        original=sys.argv
        try:
            sys.argv=[str(ROOT/'tests/pytorch_oracle.py'),str(out/'NO_NATIVE_EXECUTION'),str(out/'oracle-provenance.json'),str(ROOT/'compatibility/pytorch-oracle-lock.json')]
            ctx=runpy.run_path(str(ROOT/'tests/pytorch_oracle.py'),run_name='same_f64_arithmetic_reference_only')
        finally:sys.argv=original
        torch,provenance=ctx['torch'],ctx['PROVENANCE']
        assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(provenance['checks'].values())
        capability=torch.backends.cpu.get_cpu_capability();assert capability==CPU_CAPABILITY,capability
        assert torch.get_num_threads()==1
        report.update(Torch_executed=True,oracle_provenance=provenance,cpu_capability=capability,torch_threads=1)
        rng=bytes(torch.get_rng_state().tolist())
        for index,spec in enumerate(CASES):
            left=setup(torch,spec['left']);right=left if spec.get('same_handle') else setup(torch,spec['right'])
            before=[tensor(torch,left,out),tensor(torch,right,out)]
            row=dict(index=index,spec=spec,inputs=before)
            try:
                result=getattr(torch,OPERATIONS[spec['operation']])(left,right)
                assert not spec.get('expected_error'), 'expected actual Torch rejection did not occur'
                row.update(result=tensor(torch,result,out),accepted=True,
                    fresh_tensor=result._cdata not in (left._cdata,right._cdata),
                    fresh_storage=result.untyped_storage()._cdata not in (left.untyped_storage()._cdata,right.untyped_storage()._cdata))
                assert row['fresh_tensor'] and row['fresh_storage']
            except RuntimeError as error:
                expected=spec.get('expected_error')
                assert expected in ('broadcast','bool_subtraction'),str(error)
                if expected=='broadcast':
                    assert type(error) is RuntimeError and 'size' in str(error) and 'match' in str(error),str(error)
                else:
                    assert type(error) is NotImplementedError and 'bool' in str(error).lower() and 'subtraction' in str(error).lower(),str(error)
                row.update(accepted=False,error=dict(type=type(error).__name__,message=str(error)))
            assert [tensor(torch,left,out),tensor(torch,right,out)]==before
            assert bytes(torch.get_rng_state().tolist())==rng
            row.update(inputs_unchanged=True,rng_unchanged=True)
            report['rows'].append(row)
            (out/('case-%03d.json'%index)).write_text(json.dumps(row,sort_keys=True,indent=2,allow_nan=False)+'\n')
        assert all(identity(p)==inputs[str(p.resolve())] for p in files)
        report.update(status='OBSERVED_SAME_F64_ARITHMETIC',cases=len(report['rows']),inputs_unchanged=True,rng_unchanged=True,rng_sha256=sha(rng))
    except BaseException as error:
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','cases','expected_cases','error')}),flush=True)
    return 0 if report['status']=='OBSERVED_SAME_F64_ARITHMETIC' else 1

if __name__=='__main__':raise SystemExit(main())
