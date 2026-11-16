#!/usr/bin/env python3
"""Observe real int64 -> binary32 conversion without a Python float intermediate."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import struct
import sys
import traceback

SUPPORT='ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
CHECKS={'python_matches','platform_matches','version_backend_matches',
        'git_matches_oracle_build_pin','distribution_version_matches','direct_url_matches',
        'direct_url_archive_sha256_matches','pip_report_url_matches',
        'pip_report_archive_sha256_matches','pip_report_version_matches','pip_report_requested'}
MASK=(1<<64)-1
MIN=-(1<<63)
MAX=(1<<63)-1


def sha(raw):return hashlib.sha256(raw).hexdigest()


def plan():
    groups=[]
    values=[]
    for sign in (1,-1):
        for delta in (-3,-2,-1,0,1,2,3):values.append(sign*((1<<24)+delta))
    groups.append({'name':'signed-2pow24-neighbors','signed_inputs':values})
    for exponent in (53,62):
        values=[];shift=exponent-23;half=1<<(shift-1)
        for mantissa in ((1<<23),(1<<23)+1,(1<<23)+2,(1<<23)+3,(1<<23)+17):
            center=mantissa<<shift
            for delta in (-half-1,-half,-half+1,-1,0,1,half-1,half,half+1):
                for sign in (1,-1):
                    value=sign*(center+delta)
                    if MIN<=value<=MAX:values.append(value)
        groups.append({'name':f'signed-2pow{exponent}-half-ulp','signed_inputs':values})
    groups.append({'name':'extrema-and-small-integers',
                   'signed_inputs':[MIN,MIN+1,MIN+2,MIN+3,MIN+((1<<39)-1),
                                    MAX,MAX-1,MAX-2,MAX-3,MAX-((1<<39)-1),
                                    -17,-3,-2,-1,0,1,2,3,17]})
    value=0x6A09E667F3BCC909;values=[]
    for _ in range(1024):
        value^=(value<<13)&MASK;value^=value>>7;value^=(value<<17)&MASK;value&=MASK
        values.append(value if value<(1<<63) else value-(1<<64))
    groups.append({'name':'deterministic-xorshift64-1024','signed_inputs':values})
    assert len(groups)==5 and all(MIN<=x<=MAX and isinstance(x,int) for g in groups for x in g['signed_inputs'])
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','lock','repo','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    assert Path.cwd().resolve()==a.repo.resolve()
    a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists()
    inputs={Path(__file__).resolve():Path(__file__).read_bytes(),a.reference.resolve():a.reference.read_bytes(),a.lock.resolve():a.lock.read_bytes()}
    assert sha(inputs[a.reference.resolve()])==SUPPORT and sha(inputs[a.lock.resolve()])==LOCK
    planned=plan()
    common={'schema':'heptatorch.int64-f32-rne-observations.v1',
            'probe_sha256':sha(inputs[Path(__file__).resolve()]),'primary_source_ref':'31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457',
            'case_plan':planned,'expected_cases':len(planned),'expected_integers':sum(len(g['signed_inputs']) for g in planned),
            'integer_input_construction':'Python integers -> torch.int64 Tensor directly; no Python float/float64 intermediate.',
            'operations':['Tensor.to(torch.float32)','torch.add(int64_tensor, torch.zeros((), dtype=torch.float32))'],
            'native_executed':False,'native_values_compared':0,'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590}
    if a.prepare_only:
        report={**common,'status':'PREPARED_NOT_RUN','torch_executed':False,'cases':0,'rows':[]}
    else:
        old=sys.argv
        try:
            sys.argv=[str(a.reference),str(a.repo/'reference-only-no-native-execution'),str(a.output.with_suffix('.provenance.json')),str(a.lock)]
            ctx=runpy.run_path(str(a.reference),run_name='int64_f32_rne_provenance_only')
        finally:sys.argv=old
        t,prov=ctx['torch'],ctx['PROVENANCE']
        assert prov['status']=='PASS' and set(prov['checks'])==CHECKS and all(v is True for v in prov['checks'].values())
        rows=[];errors=[]
        for index,g in enumerate(planned):
            try:
                before=bytes(t.get_rng_state().tolist())
                value=t.tensor(g['signed_inputs'],dtype=t.int64)
                direct=value.to(dtype=t.float32)
                promoted=t.add(value,t.zeros((),dtype=t.float32))
                after=bytes(t.get_rng_state().tolist())
                input_raw=bytes(value.view(t.uint8).tolist())
                direct_raw=bytes(direct.view(t.uint8).tolist())
                promoted_raw=bytes(promoted.view(t.uint8).tolist())
                rows.append({'index':index,'name':g['name'],'status':'OBSERVED',
                    'signed_inputs':g['signed_inputs'],'input_i64_le_hex':input_raw.hex(),
                    'input_bytes':len(input_raw),'direct_f32_le_hex':direct_raw.hex(),
                    'promoted_add_f32_le_hex':promoted_raw.hex(),
                    'direct_u32_words':list(struct.unpack('<'+'I'*direct.numel(),direct_raw)),
                    'promoted_u32_words':list(struct.unpack('<'+'I'*promoted.numel(),promoted_raw)),
                    'direct_dtype':str(direct.dtype),'promoted_dtype':str(promoted.dtype),
                    'direct_strides':list(direct.stride()),'promoted_strides':list(promoted.stride()),
                    'direct_version':direct._version,'promoted_version':promoted._version,
                    'rng_before_hex':before.hex(),'rng_after_hex':after.hex(),'rng_unchanged':before==after})
            except BaseException as error:
                event={'index':index,'name':g['name'],'status':'PROBE_ERROR','error':repr(error),'traceback':traceback.format_exc()}
                rows.append(event);errors.append(event)
        report={**common,'status':'OBSERVED' if not errors else 'PROBE_ERROR',
                'torch_executed':True,'oracle_provenance':prov,'cases':len(rows),'rows':rows,
                'probe_errors':errors,'probe_error_count':len(errors),
                'cpu_capability':t.backends.cpu.get_cpu_capability(),'threads':t.get_num_threads()}
    assert all(p.read_bytes()==raw for p,raw in inputs.items())
    report['source_files_unchanged']=True
    report['inputs_sha256']={str(p):sha(raw) for p,raw in inputs.items()}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','expected_integers','probe_sha256')}))
    return int(report['status']=='PROBE_ERROR')


if __name__=='__main__':raise SystemExit(main())
