#!/usr/bin/env python3
"""Check native admission against real C++ double and exact integer threshold."""
import argparse,hashlib,importlib.util,json,shutil,struct,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True

SHA=lambda b:hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('executable','rng-package','cpp-reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--torch-observation',type=Path)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    package=a.rng_package.resolve();source=Path(__file__).with_name('range_reference.cpp')
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    wire=module('range_wire',package/'protocol.py');mt=module('range_mt',package/'reference.py');fp=module('range_fp',package/'reference_fp.py')
    corpus=module('range_probe',Path(__file__).with_name('uniform_range_probe.py')).corpus()
    binary=out/'double-range-reference';compiled=subprocess.run([shutil.which('g++'),'-std=c++17','-O2',str(source),'-o',str(binary)],capture_output=True)
    (out/'reference-build.stdout').write_bytes(compiled.stdout);(out/'reference-build.stderr').write_bytes(compiled.stderr)
    assert compiled.returncode==0 and not compiled.stderr
    inputs=b''.join(struct.pack('<2I',lo,hi) for lo,hi in corpus)
    response=subprocess.run([str(binary)],input=inputs,capture_output=True,check=True);assert not response.stderr and len(response.stdout)==16*len(corpus)
    (out/'reference-input.bin').write_bytes(inputs);(out/'reference-output.bin').write_bytes(response.stdout)
    oracle=[struct.unpack_from('<2Q',response.stdout,i*16) for i in range(len(corpus))]
    torch_rows=None
    if a.torch_observation:
        assert SHA(a.torch_observation.read_bytes())=='467e2de683758881175bbfb82446a9d174fac91feb686baea7f9cbe40de1d70a'
        actual=json.loads(a.torch_observation.read_text());assert actual['status']=='OBSERVED'
        assert actual['probe_sha256']=='a607164e51891c0c45922b5cd596e14fb8ecc6f176e0585757aebb56e106e29f'
        assert actual['oracle_provenance']['status']=='PASS' and len(actual['oracle_provenance']['checks'])==11 and all(actual['oracle_provenance']['checks'].values())
        torch_rows={(r['case'],r['elements']):r for r in actual['rows']}
    s=wire.Session(a.executable,directory=out/'frames');rows=[];rollback=0
    # Unit 2^-149 represents every finite binary32 endpoint as an integer.
    # FLT_MAX is even in binary64; the upper RNE boundary is inclusive.
    def integer(word):
        m,e=fp.parts(word);return m<<(e+149)
    threshold=integer(0x7f7fffff)+(1<<(74+149))
    for i,(lo,hi) in enumerate(corpus):
        exact=integer(hi)-integer(lo)<=threshold;assert exact==bool(oracle[i][0])
        for count in (0,4):
            r=s.run([wire.full((count,),-7),(60,99)]);t,g=[x['value'] for x in r];model=mt.Reference(a.cpp_reference.resolve(),99)
            if exact:s.run([(68,t,g,lo,hi)])
            else:s.error([(68,t,g,lo,hi)],7);rollback+=1
            result=s.run([(13,t),(63,g),(39,t,0,count),(53,)])
            expected=[fp.uniform(model.word(),lo,hi) for _ in range(count)] if exact else [wire.bits(-7)]*count
            bits=list(struct.unpack('<'+'I'*count,result[2]['payload']))
            assert bits==expected and result[1]['payload']==model.state()
            version=wire.meta(result[0])['version'];assert version==int(exact)
            row={'case':i,'from_bits':lo,'to_bits':hi,'elements':count,'accepted':exact,'version':version,
                'state_sha256':SHA(result[1]['payload']),'output_bits':bits,
                'generator_state_unchanged':not exact or count==0,'cpp_binary64_difference_bits':oracle[i][1]}
            if torch_rows is not None:
                actual=torch_rows[i,count]
                for key in ('from_bits','to_bits','elements','accepted','version','state_sha256','output_bits','generator_state_unchanged'):
                    assert row[key]==actual[key],(i,count,key,row[key],actual[key])
                row['actual_fixed_torch_exact_match']=True
            rows.append(row);s.run([(41,t),(66,g),(33,0),(53,)])
    s.finish();report={'schema':'heptatorch.uniform-binary64-admission-native.v1','status':'PASS','cases':len(rows),
        'executable_sha256':SHA(a.executable.read_bytes()),'reference_source_sha256':SHA(source.read_bytes()),
        'reference_executable_sha256':SHA(binary.read_bytes()),'native_executed':True,'pytorch_executed_here':False,
        'actual_fixed_torch_observation_compared':torch_rows is not None,'actual_observation_sha256':SHA(a.torch_observation.read_bytes()) if a.torch_observation else None,
        'exact_threshold':'FLT_MAX + 2^74, inclusive because the binary64 FLT_MAX significand is even',
        'cases':len(rows),'whole_owner_rollbacks':rollback,'frames':s.frames,'rows':rows}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))

if __name__=='__main__':main()
