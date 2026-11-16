"""Strict native HIA byte-packed signed int8 operations and locked PyTorch comparison.

The production binary performs all arithmetic, bitwise, packet IO and refusal.
This independent reference is test-only. No upstream schema is claimed complete.
"""
from pathlib import Path
import argparse,hashlib,json,random,runpy,struct,subprocess,sys,traceback,os,time

SHA=lambda b:hashlib.sha256(b).hexdigest()
SOURCE_SHA="7ca1765e2cede95ec19b64742f69816581733619b5b1b844f57aa72b743ed8d3"
COMPILER_SHA="87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54"
ELF_SHA=None  # Filled only after independently successful reproducible compilation.

def one(mode,a,b):
    if mode==0:return (a+b)&255
    if mode==1:return (a-b)&255
    if mode==2:return (a*b)&255
    if mode==3:return a&b
    if mode==4:return a|b
    if mode==5:return a^b
    raise ValueError(mode)

def corpus():
    exhaustive=[(a,b) for a in range(256) for b in range(256)]
    for m in range(6):
        yield f'empty-{m}',m,[]
        yield f'exhaustive-{m}',m,exhaustive
        r=random.Random(20261116+1009*m)
        yield f'100k-random-{m}',m,[(r.randrange(256),r.randrange(256)) for _ in range(100000)]
    r=random.Random(836485)
    yield '100k-admission-boundary',2,[(r.randrange(256),r.randrange(256)) for _ in range(100000)]

def wire(mode,pairs):
    return struct.pack('<II',mode,len(pairs))+b''.join(bytes((a,b)) for a,b in pairs)

def signed(a):return a if a<128 else a-256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',choices=('cpython','pytorch'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--expected-elf-sha256',default='')
    a=p.parse_args()
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    source=a.source.resolve();compiler=a.compiler.resolve();repo=a.repo.resolve()
    raw=source.read_bytes()
    assert SHA(raw)==SOURCE_SHA
    assert SHA(compiler.read_bytes())==COMPILER_SHA
    observer_path=Path(__file__).with_name('integer_byte_evidence.py')
    assert SHA(observer_path.read_bytes())=='6a5ff8a8a6f920f8734f9c5a26533ba26d71965d8de4fbbf1e7f7aa1c631f453'
    evidence=runpy.run_path(str(observer_path),run_name='integer_byte_observer')['Evidence'](
        output,'i8',a.reference,source,compiler,Path(__file__),repo)
    images=[]
    for i in range(2):
        r=subprocess.run([str(compiler)],input=raw,capture_output=True,env={'PATH':''},timeout=45)
        (output/f'compiler-{i}.stderr').write_bytes(r.stderr)
        evidence.compile_result(i,r)
        assert r.returncode==0 and not r.stderr and r.stdout.startswith(b'\x7fELF')
        images.append(r.stdout)
    assert images[0]==images[1]
    assert not a.expected_elf_sha256 or SHA(images[0])==a.expected_elf_sha256
    exe=output/'i8-native';exe.write_bytes(images[0]);exe.chmod(0o700)
    provenance=None;torch=None
    if a.reference=='pytorch':
        support=repo/'tests/pytorch_oracle.py'
        lock=repo/'compatibility/pytorch-oracle-lock.json'
        original=sys.argv;old=os.getcwd()
        try:
            os.chdir(repo)
            sys.argv=[str(support),str(exe),str(output/'provenance.json'),str(lock)]
            context=runpy.run_path(str(support),run_name='i8_locked_pytorch_reference')
        finally:
            sys.argv=original;os.chdir(old)
        torch=context['torch'];provenance=context['PROVENANCE']
        assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(v is True for v in provenance['checks'].values())
        torch.set_num_threads(1)
    rows=[]
    def execute(raw_input):
        evidence.request(raw_input)
        proc=subprocess.run([str(exe)],input=raw_input,capture_output=True,timeout=90,env={'PATH':''})
        evidence.native_result(proc)
        assert proc.returncode==0 and not proc.stderr,(proc.returncode,proc.stderr[:200])
        return proc.stdout
    for name,mode,pairs in corpus():
        row={'case':name,'mode':mode,'count':len(pairs),'status':'FAIL'}
        evidence.begin_case(name)
        try:
            request=wire(mode,pairs);actual=execute(request)
            expected=bytes(one(mode,x,y) for x,y in pairs)
            evidence.reference_bytes('cpython',expected)
            if torch is not None and pairs:
                left=torch.tensor([signed(x) for x,y in pairs],dtype=torch.int8)
                right=torch.tensor([signed(y) for x,y in pairs],dtype=torch.int8)
                ops=(lambda a,b:a+b,lambda a,b:a-b,lambda a,b:a*b,torch.bitwise_and,torch.bitwise_or,torch.bitwise_xor)
                want=bytes(v&255 for v in ops[mode](left,right).tolist())
                evidence.reference_bytes('pytorch',want)
                if want!=expected:
                    row['reference_disagreements']=sum(x!=y for x,y in zip(want,expected))
                    raise AssertionError('PyTorch and independent CPython mismatch')
                expected=want
            evidence.expected_bytes(expected,'payload','pytorch' if torch is not None and pairs else ('cpython' if torch is None else 'empty-no-framework-operation'))
            assert actual==bytes(4)+expected,(name,len(actual),len(expected),next(((i,v,e) for i,(v,e) in enumerate(zip(actual[4:],expected)) if v!=e),None))
            row.update(status='PASS',input_sha256=SHA(request),output_sha256=SHA(actual),expected_sha256=SHA(expected),compared_values=len(pairs))
        except Exception as e:row.update(error=repr(e),traceback=traceback.format_exc()[:3000])
        rows.append(row)
    refuses=[
      ('short-header',bytes(7)),('unknown-mode',struct.pack('<II',6,0)),
      ('mode-fmax',struct.pack('<II',0xffffffff,0)),
      ('count-over',struct.pack('<II',0,100001)),('count-u32max',struct.pack('<II',0,0xffffffff)),
      ('truncated-pair',struct.pack('<II',0,1)+bytes(1)),('trailing-empty',struct.pack('<II',1,0)+b'x'),
      ('trailing-pair',wire(2,[(0,255)])+b'x'),('count-mismatch',struct.pack('<II',0,2)+bytes(2)),
    ]
    for name,request in refuses:
        row={'case':'refuse-'+name,'status':'FAIL'}
        evidence.begin_case('refuse-'+name)
        evidence.expected_bytes(b'\x01\x00\x00\x00','full-response','protocol-literal')
        try:
            actual=execute(request)
            assert actual==b'\x01\x00\x00\x00',actual[:32].hex()
            row.update(status='PASS',input_sha256=SHA(request),output_sha256=SHA(actual))
        except Exception as e:row.update(error=repr(e),traceback=traceback.format_exc()[:3000])
        rows.append(row)
    passed=(len(rows)==28 and all(x['status']=='PASS' for x in rows)
            and source.read_bytes()==raw and SHA(compiler.read_bytes())==COMPILER_SHA
            and exe.read_bytes()==images[0])
    report={'schema':'heptatorch.native-packed-int8-elementwise.v1','status':'PASS' if passed else 'FAIL',
       'reference':a.reference,'upstream_commit':'31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457',
       'source_sha256':SHA(raw),'executable_sha256':SHA(images[0]),'compiler_sha256':COMPILER_SHA,
       'two_independent_compiler_invocations_identical':True,'cases':len(rows),'failed_cases':sum(r['status']!='PASS' for r in rows),
       'compared_values':sum(r.get('compared_values',0) for r in rows),'provenance':provenance,
       'oracle_source_sha256':SHA(Path(__file__).read_bytes()),'rows':rows,
       'scope':'byte-packed CPU int8 add/sub/mul and bitwise operations only; no complete torch schema or Tensor dispatcher',
       'full_rewrite_complete':False,'pytorch_schema_denominator':2590,'full_system_gates_closed':0}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    evidence.finalize(output/'report.json')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','provenance')},sort_keys=True))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
