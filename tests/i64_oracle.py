"""Source-bound native HIA int64 binary32-independent arithmetic qualification.

CPython and locked PyTorch are independent references. The HIA ELF alone
executes every arithmetic, bitwise, packet read, write, and refusal operation.
"""
import argparse
import hashlib
import json
import random
import runpy
import struct
import subprocess
import sys
import traceback
from pathlib import Path

SOURCE_SHA = '251705542f4e117e3c9cdaec36cead8ab38252b24ebfef1104d765c546fd0716'
IMAGE_SHA = 'fbb6cf1648385be200c94fc14ed73d1539cba3b0239aeea4f7866058c8395ebc'
COMPILER_SHA = '87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54'
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
MASK = (1 << 64) - 1
SHA = lambda data: hashlib.sha256(data).hexdigest()


def signed(x):
    x &= MASK
    return x if x < 1 << 63 else x - (1 << 64)


def source_reference(mode, pairs):
    def one(a,b):
        if mode == 0: return (a+b)&MASK
        if mode == 1: return (a-b)&MASK
        if mode == 2: return (a*b)&MASK
        if mode == 3: return a&b
        if mode == 4: return a|b
        if mode == 5: return a^b
        raise ValueError(mode)
    return [one(a,b) for a,b in pairs]


def corpus():
    values=[0,1,2,3,7,255,256,1024,(1<<16)-1,1<<16,
            (1<<31)-1,1<<31,(1<<32)-1,1<<32,
            (1<<53)-1,1<<53,(1<<63)-1,1<<63,MASK,MASK-1,MASK-2,
            0x5555555555555555,0xaaaaaaaaaaaaaaaa]
    edges=[(a,b) for a in values for b in values]
    for mode in range(6):
        yield f'empty-mode-{mode}', mode, []
        yield f'edge-product-mode-{mode}', mode, edges
        rng=random.Random(192830+(mode*19))
        yield f'10k-random-mode-{mode}', mode, [(rng.getrandbits(64),rng.getrandbits(64)) for _ in range(10000)]
        yield f'48k-boundary-mode-{mode}', mode, [(rng.getrandbits(64),rng.getrandbits(64)) if i%13
            else edges[i%len(edges)] for i in range(48000)]


def as_wire(mode, pairs):
    wire=bytearray(struct.pack('<II',mode,len(pairs)))
    for a,b in pairs:wire.extend(struct.pack('<QQ',a,b))
    return bytes(wire)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('tools/native-i64/i64_elementwise.hepta'))
    p.add_argument('--reference',choices=('cpython','pytorch'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    original_source=a.source.read_bytes()
    compiler=a.compiler.resolve()
    assert SHA(original_source)==SOURCE_SHA
    assert SHA(compiler.read_bytes())==COMPILER_SHA
    binary=[]
    for generation in range(2):
        run=subprocess.run([str(compiler)],input=original_source,capture_output=True,
                           timeout=30,env={'PATH':''})
        (out / f'compiler-{generation}.stderr').write_bytes(run.stderr)
        assert run.returncode==0 and not run.stderr and run.stdout.startswith(b'\x7fELF')
        binary.append(run.stdout)
    assert binary[0]==binary[1] and SHA(binary[0])==IMAGE_SHA
    exe=out/'i64-native';exe.write_bytes(binary[0]);exe.chmod(0o700)

    torch=None
    provenance=None
    if a.reference=='pytorch':
        support=Path('tests/pytorch_oracle.py').resolve()
        orig_argv=sys.argv
        try:
            sys.argv=[str(support),str(exe),str(out/'provenance-report.json'),
                      str(Path('compatibility/pytorch-oracle-lock.json').resolve())]
            context=runpy.run_path(str(support),run_name='i64_pinned_source_provenance')
        finally:sys.argv=orig_argv
        torch=context['torch'];provenance=context['PROVENANCE']
        assert provenance['status']=='PASS' and len(provenance['checks'])==11
        assert all(value is True for value in provenance['checks'].values())
        torch.set_num_threads(1)

    def run_native(wire):
        run=subprocess.run([str(exe)],input=wire,capture_output=True,
                           timeout=45,env={'PATH':''})
        assert run.returncode==0 and not run.stderr,(run.returncode,run.stderr[:100])
        return run.stdout

    rows=[]
    for name,mode,pairs in corpus():
        row={'case':name,'mode':mode,'count':len(pairs),'status':'FAIL','mismatches':[]}
        try:
            wire=as_wire(mode,pairs)
            response=run_native(wire)
            assert len(response)==4+8*len(pairs) and response[:4]==bytes(4), (len(response),response[:16].hex())
            want=source_reference(mode,pairs)
            if torch is not None and pairs:
                lhs=torch.tensor([signed(a) for a,b in pairs],dtype=torch.int64)
                rhs=torch.tensor([signed(b) for a,b in pairs],dtype=torch.int64)
                op=(lambda a,b:a+b,lambda a,b:a-b,lambda a,b:a*b,
                    torch.bitwise_and,torch.bitwise_or,torch.bitwise_xor)[mode]
                want_t=[v&MASK for v in op(lhs,rhs).tolist()]
                if want_t!=want:
                    row['reference_crosscheck_errors']=[{'index':i,'cpython':x,'torch':y}
                        for i,(x,y) in enumerate(zip(want,want_t)) if x!=y][:8]
                    raise AssertionError('CPython and PyTorch independent references disagree')
                want=want_t
            raw_expected=b''.join(struct.pack('<Q',v) for v in want)
            raw_actual=response[4:]
            if raw_actual!=raw_expected:
                got=struct.unpack('<'+str(len(pairs))+'Q',raw_actual) if pairs else []
                row['mismatches']=[{'index':i,'actual':v,'want':expected}
                    for i,(v,expected) in enumerate(zip(got,want)) if v!=expected][:8]
                raise AssertionError('native int64 bitwise mismatch')
            row.update(status='PASS',compared_values=len(pairs),input_sha256=SHA(wire),
                       response_sha256=SHA(response),reference_sha256=SHA(raw_expected))
        except Exception as e:
            row.update(error=str(e)[:1000],traceback=traceback.format_exc()[:3000])
        rows.append(row)
    refusals=[
       ('short-header',bytes(7)),('unknown-mode',struct.pack('<II',6,0)),
       ('u32-mode',struct.pack('<II',0xffffffff,0)),
       ('truncated-pair',struct.pack('<II',0,1)+bytes(15)),
       ('count-too-large',struct.pack('<II',1,48001)),
       ('count-max',struct.pack('<II',0,0xffffffff)),
       ('payload-after-empty',struct.pack('<II',2,0)+b'X'),
       ('payload-after-pair',as_wire(3,[(1,2)])+b'X'),
       ('wrong-count',struct.pack('<II',4,2)+struct.pack('<QQ',3,4)),
    ]
    for name,wire in refusals:
        row={'case':'refuse-'+name,'status':'FAIL'}
        try:
            response=run_native(wire)
            assert response==b'\x01\0\0\0',response[:32].hex()
            row.update(status='PASS',input_sha256=SHA(wire),response_sha256=SHA(response))
        except Exception as e:row.update(error=str(e)[:1000],traceback=traceback.format_exc()[:3000])
        rows.append(row)
    unchanged=a.source.read_bytes()==original_source and exe.read_bytes()==binary[0] and SHA(compiler.read_bytes())==COMPILER_SHA
    passed=len(rows)==33 and len(set(r['case'] for r in rows))==33 and all(r['status']=='PASS' for r in rows) and unchanged
    report={'schema':'heptatorch.native-int64-elementwise.v1','status':'PASS' if passed else 'FAIL',
            'reference':a.reference,'upstream_commit':PIN,'source_sha256':SHA(original_source),
            'native_elf_sha256':SHA(binary[0]),'compiler_sha256':COMPILER_SHA,
            'oracle_provenance':provenance,'tests':len(rows),
            'failed_tests':sum(r['status']!='PASS' for r in rows),
            'compared_values':sum(r.get('compared_values',0) for r in rows),
            'native_compilations_byte_identical':True,'producer_and_elf_unchanged':unchanged,
            'oracle_script_sha256':SHA(Path(__file__).read_bytes()),
            'scope':'bounded contiguous CPU int64 elementwise add/sub/mul/bitwise; no general Tensor dispatcher, shapes, broadcasting or full upstream schema',
            'rows':rows}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','oracle_provenance')},sort_keys=True))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
