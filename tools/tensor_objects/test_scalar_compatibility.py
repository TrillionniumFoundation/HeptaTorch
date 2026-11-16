"""Replay the unchanged canonical scalar fixture through the v3 envelope."""
from pathlib import Path
import argparse, hashlib, json, struct, sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from protocol import Session, parse, CLIENT

def sha(raw): return hashlib.sha256(raw).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--input-fixture',type=Path,required=True)
    ap.add_argument('--output-fixture',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),Path(__file__).with_name('protocol.py'),CLIENT,
           args.executable,args.input_fixture,args.output_fixture]
    originals={str(p):p.read_bytes() for p in paths}
    request=bytes.fromhex(args.input_fixture.read_text())
    expected=bytes.fromhex(args.output_fixture.read_text())
    count=struct.unpack_from('<Q',request,8)[0]
    assert count==894 and len(request)==32+count*64
    assert all(struct.unpack_from('<Q',request,32+i*64)[0]==14 for i in range(count))
    assert len(expected)==32+count*32 and struct.unpack_from('<Q',expected,24)[0]==0
    session=Session(args.executable,directory=args.output/'interactive')
    actual=session.exchange(b'HTORCH03'+request[8:])
    status,rows=parse(actual)
    assert status==0 and len(rows)==count
    assert actual[32:]==expected[32:], 'Scalar results differ from the unchanged canonical fixture'
    session.finish()
    for path in paths: assert path.read_bytes()==originals[str(path)]
    report={'schema':'heptatorch.tensor-object-scalar-compatibility.v1','status':'PASS',
            'frames':session.frames,'exact_scalar_results':count,
            'request_transformation':'Only replace the 8-byte request magic with HTORCH03',
            'response_comparison':'All 894 command records byte-identical; only the versioned header differs',
            'input_sha256':{path:sha(raw) for path,raw in originals.items()},
            'stdin_open_before_response':True}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS','exact_scalar_results':count}))

if __name__=='__main__': main()
