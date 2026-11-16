"""Exact native result-reference, typed-ID and whole-frame rollback checks."""
from pathlib import Path
import argparse, hashlib, json, struct, sys, traceback
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import Session, Payload, tensor, full, meta, bits, reference as ref, CLIENT

def words(*values): return struct.pack('<'+'Q'*len(values), *values)
def floats(*values): return struct.pack('<'+'f'*len(values), *values)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('protocol.py'),CLIENT,args.executable]
    before={str(p):sha(p) for p in sources}
    report={'schema':'heptatorch.tensor-object-result-reference.v1','status':'RUNNING',
            'input_sha256':before,'cases':[],'frames':0,'exact_values':0,'rollback_bytes':0}
    def save(): (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def run_case(name,check):
        session=Session(args.executable,directory=args.output/name)
        row={'name':name,'status':'RUNNING'}; report['cases'].append(row); save()
        try:
            values,rollback=check(session)
            session.finish()
            row.update(status='PASS',frames=session.frames,exact_values=values,rollback_bytes=rollback,
                       stdin_open_before_every_response=True)
            report['frames']+=session.frames; report['exact_values']+=values;report['rollback_bytes']+=rollback
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
            report['status']='FAIL';save();raise
        finally:
            (args.output/name/'assertions.json').write_text(json.dumps(row,indent=2)+'\n');save()

    def graph(s):
        rows=s.run([tensor((2,),[1,2],1),(2,3,ref(1),ref(1)),(4,7,ref(2)),
                    (7,ref(3)),(11,ref(1)),(13,ref(1)),(46,ref(1)),
                    (49,ref(7),0,2),(19,ref(1)),(13,ref(9)),(55,ref(1)),(13,ref(11)),(53,)])
        assert rows[4]['payload']==floats(2,4)
        assert rows[7]['payload']==floats(1,2)
        x,detached,data=meta(rows[5]),meta(rows[9]),meta(rows[11])
        assert rows[5]['value']==rows[0]['value']
        assert x['storage_id']==detached['storage_id']==data['storage_id']
        assert x['version_id']==detached['version_id']!=data['version_id']
        assert x['version']==detached['version']==data['version']==0
        rollback=s.error([(10,ref(1))],1)
        assert s.run([(10,rows[0]['value'])])[0]['payload']==floats(1,2)
        return 13,rollback
    run_case('graph-saved-alias-and-cross-frame-refusal',graph)

    def lists(s):
        rows=s.run([tensor((2,),[2,3],1),
                    (34,0,Payload(words(3,ref(1),ref(1),ref(1)))),
                    (35,0,Payload(words(2,ref(1),ref(1)))),
                    (4,7,ref(2)),(4,7,ref(3)),(7,ref(4)),(7,ref(5)),
                    (11,ref(1)),(10,ref(2)),(10,ref(3)),(53,)])
        assert rows[7]['payload']==floats(5,5)
        assert rows[8]['payload']==floats(2,3,2,3,2,3)
        assert rows[9]['payload']==floats(2,3,2,3)
        return 12,0
    run_case('cat-stack-config-reference-gradient',lists)

    def optimizer(s):
        adam=Payload(words(bits(0),bits(.5),bits(.75),bits(2**-14),bits(0),0))
        momentum=Payload(words(bits(.125),bits(.5),bits(0),bits(0),0))
        rows=s.run([tensor((2,),[1,2],1),full((2,),0),full((2,),0),
                    (2,3,ref(1),ref(1)),(4,7,ref(4)),(7,ref(5)),
                    (24,ref(1),ref(2),ref(3),1,adam),(13,ref(1)),(13,ref(2)),(13,ref(3)),
                    (10,ref(2)),(10,ref(3)),full((2,),0),
                    (25,ref(1),ref(13),0,1,momentum),(10,ref(1)),(10,ref(13)),(13,ref(1)),(53,)])
        assert [meta(rows[i])['version'] for i in [7,8,9,16]]==[1,1,2,2]
        assert rows[10]['payload']==floats(1,2)
        assert rows[11]['payload']==floats(1,4)
        assert rows[14]['payload']==floats(.75,1.5)
        assert rows[15]['payload']==floats(2,4)
        return 12,0
    run_case('optimizer-reference-parameter-and-state',optimizer)

    def refusals(s):
        x=s.run([tensor((2,),[7,9])])[0]['value'];rollback=0
        requests=[
            [(10,0xFE000000)],
            [(10,ref(1))],
            [full((),1),(10,ref(2))],
            [full((),1),(10,ref(3))],
            [(14,4,bits(1),bits(2)),(10,ref(1))],
            [(46,x),(10,ref(1))],
            [(12,x,0,bits(77)),(41,x),(10,x)],
            [(10,(1<<32)|ref(1))],
            [full((),2),(41,ref(1)),(10,ref(1))],
        ]
        for commands in requests: rollback+=s.error(commands,1)
        assert s.run([(10,x),(53,)])[0]['payload']==floats(7,9)
        return 2,rollback
    run_case('nine-invalid-reference-rollback-cases',refusals)

    def maximum(s):
        commands=[full((),3)]
        commands.extend([(14,4,bits(1),bits(2))]*16380)
        commands.extend([(13,ref(1)),(10,ref(16382))])
        assert len(commands)==16383
        rows=s.run(commands)
        assert rows[-2]['value']==rows[0]['value']
        assert rows[-1]['payload']==floats(3)
        assert all(row['value']==1 for row in rows[1:-2])
        assert s.run([(10,rows[0]['value'])])[0]['payload']==floats(3)
        return 16383,0
    run_case('maximum-record-count-and-last-prior-reference',maximum)
    assert {str(p):sha(p) for p in sources}==before
    report['status']='PASS';save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','input_sha256')}))

if __name__=='__main__': main()
