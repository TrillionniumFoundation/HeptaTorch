"""Independent exact-dyadic values and actual auxiliary Storage reclamation."""
from pathlib import Path
import argparse, hashlib, json, struct, traceback, sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from protocol import Session, Payload, tensor, packet, parse, bits

def words(*values): return struct.pack('<'+'Q'*len(values),*values)
def stats(row):
    assert row['opcode']==54 and len(row['payload'])==96
    return struct.unpack('<12Q',row['payload'])
def pair(a,b): return (a<<32)|b

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[]
    def case(name,parents,make,gradients,value,keeps_output=False):
        session=Session(args.executable,directory=args.output/name)
        row={'name':name,'status':'RUNNING','executable_sha256':session.binary_sha256}
        try:
            ids=[session.run([tensor(shape,values,1)])[0]['value'] for shape,values in parents]
            before=stats(session.run([(54,)])[0])
            output=session.run([make(ids)])[0]['value']
            loss=session.run([(4,7,output)])[0]['value']
            observed=session.run([(10,loss)])[0]['payload']
            assert observed==struct.pack('<f',value),(name,observed,value)
            weak=session.run([(42,output)])[0]['value']
            session.run([(41,output)])
            if keeps_output:
                held=session.run([(43,weak)])[0]['value'];session.run([(45,held)])
            else: session.error([(43,weak)],6)
            session.run([(7,loss),(53,)])
            values=0
            for tid,expected in zip(ids,gradients):
                observed=session.run([(11,tid)])[0]['payload']
                assert observed==struct.pack('<'+'f'*len(expected),*expected),(name,observed,expected)
                values+=len(expected)
            session.run([(41,loss),(44,weak),(53,)])
            after=stats(session.run([(54,)])[0])
            assert after[:2]==before[:2],(name,'capacity changed unexpectedly',before,after)
            assert after[4:11]==before[4:11],(name,'Storage/Version/Hold leak',before,after)
            session.finish()
            row.update(status='PASS',frames=session.frames,exact_float32_values=values+1,
                       before_stats=before,after_stats=after,stdin_open_before_every_response=True)
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
            raise
        finally:
            (args.output/name/'assertions.json').write_text(json.dumps(row,indent=2)+'\n')
            cases.append(row)
            report={'schema':'heptatorch.tensor-object-aux-lifetime.v1','status':'PASS' if all(x['status']=='PASS' for x in cases) else 'FAIL',
                'native_sha256':hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases,
                'frames':sum(x.get('frames',0) for x in cases),
                'exact_float32_values':sum(x.get('exact_float32_values',0) for x in cases)}
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    counts=[1,2,1,2,4,2,1,2,1]
    case('conv-config', [((1,1,3,3),[1]*9),((1,1,2,2),[1]*4)],
         lambda t:(22,t[0],t[1],pair(1,1),0,pair(1,1),1),[counts,[4]*4],16)
    case('avgpool-config', [((1,1,3,3),[1]*9)],
         lambda t:(23,23,t[0],pair(2,2),pair(1,1),0,1),[[x/4 for x in counts]],4)
    case('maxpool-config-and-indices', [((1,1,3,3),list(range(9)))],
         lambda t:(23,24,t[0],pair(2,2),pair(1,1),0,0),[[0,0,0,0,1,1,0,1,1]],24)
    case('index-select-indices', [((2,3),list(range(6)))],
         lambda t:(30,t[0],1,3,Payload(words(2,0,2))),[[1,0,2,1,0,2]],17)
    case('gather-indices', [((2,3),list(range(6)))],
         lambda t:(31,t[0],1,Payload(words(2,2,2,2,0,1,1))),[[1,0,1,0,2,0]],10)
    for code,label in [(34,'cat'),(35,'stack')]:
        case(label+'-parents', [((2,2),[1]*4)],
             lambda t,code=code:(code,1,Payload(words(2,t[0],t[0]))),[[2]*4],8)
    case('scatter-config', [((2,3),list(range(6))),((2,2),[1]*4)],
         lambda t:(36,t[0],t[1],1,Payload(words(2,2,2,2,0,1,1))),[[1]*6,[1]*4],19)
    case('index-add-config', [((2,3),[0]*6),((2,2),[1]*4)],
         lambda t:(37,t[0],t[1],1,Payload(words(1,2,2,0)),bits(.5)),[[1]*6,[.5]*4],2)
    case('relu-saved-result', [((3,),[-2,1,3])], lambda t:(3,5,t[0]),[[0,1,1]],4,True)
    case('softmax-saved-result', [((2,),[0,0])], lambda t:(21,20,t[0],0),[[0,0]],1,True)
    print(json.dumps({'status':'PASS','cases':len(cases),'frames':sum(r['frames'] for r in cases)}))

if __name__=='__main__':main()
