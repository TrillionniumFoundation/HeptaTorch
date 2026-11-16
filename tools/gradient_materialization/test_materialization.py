"""Bit-preserving first-gradient and transient-state native qualification.

This is an independent wire test. It does not import authoring or arithmetic
implementation code. Raw words below exercise copy semantics; no host float
conversion is used to construct or compare a NaN or signed zero.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,signal,struct,traceback

SHA=lambda raw:hashlib.sha256(raw).hexdigest()
PATTERN=[0,0x80000000,1,0x80000001,0x00800000,0x007fffff,0x3f800000,
         0xc0000000,0x7f7fffff,0x7f800000,0xff800000,0x7fc12345,
         0xffc12345,0x7f812345,0xff812345,0x3f000001]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--expected-source-sha256',required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    inputs={p.resolve():p.read_bytes() for p in (Path(__file__),args.binary,args.source,args.protocol)}
    assert SHA(inputs[args.binary.resolve()])==args.expected_sha256
    assert SHA(inputs[args.source.resolve()])==args.expected_source_sha256
    assert SHA(inputs[args.protocol.resolve()])=='5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec=importlib.util.spec_from_file_location('materialization_wire',args.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    def tensor(s,shape,values,requires=0):
        payload=wire.Payload(struct.pack('<'+'I'*len(values),*values),4)
        return s.run([(1,len(shape),requires,*shape,*([0]*(4-len(shape))),payload)])[0]['value']
    def full(s,shape,value,requires=0):return s.run([wire.full(shape,value,requires)])[0]['value']
    def gradient(s,t,expected):
        record=s.run([(11,t)])[0]
        assert record['value']==len(expected)
        actual=list(struct.unpack('<'+'I'*len(expected),record['payload']))
        assert actual==expected,(actual,expected)
        return len(expected)
    def markers(s):
        raw=s.owner()['bytes'];q=lambda at:struct.unpack_from('<Q',raw,at)[0]
        data_base=139264+144+424*q(139264+16)
        checked=0
        for index in range(1,q(8)):
            tagged=q(4096+(index>>8)*8)
            assert tagged>>63
            node=data_base+(tagged&((1<<63)-1))+(index&255)*384
            if q(node+304):
                assert [q(node+off) for off in (336,368,376)]==[0,0,0],index
                checked+=3
        return checked
    report={'schema':'heptatorch.gradient-materialization-native-qualification.v1','status':'RUNNING',
            'native_executable_sha256':args.expected_sha256,'native_source_sha256':args.expected_source_sha256,
            'input_sha256':{str(p):SHA(b) for p,b in inputs.items()},'native_executed':True,
            'pinned_Torch_executed':False,'rows':[],'frames':0,'exact_gradient_words':0,'zero_transient_fields':0}
    def save():(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def case(name,body):
        s=wire.Session(args.binary,directory=args.output/name,address_limit=64<<20)
        row={'case':name,'status':'RUNNING'};report['rows'].append(row);save()
        try:
            words,fields,details=body(s);s.run([(53,)]);fields+=markers(s);s.finish()
            row.update(status='PASS',frames=s.frames,exact_gradient_words=words,zero_transient_fields=fields,details=details)
            report['frames']+=s.frames;report['exact_gradient_words']+=words;report['zero_transient_fields']+=fields
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());report['status']='FAIL'
            if s.poll() is None:os.kill(s.pid,signal.SIGKILL)
            try:s.finish(expected=-signal.SIGKILL)
            except BaseException:pass
            save();raise
        finally:(args.output/name/'assertions.json').write_text(json.dumps(row,indent=2)+'\n');save()
    def raw_copies(s):
        count=0;fields=0;details=[]
        for opcode in (29,57,58):
            for operation in ('leaf','clone','view','transpose'):
                x=full(s,(4,4),2,1);y=x;shape=(4,4);expected=PATTERN
                if operation=='clone':y=s.run([(18,x)])[0]['value']
                elif operation=='view':y=s.run([(17,x,1,16)])[0]['value'];shape=(16,)
                elif operation=='transpose':
                    y=s.run([(6,x,0,1)])[0]['value']
                    expected=[PATTERN[c*4+r] for r in range(4) for c in range(4)]
                seed=tensor(s,shape,PATTERN);record=s.run([(opcode,y,seed,0)])[0]
                assert record['value']==(0 if opcode==58 else 1)
                count+=gradient(s,x,expected);fields+=markers(s)
                assert s.run([(59,x)])[0]['value']==1
                details.append({'entry':opcode,'operation':operation,'expected_words':expected})
        return count,fields,details
    case('raw-first-gradient-copy-across-three-explicit-backward-entries',raw_copies)
    def legacy_scalar(s):
        x=full(s,(),2,1);minus_zero=tensor(s,(),[0x80000000])
        y=s.run([(2,3,x,minus_zero)])[0]['value']
        s.run([(7,y)]);count=gradient(s,x,[0x80000000]);fields=markers(s)
        s.run([(7,y)]);count+=gradient(s,x,[0x80000000]);fields+=markers(s)
        return count,fields,{'legacy_scalar_first_and_existing_gradient_words':[0x80000000,0x80000000]}
    case('legacy-implicit-root-negative-zero',legacy_scalar)
    def repeated_parent(s):
        x=full(s,(4,),2,1)
        a=s.run([(18,x)])[0]['value'];b=s.run([(18,x)])[0]['value']
        root=s.run([(2,1,a,b)])[0]['value'];seed=tensor(s,(4,),[0x80000000,0,0x3f800000,0xbf800000])
        s.run([(58,root,seed,0)])
        count=gradient(s,x,[0x80000000,0,0x40000000,0xc0000000]);fields=markers(s)
        s.run([(58,root,seed,0)])
        count+=gradient(s,x,[0x80000000,0,0x40800000,0xc0800000]);fields+=markers(s)
        return count,fields,{'different_ready_nodes_reset_the_bitmap':True,'existing_gradient_adds_after_first_copy':True}
    case('separate-node-input-buffer-merges-and-reused-graph',repeated_parent)
    def mixed_reduce(s):
        x=full(s,(2,),2,1);direct=s.run([(18,x)])[0]['value']
        selected=s.run([(31,x,0,wire.Payload(struct.pack('<2Qq',1,1,0)))])[0]['value']
        listing=wire.Payload(struct.pack('<3Q',2,direct,selected))
        root=s.run([(34,0,listing)])[0]['value'];seed=tensor(s,(3,),[0x80000000]*3)
        s.run([(58,root,seed,0)])
        count=gradient(s,x,[0,0]);fields=markers(s)
        # A later failed node must leave no borrowed work pointer or bitmap live.
        a=full(s,(),2,1);coefficient=full(s,(),3)
        product=s.run([(2,3,a,coefficient)])[0]['value'];outer=s.run([(2,3,product,product)])[0]['value']
        s.run([(12,coefficient,0,wire.bits(4))]);result=s.run([(58,outer,0,0)])[0]
        assert result['value']==6;fields+=markers(s)
        assert s.run([(59,a)])[0]['value']==0
        result=s.run([(58,outer,0,0)])[0];assert result['value']==10;fields+=markers(s)
        return count,fields,{'scatter_reduction_keeps_positive_zero':True,'runtime_error_transient_fields_clear':True}
    case('direct-and-scatter-gradients-plus-runtime-error-cleanup',mixed_reduce)
    for p,data in inputs.items():assert p.read_bytes()==data,p
    (args.output/'executed-test.py').write_bytes(inputs[Path(__file__).resolve()])
    report['status']='PASS';save();print(json.dumps({k:report[k] for k in ('status','frames','exact_gradient_words','zero_transient_fields')}))

if __name__=='__main__':main()
