"""Independent native controls for additive default-backward command 57.

No target source, compiler, authoring helper or numerical implementation is
imported. Numeric expectations below are exact small dyadic examples, fixed
before native execution. A pinned helper only frames I/O and reads our own
child's committed owner for rollback checks.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import struct
import traceback


OBSERVER_SHA='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
CASE_NAMES=(
    'no-Saved-repeats-after-public-intermediate-drop',
    'released-Saved-marker-cannot-be-revived-by-legacy-entries',
    'legacy-and-explicit-retention-then-release',
    'public-Saved-and-alias-outlive-graph-Saved-release',
    'AccumulateGrad-keeps-dropped-leaf-until-root-drop',
    'nongrad-coefficient-expires-on-default-backward',
    'empty-index-Saved-is-released',
    'scatter-self-only-remains-repeatable',
    'index-add-self-only-remains-repeatable',
    'shared-root-versus-unvisited-independent-root',
    'flags-shape-version-and-allocation-errors-roll-back',
    'later-command-error-rolls-back-successful-release',
    'released-storage-extent-reuse-does-not-revive-old-identities',
    'command57-output-seed-result-references-and-scalar-result',
)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def bits(value): return struct.unpack('<I',struct.pack('<f',value))[0]


class Blob:
    def __init__(self,data): self.data=bytes(data)


def tensor(shape,values,requires=False):
    return (1,len(shape),int(requires),*shape,*([0]*(4-len(shape))),
            Blob(struct.pack('<'+'f'*len(values),*values)))


def encode(commands):
    end=32+64*len(commands);payload=bytearray();records=bytearray()
    for command in commands:
        words=[]
        for word in command:
            if isinstance(word,Blob):
                while (end+len(payload))%8: payload.append(0)
                words.append(end+len(payload));payload.extend(word.data)
            else: words.append(int(word)&((1<<64)-1))
        assert len(words)<=8
        records.extend(struct.pack('<8Q',*words,*([0]*(8-len(words)))))
    return b'HTORCH03'+struct.pack('<3Q',len(commands),end+len(payload),0)+records+payload


def decode(raw):
    assert len(raw)>=32
    magic,version,count,status=struct.unpack_from('<8s3Q',raw)
    assert magic==b'HTORSP03' and version==3
    if status:
        assert len(raw)==32 and count==0
        return status,[]
    at=32;records=[]
    for _ in range(count):
        assert at+32<=len(raw)
        code,error,value,size=struct.unpack_from('<4Q',raw,at);at+=32
        assert error==0 and at+size<=len(raw)
        records.append({'code':code,'value':value,'payload':raw[at:at+size]});at+=size
    assert at==len(raw)
    return 0,records


def words(record):
    assert len(record['payload'])%8==0
    return struct.unpack('<'+'Q'*(len(record['payload'])//8),record['payload'])


class Check:
    def __init__(self,session):
        self.session=session;self.frames=[];self.comparisons=0;self.rollback_bytes=0

    def run(self,commands,status=0,label=''):
        before=self.session.owner() if status else None
        request=encode(commands);response=self.session.exchange(request)
        observed,records=decode(response)
        row={'frame':self.session.frames,'label':label,'expected_status':status,'actual_status':observed,
             'request_sha256':sha(request),'response_sha256':sha(response)}
        self.frames.append(row)
        assert observed==status,(label,observed,status)
        if status:
            after=self.session.owner()
            assert before==after,('committed owner changed on failed frame',label)
            self.rollback_bytes+=before['length']
            row.update(entire_owner_unchanged=True,owner_bytes=before['length'],owner_sha256=sha(before['bytes']))
        else:
            assert len(records)==len(commands)
            assert [r['code'] for r in records]==[c[0] for c in commands]
        return records

    def one(self,*command,status=0,label=''):
        result=self.run([command],status,label)
        return None if status else result[0]

    def make(self,*command): return self.one(*command)['value']
    def new(self,shape,values,requires=False): return self.make(*tensor(shape,values,requires))

    def values(self,native,expected,gradient=False):
        record=self.one(11 if gradient else 10,native,label='exact values' if not gradient else 'exact leaf gradient')
        assert record['value']==len(expected) and len(record['payload'])==len(expected)*4
        actual=struct.unpack('<'+'I'*len(expected),record['payload'])
        wanted=tuple(bits(v) for v in expected)
        assert actual==wanted,(actual,wanted)
        self.comparisons+=len(expected)

    def expiry(self,weak,expired):
        if expired: self.one(43,weak,status=6,label='expired Weak cannot lock')
        else:
            hold=self.make(43,weak)
            self.one(45,hold,label='release temporary public Hold')
        self.comparisons+=1

    def audit(self):
        assert self.one(53,label='final object audit')['value']==1
        self.comparisons+=1


def graph(c):
    x=c.new([2],[1.,2.],True);square=c.make(2,3,x,x);loss=c.make(4,7,square)
    return x,square,loss


def no_saved(c):
    x=c.new([2],[1.,2.],True);constant=c.new([], [2.])
    middle=c.make(2,1,x,constant);copy=c.make(18,middle);loss=c.make(4,7,copy)
    weak_a,weak_b=c.make(42,middle),c.make(42,copy)
    c.run([(41,middle),(41,copy)]);c.expiry(weak_a,True);c.expiry(weak_b,True)
    before=words(c.one(54))
    c.one(57,loss,0,0);c.values(x,[1.,1.],True)
    after=words(c.one(54));assert after[4]>before[4] and after[10]<before[10]
    c.one(57,loss,0,0);c.values(x,[2.,2.],True)
    c.one(7,loss);c.values(x,[3.,3.],True)
    c.run([(44,weak_a),(44,weak_b)])
    return {'before_stats':before,'after_first_default_stats':after,'temporary_Holds_reclaimed':True}


def released_marker(c):
    x,square,loss=graph(c);seed=c.new([], [1.])
    c.one(57,loss,0,0);c.values(x,[2.,4.],True);c.values(square,[1.,4.])
    for command in [(57,loss,0,0),(57,loss,seed,1),(7,loss),(29,loss,seed)]:
        c.run([command],10,'a live payload cannot recreate released Saved records')
    c.values(x,[2.,4.],True)


def retention(c):
    x,_,loss=graph(c)
    c.one(7,loss);c.values(x,[2.,4.],True)
    c.one(57,loss,0,1);c.values(x,[4.,8.],True)
    c.one(57,loss,0,0);c.values(x,[6.,12.],True)
    c.one(57,loss,0,1,status=10);c.values(x,[6.,12.],True)


def public_saved(c):
    x=c.new([2],[1.,-2.],True);output=c.make(3,5,x);square=c.make(2,3,output,output);loss=c.make(4,7,square)
    weak,saved=c.make(42,output),c.make(46,output)
    c.run([(41,output),(41,square)])
    c.one(57,loss,0,0);c.values(x,[2.,0.],True);c.expiry(weak,False)
    assert c.one(47,saved)['value']==1
    record=c.one(49,saved,0,2);assert record['payload']==struct.pack('<2f',1.,0.)
    alias=c.make(56,saved);c.values(alias,[1.,0.]);c.one(57,loss,0,0,status=10)
    c.run([(48,saved),(41,alias)]);c.expiry(weak,True);c.one(44,weak)
    c.comparisons+=3


def leaf_lifetime(c):
    x=c.new([2],[1.,2.],True);one=c.new([], [1.]);middle=c.make(2,1,x,one);loss=c.make(4,7,middle)
    weak=c.make(42,x);c.run([(41,middle),(41,x)]);c.expiry(weak,False)
    c.one(57,loss,0,0);c.expiry(weak,False);c.one(13,x,status=1)
    c.one(41,loss);c.expiry(weak,True);c.one(44,weak)


def coefficient(c):
    x=c.new([2],[1.,2.],True);factor=c.new([2],[3.,4.]);middle=c.make(2,3,x,factor);loss=c.make(4,7,middle)
    weak=c.make(42,factor);c.run([(41,factor),(41,middle)]);c.expiry(weak,False)
    c.one(57,loss,0,0);c.values(x,[3.,4.],True);c.expiry(weak,True);c.one(44,weak)


def empty_index(c):
    x=c.new([2,3],[1.,2.,3.,4.,5.,6.],True)
    output=c.make(30,x,1,0,Blob(b''));seed=c.new([2,0],[])
    c.one(57,output,seed,0);c.values(x,[0.]*6,True)
    c.one(57,output,seed,0,status=10);c.one(29,output,seed,status=10)


def self_only(c,index_add=False):
    x=c.new([2,3],[1.,2.,3.,4.,5.,6.],True)
    if index_add:
        source=c.new([2,2],[.5,1.,1.5,2.]);config=Blob(struct.pack('<2Q2q',1,2,2,0))
        output=c.make(37,x,source,1,config,bits(.5))
    else:
        source=c.new([2,3],[.5,1.,1.5,2.,2.5,3.])
        config=Blob(struct.pack('<3Q6q',2,2,3,2,0,2,1,1,0))
        output=c.make(36,x,source,1,config)
    c.one(41,source)
    seed=c.new([2,3],[.5,1.,1.5,2.,2.5,3.])
    before=words(c.one(54));c.one(57,output,seed,0);after=words(c.one(54))
    c.values(x,[.5,1.,1.5,2.,2.5,3.],True)
    c.one(57,output,seed,0);c.values(x,[1.,2.,3.,4.,5.,6.],True)
    assert after[4]>before[4]
    return {'before_stats':before,'after_first_default_stats':after}


def roots(c):
    x,shared,left=graph(c);one=c.new([], [1.]);middle=c.make(2,1,shared,one);right=c.make(4,7,middle)
    independent=c.make(2,3,x,x);other=c.make(4,7,independent)
    c.one(57,left,0,0);c.values(x,[2.,4.],True)
    c.one(57,right,0,0,status=10)
    c.one(57,other,0,0);c.values(x,[4.,8.],True)


def failures(c):
    x,_,loss=graph(c);constant=c.new([], [1.]);bad_seed=c.new([1],[1.]);sid=words(c.one(13,x))[11]
    for flag in (2,3,1<<63,(1<<64)-1): c.one(57,loss,0,flag,status=7,label='unknown flag bits')
    for bad in (0,1,sid,x+(1<<32)): c.one(57,bad,0,0,status=1,label='invalid output identity')
    c.one(57,loss,sid,0,status=1,label='wrong-kind seed')
    c.one(57,loss,bad_seed,0,status=2,label='exact seed shape required')
    c.one(57,constant,0,0,status=5,label='output requires gradient')
    c.one(57,x,0,0,status=5,label='non-scalar output requires an explicit seed')
    c.run([(12,x,0,bits(9.)),(57,loss,0,0)],6,'version failure rolls back preceding mutation')
    c.run([(12,x,0,bits(9.)),(38,2,0,4096,4096,0,0,0)],3,'real address-limit allocation failure rolls back preceding mutation')
    c.values(x,[1.,2.]);c.values(x,[0.,0.],True)
    assert words(c.one(13,x))[13]==0
    c.one(57,loss,0,0);c.values(x,[2.,4.],True)


def later_error(c):
    x,_,loss=graph(c)
    c.run([(57,loss,0,0),(999,)],4,'later command rejects an otherwise successful default backward')
    c.values(x,[0.,0.],True)
    c.one(57,loss,0,0);c.values(x,[2.,4.],True)
    c.one(57,loss,0,0,status=10)


def extent_reuse(c):
    x=c.new([2],[1.,2.],True);factor=c.new([2],[3.,4.]);meta=words(c.one(13,factor));sid,vid=meta[11:13]
    original=words(c.one(51,sid));extent=original[2]
    product=c.make(2,3,x,factor);loss=c.make(4,7,product);weak=c.make(42,factor)
    c.run([(41,factor),(41,product)]);c.one(57,loss,0,0);c.expiry(weak,True)
    dead=words(c.one(51,sid));assert dead[7]==1 and dead[5]==0
    reused=None
    for index in range(32):
        fresh=c.new([2],[7.,8.]);newmeta=words(c.one(13,fresh));storage=words(c.one(51,newmeta[11]))
        c.values(fresh,[7.,8.])
        if storage[2]==extent:
            reused={'allocation_index':index+1,'new_TensorID':fresh,'new_StorageID':newmeta[11],'old_StorageID':sid,'extent':extent}
            break
    assert reused is not None,'free interior extent was not reused within the bounded exact fixture'
    c.one(13,factor,status=1);c.one(52,vid,status=1);c.expiry(weak,True)
    c.one(57,loss,0,0,status=10)
    c.one(44,weak);c.one(51,sid,status=1)
    c.one(33,0);assert words(c.one(54))[11]==0
    return reused


def references(c):
    ref=lambda i:0xfe000000|i
    commands=[tensor([2],[1.,2.],True),(2,3,ref(1),ref(1)),(4,7,ref(2)),tensor([], [1.]),
              (57,ref(3),ref(4),0),(11,ref(1))]
    records=c.run(commands,label='same-frame output and seed references')
    assert records[4]['value']==1 and records[4]['payload']==b''
    assert records[5]['payload']==struct.pack('<2f',2.,4.)
    x=records[0]['value'];c.comparisons+=3
    # Fresh graph is needed because the first one has just released its Saved.
    c.run([tensor([], [2.],True),(57,ref(1),0,0),(18,ref(2))],1,
          'backward scalar result is not a published Tensor reference')
    c.values(x,[2.,4.],True)


FUNCTIONS=(no_saved,released_marker,retention,public_saved,leaf_lifetime,coefficient,empty_index,
           self_only,lambda c:self_only(c,True),roots,failures,later_error,extent_reuse,references)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--expected-source-sha256',required=True)
    parser.add_argument('--transport-observer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    for name in ('binary','source','transport_observer','output'): setattr(args,name,getattr(args,name).resolve())
    assert sha(args.binary.read_bytes())==args.expected_sha256
    assert sha(args.source.read_bytes())==args.expected_source_sha256
    assert sha(args.transport_observer.read_bytes())==OBSERVER_SHA
    args.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('independent_backward_release_observer',args.transport_observer)
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    source_hash=sha(Path(__file__).read_bytes());archive=args.output/'executed-sources';archive.mkdir()
    for path in (Path(__file__),args.transport_observer): (archive/path.name).write_bytes(path.read_bytes())
    rows=[]
    for index,(name,fn) in enumerate(zip(CASE_NAMES,FUNCTIONS)):
        directory=args.output/f'case-{index+1:02d}'
        session=transport.OwnerSession(args.binary,directory=directory,address_limit=64<<20);check=Check(session)
        row={'case':name,'status':'FAIL','reference':'Independent fixed native contract, not a live Torch differential'}
        try:
            details=fn(check);check.audit();session.finish()
            row.update(status='PASS',details=details)
        except Exception as error:
            row.update(exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            if session.poll() is None: os.kill(session.pid,signal.SIGKILL)
            try: session.finish(expected=-signal.SIGKILL)
            except Exception as cleanup: row['cleanup_exception']=repr(cleanup)
        row.update(frames=check.frames,exact_scalar_or_binary32_comparisons=check.comparisons,
                   entire_committed_owner_rollback_bytes=check.rollback_bytes)
        (directory/'report.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');rows.append(row)
    assert len(rows)==len(CASE_NAMES)==14
    assert sha(args.binary.read_bytes())==args.expected_sha256 and sha(args.source.read_bytes())==args.expected_source_sha256
    assert sha(Path(__file__).read_bytes())==source_hash
    report={'schema':'heptatorch.native-backward-release-independent-controls.v1',
            'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
            'native_executable_sha256':args.expected_sha256,'native_source_sha256':args.expected_source_sha256,
            'test_source_sha256':source_hash,'transport_observer_sha256':OBSERVER_SHA,
            'native_executed':True,'pinned_Torch_executed':False,'cases':len(rows),
            'passed':sum(r['status']=='PASS' for r in rows),'frames':sum(len(r['frames']) for r in rows),
            'exact_scalar_or_binary32_comparisons':sum(r['exact_scalar_or_binary32_comparisons'] for r in rows),
            'entire_committed_owner_rollback_bytes':sum(r['entire_committed_owner_rollback_bytes'] for r in rows),
            'rows':rows,'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','passed','frames','exact_scalar_or_binary32_comparisons','entire_committed_owner_rollback_bytes')}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
