"""Protocol/ownership smoke; no numerical parity oracle is implemented here."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, struct

def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-native-sha256',required=True)
    p.add_argument('--expected-source-sha256',required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__).resolve(),a.binary.resolve(),a.source.resolve(),a.protocol.resolve()]
    inputs={f:f.read_bytes() for f in files}
    assert sha(inputs[a.binary.resolve()])==a.expected_native_sha256
    assert sha(inputs[a.source.resolve()])==a.expected_source_sha256
    spec=importlib.util.spec_from_file_location('retained_smoke_wire',a.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    s=wire.Session(a.binary.resolve(),directory=out/'raw')
    report={'schema':'heptatorch.retained-native-smoke.v1','status':'RUNNING',
        'source_sha256':a.expected_source_sha256,'native_sha256':a.expected_native_sha256,
        'native_executed':True,'Torch_executed_here':False,'numerical_parity_claimed':False,
        'inputs_sha256':{str(f):sha(raw) for f,raw in inputs.items()},'observations':[]}
    def one(*cmd):return s.run([cmd])[0]
    def snapshot(t):
        meta,values=s.run([(13,t),(39,t,0,2)])
        return {'meta':wire.meta(meta),'raw_binary32_words':list(struct.unpack('<2I',values['payload']))}
    try:
        result=s.run([wire.tensor([2],[1.,2.],1),(2,3,wire.reference(1),wire.reference(1)),
            (93,wire.reference(2)),(94,wire.reference(2))])
        x,q=result[0]['value'],result[1]['value']
        assert result[2]['value']==1 and result[3]['value']==1
        assert one(93,x)['value']==1 and one(94,x)['value']==0
        assert one(91,q)['value']==0 and one(59,q)['value']==0
        seed=one(*wire.tensor([2],[1.,-0.]))['value']
        before=snapshot(q)
        reply=one(58,q,seed,1)
        assert reply['value']==0
        first=one(91,q)['value'];first_snapshot=snapshot(first)
        assert first and one(59,q)['value']==1 and first_snapshot['meta']['version']==0
        assert one(53)['value']==1
        reply=one(58,q,seed,1);assert reply['value']==0
        second=one(91,q)['value'];second_snapshot=snapshot(second)
        assert second!=first and second_snapshot['meta']['storage_id']!=first_snapshot['meta']['storage_id']
        assert second_snapshot['meta']['version']==0
        assert snapshot(first)==first_snapshot
        assert snapshot(q)==before
        assert one(53)['value']==1
        one(8,q);assert one(59,q)['value']==1
        assert snapshot(second)['meta']['version']==1
        one(70,q);assert one(59,q)['value']==0 and one(91,q)['value']==0
        assert one(94,q)['value']==1 and one(53)['value']==1
        report['observations'].append({'first_gradient':first_snapshot,'second_gradient':second_snapshot,
            'independent_new_Tensor_and_Storage':True,'held_old_gradient_unchanged':True,
            'retention_does_not_mutate_parameter':True,'zero_and_None_keep_retention_flag':True})
        owner=s.owner()['bytes']
        assert owner[64:72]==b'HTRET001' and owner[139264:139272]==b'HTORET01'
        report.update(status='PASS',new_state_tags_observed=True)
    except Exception as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        report['frames']=s.frames;report['transport']=s.finish()
        assert all(f.read_bytes()==raw for f,raw in inputs.items())
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','frames','source_sha256','native_sha256','numerical_parity_claimed')}))
if __name__=='__main__':main()
