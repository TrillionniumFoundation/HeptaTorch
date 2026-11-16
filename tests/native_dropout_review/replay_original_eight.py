"""Additive exact replay of native_dropout rows 167..174 in the immutable 183 report."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback

REFERENCE='c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467'
def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','protocol','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs={f.resolve():f.read_bytes() for f in (a.binary,a.source,a.protocol,a.reference,Path(__file__))}
    assert sha(inputs[a.reference.resolve()])==REFERENCE
    reference=json.loads(inputs[a.reference.resolve()]);assert reference['cases']==183 and reference['probe_error_count']==0
    assert len(reference['oracle_provenance']['checks'])==11 and all(reference['oracle_provenance']['checks'].values())
    spec=importlib.util.spec_from_file_location('original_native_dropout_wire',a.protocol)
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    rows=[]
    for index in range(167,175):
        actual=reference['rows'][index];assert actual['family']=='native-dropout-contrast'
        row={'index':index,'name':actual['name'],'status':'RUNNING','checks':0,'differences':[]}
        case=out/str(index);case.mkdir();s=w.Session(a.binary.resolve(),directory=case/'raw')
        def one(*cmd):return s.run([cmd])[0]
        def eq(name,got,want):
            row['checks']+=1
            if got!=want:row['differences'].append({'field':name,'actual':got,'expected':want})
        def snapshot(t,ref,name):
            m=w.meta(one(13,t));info=struct.unpack('<4Q',one(96,t)['payload'])
            raw=one(98,t,0,m['numel'])['payload']
            vals=list(raw) if info[0] else list(struct.unpack('<'+'I'*m['numel'],raw))
            result={'dtype':'torch.bool' if info[0] else 'torch.float32','shape':m['shape'][:m['rank']],
                    'strides':m['strides'][:m['rank']],'storage_offset':m['storage_offset']//info[1],
                    'version':m['version'],'requires_grad':bool(m['requires_grad']),'is_leaf':bool(info[3]),'words':vals}
            for key,value in result.items():eq(name+'.'+key,value,ref[key])
            eq(name+'.actual_width',info[1],1 if ref['dtype']=='torch.bool' else 4)
            eq(name+'.true_byte_extent',m['storage_bytes'],len(ref['words'])*info[1])
            return m
        try:
            v=actual['input_before'];shape=v['shape'];raw=struct.pack('<'+'I'*len(v['words']),*v['words'])
            x=one(1,len(shape),0,*shape,*([0]*(4-len(shape))),w.Payload(raw,4))['value']
            gen=one(60,1729)['value'];state=bytes.fromhex(reference['states'][actual['state_before']]['hex'])
            one(64,gen,w.Payload(state),5056)
            request=(99,x,gen,actual['probability_double_bits'],1)
            if actual['accepted']:
                r=one(*request);y=r['value'];mask=struct.unpack('<Q',r['payload'])[0]
                xm=snapshot(x,actual['input_after'],'input');ym=snapshot(y,actual['results'][0],'output')
                snapshot(mask,actual['results'][1],'mask')
                eq('output_same_Tensor',y==x,actual['result0_is_input'])
                eq('output_same_Storage',ym['storage_id']==xm['storage_id'],actual['result0_aliases_input_storage'])
                one(41,mask)
                if y!=x:one(41,y)
            else:
                before=s.owner();status,records=w.parse(s.exchange(w.packet([request])));after=s.owner()
                eq('invalid_probability_status',status,7);eq('no_failure_records',len(records),0)
                eq('whole_owner_rollback',sha(after['bytes']),sha(before['bytes']))
                eq('same_owner_mapping',after['pointer'],before['pointer'])
                snapshot(x,actual['input_after'],'input')
            state=one(63,gen)['payload'];eq('complete_5056B_state',sha(state),actual['state_after']);eq('state_bytes',len(state),5056)
            one(41,x);one(66,gen);eq('audit_after_release',one(53)['value'],1)
            row['status']='FAIL' if row['differences'] else 'PASS'
        except BaseException as e:row.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
        finally:
            row['frames']=s.frames
            try:s.finish()
            except BaseException as e:row.update(status='FAIL',transport_error=repr(e))
            (case/'report.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    assert all(f.read_bytes()==raw for f,raw in inputs.items())
    report={'schema':'heptatorch.native-dropout-original-eight-additive.v1','status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
            'source_sha256':sha(inputs[a.source.resolve()]),'native_sha256':sha(inputs[a.binary.resolve()]),'reference_sha256':REFERENCE,
            'reference_indices':list(range(167,175)),'rows':rows,'cases':8,'passed':sum(r['status']=='PASS' for r in rows),
            'frames':sum(r['frames'] for r in rows),'checks':sum(r['checks'] for r in rows),
            'original_183_report_and_replayer_unchanged':True,'Torch_executed_here':False,'full_pytorch_rewrite_complete':False,
            'inputs_sha256':{str(f):sha(raw) for f,raw in inputs.items()}}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','cases','passed','frames','checks')}))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
