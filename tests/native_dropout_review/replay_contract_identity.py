"""Strict replay of the original28 real outcomes, including same-T empty autograd history."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import struct
import traceback

REF='bc5a490f1623a4078050829798f502b0be934722f371adbcbb5612aaeb1a563c'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','protocol','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    files=(a.binary,a.source,a.protocol,a.reference,Path(__file__))
    inputs={f.resolve():f.read_bytes() for f in files};assert sha(inputs[a.reference.resolve()])==REF
    actual=json.loads(inputs[a.reference.resolve()]);assert actual['probe_error_count']==0 and actual['cases']==28
    assert all(actual['oracle_provenance']['checks'].values())
    spec=importlib.util.spec_from_file_location('dtype_contract_wire',a.protocol)
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    rows=[]
    for case in actual['rows']:
        row={'index':case['index'],'name':case['name'],'status':'RUNNING','checks':0,'differences':[]}
        directory=out/('case-'+str(case['index']).zfill(2));directory.mkdir()
        s=w.Session(a.binary.resolve(),directory=directory/'raw')
        def one(*cmd):return s.run([cmd])[0]
        def eq(name,got,wanted):
            row['checks']+=1
            if got!=wanted:row['differences'].append({'field':name,'actual':got,'expected':wanted})
        def error(commands,expected):
            before=s.owner();got,records=w.parse(s.exchange(w.packet(commands)));after=s.owner()
            eq('admission_status',got,expected);eq('failure_no_records',len(records),0)
            eq('failure_owner_pointer',after['pointer'],before['pointer'])
            eq('failure_owner_bytes',sha(after['bytes']),sha(before['bytes']))
            row['rollback_bytes']=row.get('rollback_bytes',0)+before['length']
        def snapshot(t,expected,name):
            m=w.meta(one(13,t));info=struct.unpack('<4Q',one(96,t,0)['payload'])
            raw=one(98,t,0,m['numel'])['payload']
            vals=list(raw) if info[0]==1 else list(struct.unpack('<'+'I'*m['numel'],raw))
            got={'dtype':'torch.bool' if info[0]==1 else 'torch.float32','shape':m['shape'][:m['rank']],
                 'strides':m['strides'][:m['rank']],'element_size':info[1],'storage_bytes':m['storage_bytes'],
                 'storage_offset_bytes':m['storage_offset'],'version':m['version'],
                 'requires_grad':bool(m['requires_grad']),'is_leaf':bool(info[3]),'binary_words':vals}
            if expected is not None:
                for key,value in got.items():eq(name+'.'+key,value,expected[key])
            return {'meta':m,'info':info,'values':vals}
        def gradient(t,expected,name):
            g=one(91,t)['value'];eq(name+'.None',g==0,expected is None)
            if g:
                snapshot(g,expected,name);one(41,g)
        def setup(expected):
            shape=expected['shape'];strides=expected['strides'];offset=expected['storage_offset_elements']
            count=expected['storage_bytes']//4;values=[0]*count
            for coords,value in zip(itertools.product(*(range(n) for n in shape)),expected['binary_words']):
                values[offset+sum(i*j for i,j in zip(coords,strides))]=value
            raw=struct.pack('<'+'I'*count,*values)
            base=one(1,1,0,count,0,0,0,w.Payload(raw,4))['value']
            cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),offset)
            x=one(69,base,w.Payload(cfg),80)['value']
            if expected['requires_grad']:one(90,x,1)
            one(41,base)
            return x
        def backward(output,seed,x,mask,ref,name,retain):
            rec=one(58,output,seed,int(retain));wanted=0
            if not ref['accepted']:
                wanted=11 if 'moved into the graph interior' in ref['error']['message'] else (10 if 'second time' in ref['error']['message'] else 6)
            eq(name+'.status',rec['value'],wanted)
            snapshot(x,ref['input_after'],name+'.input')
            snapshot(output,ref['output_after'],name+'.output')
            snapshot(mask,ref['mask_after'],name+'.mask')
            gradient(x,ref['input_gradient']['tensor'],name+'.gradient')
            state=one(63,generator)['payload'];eq(name+'.state',sha(state),ref['state_after'])
        try:
            x=setup(case['input_before']);generator=one(60,1729)['value']
            state=bytes.fromhex(actual['states'][case['state_before']]['hex'])
            one(64,generator,w.Payload(state),5056)
            training=2 if case['training'] is None else int(case['training'])
            request=(99,x,generator,case['probability_double_bits'],training)
            if not case['forward']['accepted']:
                error([request],7);eq('forward_rejected_rng',sha(one(63,generator)['payload']),case['state_after_forward'])
            else:
                rec=one(*request);output=rec['value'];mask=struct.unpack('<Q',rec['payload'])[0]
                before_out=snapshot(output,case['output'],'output');before_mask=snapshot(mask,case['mask'],'mask')
                input_after=snapshot(x,case['input_after_forward'],'input_after_forward')
                eq('output_is_input_Tensor',output==x,case['output_is_input_Tensor'])
                eq('output_aliases_input_Storage',before_out['meta']['storage_id']==input_after['meta']['storage_id'],case['output_aliases_input_Storage'])
                eq('mask_aliases_input_Storage',before_mask['meta']['storage_id']==input_after['meta']['storage_id'],case['mask_aliases_input_Storage'])
                eq('mask_aliases_output_Storage',before_mask['meta']['storage_id']==before_out['meta']['storage_id'],case['mask_aliases_output_Storage'])
                eq('forward_rng',sha(one(63,generator)['payload']),case['state_after_forward'])
                eq('mask_saved_same_Tensor',one(96,output,1)['value'],mask)
                error([(58,mask,0,0)],5)
                eq('audit_after_forward',one(53)['value'],1)
                seed_id=setup(case['seed'])
                snapshot(seed_id,case['seed'],'seed')
                retain=case['backward_path']!='default_then_retry'
                backward(output,seed_id,x,mask,case['first_backward'],'first_backward',retain)
                eq('audit_after_backward',one(53)['value'],1)
                one(92,x,0)
                if not retain:
                    error([(96,output,1)],10)
                    backward(output,seed_id,x,mask,case['retry_after_default_release'],'retry',False)
                else:
                    one(95,x,w.bits(.25))
                    snapshot(x,case['input_mutation']['input_after'],'input_mutation')
                    backward(output,seed_id,x,mask,case['backward_after_input_mutation'],'backward_after_input_mutation',True)
                    one(92,x,0);one(100,mask,0,0)
                    snapshot(mask,case['mask_mutation']['mask_after'],'mask_mutation')
                    error([(96,output,1)],6)
                    backward(output,seed_id,x,mask,case['backward_after_mask_mutation'],'backward_after_mask_mutation',False)
                eq('audit_after_final_event',one(53)['value'],1)
                for t in (seed_id,output,mask):
                    if t!=x:one(41,t)
            one(41,x);one(66,generator);eq('audit_final_release',one(53)['value'],1)
            if row['status']!='OPEN':row['status']='FAIL' if row['differences'] else 'PASS'
        except BaseException as e:
            row.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
        finally:
            row['frames']=s.frames
            try:s.finish()
            except BaseException as e:row.update(status='FAIL',transport_error=repr(e))
            (directory/'report.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    assert all(f.read_bytes()==raw for f,raw in inputs.items())
    report={'schema':'heptatorch.native-dropout-initial-reference-replay.v1','status':'FAIL' if any(r['status']=='FAIL' for r in rows) else 'PASS',
        'reference_sha256':REF,'source_sha256':sha(inputs[a.source.resolve()]),'native_sha256':sha(inputs[a.binary.resolve()]),
        'inputs_sha256':{str(f):sha(raw) for f,raw in inputs.items()},'cases':len(rows),
        'pass':sum(r['status']=='PASS' for r in rows),'fail':sum(r['status']=='FAIL' for r in rows),'open':sum(r['status']=='OPEN' for r in rows),
        'frames':sum(r['frames'] for r in rows),'checks':sum(r['checks'] for r in rows),'rows':rows,
        'Torch_executed_here':False,'reference_replayed':True,'full_pytorch_rewrite_complete':False,
        'private_saved_scalar_query':'Not exposed by the wire; scalar private attributes are not counted as matched.',
        'old_open_empty_forward_not_reclassified':True}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','pass','fail','open','frames','checks')}))
    return int(report['fail']!=0)


if __name__=='__main__':raise SystemExit(main())
