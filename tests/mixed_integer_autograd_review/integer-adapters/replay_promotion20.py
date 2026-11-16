"""Strict raw-value/layout replay of the forty original mixed operations.

Explicit Torch to() calls are reference diagnostics for CPU promotion order,
not a claimed native public Tensor.to API. Legacy F32 command69 supplies the
same input layout/S/V through its documented detached-alias contract; its
Python _base/is_view semantics are not advertised as a new F32 view API.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import struct
import traceback
from native_observer import Native,sha

REFERENCE='91623503499e1e8868d84b9154aa9b36aae5c0acda0d8f8804ffc0f48ab8b765'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('binary','source','protocol','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    files=[a.binary,a.source,a.protocol,a.protocol.with_name('owner_client.py'),a.reference,Path(__file__),Path(__file__).with_name('native_observer.py')]
    inputs={p.resolve():p.read_bytes() for p in files};assert sha(inputs[a.reference.resolve()])==REFERENCE
    actual=json.loads(inputs[a.reference.resolve()]);assert actual['status']=='OBSERVED' and actual['cases']==20 and actual['probe_error_count']==0
    assert len(actual['oracle_provenance']['checks'])==11 and all(v is True for v in actual['oracle_provenance']['checks'].values())
    rows=[]
    for ref in actual['rows']:
        directory=out/f"case-{ref['index']:02d}";directory.mkdir();n=Native(a.binary,a.source,a.protocol,directory)
        row={'index':ref['index'],'name':ref['name'],'status':'RUNNING','checks':0,'raw_words':0,'rng_bytes':0,'differences':[],'mapped_original_mixed_operations':0}
        tensors={};storages={};roots={};results=[];obs=ref['observation']
        def eq(label,got,want):
            row['checks']+=1
            if got!=want:row['differences'].append({'field':label,'actual':got,'expected':want})
        def identity(table,key,value,label):
            if key in table:eq(label+'.stable',value,table[key])
            else:eq(label+'.distinct',value not in table.values(),True);table[key]=value
        def snapshot(token,expected,label):
            value=n.tensor(token)
            for field in ('dtype','element_size','numel','shape','storage_bytes','version','requires_grad','is_leaf','grad_fn','logical_bytes','logical_bytes_hex','storage_bytes_hex'):
                eq(label+'.'+field,value[field],expected[field])
            eq(label+'.strides',value['strides'],[int(x) for x in expected['strides_elements_decimal']])
            eq(label+'.offset',value['storage_offset_elements'],int(expected['storage_offset_elements_decimal']))
            if expected['dtype']=='torch.int64' or expected['base_tensor_identity'] is None:
                eq(label+'.is_view',value['is_view'],expected['is_view'])
            if expected['dtype']=='torch.int64' and expected['base_tensor_identity'] is not None:
                eq(label+'.actual_base_Tensor',value['native_base_incarnation'],tensors[expected['base_tensor_identity']])
            identity(tensors,expected['tensor_identity'],value['native_incarnation'],label+'.Tensor')
            identity(storages,expected['storage_identity'],value['native_storage_id'],label+'.Storage')
            row['raw_words']+=value['numel']
        def originals(refs,label):
            eq(label+'.root_names',sorted(roots),sorted(refs))
            for key,token in roots.items():snapshot(token,refs[key],label+'.'+key)
        def rng(key,label):
            wanted=actual['states'][key]['hex'];assert len(bytes.fromhex(wanted))==5056 and sha(bytes.fromhex(wanted))==key
            eq(label,n.one(63,gen)['payload'].hex(),wanted);row['rng_bytes']+=5056
        try:
            gen=n.one(60,1729)['value'];rng(ref['state_before'],'initial.RNG')
            for kind,dtype in (('i64',2),('f32',0)):
                root_ref=next(x for x in obs['constructions'] if x['name']==kind+'_root')['returned']
                assert root_ref['shape']==[32]
                root=n.create(dtype,[32],bytes.fromhex(root_ref['logical_bytes_hex']));roots[kind+'_root']=root
                snapshot(root,root_ref,'construction.'+kind+'_root')
                stride=[int(x) for x in ref[kind+'_strides_elements_decimal']];offset=int(ref[kind+'_offset_elements_decimal'])
                alias=n.view(root,ref['shape'],stride,offset);roots[kind]=alias
                view_ref=next(x for x in obs['constructions'] if x['name']==kind+'_view')['returned']
                snapshot(alias,view_ref,'construction.'+kind+'_view')
            originals(obs['constructed'],'constructed')
            eq('common_dtype',obs['common_dtype'],'torch.float32')
            left,right=(roots['i64'],roots['f32']) if ref['order']=='i64-first' else (roots['f32'],roots['i64'])
            for event,op in zip(obs['events'][2:],(1,7)):
                assert event['name']==('torch.add(original_lhs, original_rhs)' if op==1 else 'torch.true_divide(original_lhs, original_rhs)')
                originals(event['before_originals'],event['name']+'.before');rng(event['state_before_call'],event['name']+'.RNGbefore')
                result=n.one(103,op,left,right)['value'];results.append(result)
                eq(event['name']+'.accepted',True,event['operation']['accepted'])
                snapshot(result,event['returned'],event['name']+'.returned')
                originals(event['after_originals'],event['name']+'.after');rng(event['state_after_call'],event['name']+'.RNGafter')
                eq(event['name']+'.audit',n.one(53)['value'],1);row['mapped_original_mixed_operations']+=1
            originals(obs['final_originals'],'final');rng(ref['state_after'],'final.RNG')
            for token in list(roots.values())+results:n.one(41,token)
            n.one(66,gen);eq('audit_after_all_drop',n.one(53)['value'],1)
            stats=struct.unpack('<12Q',n.one(54)['payload']);eq('remaining_owned_live_counts',list(stats[5:11]),[0,1,0,0,0,1])
            row['status']='PASS' if not row['differences'] else 'FAIL'
        except BaseException as error:row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        finally:
            row['native_frames']=n.s.frames
            try:row['transport']=n.finish()
            except BaseException as error:row.update(status='FAIL',transport_error=repr(error))
            (directory/'report.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    assert all(p.read_bytes()==raw for p,raw in inputs.items())
    report={'schema':'heptatorch.int64-mixed-original40-replay.v1','status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
            'cases':20,'passed':sum(r['status']=='PASS' for r in rows),'original_mixed_operations':sum(r['mapped_original_mixed_operations'] for r in rows),
            'native_frames':sum(r['native_frames'] for r in rows),'exact_checks':sum(r['checks'] for r in rows),'raw_words':sum(r['raw_words'] for r in rows),'rng_bytes':sum(r['rng_bytes'] for r in rows),'rows':rows,
            'source_sha256':sha(inputs[a.source.resolve()]),'native_sha256':sha(inputs[a.binary.resolve()]),'reference_sha256':REFERENCE,
            'native_executed':True,'torch_executed_here':False,'all_inputs_unchanged':True,'inputs_sha256':{str(p):sha(raw) for p,raw in inputs.items()},
            'public_Tensor_to_implemented':False,'reference_to_calls_diagnostic_only':40,
            'legacy_F32_as_strided_input_boundary':'Command69 is a detached alias with identical values/shape/strides/S/V; F32 _base/is_view is not mapped as a new view API. Numeric/layout/counter assertions are exact.',
            'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','cases','passed','original_mixed_operations','native_frames','exact_checks')}))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
