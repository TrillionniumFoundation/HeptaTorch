"""Strict replay of fourteen fixed, actually observed public Saved lifetime recipes.

This host is a wire/owner observer. It performs no derivative or Tensor
evaluation and never constructs expected values from candidate output.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import lzma
from pathlib import Path
import re
import struct
import traceback

REFERENCE = 'c80ee752f7d7fffe3e808eb4720de0a0447e146e1b20eadc4b219dbbff96c3b5'
PROBE = 'eca3e273d0afbb3265fca7c73451950d011f41fa1670a7d695911776657296ae'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
MASK = (1 << 64)-1
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def json_diagnostic(value):
    if isinstance(value, bytes):
        return {'diagnostic_type':'bytes','hex':value.hex(),'sha256':sha(value)}
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('binary','source','build-report','protocol','observer','reference','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    for key in ('expected-source-sha256','expected-native-sha256'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--indices',type=str)
    a = parser.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    paths = [a.binary,a.source,a.build_report,a.protocol,a.protocol.with_name('owner_client.py'),a.observer,a.reference,Path(__file__)]
    inputs = {p.resolve():p.read_bytes() for p in paths}
    assert sha(inputs[a.binary.resolve()]) == a.expected_native_sha256
    assert sha(inputs[a.source.resolve()]) == a.expected_source_sha256
    assert sha(inputs[a.reference.resolve()]) == REFERENCE
    assert sha(inputs[a.protocol.resolve()]) == PROTOCOL
    reference = json.loads(inputs[a.reference.resolve()])
    assert reference['status']=='OBSERVED' and reference['cases']==14 and reference['probe_error_count']==0
    assert reference['probe_sha256']==PROBE and reference['source_files_unchanged'] is True
    provenance=reference['oracle_provenance'];assert provenance['status']=='PASS' and len(provenance['checks'])==11 and all(v is True for v in provenance['checks'].values())
    build=json.loads(inputs[a.build_report.resolve()]);assert build['status']=='PASS'
    assert len(build['units'])==11 and build['descriptor_bytes']==640
    assert sha(inputs[a.observer.resolve()])=='32bce0992604b879a466c4c32fe5b7b378b75a368d9b2dd9494290eef95ebaee'
    assert sha(inputs[a.protocol.with_name('owner_client.py').resolve()])=='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
    assert build['source_sha256']==a.expected_source_sha256 and build['native_sha256']==a.expected_native_sha256
    for unit in build['units']:
        for suffix,field in (('.hepta','source_sha256'),('.o','object_sha256')):
            path=a.build_report.parent/(unit['name']+suffix);raw=path.read_bytes();assert sha(raw)==unit[field];inputs[path.resolve()]=raw
    spec=importlib.util.spec_from_file_location('saved_public14_native_observer',a.observer)
    observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
    selected=list(range(14)) if a.indices is None else [int(i) for i in a.indices.split(',')]
    assert len(set(selected))==len(selected) and all(0<=i<14 for i in selected)
    rows=[]
    def pack32(values):return struct.pack('<'+'f'*len(values),*values)
    def pack64(values):return struct.pack('<'+'Q'*len(values),*(v&MASK for v in values))
    for case in [reference['rows'][i] for i in selected]:
        directory=out/('case-'+str(case['index']).zfill(2));directory.mkdir()
        row={'index':case['index'],'name':case['name'],'status':'RUNNING','checks':0,'exact_words':0,'rollback_bytes':0,'rng_bytes':0,'differences':[],'phases':[]}
        n=observer.Native(a.binary,a.source,a.protocol,directory)
        live={}; roots=[]; previous={}; weak=None; generator=None
        def eq(label,got,want):
            row['checks']+=1
            if got!=want:
                row['differences'].append({'field':label,'actual':got,'expected':want})
        def one(*cmd):return n.one(*cmd)
        def create(dtype,shape,values,requires=0):
            return n.create(dtype,shape,pack32(values) if dtype==0 else pack64(values),requires)
        def drop(token):one(41,token)
        def owner_file(data):
            digest=sha(data);path=directory/(digest+'.owner.xz')
            if not path.exists():path.write_bytes(lzma.compress(data))
            return path.name
        def rejected(command,status,label):
            before=n.s.owner();got,records=n.w.parse(n.s.exchange(n.w.packet([command])));after=n.s.owner()
            eq(label+'.status',got,status);eq(label+'.records',records,[])
            eq(label+'.owner.pointer',after['pointer'],before['pointer']);eq(label+'.owner.length',after['length'],before['length'])
            eq(label+'.owner.sha256',sha(after['bytes']),sha(before['bytes']))
            assert after['bytes']==before['bytes'],('full owner rollback',label)
            row['rollback_bytes']+=len(before['bytes'])
            row.setdefault('rollbacks',[]).append({'label':label,'status':got,'before_pointer':before['pointer'],'after_pointer':after['pointer'],
                'bytes':len(before['bytes']),'sha256':sha(before['bytes']),'raw':owner_file(before['bytes'])})
            return got
        def status_of(expected):
            if expected['accepted']:return 0
            message=expected['error']['message']
            if 'modified by an inplace operation' in message:return 6
            if 'already been freed' in message or 'backward through the graph a second time' in message:return 10
            raise AssertionError(('unmapped actual error',expected['error']))
        def node_name(memory,index):
            node=memory.node(index)
            return {0:'AccumulateGrad',1:'AddBackward0',2:'SubBackward0',3:'MulBackward0',4:'DivBackward0',7:'SumBackward0',10:'TransposeBackward0',16:'ViewBackward0',17:'CloneBackward0'}.get(node[5], 'UNMAPPED-'+str(node[5]))
        def observe_tensor(token):
            got=n.tensor(token);memory=n.memory();node=memory.node(got['native_node'])
            got['storage_offset_bytes']=got['storage_offset_elements']*got['element_size']
            got['grad_fn']=None if got['is_leaf'] else node_name(memory,got['native_node'])
            got['next_functions']=[] if got['is_leaf'] else [node_name(memory,node[6+k]) if node[9+k] else None for k in range(2)]
            return got
        def compare_tensor(label,got,want,current):
            keys=('dtype','element_size','numel','shape','strides','storage_bytes','storage_offset_elements','storage_offset_bytes','version','requires_grad','is_leaf','is_view','grad_fn','next_functions','logical_bytes_hex')
            for key in keys:eq(label+'.'+key,got[key],want[key])
            eq(label+'.base_is_None',got['native_base_incarnation'] is None,want['base_tensor_identity'] is None)
            row['exact_words']+=want['numel']
            identity=(got['native_incarnation'],got['native_storage_id'],want['tensor_identity'],want['storage_identity'])
            if label in previous:
                old=previous[label]
                eq(label+'.Tensor_stable',old[0]==identity[0],old[2]==identity[2])
                eq(label+'.Storage_stable',old[1]==identity[1],old[3]==identity[3])
            current[label]=identity
            if got['native_base_incarnation'] is not None:
                current[label+'._base']=(got['native_base_incarnation'],got['native_storage_id'],want['base_tensor_identity'],want['storage_identity'])
        def snapshot(want,label):
            current={}; details={'label':label,'tensors':{},'gradients':{},'saved':{}}
            eq(label+'.live_names',sorted(live),sorted(want['tensors']))
            for name,token in live.items():
                got=observe_tensor(token);compare_tensor('Tensor.'+name,got,want['tensors'][name],current);details['tensors'][name]=got
            for name,event in want['gradients'].items():
                before=n.memory();p=before.node(before.public_node(live[name]));g=p[48]
                was_public=False if not g else bool(before.node(g)[25])
                presence=one(59,live[name])['value'];token=one(91,live[name])['value']
                eq(label+'.G.'+name+'.None',token==0,event['tensor'] is None)
                eq(label+'.G.'+name+'.presence',presence,0 if event['tensor'] is None else 1)
                if token:
                    got=observe_tensor(token);compare_tensor('Gradient.'+name,got,event['tensor'],current);details['gradients'][name]=got
                    if not was_public:drop(token)
                else:details['gradients'][name]=None
            for name,saved in want['saved'].items():
                eq(label+'.saved.'+name+'.grad_fn',details['tensors'][name]['grad_fn'],saved['grad_fn'])
                details['saved'][name]={}
                for attribute,event in saved['attributes'].items():
                    if attribute=='_saved_alpha':
                        # The admitted add/sub overload is alpha=1; no scalar alpha field exists.
                        eq(label+'.saved.'+name+'.alpha',1,event['value']);continue
                    slot={'_saved_self':0,'_saved_other':1}[attribute]
                    expected_status=status_of(event)
                    item_label='Saved.'+name+'.'+attribute
                    if expected_status:
                        rejected((107,live[name],slot),expected_status,label+'.'+item_label)
                        details['saved'][name][attribute]={'status':expected_status};continue
                    memory=n.memory();public_before={memory.node(i)[40] for i in range(1,memory.q(8)) if memory.node(i)[38] and memory.node(i)[25]}
                    token=one(107,live[name],slot)['value']
                    eq(label+'.'+item_label+'.None',token==0,'tensor' not in event)
                    if not token:
                        eq(label+'.'+item_label+'.value',None,event.get('value'));continue
                    got=observe_tensor(token);compare_tensor(item_label,got,event['tensor'],current)
                    eq(label+'.'+item_label+'.live_Tensor_matches',[key for key in live if details['tensors'][key]['native_incarnation']==got['native_incarnation']],event['live_Tensor_matches'])
                    eq(label+'.'+item_label+'.live_Storage_matches',[key for key in live if details['tensors'][key]['native_storage_id']==got['native_storage_id']],event['live_Storage_matches'])
                    details['saved'][name][attribute]=got
                    if got['native_incarnation'] not in public_before:drop(token)
            pairs=list(current.items())
            for (left,l),(right,r) in itertools.combinations(pairs,2):
                eq(label+'.'+left+':'+right+'.Tensor_identity',l[0]==r[0],l[2]==r[2])
                eq(label+'.'+left+':'+right+'.Storage_identity',l[1]==r[1],l[3]==r[3])
            previous.clear();previous.update(current)
            if weak is not None:
                expired=want['original_Storage_expired']
                if expired:rejected((43,weak),6,label+'.weak_expired')
                else:
                    hold=one(43,weak)['value'];eq(label+'.weak_lock_live',hold!=0,True);one(45,hold)
                details['original_Storage_expired']=expired
            raw=one(63,generator)['payload'];eq(label+'.complete_rng',raw.hex(),reference['rng_states'][want['rng_state']]['hex']);row['rng_bytes']+=len(raw)
            eq(label+'.audit',one(53)['value'],1)
            row['phases'].append(details)
        try:
            generator=one(60,1729)['value']
            floating=[1.5,-2.,4.,.5,8.,-4.]
            integers=[2,3,5,7,16777217,-16777219]
            live['x']=create(0,[2,3],floating,1)
            original_name='x' if case['family']=='self' else 'i'
            if case['family']!='self':
                if case['layout']=='integer_holes':
                    base=create(2,[24],list(range(2,26)))
                    live['i']=n.view(base,[2,3],[9,2],1);drop(base)
                else:
                    live['i']=create(2,[2,3],integers)
            weak=one(42,live[original_name])['value']
            snapshot(case['input_before'],'input_before')
            if case['family']=='self':
                live['y']=one(2,3 if case['operation']=='mul' else 4,live['x'],live['x'])['value']
            else:
                left,right=(live['x'],live['i']) if case['float_side']=='left' else (live['i'],live['x'])
                live['y']=one(103,3 if case['operation']=='mul' else 7,left,right)['value']
            live['seed']=create(0,[2,3],[-0.,1.,-1.,.5,2.,-2.])
            eq('forward.accepted',True,case['forward']['accepted'])
            snapshot(case['after_forward'],'after_forward')
            for event in case['events']:
                label=event['event']
                if label=='drop_original_public':
                    drop(live.pop(event['target']))
                elif label=='hold_saved_public':
                    slot={'_saved_self':0,'_saved_other':1}[event['attribute']]
                    expected_status=status_of(event['result'])
                    command=(107,live['y'],slot)
                    if expected_status:
                        rejected(command,expected_status,label+'.'+event['name'])
                    else:
                        token=one(*command)['value']
                        eq(label+'.'+event['name']+'.Tensor_present',token!=0,'tensor' in event['result'])
                        if token:
                            live[event['name']]=token
                elif label=='backward_default':
                    record=one(58,live['y'],live['seed'],0)
                    error,completed,leaves,failing=struct.unpack('<4Q',record['payload'])
                    eq(label+'.runtime_status',error,status_of(event['result']))
                    eq(label+'.record_error',record['value'],error)
                    row.setdefault('backwards',[]).append(dict(event=label,runtime_status=error,
                        completed_nodes=completed,completed_leaves=leaves,failing_opcode=failing))
                elif label in ('drop_graph','drop_graph_after_backward'):
                    drop(live.pop('y'))
                elif label=='drop_all_held_saved':
                    # A native public capability has one token. Both Python aliases are
                    # deliberately removed together in the fixed reference recipe.
                    tokens={live.pop(name) for name in event['names']}
                    for token in tokens:
                        if token not in live.values():
                            drop(token)
                else:
                    raise AssertionError(('unmapped actual event',label))
                snapshot(event['after'],label)
            for token in set(live.values()):drop(token)
            live.clear();one(44,weak);weak=None;one(66,generator);generator=None
            eq('final.audit',one(53)['value'],1)
            stats=struct.unpack('<12Q',one(54)['payload']);eq('final.live_counts',stats[5:11],(0,1,0,0,0,1))
            row['status']='PASS' if not row['differences'] else 'FAIL'
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
            try:
                owner=n.s.owner();row['failed_owner']={'bytes':owner['length'],'sha256':sha(owner['bytes']),'file':owner_file(owner['bytes'])}
            except BaseException as second:row['owner_read_error']=repr(second)
        finally:
            row['frames']=n.s.frames
            try:row['transport']=n.finish()
            except BaseException as error:row.update(status='FAIL',transport_error=repr(error))
            (directory/'report.json').write_text(json.dumps(row,indent=2,default=json_diagnostic)+'\n');rows.append(row)
    unchanged=all(p.read_bytes()==raw for p,raw in inputs.items());assert unchanged
    report={'schema':'heptatorch.saved-public-lifetime14-replay.v1',
        'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL','cases':len(rows),'passed':sum(r['status']=='PASS' for r in rows),
        'failed':sum(r['status']!='PASS' for r in rows),'frames':sum(r['frames'] for r in rows),'checks':sum(r['checks'] for r in rows),
        'exact_words':sum(r['exact_words'] for r in rows),'rollback_bytes':sum(r['rollback_bytes'] for r in rows),'rng_bytes':sum(r['rng_bytes'] for r in rows),
        'reference_sha256':REFERENCE,'probe_sha256':PROBE,'source_sha256':a.expected_source_sha256,'native_sha256':a.expected_native_sha256,
        'reference_indices':selected,'rows':rows,'inputs_sha256':{str(p):sha(raw) for p,raw in inputs.items()},'all_inputs_unchanged':unchanged,
        'provenance_checks':provenance['checks'],'native_executed':True,'Torch_executed_here':False,
        'numeric_comparison':'EXACT raw bytes, with no tolerance','saved_accessor':'107 returns original Tensor; held Saved capabilities survive graph release; temporary observer tokens are released immediately',
        'backward_surface':'58 partial-commit runtime statuses; Python exception wording not claimed',
        'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590}
    (out/'report.json').write_text(json.dumps(report,indent=2,default=json_diagnostic)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','passed','failed','frames','checks','exact_words','rollback_bytes')}))
    return int(report['failed']!=0)


if __name__=='__main__':raise SystemExit(main())
