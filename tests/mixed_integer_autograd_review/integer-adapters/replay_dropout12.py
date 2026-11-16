"""Replay the final twelve native_dropout actual54 recipes against native HIA.

Expected values, metadata and identity relations come from the immutable real
Torch report. Physical Storage bytes are read from the owned child without
creating observer Tensor aliases or writing into native memory.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import struct
import traceback

REFERENCE = '4c454e443c4147c740ea400cb1195bcbcb80e16e5b50debce54c068fa9ae712b'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
MASK = (1 << 64)-1


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def frame(dtype, shape, values, requires=0):
    raw = struct.pack('<'+'Q'*len(values), *(x & MASK for x in values))
    config = struct.pack('<8Q',len(shape),requires,*shape,*([0]*(4-len(shape))),160,len(raw))
    return b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)+struct.pack('<8Q',97,dtype,96,0,0,0,0,0)+config+raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','protocol','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs={x.resolve():x.read_bytes() for x in (a.binary,a.source,a.protocol,a.reference,Path(__file__),a.protocol.with_name('owner_client.py'))}
    assert sha(inputs[a.reference.resolve()])==REFERENCE
    assert sha(inputs[a.protocol.resolve()])==PROTOCOL
    actual=json.loads(inputs[a.reference.resolve()])
    assert actual['status']=='OBSERVED' and actual['cases']==54 and actual['probe_error_count']==0
    assert len(actual['oracle_provenance']['checks'])==11 and all(v is True for v in actual['oracle_provenance']['checks'].values())
    # These inherited three bodies determine the read-only physical observer.
    source=inputs[a.source.resolve()]
    bodies={name:re.search(rb'^fn '+name.encode()+rb'\(.*?^end\n',source,re.M|re.S).group()
            for name in ('object_address','so_data_base','so_slot_at')}
    assert b'{"value":139264}' in bodies['object_address']
    assert b'{"value":552}' in bodies['so_data_base'] and b'{"value":144}' in bodies['so_data_base']
    assert b'{"value":128}' in bodies['so_slot_at'] and b'{"value":64}' in bodies['so_slot_at']
    spec=importlib.util.spec_from_file_location('integer_storage_wire',a.protocol)
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    rows=[]
    minimum=-(1<<63);maximum=(1<<63)-1;edge=[minimum,maximum,-1,0,1,3]
    for case in actual['rows'][42:54]:
        row={'index':case['index'],'name':case['name'],'status':'RUNNING','checks':0,
             'differences':[],'rollback_bytes':0}
        directory=out/('case-'+str(case['index']).zfill(2));directory.mkdir()
        s=w.Session(a.binary.resolve(),directory=directory/'raw')
        identities=[]
        def eq(name,got,want):
            row['checks']+=1
            if got!=want:row['differences'].append({'field':name,'actual':got,'expected':want})
        def one(*cmd):return s.run([cmd])[0]
        def create(shape,values,requires=0):
            status,records=w.parse(s.exchange(frame(2,shape,values,requires)))
            assert status==0 and len(records)==1,('create',status)
            return records[0]['value']
        def view(base,shape,strides,offset=0):
            cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),offset)
            return one(69,base,w.Payload(cfg),80)['value']
        def physical(m):
            owner=s.owner()['bytes'];q=lambda off:struct.unpack_from('<Q',owner,off)[0]
            slots=q(139264+16);sid=m['storage_id'];slot=sid&0xffffff
            assert sid>>24&255==2 and 1<=slot<=slots
            off=139264+128+(slots+slot-1)*64
            rec=struct.unpack_from('<8Q',owner,off)
            assert rec[0]==sid>>32 and rec[1]==1 and rec[5]>0 and rec[7]==0
            assert rec[3]==m['storage_bytes'] and rec[4]>=rec[3]
            start=139264+552*slots+144+rec[2]
            assert start<=len(owner) and rec[3]<=len(owner)-start
            return owner[start:start+rec[3]]
        def snapshot(token,expected,label):
            m=w.meta(one(13,token));info=struct.unpack('<4Q',one(96,token,0)['payload'])
            logical=one(98,token,0,m['numel'])['payload']
            assert m['storage_offset']%info[1]==0
            got={'dtype':{0:'torch.float32',1:'torch.bool',2:'torch.int64'}[info[0]],
                 'element_size':info[1],'shape':m['shape'][:m['rank']],
                 'strides':m['strides'][:m['rank']],'numel':m['numel'],
                 'storage_bytes':m['storage_bytes'],'storage_offset_elements':m['storage_offset']//info[1],
                 'version':m['version'],'requires_grad':bool(m['requires_grad']),
                 'is_leaf':bool(info[3]),'grad_fn':None if info[3] else 'UNMAPPED',
                 'logical_bytes':len(logical),'logical_bytes_hex':logical.hex(),
                 'storage_bytes_hex':physical(m).hex()}
            for key,value in got.items():eq(label+'.'+key,value,expected[key])
            identities.append((label,info[2],m['storage_id'],expected['tensor_identity'],expected['storage_identity']))
        try:
            generator=one(60,1729)['value']
            observation=case['observation'];roots={};op='native_dropout'
            if case['layout']=='empty':
                roots['base']=create([0],[])
                roots['x']=view(roots['base'],[2,0,3],[17,101,2],7)
            else:
                base=create([2,3],edge)
                roots['x']=one(6,base,0,1)['value'];one(41,base)
            command=(105,roots['x'],generator,struct.unpack('<Q',struct.pack('<d',case['probability']))[0],int(case['train']))
            eq('before.named_roots',sorted(roots),sorted(observation['before']))
            for name,t in roots.items():snapshot(t,observation['before'][name],'before.'+name)
            eq('state_before_operation',one(63,generator)['payload'].hex(),actual['states'][observation['state_before_operation']]['hex'])
            stats_before=struct.unpack('<12Q',one(54)['payload'])
            rec=one(*command)
            runtime,mask_token=struct.unpack('<2Q',rec['payload'])
            result=rec['value'];mask=mask_token
            eq('operation.accepted',runtime==0,observation['operation']['accepted'])
            for name,t in roots.items():snapshot(t,observation['after'][name],'after.'+name)
            if runtime==0:
                eq('operation.runtime_error',runtime,0)
                snapshot(result,observation['results'][0],'output')
                snapshot(mask,observation['results'][1],'mask')
            else:
                eq('operation.runtime_error',runtime,12)
                eq('operation.reference_exception_type',observation['operation']['error']['type'],'RuntimeError')
                eq('operation.no_output_tokens',(result,mask),(0,0))
                eq('result.None',observation['results'],[None])
                stats_after=struct.unpack('<12Q',one(54)['payload'])
                eq('temporary.live_counts_released',stats_after[5:11],stats_before[5:11])
                eq('temporary.extents_returned',stats_after[4],stats_before[4])
            for left,right in itertools.combinations(identities,2):
                eq(left[0]+':'+right[0]+'.Tensor_identity',left[1]==right[1],left[3]==right[3])
                eq(left[0]+':'+right[0]+'.Storage_identity',left[2]==right[2],left[4]==right[4])
            eq('state_after_operation',one(63,generator)['payload'].hex(),actual['states'][observation['state_after_operation']]['hex'])
            eq('audit_after_operation',one(53)['value'],1)
            for token in set(roots.values())|({result,mask} if result != 0 else set()):one(41,token)
            one(66,generator);eq('audit_after_release',one(53)['value'],1)
            stats=struct.unpack('<12Q',one(54)['payload'])
            eq('remaining_owned_live_counts',stats[5:11],(0,1,0,0,0,1))
            row['status']='FAIL' if row['differences'] else 'PASS'
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        finally:
            row['frames']=s.frames
            try:row['transport']=s.finish()
            except BaseException as error:row.update(status='FAIL',transport_error=repr(error))
            (directory/'report.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    unchanged=all(p.read_bytes()==raw for p,raw in inputs.items());assert unchanged
    report={'schema':'heptatorch.integer-dropout-actual12-replay.v1',
        'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
        'source_sha256':sha(inputs[a.source.resolve()]),'native_sha256':sha(inputs[a.binary.resolve()]),
        'reference_sha256':REFERENCE,'cases':len(rows),'passed':sum(r['status']=='PASS' for r in rows),
        'failed':sum(r['status']!='PASS' for r in rows),'frames':sum(r['frames'] for r in rows),
        'checks':sum(r['checks'] for r in rows),'rollback_bytes':sum(r['rollback_bytes'] for r in rows),
        'rows':rows,'inputs_sha256':{str(p):sha(raw) for p,raw in inputs.items()},
        'physical_observer_bodies_sha256':{n:sha(raw) for n,raw in bodies.items()},
        'all_inputs_unchanged':unchanged,'native_executed':True,'Torch_executed_here':False,
        'reference_indices':list(range(42,54)),'additional_reference_rows_qualified':0,
        'initial_equal_Version_does_not_claim_shared_Version':True,
        'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','passed','failed','frames','checks')}))
    return int(report['failed']!=0)


if __name__=='__main__':raise SystemExit(main())
