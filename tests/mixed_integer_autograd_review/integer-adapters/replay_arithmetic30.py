"""Replay the thirty arithmetic/promotion fixed actual54 recipes against native HIA.

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
    fmt={0:'I',1:'B',2:'Q'}[dtype]
    raw = struct.pack('<'+fmt*len(values), *(x & MASK for x in values))
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
    for case in actual['rows'][12:42]:
        row={'index':case['index'],'name':case['name'],'status':'RUNNING','checks':0,
             'differences':[],'rollback_bytes':0}
        directory=out/('case-'+str(case['index']).zfill(2));directory.mkdir()
        s=w.Session(a.binary.resolve(),directory=directory/'raw')
        identities=[]
        def eq(name,got,want):
            row['checks']+=1
            if got!=want:row['differences'].append({'field':name,'actual':got,'expected':want})
        def one(*cmd):return s.run([cmd])[0]
        def create(shape,values,requires=0,dtype=2):
            status,records=w.parse(s.exchange(frame(dtype,shape,values,requires)))
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
            observation=case['observation'];roots={};op=case['operation'];family=case['family']
            roots['base' if family in ('binary','inplace') else 'x']=create([2,3],edge)
            operations={'add':1,'sub':2,'mul':3,'bitwise_and':4,'bitwise_or':5,'bitwise_xor':6}
            if family=='binary':
                roots['other']=create([2,3],[-1,2,5,7,maximum,minimum])
                if case['layout']=='transpose':
                    roots['x']=one(6,roots['base'],0,1)['value']
                    roots['y']=one(6,roots['other'],0,1)['value']
                elif case['layout']=='broadcast':
                    roots['x']=roots['base'];roots['y']=create([3],[minimum,1,maximum])
                else:
                    roots['x']=roots['base'];roots['y']=roots['other']
                command=(103,operations[op],roots['x'],roots['y'])
            elif family=='inplace':
                roots['x']=one(6,roots['base'],0,1)['value']
                roots['y']=create([3,1],[-1,1,3])
                command=(104,operations[op],roots['x'],roots['y'])
            elif family=='promotion':
                if op=='i64-f32-add':
                    roots['y']=create([3],[0x3f000000,0xc0000000,0x40800000],dtype=0)
                    command=(103,1,roots['x'],roots['y'])
                elif op=='f32-i64-mul':
                    roots['y']=create([2,1],[0x3f000000,0xc0000000],dtype=0)
                    command=(103,3,roots['y'],roots['x'])
                elif op in ('i64-bool-add','bool-i64-mul'):
                    roots['y']=create([3],[1,0,1],dtype=1)
                    command=(103,1,roots['x'],roots['y']) if op=='i64-bool-add' else (103,3,roots['y'],roots['x'])
                elif op=='i64-true-divide':
                    roots['y']=create([3],[2,3,-1])
                    command=(103,7,roots['x'],roots['y'])
                elif op=='requires-grad-inplace':command=(90,roots['x'],1)
                else:raise AssertionError(op)
            else:raise AssertionError(family)
            eq('before.named_roots',sorted(roots),sorted(observation['before']))
            for name,t in roots.items():snapshot(t,observation['before'][name],'before.'+name)
            eq('state_before_operation',one(63,generator)['payload'].hex(),actual['states'][observation['state_before_operation']]['hex'])
            result=None
            if op=='requires-grad-inplace':
                before=s.owner();status,records=w.parse(s.exchange(w.packet([command])));after=s.owner()
                eq('operation.accepted',status==0,observation['operation']['accepted'])
                eq('operation.status',status,5);eq('failure.records',records,[])
                eq('failure.owner.pointer',after['pointer'],before['pointer'])
                eq('failure.owner.length',after['length'],before['length'])
                eq('failure.owner.bytes',sha(after['bytes']),sha(before['bytes']))
                row['rollback_bytes']+=before['length']
                (directory/'owner-before-error.bin').write_bytes(before['bytes'])
                (directory/'owner-after-error.bin').write_bytes(after['bytes'])
            else:
                result=one(*command)['value']
                eq('operation.accepted',True,observation['operation']['accepted'])
            for name,t in roots.items():snapshot(t,observation['after'][name],'after.'+name)
            if result is not None:snapshot(result,observation['results'][0],'result')
            else:eq('result.None',observation['results'],[None])
            for left,right in itertools.combinations(identities,2):
                eq(left[0]+':'+right[0]+'.Tensor_identity',left[1]==right[1],left[3]==right[3])
                eq(left[0]+':'+right[0]+'.Storage_identity',left[2]==right[2],left[4]==right[4])
            eq('state_after_operation',one(63,generator)['payload'].hex(),actual['states'][observation['state_after_operation']]['hex'])
            eq('audit_after_operation',one(53)['value'],1)
            for token in set(roots.values())|({result} if result is not None else set()):one(41,token)
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
    report={'schema':'heptatorch.integer-arithmetic-actual30-replay.v1',
        'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
        'source_sha256':sha(inputs[a.source.resolve()]),'native_sha256':sha(inputs[a.binary.resolve()]),
        'reference_sha256':REFERENCE,'cases':len(rows),'passed':sum(r['status']=='PASS' for r in rows),
        'failed':sum(r['status']!='PASS' for r in rows),'frames':sum(r['frames'] for r in rows),
        'checks':sum(r['checks'] for r in rows),'rollback_bytes':sum(r['rollback_bytes'] for r in rows),
        'rows':rows,'inputs_sha256':{str(p):sha(raw) for p,raw in inputs.items()},
        'physical_observer_bodies_sha256':{n:sha(raw) for n,raw in bodies.items()},
        'all_inputs_unchanged':unchanged,'native_executed':True,'Torch_executed_here':False,
        'reference_indices':list(range(12,42)),'additional_reference_rows_qualified':0,
        'initial_equal_Version_does_not_claim_shared_Version':True,
        'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','passed','failed','frames','checks')}))
    return int(report['failed']!=0)


if __name__=='__main__':raise SystemExit(main())
