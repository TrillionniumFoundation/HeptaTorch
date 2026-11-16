"""Replay fixed real int64 alias mutations with actual native Scalar command106.

Opaque Tensor/Storage relations, Version changes, typed bytes, physical extents
and view flags are observed. Python wrapper addresses and _base pointers remain
diagnostic; no shared Version inference is made from equal initial counters.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import traceback

REFERENCE='335dd5cf5a8d3db1fb19d441c3eb5875f8ef940a11f1df9b2c8688bd3e095631'
PROTOCOL='7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
MASK=(1<<64)-1
sha=lambda b:hashlib.sha256(b).hexdigest()


def frame(shape,raw):
    config=struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),160,len(raw))
    return b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)+struct.pack('<8Q',97,2,96,0,0,0,0,0)+config+raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','protocol','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs={p.resolve():p.read_bytes() for p in (a.binary,a.source,a.protocol,a.protocol.with_name('owner_client.py'),a.reference,Path(__file__))}
    assert sha(inputs[a.reference.resolve()])==REFERENCE and sha(inputs[a.protocol.resolve()])==PROTOCOL
    actual=json.loads(inputs[a.reference.resolve()]);assert actual['status']=='OBSERVED' and actual['cases']==8 and actual['probe_error_count']==0
    assert len(actual['oracle_provenance']['checks'])==11 and all(v is True for v in actual['oracle_provenance']['checks'].values())
    source=inputs[a.source.resolve()]
    bodies={n:re.search(rb'^fn '+n.encode()+rb'\(.*?^end\n',source,re.M|re.S).group() for n in ('object_address','so_data_base','so_slot_at','dptr')}
    assert b'{"value":139264}' in bodies['object_address'] and b'{"value":552}' in bodies['so_data_base'] and b'{"value":640}' in bodies['dptr']
    spec=importlib.util.spec_from_file_location('alias8_wire',a.protocol);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    rows=[]
    for ref in actual['rows']:
        directory=out/f"case-{ref['index']:02d}";directory.mkdir();s=w.Session(a.binary.resolve(),directory=directory/'raw')
        row={'index':ref['index'],'name':ref['name'],'status':'RUNNING','checks':0,'raw_words':0,'rng_bytes':0,'snapshots':[],'differences':[]}
        tensors={};storages={};observation=ref['observation']
        def eq(name,got,want):
            row['checks']+=1
            if got!=want:
                def diagnostic(v):return {'bytes':len(v),'sha256':sha(v),'first64':v[:64].hex()} if isinstance(v,bytes) else v
                row['differences'].append({'field':name,'actual':diagnostic(got),'expected':diagnostic(want)})
        def one(*command):return s.run([command])[0]
        def identity(table,key,value,label):
            if key in table:eq(label+'.stable',value,table[key])
            else:eq(label+'.distinct',value not in table.values(),True);table[key]=value
        def raw_parts(token):
            owner=s.owner()['bytes'];q=lambda at:struct.unpack_from('<Q',owner,at)[0]
            assert owner[64:72]==b'HTMXAD02' and owner[139264:139272]==b'HTOMXD02'
            slots=q(139280);slot=token&0xffffff;assert token>>24&255==1 and 0<slot<=slots
            tp=139264+128+(slot-1)*64;assert q(tp)==token>>32 and q(tp+8)==1
            node=q(tp+48);assert 0<node<q(8)
            data=139264+144+552*slots;page=q(4096+(node>>8)*8);assert page>>63==1
            dp=data+(page&((1<<63)-1))+(node&255)*640;assert q(dp+304)==1 and q(dp+192)==token
            sid=q(dp+216);ss=sid&0xffffff;assert sid>>24&255==2 and 0<ss<=slots
            sp=139264+128+(slots+ss-1)*64;assert q(sp)==sid>>32 and q(sp+8)==1 and q(sp+40)>0 and q(sp+56)==0
            at=data+q(sp+16);n=q(sp+24);assert at<=len(owner) and n<=len(owner)-at
            return {'incarnation':q(dp+320),'dtype':q(dp+576),'is_view':bool(q(dp+408)),
                    'storage_id':sid,'version_id':q(dp+224),'physical':owner[at:at+n]}
        def snapshot(token,expected,label):
            m=w.meta(one(13,token));info=struct.unpack('<4Q',one(96,token,0)['payload']);raw=one(98,token,0,m['numel'])['payload'];fields=raw_parts(token)
            got={'dtype':'torch.int64' if info[0]==2 else str(info[0]),'element_size':info[1],
                 'numel':m['numel'],'shape':m['shape'][:m['rank']],'strides':m['strides'][:m['rank']],
                 'storage_offset_elements':m['storage_offset']//8,'storage_bytes':m['storage_bytes'],
                 'version':m['version'],'requires_grad':bool(m['requires_grad']),'is_leaf':bool(info[3]),
                 'is_view':fields['is_view'],'grad_fn':None if info[3] else 'UNMAPPED',
                 'logical_bytes':len(raw),'logical_bytes_hex':raw.hex(),'storage_bytes_hex':fields['physical'].hex(),
                 'signed_values_decimal':[str(x) for x in struct.unpack('<'+'q'*m['numel'],raw)]}
            for key,value in got.items():eq(label+'.'+key,value,expected[key])
            eq(label+'.descriptor_incarnation',fields['incarnation'],info[2]);eq(label+'.descriptor_dtype',fields['dtype'],2)
            eq(label+'.descriptor_storage',fields['storage_id'],m['storage_id']);eq(label+'.descriptor_version',fields['version_id'],m['version_id'])
            identity(tensors,expected['tensor_identity'],info[2],label+'.Tensor');identity(storages,expected['storage_identity'],m['storage_id'],label+'.Storage')
            row['raw_words']+=m['numel'];row['snapshots'].append({'label':label,'token':token,'incarnation':info[2],'storage_id':m['storage_id'],'version_id':m['version_id'],'version':m['version'],'reference_python_identity_diagnostic_only':expected['python_identity'],'reference_base_identity_diagnostic_only':expected['base_tensor_identity']})
            return m
        def pair(base,alias,expected,label):
            b=snapshot(base,expected['base'],label+'.base')
            if alias is None:eq(label+'.alias_absent',expected['alias'],None);return b,None
            return b,snapshot(alias,expected['alias'],label+'.alias')
        def rng(generator,key,label):
            want=bytes.fromhex(actual['states'][key]['hex']);assert len(want)==5056 and sha(want)==key
            eq(label,one(63,generator)['payload'],want);row['rng_bytes']+=len(want)
        try:
            gen=one(60,1729)['value'];rng(gen,observation['state_before_base'],'state_before_base')
            raw=b'' if ref['empty'] else struct.pack('<6q',-(1<<63),(1<<63)-1,-1,0,1,3)
            status,records=w.parse(s.exchange(frame(ref['base_shape'],raw)));assert status==0 and len(records)==1;base=records[0]['value']
            eq('base_construction.accepted',True,observation['base_construction']['accepted']);pair(base,None,observation['initial_base'],'initial_base');rng(gen,observation['state_after_base'],'state_after_base')
            kind=ref['construction']
            if kind=='detach':alias=one(19,base)['value']
            elif kind=='data-alias':alias=one(55,base)['value']
            elif kind=='clone-transpose':
                temp=one(6,base,0,1)['value'];alias=one(18,temp)['value'];one(41,temp)
            elif kind=='as-strided':
                shape,strides,offset=([2,0,3],[17,101,2],7) if ref['empty'] else ([2,2],[3,1],1)
                cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),offset)
                alias=one(69,base,w.Payload(cfg),80)['value']
            else:raise AssertionError(kind)
            eq('alias_construction.accepted',True,observation['alias_construction']['accepted'])
            b,a_meta=pair(base,alias,observation['after_alias_construction'],'after_alias');rng(gen,observation['state_after_alias'],'state_after_alias')
            events=observation['events'];assert len(events)==2
            for index,event in enumerate(events):
                pair(base,alias,event['before'],f'event{index}.before');rng(gen,event['state_before_call'],f'event{index}.RNGbefore')
                if index==0:
                    assert event['name']=='alias.fill_(INT64_MIN)';record=one(100,alias,1<<63,0);returned=alias
                    eq('fill_native_success_scalar',record['value'],1)
                else:
                    assert event['name']=='base.add_(1)';record=one(106,base,1);returned=record['value'];eq('Scalar_add_returns_same_public_token',returned,base)
                eq(f'event{index}.accepted',True,event['operation']['accepted']);snapshot(returned,event['returned'],f'event{index}.returned');pair(base,alias,event['after'],f'event{index}.after');rng(gen,event['state_after_call'],f'event{index}.RNGafter')
            shared_witness=(events[0]['after']['base']['version']==events[0]['before']['base']['version']+1 and events[1]['after']['alias']['version']==events[1]['before']['alias']['version']+1)
            eq('native_Version_identity_matches_two_actual_mutation_directions',b['version_id']==a_meta['version_id'],shared_witness)
            pair(base,alias,observation['final'],'final');rng(gen,ref['state_after'],'state_after')
            eq('audit',one(53)['value'],1)
            for token in (base,alias):one(41,token)
            one(66,gen);eq('audit_after_release',one(53)['value'],1)
            stats=struct.unpack('<12Q',one(54)['payload']);eq('remaining_owned_live_counts',stats[5:11],(0,1,0,0,0,1))
            row['status']='PASS' if not row['differences'] else 'FAIL'
        except BaseException as error:row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        finally:
            row['native_frames']=s.frames
            try:row['transport']=s.finish()
            except BaseException as error:row.update(status='FAIL',transport_error=repr(error))
            (directory/'report.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    assert all(p.read_bytes()==raw for p,raw in inputs.items())
    report={'schema':'heptatorch.int64-alias8-native-replay.v1','status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
            'cases':len(rows),'passed':sum(r['status']=='PASS' for r in rows),'rows':rows,
            'native_frames':sum(r['native_frames'] for r in rows),'exact_checks':sum(r['checks'] for r in rows),
            'raw_words':sum(r['raw_words'] for r in rows),'rng_bytes':sum(r['rng_bytes'] for r in rows),
            'source_sha256':sha(inputs[a.source.resolve()]),'native_sha256':sha(inputs[a.binary.resolve()]),
            'reference_sha256':REFERENCE,'native_executed':True,'torch_executed_here':False,'all_inputs_unchanged':True,
            'inputs_sha256':{str(p):sha(raw) for p,raw in inputs.items()},'physical_observer_bodies_sha256':{n:sha(raw) for n,raw in bodies.items()},
            'Scalar_add_is_real_native106':True,'Tensor_RHS_substitution':False,
            'Version_identity_qualified_by_two_mutation_directions':True,'initial_equal_counter_identity_inference':False,
            'fill_wire_result_contract':'Native100 returns success scalar1; actual self-return snapshot reads the same bound Tensor, without constructing an alias.',
            'diagnostic_only':['Python wrapper addresses','_base Python object pointer'],
            'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','cases','passed','native_frames','exact_checks','raw_words','rng_bytes')}))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
