"""Native bool Saved-Tensor identity, weak/strong lifetime and migration controls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback


def sha(raw):return hashlib.sha256(raw).hexdigest()


def raw_create(dtype,shape,raw):
    return (b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)+struct.pack('<8Q',97,dtype,96,0,0,0,0,0)
            +struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),160,len(raw))+raw)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    paths=(a.binary,a.source,a.protocol,Path(__file__));inputs={p.resolve():p.read_bytes() for p in paths}
    spec=importlib.util.spec_from_file_location('typed_lifetime_wire',a.protocol)
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    s=w.Session(a.binary.resolve(),directory=out/'raw');checks=[];rollback=0
    def one(*args):return s.run([args])[0]
    def check(name,condition,details=None):
        assert condition,(name,details);checks.append({'name':name,'status':'PASS','details':details})
    def error(commands,status):
        nonlocal rollback
        before=s.owner();got,rows=w.parse(s.exchange(w.packet(commands)));after=s.owner()
        check('expected_status_'+str(status),got==status and not rows,(got,status))
        check('whole_owner_rollback',(before['pointer'],before['length'],sha(before['bytes']))==(after['pointer'],after['length'],sha(after['bytes'])))
        rollback+=before['length']
    def meta(t):return w.meta(one(13,t))
    def info(t):return struct.unpack('<4Q',one(96,t)['payload'])
    def create(shape,raw):
        status,rows=w.parse(s.exchange(raw_create(1,shape,raw)));assert status==0,status
        return rows[0]['value']
    report={'schema':'heptatorch.typed-native-lifetimes.v1','status':'RUNNING','source_sha256':sha(inputs[a.source.resolve()]),
            'native_sha256':sha(inputs[a.binary.resolve()]),'native_executed':True,'Torch_executed_here':False,
            'inputs_sha256':{str(p):sha(raw) for p,raw in inputs.items()}}
    try:
        x=one(*w.tensor([8],[.25]*8,1))['value'];gen=one(60,1729)['value']
        rec=one(99,x,gen,0x3fe0000000000000,1);y=rec['value'];mask=struct.unpack('<Q',rec['payload'])[0]
        mask_info=info(mask);mask_meta=meta(mask);weak=one(42,mask)['value'];one(41,mask)
        hold=one(43,weak)['value'];check('Saved_retains_actual_mask_storage',hold!=0)
        error([(50,hold,0,1)],7);error([(50,hold,0,0)],7)
        one(45,hold)
        again=one(96,y,1)['value']
        check('republish_same_Tensor_new_token',again!=mask and info(again)==mask_info)
        check('republish_same_Storage_Version',meta(again)==mask_meta)
        one(41,again);seed=one(*w.tensor([8],[1]*8))['value']
        check('default_backward',one(58,y,seed,0)['value']==0)
        error([(43,weak)],6);error([(96,y,1)],10);error([(98,mask,0,1)],1)
        check('audit_after_actual_Saved_Tensor_release',one(53)['value']==1)
        one(41,y);one(44,weak);one(70,x)

        rec=one(99,x,gen,0x3fe0000000000000,1);y=rec['value'];mask=struct.unpack('<Q',rec['payload'])[0]
        weak=one(42,mask)['value'];saved=one(46,mask)['value'];one(41,mask)
        check('retained_backward_keeps_mask_Tensor',one(58,y,seed,1)['value']==0)
        again=one(96,y,1)['value'];one(100,again,0,0);one(41,again)
        rec=one(58,y,seed,0);check('mask_mutation_partial_error',rec['value']==6)
        hold=one(43,weak)['value'];one(45,hold)
        one(41,y);hold=one(43,weak)['value'];one(45,hold)
        check('public_Saved_keeps_Storage_after_AD_mask_Tensor_died',one(53)['value']==1)
        error([(47,saved)],6);one(48,saved);error([(43,weak)],6);one(44,weak)
        one(70,x);one(41,x);one(41,seed);one(66,gen)

        a0=create([6],b'\x00\x01\x01\x00\x01\x00');before=meta(a0)
        saved=one(46,a0)['value'];weak=one(42,a0)['value']
        pages=[]
        for i in range(270):pages.append(create([0],b''))
        sizes=[];large=[]
        for _ in range(3):
            t=create([2,200000],bytes((1,0))*200000);large.append(t);sizes.append(s.owner(False)['length'])
        check('real_descriptor_and_owner_growth',len(set(sizes))>1 and meta(a0)==before,sizes)
        check('after_growth_byte_identity',one(98,a0,0,6)['payload']==b'\x00\x01\x01\x00\x01\x00')
        for t in large:
            check('large_bool_real_byte_extent',meta(t)['storage_bytes']==400000)
            check('large_bool_tail_values',one(98,t,399994,6)['payload']==bytes((1,0))*3)
        check('audit_after_table_pool_page_growth',one(53)['value']==1)
        one(33,0)
        unpacked=one(56,saved)['value'];check('Saved_bool_dtype_survives_graph_reset',info(unpacked)[:2]==(1,1) and one(98,unpacked,0,6)['payload']==b'\x00\x01\x01\x00\x01\x00')
        check('audit_after_reset_unpack',one(53)['value']==1)
        one(41,unpacked);one(48,saved);error([(43,weak)],6);one(44,weak)
        check('final_audit',one(53)['value']==1)
        report['status']='PASS'
    except BaseException as e:report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
    finally:
        report.update(checks=checks,frames=s.frames,rollback_bytes=rollback)
        try:s.finish()
        except BaseException as e:report.update(status='FAIL',transport_error=repr(e))
        assert all(p.read_bytes()==raw for p,raw in inputs.items())
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','frames','rollback_bytes')}))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
