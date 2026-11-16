"""Native typed-storage smoke. Expected bytes are explicit inputs, not model output."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def create_frame(dtype, shape, raw, requires=0):
    config = struct.pack('<8Q', len(shape), requires, *shape, *([0] * (4-len(shape))), 160, len(raw))
    return b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)+struct.pack('<8Q',97,dtype,96,0,0,0,0,0)+config+raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'protocol', 'output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs={f.resolve():f.read_bytes() for f in (a.binary,a.source,a.protocol,Path(__file__))}
    spec=importlib.util.spec_from_file_location('dtype_slice_wire',a.protocol)
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    s=w.Session(a.binary.resolve(),directory=out/'raw')
    report={'schema':'heptatorch.native-dtype-storage-slice.v1','status':'RUNNING',
            'native_sha256':sha(inputs[a.binary.resolve()]),'source_sha256':sha(inputs[a.source.resolve()]),
            'inputs_sha256':{str(f):sha(raw) for f,raw in inputs.items()},'native_executed':True,
            'Torch_executed_here':False,'checks':[],'rollback_bytes':0}
    def one(*cmd): return s.run([cmd])[0]
    def check(name,condition,details=None):
        assert condition,(name,details)
        report['checks'].append({'name':name,'status':'PASS','details':details})
    def create(dtype,shape,raw,requires=0):
        status,rows=w.parse(s.exchange(create_frame(dtype,shape,raw,requires)))
        assert status==0 and len(rows)==1,('create',status)
        return rows[0]['value']
    def meta(t):return w.meta(one(13,t))
    def error(commands,status):
        before=s.owner();body=s.exchange(w.packet(commands));got,rows=w.parse(body);after=s.owner()
        assert got==status and not rows,('error status',got,status)
        assert (before['pointer'],before['length'],sha(before['bytes'])) == (after['pointer'],after['length'],sha(after['bytes'])),'rollback mismatch'
        report['rollback_bytes']+=before['length']
    try:
        data=bytes((0,1,1,0,1,0));x=create(1,[2,3],data)
        m=meta(x);info=struct.unpack('<4Q',one(96,x,0)['payload'])
        check('real_bool_storage',m['storage_bytes']==6 and info[:2]==(1,1),(m,info))
        check('raw_typed_read',one(98,x,0,6)['payload']==data)
        check('audit_after_bool_create',one(53)['value']==1)
        view=one(69,x,w.Payload(struct.pack('<10Q',2,3,2,0,0,1,3,0,0,0)),80)['value']
        logical=bytes((0,0,1,1,1,0))
        check('noncontiguous_bool_view',one(98,view,0,6)['payload']==logical and meta(view)['strides'][:2]==[1,3])
        clone=one(18,view)['value'];detached=one(19,view)['value']
        check('clone_preserves_bool_layout',meta(clone)['strides'][:2]==[1,3] and meta(clone)['storage_bytes']==6 and meta(clone)['storage_id']!=m['storage_id'] and one(98,clone,0,6)['payload']==logical)
        check('detach_shares_actual_storage_version',meta(detached)['storage_id']==m['storage_id'] and meta(detached)['version_id']==m['version_id'])
        saved=one(46,view)['value'];unpacked=one(56,saved)['value']
        check('saved_unpack_keeps_true_dtype',struct.unpack('<4Q',one(96,unpacked,0)['payload'])[:2]==(1,1) and meta(unpacked)['storage_id']==m['storage_id'] and one(98,unpacked,0,6)['payload']==logical)
        check('audit_after_views_and_saved',one(53)['value']==1)
        g=one(60,1729)['value'];rng_before=one(63,g)['payload']
        for cmd,status in [((10,x),7),((39,x,0,6),7),((95,x,w.bits(1)),7),((90,x,1),5),((68,x,g,w.bits(0),w.bits(1),1),7),((2,1,x,x),7),((58,x,0,0),5),((12,x,0,2),7)]:
            error([cmd],status)
        check('bool_f32_gradient_rng_rejections',one(63,g)['payload']==rng_before)
        before=s.owner();status,rows=w.parse(s.exchange(create_frame(1,[2],b'\x00\x01',1)));after=s.owner()
        check('bool_requires_create_rejected',status==5 and not rows and before==after)
        one(100,view,0,0)
        check('bool_fill_shared_version',one(98,x,0,6)['payload']==bytes(6) and meta(x)['version']==1 and one(98,clone,0,6)['payload']==logical)
        error([(47,saved)],6)
        for t in (view,x,detached,unpacked,clone):one(41,t)
        one(48,saved);one(66,g)
        check('final_all_owned_rows_released',one(53)['value']==1)
        stats=struct.unpack('<12Q',one(54)['payload'])
        check('only_descriptor_page_storage_and_hold_remain',stats[5:11]==(0,1,0,0,0,1) and stats[4]==stats[1]-256*640,stats)
        report['status']='PASS'
    except BaseException as e:
        report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
    finally:
        report['frames']=s.frames
        try:report['transport']=s.finish()
        except BaseException as e:report.update(status='FAIL',transport_error=repr(e))
        assert all(f.read_bytes()==raw for f,raw in inputs.items())
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','frames','source_sha256','native_sha256')}))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
