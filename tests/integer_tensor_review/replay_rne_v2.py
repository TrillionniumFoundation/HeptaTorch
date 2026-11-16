"""Compare genuine HIA i64->f32 promotion against fixed actual Torch raw bytes.

No Python float constructs inputs or expected outputs. Direct Tensor.to was
observed separately and is cross-checked, not advertised as a native public API.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback

REFERENCE='dc96ca9a2a224cb5878de35debda36d9be7c1bf522d1bfa28a9354a8cb4c5962'
PROBE='8083747140bd717972dde78967be7f93da3e2ff7d9f00e7dd7a1797270ef8268'
PROTOCOL='7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
sha=lambda b:hashlib.sha256(b).hexdigest()


def frame(dtype,shape,raw):
    cfg=struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),160,len(raw))
    return b'HTORCH03'+struct.pack('<3Q',1,160+len(raw),0)+struct.pack('<8Q',97,dtype,96,0,0,0,0,0)+cfg+raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','protocol','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs={v.resolve():v.read_bytes() for v in (a.binary,a.source,a.protocol,a.protocol.with_name('owner_client.py'),a.reference,Path(__file__))}
    assert sha(inputs[a.reference.resolve()])==REFERENCE
    assert sha(inputs[a.protocol.resolve()])==PROTOCOL
    actual=json.loads(inputs[a.reference.resolve()]);assert actual['status']=='OBSERVED' and actual['cases']==5 and actual['expected_integers']==1237 and actual['probe_error_count']==0 and actual['probe_sha256']==PROBE
    assert len(actual['oracle_provenance']['checks'])==11 and all(v is True for v in actual['oracle_provenance']['checks'].values())
    spec=importlib.util.spec_from_file_location('rne_native_wire',a.protocol);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    rows=[]
    for index,ref in enumerate(actual['rows']):
        row={'index':index,'name':ref['name'],'status':'RUNNING','integers':len(ref['signed_inputs']),'checks':0,'differences':[]}
        directory=out/f'case-{index:02d}';directory.mkdir();s=w.Session(a.binary.resolve(),directory=directory/'raw')
        def eq(name,got,want):
            row['checks']+=1
            if got!=want:row['differences'].append({'field':name,'actual':got,'expected':want})
        def one(*cmd):return s.run([cmd])[0]
        def create(dtype,shape,raw):
            status,records=w.parse(s.exchange(frame(dtype,shape,raw)));assert status==0 and len(records)==1
            return records[0]['value']
        try:
            raw=bytes.fromhex(ref['input_i64_le_hex']);values=struct.unpack('<'+'q'*row['integers'],raw)
            assert list(values)==ref['signed_inputs'] and len(raw)==ref['input_bytes']
            x=create(2,[row['integers']],raw);zero=create(0,[],b'\0'*4)
            result=one(103,1,x,zero)['value'];got=one(98,result,0,row['integers'])['payload']
            expected=bytes.fromhex(ref['promoted_add_f32_le_hex']);direct=bytes.fromhex(ref['direct_f32_le_hex'])
            assert list(struct.unpack('<'+'I'*row['integers'],expected))==ref['promoted_u32_words']
            assert list(struct.unpack('<'+'I'*row['integers'],direct))==ref['direct_u32_words']
            eq('actual_direct_cast_and_promoted_add_identical',direct,expected)
            assert len(got)==len(expected)
            for n,(native,want) in enumerate(zip(struct.unpack('<'+'I'*row['integers'],got),ref['promoted_u32_words'])):
                eq(f'raw_binary32[{n}]',native,want)
            m=w.meta(one(13,result));info=struct.unpack('<4Q',one(96,result,0)['payload'])
            eq('shape',m['shape'][:m['rank']],[row['integers']]);eq('strides',m['strides'][:m['rank']],ref['promoted_strides'])
            eq('version',m['version'],ref['promoted_version']);eq('dtype',info[0],0);eq('width',info[1],4)
            eq('input_unmodified',one(98,x,0,row['integers'])['payload'],raw)
            eq('audit',one(53)['value'],1)
            for token in (x,zero,result):one(41,token)
            eq('audit_after_drop',one(53)['value'],1)
            row['status']='PASS' if not row['differences'] else 'FAIL'
        except BaseException as error:row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        finally:
            try:row['transport']=s.finish()
            except BaseException as error:row.update(status='FAIL',close_error=repr(error))
            row['native_frames']=s.frames
            (directory/'report.json').write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row)
    assert all(p.read_bytes()==raw for p,raw in inputs.items())
    report={'schema':'heptatorch.int64-rne-native-replay.v1','status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
            'cases':len(rows),'integers':sum(r['integers'] for r in rows),'exact_checks':sum(r['checks'] for r in rows),
            'native_frames':sum(r['native_frames'] for r in rows),'rows':rows,
            'source_sha256':sha(inputs[a.source.resolve()]),'native_sha256':sha(inputs[a.binary.resolve()]),
            'reference_sha256':REFERENCE,'reference_probe_sha256':PROBE,'native_executed':True,'torch_executed_here':False,
            'native_entry':'103 add int64 Tensor and binary32 scalar zero','public_Tensor_to_implemented':False,
            'all_inputs_unchanged':True,'inputs_sha256':{str(p):sha(raw) for p,raw in inputs.items()},
            'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','cases','integers','exact_checks','native_frames')}))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
