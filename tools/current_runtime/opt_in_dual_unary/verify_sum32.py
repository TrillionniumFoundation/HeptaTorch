#!/usr/bin/env python3
"""Independently verify the actual terminal Sum32 report and transport evidence."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys
import tarfile
import types

SOURCE='f4c43f037a3fb046425f7634215ec382aa5f8d6b0e7abe126ef64236adb5753a'
IMAGE='5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72'
QUALIFIER='ad0beeee9884c313db46a089011043e634cc255d06718f6349b3377ecfc98618'

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def selected(repo,relative):
    p=repo/relative;raw=p.read_bytes()
    need(p.resolve()==p and sha(raw)==QUALIFIER,'fixed canonical independent reader differs')
    return p,raw

def load(p,raw,name,**values):
    module=types.ModuleType(name);module.__file__=str(p);module.__dict__.update(values)
    exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),module.__dict__)
    return module

def inspect_packets(raw, response, census):
    at=rt=count=0
    while at<len(raw):
        need(at+16<=len(raw) and rt+16<=len(response),'truncated outer frame')
        magic,n=struct.unpack_from('<8sQ',raw,at);at+=16
        rmagic,m=struct.unpack_from('<8sQ',response,rt);rt+=16
        need(magic==b'HTOWNR01' and rmagic==b'HTOWNS01' and at+n<=len(raw)
             and rt+m<=len(response) and n>=32 and m>=32 and raw[at:at+8]==b'HTORCH03'
             and response[rt:rt+8]==b'HTORSP03','invalid paired historical native frame')
        c,extent,reserved=struct.unpack_from('<3Q',raw,at+8)
        need(32+64*c<=n==extent and reserved==0,'invalid complete native request table')
        op=[struct.unpack_from('<Q',raw,at+32+64*i)[0] for i in range(c)]
        need(not (set(op)&{3,127}),'request leaves exact unchanged source domain')
        version,returned,status=struct.unpack_from('<3Q',response,rt+8)
        need(version==3,'historical response version differs')
        if status:
            need(returned==0 and m==32,'complete original refusal expected')
        else:
            need(returned==c,'complete original response census expected')
            pos=rt+32
            for word in op:
                need(pos+32<=rt+m,'truncated response record')
                code,error,value,size=struct.unpack_from('<4Q',response,pos)
                need(code==word and pos+32+size<=rt+m,'original record identity or extent')
                pos+=32+size
            need(pos==rt+m,'extra original response bytes')
        census.update(op);count+=1;at+=n;rt+=m
    need(rt==len(response),'unpaired response bytes')
    return count

def independent(repo,out,build_sha):
    p,raw=selected(repo,'tests/qualify_current_runtime.py')
    fn,=[node for node in ast.parse(raw).body if isinstance(node,ast.FunctionDef) and node.name=='qualify_sum32']
    original=ast.get_source_segment(raw.decode(),fn).encode()+b'\n'
    reader=load(p,original,'unchanged_independent_sum32_readback',IMAGE=IMAGE,require=need,
        SUM32_REFERENCE='c14e4b18f657429ad5dac5498c3abcae303b6a7e7550db7d15a3c9179bdcd728')
    proc=out.parent/'worker-execution'
    receipt=json.loads((proc/'runner.json').read_bytes())
    original_capture=Path(__file__).with_name('replay_sum32_original_capture.py')
    capture_sha=sha(original_capture.read_bytes())
    wrapper=out.parent/'worker.py'
    need(receipt['started'] is True and receipt['exit_code']==0 and receipt['stdout_EOF'] is True
         and receipt['stderr_EOF'] is True and receipt['stderr_bytes']==0
         and receipt['forced_cleanup'] is False and not receipt.get('timed_out',False)
         and not receipt['cleanup_errors'] and not receipt['running_descendants_after_cleanup']
         and receipt['elapsed_seconds']<receipt['timeout_seconds']
         and receipt['argv'][1:]==['-I','-S','-B',str(wrapper)]
         and capture_sha.encode() in wrapper.read_bytes(),'actual original worker did not complete')
    for stream in ('stdout','stderr'):
        data=(proc/('runner.'+stream)).read_bytes()
        need(len(data)==receipt[stream+'_bytes'] and sha(data)==receipt[stream+'_sha256'],'supervised stream differs')
    report=json.loads((out/'report.json').read_bytes())
    counts=reader.qualify_sum32(report,build_sha)
    need(report['source_sha256']==SOURCE and report['native_sha256']==IMAGE,'actual candidate report identity')
    census=set(); sessions=[]
    for i, row in enumerate(report['transports']):
        d=out/('case-%02d'%i)/'raw'
        files=('raw/requests.frames.bin','raw/responses.frames.bin','report.json')
        if all((d.parent/n).is_file() for n in files):
            payload={n:(d.parent/n).read_bytes() for n in files}
        else:
            archive=out.parent/'closed-case-payload-archives'/f'case-{i:02d}.tar.gz'
            pins=json.loads(archive.with_suffix('.json').read_bytes())
            need(sha(archive.read_bytes())==pins['archive_sha256'],'closed-case container identity differs')
            payload={}
            with tarfile.open(archive,'r:gz') as packet:
                for m in packet:
                    expected=pins['members'][m.name]
                    content=packet.extractfile(m).read()
                    need(m.isfile() and m.mode==expected['mode'] and len(content)==expected['bytes']
                         and sha(content)==expected['sha256'],'physical archive stream differs')
                    payload[m.name]=content
            need(set(payload)=={'raw/requests.frames.bin','raw/responses.frames.bin','report.json'},'closed-case census differs')
        raw=payload['raw/requests.frames.bin'];case=json.loads(payload['report.json'])
        ledger=json.loads((d/'transport.json').read_bytes())
        need(case==report['cases'][i] and all(ledger[k]==v for k,v in row.items()),'original case/transport readback differs')
        rsp=payload['raw/responses.frames.bin']
        count=inspect_packets(raw,rsp,census)
        need(count==ledger['frames'],'physical request frame count differs')
        sessions.append(dict(index=i,frames=count,request_sha256=sha(raw),response_sha256=sha(rsp)))
    return dict(status='PASS_RETAINED_SUM32',readback=counts,physical_sessions=sessions,opcodes=sorted(census),
        native_image_sha256=IMAGE,source_sha256=SOURCE,actual_build_sha256=build_sha,
        original_report_sha256=sha((out/'report.json').read_bytes()), production_activated=False,
        old_pass_transferred=False, full_candidate_suite_qualification=False)

if __name__=='__main__':
    need(not sys.flags.optimize,'assertions must remain enabled')
    repo,actual,build=map(lambda p:Path(p).resolve(),sys.argv[1:])
    print(json.dumps(independent(repo,actual,sha((build/'build.json').read_bytes())),sort_keys=True,indent=2))
