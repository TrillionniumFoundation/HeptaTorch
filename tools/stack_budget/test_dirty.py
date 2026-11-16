#!/usr/bin/env python3
"""Observe actual native ABI3 compiler/linker on clean, dirty and reused scratch."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import mmap
import struct
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sources',type=Path,required=True)
parser.add_argument('--native',type=Path,required=True)
parser.add_argument('--review',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();S=args.sources.resolve();N=args.native.resolve();R=args.review.resolve();O=args.output.resolve();O.mkdir(parents=True,exist_ok=False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
pin=json.loads((N/'bootstrap.json').read_text());adapt=json.loads((S/'adaptation.json').read_text())
libc=ctypes.CDLL(None);libc.mprotect.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
rows=[];arenas=[]


def callable_entry(raw,hole):
    assert raw[hole-1]==0xe8
    entry=hole+4+struct.unpack_from('<i',raw,hole)[0]
    memory=mmap.mmap(-1,len(raw)+4096,prot=mmap.PROT_READ|mmap.PROT_WRITE)
    base=ctypes.addressof(ctypes.c_char.from_buffer(memory));memory[:len(raw)]=raw
    # Test-only SysV observation bridge preserves all callee-saved registers.
    # The tested HIA entry still receives three disjoint borrowed descriptors.
    bridge=bytes.fromhex('55 48 89 e5 53 41 54 41 55 41 56 41 57 48 83 ec 08 48 b8')+struct.pack('<Q',base+entry)
    bridge+=bytes.fromhex('ff d0 48 83 c4 08 41 5f 41 5e 41 5d 41 5c 5b c9 c3')
    at=(len(raw)+15)&-16;memory[at:at+len(bridge)]=bridge
    assert libc.mprotect(base,len(raw)+4096,mmap.PROT_READ|mmap.PROT_EXEC)==0
    arenas.append(memory)
    return ctypes.CFUNCTYPE(None,ctypes.c_void_p,ctypes.c_void_p)(base+at),entry,sha(bridge)


for name,executable,expected,request,workspace_bytes,output_bytes,profile in (
    ('compiler',N/'stack-module-compiler',pin['compiler_sha256'],(R/'fanout.hepta').read_bytes(),2097152,1048576,'compiler'),
    ('linker',N/'stack-linker-qualification',pin['linker_sha256']['qualification'],(R/'fanout-path-not-sum.request').read_bytes(),16777216,8388608,'linker')):
    raw=executable.read_bytes();assert sha(raw)==expected
    hole=adapt['linkers'][profile]['call_holes'][0][1]
    call,entry,bridge_sha=callable_entry(raw,hole)
    cli=subprocess.run([str(executable)],input=request,capture_output=True,env={'PATH':''},timeout=30)
    assert cli.returncode==0 and not cli.stderr
    (O/(name+'.cli.stdout')).write_bytes(cli.stdout)
    old_workspace=None
    for mode in ('clean','dirty','reused'):
        source=ctypes.create_string_buffer(b'\xc3'*64+request+b'\xc3'*64)
        output=ctypes.create_string_buffer(b'\xa7'*(output_bytes+128))
        if mode=='reused':workspace=old_workspace
        else:workspace=ctypes.create_string_buffer(b'\x6d'*64+bytes([0x5a if mode=='dirty' else 0])*workspace_bytes+b'\x6d'*64)
        arguments=(ctypes.c_uint64*6)(ctypes.addressof(source)+64,len(request),ctypes.addressof(output)+64,output_bytes,
                                     ctypes.addressof(workspace)+64,workspace_bytes)
        results=(ctypes.c_uint64*2)()
        call(arguments,results)
        count=results[0];assert count==len(cli.stdout),(name,mode,count,len(cli.stdout))
        out=bytes(output)[64:64+count]
        assert out==cli.stdout
        assert bytes(source)[:64]==bytes(source)[64+len(request):128+len(request)]==b'\xc3'*64
        assert bytes(source)[64:64+len(request)]==request
        assert bytes(output)[:64]==bytes(output)[64+output_bytes:128+output_bytes]==b'\xa7'*64
        assert bytes(workspace)[:64]==bytes(workspace)[64+workspace_bytes:128+workspace_bytes]==b'\x6d'*64
        old_workspace=workspace
        (O/(name+'-'+mode+'.stdout')).write_bytes(out)
        rows.append({'tool':name,'mode':mode,'status':'PASS','private_entry_offset':entry,'output_sha256':sha(out),
                     'output_bytes':count,'input_unchanged':True,'all_buffer_canaries_intact':True,
                     'workspace_bytes':workspace_bytes,'test_only_bridge_sha256':bridge_sha})
    assert executable.read_bytes()==raw
report={'schema':'heptatorch.stack-budget-native-dirty-workspace.v1','status':'PASS','cases':len(rows),
        'compiler_sha256':pin['compiler_sha256'],'linker_sha256':pin['linker_sha256']['qualification'],
        'reviewer_sha256':sha(Path(__file__).read_bytes()),'rows':rows,
        'host_libc_mprotect_used_only_for_observation_mapping':True,'target_allocator_or_numeric_runtime_on_host':False,
        'scope':'Native compiler/linker scratch initialization and bounds; this host observation bridge is not the production startup or its stack proof.'}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
