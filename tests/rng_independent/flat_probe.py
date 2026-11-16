"""Observe the actual pinned native RNG six-word bridge, in forked processes."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import mmap
import os
import resource
import struct

SHA=lambda b:hashlib.sha256(b).hexdigest()
IMAGE_PINS={'trap':'1a1daa8390af5d79e257973f0e4ce43ca759265e75d917ce4cf1c6cff0cac657',
            'return-zero':'cffc7c2f6574bf08a184e7d4f84e41a38ce0fb9be2f645bf9c332c93c2621eac'}


def observe(image, request, out_cap, work_cap, out_size=8192, work_size=8192):
    # This location is derived from the already-observed ELF startup call.
    assert image[:4]==b'\x7fELF' and image[0x151]==0xe8
    bridge=0x156+struct.unpack_from('<i',image,0x152)[0]
    assert bridge==0x1ea and image[bridge:bridge+8]==bytes.fromhex('554889e54883ec40')
    code=mmap.mmap(-1,len(image),prot=mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)
    code[:]=image
    incoming=mmap.mmap(-1,max(1,len(request)));incoming[:len(request)]=request
    output=mmap.mmap(-1,out_size);output[:]=bytes([0xa5])*out_size
    workspace=mmap.mmap(-1,work_size);workspace[:]=bytes([0x5a])*work_size
    def address(mapping):return ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping))
    signature=ctypes.CFUNCTYPE(ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint64,
                               ctypes.c_void_p,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_uint64)
    function=signature(address(code)+bridge)
    readfd,writefd=os.pipe();child=os.fork()
    if child==0:
        os.close(readfd);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        value=function(address(incoming),len(request),address(output),out_cap,address(workspace),work_cap)
        os.write(writefd,struct.pack('<I',value));os._exit(0)
    os.close(writefd);raw=os.read(readfd,4);os.close(readfd)
    _,status=os.waitpid(child,0)
    out,work=output[:],workspace[:]
    changed=[i for i,v in enumerate(out) if v!=0xa5]
    result={'output_capacity':out_cap,'workspace_capacity':work_cap,'wait_status':status,
            'signal':os.WTERMSIG(status) if os.WIFSIGNALED(status) else None,
            'exit_code':os.WEXITSTATUS(status) if os.WIFEXITED(status) else None,
            'returned_u32':struct.unpack('<I',raw)[0] if len(raw)==4 else None,
            'output_changed_bytes':len(changed),'output_changed_range':([min(changed),max(changed)] if changed else None),
            'output_prefix_hex':out[:32].hex(),'output_sha256':SHA(out),
            'workspace_unchanged':work==bytes([0x5a])*work_size,
            'outside_declared_output_capacity_unchanged':out[out_cap:]==bytes([0xa5])*(out_size-out_cap),
            'input_unchanged':incoming[:len(request)]==request}
    return result,out,work


def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True)
    p.add_argument('--expect-short',choices=('trap','return-zero'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    image=a.native.read_bytes();assert SHA(image)==IMAGE_PINS[a.expect_short]
    packet=b'HTRNG001'+struct.pack('<IIQQ',0,0,17,0)
    rows=[]
    for capacity in [0,1,7,8,15,16,23,24,5079,5080]:
        row,out,work=observe(image,packet,capacity,5056);row['case']='output-capacity-'+str(capacity)
        if capacity<24:
            if a.expect_short=='trap':assert row['signal']==4 and row['returned_u32'] is None
            else:assert row['signal'] is None and row['returned_u32']==0 and row['output_changed_bytes']==0
            assert row['workspace_unchanged']
        elif capacity<5080:
            assert row['signal'] is None and row['returned_u32']==24 and out[:24]==struct.pack('<3Q',3,0,0)
            assert row['workspace_unchanged']
        else:
            assert row['signal'] is None and row['returned_u32']==5080 and out[:24]==struct.pack('<3Q',0,0,5056)
            assert not row['workspace_unchanged']
        assert row['outside_declared_output_capacity_unchanged'] and row['input_unchanged']
        (a.output/(row['case']+'.output.bin')).write_bytes(out)
        rows.append(row)
    assert len(rows)==10 and a.native.read_bytes()==image
    report={'schema':'heptatorch.rng-native-flat-output-boundary.v1','status':'PASS',
            'expectation':a.expect_short,'native_sha256':SHA(image),'script_sha256':SHA(Path(__file__).read_bytes()),
            'bridge_offset':490,'native_machine_code_modified':False,'rows':rows,
            'scope':'Qualification calls the existing six-word bridge directly; stdin executable normally provides 1 MiB output and cannot trigger these short-output cases.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
