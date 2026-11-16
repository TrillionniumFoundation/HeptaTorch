"""Qualification-only transport. The service owns all Generator state."""
from pathlib import Path
import os
import select
import struct
import subprocess
import time

MASK=(1<<64)-1
OPS={'create':1,'manual_seed':2,'seed':3,'current_seed':4,'get_state':5,'set_state':6,
     'clone_state':7,'advance':8,'raw32':9,'raw64':10,'rand_f32':11,'rand_f64':12,
     'rand_f16':13,'rand_bf16':14,'rand_int64':15,'release':16,'stats':17,'dump':18,'create_default':19}

def packet(op,handle=0,count=0,a=0,b=0,payload=b''):
    if isinstance(op,str):op=OPS[op]
    return b'HTCGEN01'+struct.pack('<7Q',op,handle&MASK,count&MASK,a&MASK,b&MASK,len(payload),0)+payload

class Session:
    def __init__(self,executable,preexec_fn=None):
        self.process=subprocess.Popen([str(Path(executable).resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,env={'PATH':''},preexec_fn=preexec_fn)
        self.frames=0

    def exact(self,size):
        result=bytearray();deadline=time.monotonic()+30
        while len(result)<size:
            timeout=deadline-time.monotonic()
            if timeout<=0 or not select.select([self.process.stdout],[],[],max(0,timeout))[0]:
                raise TimeoutError(('native frame',size,len(result),self.process.poll()))
            part=os.read(self.process.stdout.fileno(),size-len(result))
            if not part:raise AssertionError(('early EOF',self.process.poll(),len(result),size))
            result+=part
        return bytes(result)

    def raw(self,body):
        self.process.stdin.write(b'HTOWNR01'+struct.pack('<Q',len(body))+body);self.process.stdin.flush()
        header=self.exact(16)
        assert header[:8]==b'HTOWNS01',header
        size=struct.unpack_from('<Q',header,8)[0];assert 64<=size<=1<<20,size
        reply=self.exact(size);assert reply[:8]==b'HTCGRS01',reply[:64]
        status,payload_bytes,live,allocation,last,epoch,reserved=struct.unpack_from('<7Q',reply,8)
        assert reserved==0 and size==64+payload_bytes,(size,payload_bytes)
        self.frames+=1
        return dict(status=status,payload=reply[64:],live=live,allocation=allocation,last=last,epoch=epoch,raw=reply)

    def call(self,op,handle=0,count=0,a=0,b=0,payload=b'',status=0):
        result=self.raw(packet(op,handle,count,a,b,payload))
        assert result['status']==status,(op,handle,count,a,b,result)
        return result

    def words(self,op,handle=0,count=0,a=0,b=0,payload=b''):
        raw=self.call(op,handle,count,a,b,payload)['payload']
        assert len(raw)%8==0
        return list(struct.unpack('<'+'Q'*(len(raw)//8),raw))

    def dump(self):
        size=self.call('stats')['allocation'];parts=[]
        for offset in range(0,size,1<<19):
            parts.append(self.call('dump',count=min(1<<19,size-offset),a=offset)['payload'])
        return b''.join(parts)

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close();self.process.stdin=None
        stdout,stderr=self.process.communicate(timeout=30)
        assert self.process.returncode==0 and not stdout and not stderr,(self.process.returncode,stdout,stderr)

    def __enter__(self):return self
    def __exit__(self,typ,value,trace):
        if typ is None:self.close()
        else:
            self.process.kill();self.process.communicate()
