"""Independent test transport and own-child observation, not a runtime."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import resource
import select
import signal
import struct
import time

MIB=1<<20
REQUEST_MAGIC=b'HTOWNR01'
RESPONSE_MAGIC=b'HTOWNS01'


def request(commands=(),slots=0,pool=0):
    records=[]
    for command in commands:
        assert 1<=len(command)<=8
        records.append(struct.pack('<8Q',*command,*([0]*(8-len(command)))))
    return b'HTOSOBJ1'+struct.pack('<QQQ',len(records),slots,pool)+b''.join(records)


def response(body):
    assert len(body)>=32
    magic,version,count,status=struct.unpack_from('<8sQQQ',body)
    assert magic==b'HTOSRES1' and version==1
    if status:
        assert count==0 and len(body)==32
        return status,[]
    records=[];at=32
    for _ in range(count):
        assert at+64<=len(body)
        code,err,a,b,c,d,e,size=struct.unpack_from('<8Q',body,at);at+=64
        assert err==0 and at+size<=len(body)
        records.append({'op':code,'values':(a,b,c,d,e),'payload':body[at:at+size]});at+=size
    assert at==len(body)
    return status,records


class OwnerSession:
    def __init__(self,executable,*,probe=False,directory=None,address_limit=None,
                 keep_read_control=False,stdin_directory=None):
        self.executable=Path(executable).resolve();self.probe=probe
        self.binary_sha256=hashlib.sha256(self.executable.read_bytes()).hexdigest()
        self.exit_code=None;self.base=None;self.frames=0;self.rows=[]
        in_r,in_w=os.pipe();out_r,out_w=os.pipe();err_r,err_w=os.pipe()
        pairs=[in_r,in_w,out_r,out_w,err_r,err_w]
        mark_r=mark_w=ack_r=ack_w=None
        if probe:
            mark_r,mark_w=os.pipe();ack_r,ack_w=os.pipe();pairs += [mark_r,mark_w,ack_r,ack_w]
        directory_fd=None
        if stdin_directory is not None:
            directory_fd=os.open(str(stdin_directory),os.O_RDONLY|os.O_DIRECTORY);pairs.append(directory_fd)
        assert all(fd>=3 for fd in pairs)
        self.pid=os.fork()
        if self.pid==0:
            try:
                os.dup2(in_r if directory_fd is None else directory_fd,0)
                os.dup2(out_w,1);os.dup2(err_w,2)
                if probe:os.dup2(mark_w,3);os.dup2(ack_r,4)
                for fd in pairs:
                    if fd>(4 if probe else 2):os.close(fd)
                signal.signal(signal.SIGPIPE,signal.SIG_DFL)
                resource.setrlimit(resource.RLIMIT_CORE,(0,0))
                if address_limit is not None:resource.setrlimit(resource.RLIMIT_AS,(address_limit,address_limit))
                os.execve(str(self.executable),[str(self.executable)],{'PATH':''})
            except BaseException as error:
                os.write(2,repr(error).encode());os._exit(127)
        os.close(out_w);os.close(err_w)
        self.read_control=in_r if keep_read_control else None
        if not keep_read_control:os.close(in_r)
        if directory_fd is not None:os.close(directory_fd)
        if probe:os.close(mark_w);os.close(ack_r)
        self.input=in_w;self.output=out_r;self.stderr=err_r
        self.marker=mark_r;self.ack=ack_w
        self.directory=Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True,exist_ok=False)
            self.requests=(self.directory/'requests.frames.bin').open('wb')
            self.responses=(self.directory/'responses.frames.bin').open('wb')

    def poll(self):
        if self.exit_code is None:
            pid,status=os.waitpid(self.pid,os.WNOHANG)
            if pid:self.exit_code=os.waitstatus_to_exitcode(status)
        return self.exit_code

    def read_exact(self,fd,n,timeout=10):
        result=bytearray();until=time.monotonic()+timeout
        while len(result)<n:
            ready,_,_=select.select([fd],[],[],max(0,until-time.monotonic()))
            assert ready,('read deadline',n,len(result),self.poll())
            part=os.read(fd,n-len(result));assert part,('early EOF',n,len(result),self.poll())
            result.extend(part)
        return bytes(result)

    def send(self,raw):
        assert self.input is not None
        at=0
        while at<len(raw):
            _,ready,_=select.select([],[self.input],[],10)
            assert ready and self.poll() is None
            count=os.write(self.input,raw[at:]);assert count>0;at+=count
        if self.directory:self.requests.write(raw);self.requests.flush()

    def receive(self):
        head=self.read_exact(self.output,16)
        magic,size=struct.unpack('<8sQ',head);assert magic==RESPONSE_MAGIC and 32<=size<=MIB
        body=self.read_exact(self.output,size)
        assert self.input is not None and self.poll() is None
        self.frames+=1;self.rows.append({'frame':self.frames,'pid':self.pid,'response_bytes':size,'stdin_open':True})
        if self.directory:self.responses.write(head+body);self.responses.flush()
        return body

    def exchange(self,body):
        self.send(REQUEST_MAGIC+struct.pack('<Q',len(body))+body)
        return self.receive()

    def assert_no_response(self,timeout=.02):
        ready,_,_=select.select([self.output],[],[],timeout)
        assert not ready and self.poll() is None

    def wait_syscall(self,number,fd,timeout=5):
        until=time.monotonic()+timeout
        while True:
            assert self.poll() is None,('exited before syscall',self.exit_code)
            raw=Path(f'/proc/{self.pid}/syscall').read_text().split()
            if raw and raw[0] not in ('running','-1'):
                values=[int(v,0) for v in raw]
                if values[0]==number and values[1]==fd:return values
            assert time.monotonic()<until,('syscall timeout',number,fd,raw)
            time.sleep(.001)

    def control_base(self):
        if self.base is None:
            call=self.wait_syscall(0,0)
            assert call[3]==16,call
            self.base=call[2]
        return self.base

    def memory(self,address,size):
        with open(f'/proc/{self.pid}/mem','rb',buffering=0) as fd:
            fd.seek(address);raw=fd.read(size)
        assert len(raw)==size
        return raw

    def owner(self,read_bytes=True):
        base=self.control_base()
        call=self.wait_syscall(0,0);assert call[2]==base and call[3]==16
        pointer,length=struct.unpack('<QQ',self.memory(base+32,16))
        live=struct.unpack('<Q',self.memory(base+192,8))[0]
        assert live==1
        assert self.memory(base+128,16)==bytes(16)
        assert self.memory(base+160,32)==bytes(32)
        assert (pointer==0)==(length==0)
        raw=None
        if read_bytes:
            assert length<=64*MIB
            raw=self.memory(pointer,length) if length else b''
        return {'pointer':pointer,'length':length,'bytes':raw,'live':live}

    def mapped(self,address):
        for row in Path(f'/proc/{self.pid}/maps').read_text().splitlines():
            lo,hi=(int(x,16) for x in row.split()[0].split('-'))
            if lo<=address<hi:return True
        return False

    def close_input(self):
        if self.input is not None:os.close(self.input);self.input=None

    def close_output(self):
        if self.output is not None:os.close(self.output);self.output=None

    def nonblocking_input(self):
        assert self.read_control is not None
        flags=fcntl.fcntl(self.read_control,fcntl.F_GETFL)
        fcntl.fcntl(self.read_control,fcntl.F_SETFL,flags|os.O_NONBLOCK)

    def observe_close(self,expected_status):
        assert self.probe
        marker=self.read_exact(self.marker,16)
        assert marker==b'HTOCLO01'+struct.pack('<Q',expected_status),marker
        call=self.wait_syscall(0,4)
        # The hook uses the same r12 mapping; rsi points to control+224.
        base=call[2]-224
        if self.base is not None:assert self.base==base
        self.base=base
        pointer,length=struct.unpack('<QQ',self.memory(base+96,16))
        assert self.memory(base+32,16)==bytes(16)
        assert self.memory(base+128,16)==bytes(16)
        assert self.memory(base+160,32)==bytes(32)
        assert self.memory(base+192,8)==bytes(8)
        assert pointer==0 or not self.mapped(pointer),('owner still mapped after close',pointer,length)
        return {'status':expected_status,'dead_argument_pointer':pointer,'dead_argument_length':length,
                'owner_mapping_absent_before_process_exit':True,'live_owner_flag':0,
                'canonical_and_result_owner_slots_cleared':True,'process_alive':self.poll() is None}

    def finish(self,expected=0,release_probe=True):
        self.close_input()
        if self.probe and release_probe:
            assert os.write(self.ack,b'!')==1;os.close(self.ack);self.ack=None
        until=time.monotonic()+5
        while self.poll() is None:
            assert time.monotonic()<until,'exit deadline'
            time.sleep(.001)
        error=os.read(self.stderr,65536)
        extra=b'' if self.output is None else os.read(self.output,65536)
        assert self.exit_code==expected and not error and not extra,(self.exit_code,error,extra)
        if self.probe:assert os.read(self.marker,65536)==b''
        for field in ('output','stderr','marker','ack','read_control'):
            fd=getattr(self,field)
            if fd is not None:os.close(fd);setattr(self,field,None)
        if self.directory:
            self.requests.close();self.responses.close()
            (self.directory/'transport.json').write_text(json.dumps({'pid':self.pid,'frames':self.frames,
                'executable_sha256':self.binary_sha256,'rows':self.rows,'exit_code':self.exit_code,
                'responses_before_stdin_eof':True},indent=2)+'\n')
        assert hashlib.sha256(self.executable.read_bytes()).hexdigest()==self.binary_sha256
