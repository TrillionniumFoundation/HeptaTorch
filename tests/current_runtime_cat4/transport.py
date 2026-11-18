"""Actual current ELF transport with the unchanged qualified owner decoder."""
from pathlib import Path
import gzip
import json
import lzma
import os
import select
import struct
import subprocess
import time
import traceback
from packet import MIB, require, response, sha


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def execute(body, binary, image_sha, directory, owner):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    record=dict(status='FAIL',native_started=False,native_executed=False,
        process_id=None,exit_code=None,stdout_EOF=False,stderr_EOF=False,
        unexpected_stdout_bytes=None,stderr_bytes=None,forced_cleanup=False,
        request_sha256=sha(body),frames=0,owner_observations=[],error=None)
    child=None;started=time.monotonic();deadline=started+120.;reply=None;after=None
    request=directory/'actual.stdin';response_file=directory/'actual.stdout'
    sent=request.open('xb');received=response_file.open('xb')

    def remaining():
        value=deadline-time.monotonic();require(value>0,'actual complete-case 120-second deadline');return value

    def read_exact(count):
        result=bytearray()
        while len(result)<count:
            require(select.select([child.stdout],[],[],remaining())[0],'native response timeout')
            try:part=os.read(child.stdout.fileno(),count-len(result))
            except BlockingIOError:continue
            require(part,'native stdout premature EOF')
            received.write(part);received.flush();result.extend(part)
        return bytes(result)

    def snapshot(label, reader):
        value=reader.snapshot();checked=owner.verify_snapshot(value)
        raw=bytearray()
        fd=os.open('/proc/%d/mem'%child.pid,os.O_RDONLY)
        try:
            for at in range(0,value['length'],65536):
                chunk=os.pread(fd,min(65536,value['length']-at),value['pointer']+at)
                require(len(chunk)==min(65536,value['length']-at),'complete physical owner read')
                raw.extend(chunk)
        finally:os.close(fd)
        reader.settled_control()
        require(sha(raw)==value['full_sha256'],'independently reread owner bytes differ')
        repeated=owner.decode_owner(bytes(raw),value['pointer']);owner.verify_snapshot(repeated)
        require(canonical(repeated)==canonical(value),'physical bytes do not reproduce full owner snapshot')
        raw_path=directory/(label+'.owner.xz');raw_path.write_bytes(lzma.compress(bytes(raw),preset=1))
        decoded=canonical(value);decoded_path=directory/(label+'.json.gz')
        with gzip.open(decoded_path,'xb',compresslevel=1) as stream:stream.write(decoded)
        record['owner_observations'].append(dict(label=label,raw_file=raw_path.name,
            raw_bytes=len(raw),raw_sha256=sha(raw),compressed_sha256=sha(raw_path.read_bytes()),
            decoded_file=decoded_path.name,decoded_sha256=sha(decoded),verified=checked))
        return value

    try:
        child=subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,bufsize=0,env={'PATH':''})
        record.update(native_started=True,native_executed=None,process_id=child.pid)
        record['executed_image_sha256']=sha(Path('/proc/%d/exe'%child.pid).read_bytes())
        require(record['executed_image_sha256']==image_sha,'actual child image differs')
        record['native_executed']=True
        for stream in (child.stdin,child.stdout):os.set_blocking(stream.fileno(),False)
        reader=owner.OwnerReader(child);before=snapshot('before',reader)
        framed=b'HTOWNR01'+struct.pack('<Q',len(body))+body;view=memoryview(framed)
        while view:
            require(select.select([],[child.stdin],[],remaining())[1],'native request timeout')
            try:count=os.write(child.stdin.fileno(),view)
            except BlockingIOError:continue
            require(count>0,'native closed stdin');sent.write(view[:count]);sent.flush();view=view[count:]
        header=read_exact(16);magic,size=struct.unpack('<8sQ',header)
        require(magic==b'HTOWNS01' and 32<=size<=MIB,'actual owned response frame')
        reply=read_exact(size);record['frames']=1
        require(child.poll() is None and not child.stdin.closed,'response must precede stdin EOF')
        status,rows=response(reply,3);record.update(actual_status=status,actual_records=len(rows),
            response_sha256=sha(reply),response_before_stdin_EOF=True)
        (directory/'actual-response-body.bin').write_bytes(reply)
        after=snapshot('after',reader)
        if status:
            record['complete_owner_rollback']=(before['pointer']==after['pointer']
                and before['length']==after['length'] and before['full_sha256']==after['full_sha256']
                and owner.semantic(before)==owner.semantic(after))
            require(record['complete_owner_rollback'],'actual refusal changed complete owner')
        child.stdin.close();record['exit_code']=child.wait(timeout=min(20,remaining()))
        os.set_blocking(child.stdout.fileno(),True)
        tail,stderr=child.stdout.read(),child.stderr.read();received.write(tail);received.flush()
        (directory/'unexpected.stdout').write_bytes(tail);(directory/'native.stderr').write_bytes(stderr)
        record.update(stdout_EOF=True,stderr_EOF=True,unexpected_stdout_bytes=len(tail),stderr_bytes=len(stderr))
        require(record['exit_code']==0 and tail==stderr==b'','native exit or complete EOF differs')
        record['status']='PASS_ACTUAL_TRANSPORT'
    except BaseException as error:
        record['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        if child is not None:
            if child.poll() is None:
                record['forced_cleanup']=True;child.kill();child.wait(timeout=10)
            record['exit_code']=child.returncode
            if not record['stdout_EOF']:
                os.set_blocking(child.stdout.fileno(),True);tail=child.stdout.read()
                received.write(tail);received.flush();record['stdout_EOF']=True
                record['unexpected_stdout_bytes']=len(tail);(directory/'failure-tail.stdout').write_bytes(tail)
            if not record['stderr_EOF']:
                stderr=child.stderr.read();record.update(stderr_EOF=True,stderr_bytes=len(stderr))
                (directory/'native.stderr').write_bytes(stderr)
            for stream in (child.stdin,child.stdout,child.stderr):
                if not stream.closed:stream.close()
        sent.close();received.close();record['wall_seconds']=time.monotonic()-started
        record['stdin_file']=dict(bytes=request.stat().st_size,sha256=sha(request.read_bytes()))
        record['stdout_file']=dict(bytes=response_file.stat().st_size,sha256=sha(response_file.read_bytes()))
        (directory/'transport.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    return reply,after,record
