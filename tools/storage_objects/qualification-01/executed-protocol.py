"""Wire encoding/decoding for native qualification only; no storage executor."""
import struct


def handle(kind, slot, generation=1):
    assert 1 <= kind <= 6 and 1 <= slot < 1 << 24 and 1 <= generation < 1 << 32
    return (generation << 32)|(kind << 24)|slot


def body(commands=(), slots=0, pool=0):
    rows = []
    for command in commands:
        assert 1 <= len(command) <= 8
        rows.append(struct.pack('<8Q', *command, *([0]*(8-len(command)))))
    return b'HTOSOBJ1'+struct.pack('<3Q',len(rows),slots,pool)+b''.join(rows)


def trace(frames):
    return b'HTOSTRC1'+struct.pack('<Q',len(frames))+b''.join(struct.pack('<Q',len(f))+f for f in frames)


def parse_body(raw):
    assert len(raw) >= 32, len(raw)
    magic,version,count,status = struct.unpack_from('<8s3Q',raw)
    assert magic == b'HTOSRES1' and version == 1
    if status:
        assert count == 0 and len(raw) == 32
        return {'status':status, 'records':[], 'raw':raw}
    records,at = [],32
    for _ in range(count):
        assert len(raw)-at >= 64
        op,code,a,b,c,d,e,size = struct.unpack_from('<8Q',raw,at)
        at += 64
        assert code == 0 and len(raw)-at >= size
        payload = raw[at:at+size]
        at += size
        records.append({'op':op,'values':(a,b,c,d,e),'payload':payload})
    assert at == len(raw)
    return {'status':0, 'records':records, 'raw':raw}


def parse_trace(raw):
    assert len(raw) >= 32
    magic,version,count,status = struct.unpack_from('<8s3Q',raw)
    assert magic == b'HTOSTRR1' and version == 1
    frames,at = [],32
    for _ in range(count):
        assert len(raw)-at >= 8
        size = struct.unpack_from('<Q',raw,at)[0]
        at += 8
        assert len(raw)-at >= size
        frames.append(parse_body(raw[at:at+size]))
        at += size
    assert at == len(raw)
    return status,frames
