"""Run fixed HeptaTorch CPU demos with only the Python standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import stat
import struct
import subprocess
import tempfile
import time

ELF_SHA = '5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d'
MANIFESTS = {
    'inference': '899f9aaecbd6621625bf2b702ab445922cb4d3d10ba03e4c5fc0a8b8eaa05065',
    'sgd': '898b3b2ccb70729d124709d608063e5eec7446161fbe36896319e9c27fecc6f1',
}
ELF_CAP = 8 * (1 << 20)
FRAME_CAP = 1 << 20
TIMEOUT = 5.0

class DemoError(RuntimeError):
    """Known demo setup, protocol or numerical failure."""

def need(ok, reason):
    if not ok:
        raise DemoError(reason)

def pinned_bytes(path, expected, cap):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as f:
        info = os.fstat(f.fileno())
        need(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= cap, 'not a bounded regular file')
        data = f.read(cap + 1)
    need(0 < len(data) <= cap and hashlib.sha256(data).hexdigest() == expected, 'SHA-256 mismatch')
    return data

def manifest(name):
    digest = MANIFESTS[name]
    p = Path(__file__).resolve().with_name(name + '.json')
    data = json.loads(pinned_bytes(p, digest, FRAME_CAP))
    need(data['native_elf_sha256'] == ELF_SHA, 'fixture image mismatch')
    return data

def stage_executable(root, data):
    p = Path(root) / 'heptatorch'
    with p.open('xb') as f:
        f.write(data)
        f.flush()
        os.fchmod(f.fileno(), 0o500)
    return p

def request(cmd, payload=b''):
    need(1 <= len(cmd) <= 8 and len(payload) <= FRAME_CAP - 96, 'request size mismatch')
    return b'HTORCH03' + struct.pack('<3Q', 1, 96 + len(payload), 0) + struct.pack(
        '<8Q', *cmd, *([0] * (8 - len(cmd)))) + payload

def factory(e):
    shape = e['shape']
    data = bytes.fromhex(e['raw_f32le_hex'])
    need(e['dtype'] == 'float32' and len(shape) <= 2, 'not a fixed F32 fixture')
    config = struct.pack('<8Q', len(shape), int(e['requires_grad']), *shape, *([0] * (4 - len(shape))), 160, len(data))
    return request((97, 0, 96), config + data)

def parse(body, op):
    need(32 <= len(body) <= FRAME_CAP, 'unexpected response body extent')
    magic, ver, count, status = struct.unpack_from('<8s3Q', body)
    need(magic == b'HTORSP03' and ver == 3, 'unexpected response magic/version')
    need(status == 0 and count == 1 and len(body) >= 64, f'native operation {op} failed (status {status})')
    code, err, value, size = struct.unpack_from('<4Q', body, 32)
    need(code == op and err == 0 and size == len(body) - 64, 'unexpected native result record')
    return value, body[64:]

def unpack_meta(payload):
    need(len(payload) == 128, 'unexpected metadata length')
    rank, n, grad, *fields = struct.unpack('<16Q', payload)
    need(rank <= 4, 'unexpected metadata rank')
    return rank, n, grad, fields[:rank], fields[4:4+rank], *fields[8:]

class Session:
    def __init__(self, exe):
        self.child = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, env={'PATH': ''}, close_fds=True)
        self.reader = self.child.stdout.fileno()
        self.writer = self.child.stdin.fileno()
        os.set_blocking(self.reader, False)
        os.set_blocking(self.writer, False)
        self.frames = 0

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.child.stdin.close()
        try:
            if typ is None:
                extra = self._read(1, time.monotonic() + TIMEOUT, allow_eof=True)
                need(not extra, 'unexpected bytes after stdin close')
                need(self.child.wait(timeout=TIMEOUT) == 0, 'native process did not exit cleanly')
        finally:
            if self.child.poll() is None:
                self.child.kill()
                self.child.wait(timeout=TIMEOUT)
            self.child.stdout.close()

    def _wait(self, read, until):
        try:
            r, w, _ = select.select([self.reader] if read else [], [] if read else [self.writer], [],
                                    max(0.0, until - time.monotonic()))
        except (OSError, ValueError) as exc:
            raise DemoError('native pipe wait failed') from exc
        need(bool(r or w), 'native I/O deadline exceeded')

    def _read(self, n, until, allow_eof=False):
        buf = bytearray()
        while len(buf) < n:
            self._wait(True, until)
            try:
                block = os.read(self.reader, n - len(buf))
            except BlockingIOError:
                continue
            if not block:
                need(allow_eof and not buf, 'unexpected native EOF')
                return b''
            buf.extend(block)
        return bytes(buf)

    def _write(self, data, until):
        data = memoryview(data)
        while data:
            self._wait(False, until)
            try:
                n = os.write(self.writer, data)
            except BlockingIOError:
                continue
            need(n > 0, 'native request write failed')
            data = data[n:]

    def exchange(self, body, op):
        end = time.monotonic() + TIMEOUT
        self._write(b'HTOWNR01' + struct.pack('<Q', len(body)) + body, end)
        magic, size = struct.unpack('<8sQ', self._read(16, end))
        need(magic == b'HTOWNS01' and 32 <= size <= FRAME_CAP, 'invalid native frame extent')
        value = parse(self._read(size, end), op)
        self.frames += 1
        return value

    def call(self, *cmd):
        return self.exchange(request(cmd), cmd[0])

def inspect(s, handle, expected):
    want = bytes.fromhex(expected['raw_f32le_hex'])
    n = len(want)//4
    rank, count, grad, shape, strides, storage, version_id, version, offset, size = unpack_meta(s.call(13, handle)[1])
    need((rank, count, grad, shape, strides) == (len(expected['shape']), n, int(expected['requires_grad']),
           expected['shape'], expected['strides']), 'tensor metadata mismatch')
    need((version, offset, size) == (expected['version'], 0, len(want)) and storage and version_id,
         'Version/extent or authority mismatch')
    count, actual = s.call(98, handle, 0, n)
    need((count, actual) == (n, want), 'raw F32 values differ from the locked fixture')
    return (handle, storage, version_id), list(struct.unpack(f'<{n}f', actual))

def roots(s, spec):
    named, expected = {}, dict(spec)
    for name, e in spec.items():
        named[name] = s.exchange(factory(e), 97)[0]
    cur = {k:inspect(s,h,expected[k])[0] for k,h in named.items()}
    need(fresh(cur.values()), 'literal roots are not independent')
    return named, expected, cur

def fresh(authorities):
    auth = list(authorities)
    return all(len({a[i] for a in auth}) == len(auth) for i in range(3))

def graph(s, stages, named, expected, authority, fixed):
    for stage, allowed in zip(stages, fixed, strict=True):
        name, cmd, e = stage['name'], stage['command'], stage['expected']
        need(tuple(cmd) == allowed, 'fixed demo graph changed')
        args = tuple(named[a] if isinstance(a, str) else a for a in cmd)
        named[name] = s.call(*args)[0]
        expected[name] = e
        for existing, h in named.items():
            ids, vals = inspect(s, h, expected[existing])
            if existing in authority:
                need(ids == authority[existing], 'live authority changed')
            else:
                authority[existing] = ids
            if existing == name:
                values = vals
        need(fresh(authority.values()), 'fresh output authority aliases another live value')
    return values

def release(s, names, named, authority, expected):
    for key in names:
        s.call(41, named.pop(key))
        del expected[key], authority[key]
    for k,h in named.items():
        need(inspect(s,h,expected[k])[0] == authority[k], 'surviving tensor authority changed')

INFER = ((32,'x','w1'), (2,1,'mm1','b1'), (3,5,'add1'), (32,'relu1','w2'), (2,1,'mm2','b2'))
TRAIN = ((32,'x','weight'), (2,2,'prediction','target'), (2,3,'error','error'), (4,7,'squared'))
TRAIN_FRESH = ((32,'x','weight'), (2,2,'prediction_fresh','target'),
               (2,3,'error_fresh','error_fresh'), (4,7,'squared_fresh'))

def inference(exe):
    data = manifest('inference')
    need(len(data['admitted']) == 3, 'unexpected inference scope')
    result = []
    for index, row in enumerate(data['admitted']):
        case = row['original_frozen']
        with Session(exe) as s:
            n,e,cur = roots(s, case['inputs'])
            y = graph(s, case['stages'], n,e,cur,INFER)
            order = list(n) if index != 1 else list(reversed(n))
            release(s,order[:1],n,cur,e)
            release(s,order[1:],n,cur,e)
            result.append({'case':case['label'],'prediction_shape':case['stages'][-1]['expected']['shape'],
                           'prediction':y,'verified_stages':5,'frames':s.frames})
    return result

def sgd(exe):
    p = manifest('sgd')
    with Session(exe) as s:
        n,e,cur = roots(s,p['input_tensors'])
        before = graph(s,p['before_update'],n,e,cur,TRAIN)
        s.call(7,n['loss'])
        g = s.call(11,n['weight'])
        gradient = bytes.fromhex(p['backward']['expected_gradient']['raw_f32le_hex'])
        need(g == (2,gradient), 'actual gradient mismatch')
        release(s,[a['name'] for a in reversed(p['before_update'])],n,cur,e)
        s.call(9,n['weight'],0x3e000000)
        e['weight'] = p['update']['weight_after']
        authority, weight = inspect(s,n['weight'],e['weight'])
        need(authority == cur['weight'] and s.call(11,n['weight']) == g, 'weight or gradient authority/values changed')
        after = graph(s,p['after_update'],n,e,cur,TRAIN_FRESH)
        release(s,[a['name'] for a in reversed(p['after_update'])]+list(reversed(p['input_tensors'])),n,cur,e)
        return {'loss_before':before[0],'loss_after':after[0],'weight':weight,
                'gradient':list(struct.unpack('<2f',g[1])),'weight_version':[0,1],'frames':s.frames}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True, type=Path, help='path to the exact pinned Linux x86-64 ELF')
    choice = p.add_mutually_exclusive_group(required=True)
    for name in ('inference','sgd','both'):
        choice.add_argument('--'+name, action='store_const', const=name, dest='profile')
    a = p.parse_args()
    try:
        binary = pinned_bytes(a.binary,ELF_SHA,ELF_CAP)
        need(binary[:4] == b'\x7fELF', 'not an ELF binary')
        with tempfile.TemporaryDirectory(prefix='heptatorch-fixed-') as root:
            exe = stage_executable(root,binary)
            run = {'binary_sha256':ELF_SHA}
            if a.profile in ('inference','both'):
                run['inference'] = inference(exe)
            if a.profile in ('sgd','both'):
                run['sgd'] = sgd(exe)
        print(json.dumps(run,sort_keys=True,indent=2))
    except (DemoError,OSError,subprocess.TimeoutExpired) as exc:
        p.exit(1, f'HeptaTorch demo: {exc}\n')

if __name__ == '__main__':
    main()
