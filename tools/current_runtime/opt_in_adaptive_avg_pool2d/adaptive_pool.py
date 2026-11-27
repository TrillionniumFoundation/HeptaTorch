"""Opt-in HIA wire client for evenly tiled F32 adaptive average pooling."""
import hashlib
import math
import os
from pathlib import Path
import select
import stat
import struct
import subprocess
import tempfile
import time

ELF_SHA = "ecd87c5eba1660def55e9b567a5c0c6ebd020a34297beea04fc5f63eb97f7467"
PARENT_ELF_SHA = "ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e"
ELF_CAP = 8 * (1 << 20)
FRAME_CAP = 1 << 20
TIMEOUT = 5.0
# Match only the documented bounded, positive, owned contiguous CPU path.
MAX_WORDS = 4096
MAX_SPATIAL = 64

class DemoError(RuntimeError):
    """Known wire, numerical or setup failure."""

class NativeStatus(DemoError):
    def __init__(self, opcode, status):
        super().__init__(f"native operation {opcode} failed (status {status})")
        self.opcode = opcode
        self.status = status

def need(ok, reason):
    if not ok:
        raise DemoError(reason)

# Fixed framed reader adapted from opt_in_current800_fixed_demo/run_demo.py.
def pinned_bytes(path, expected, cap):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as f:
        info = os.fstat(f.fileno())
        need(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= cap, 'not a bounded regular file')
        data = f.read(cap + 1)
    need(0 < len(data) <= cap and hashlib.sha256(data).hexdigest() == expected, 'SHA-256 mismatch')
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

def parse(body, op):
    need(32 <= len(body) <= FRAME_CAP, 'unexpected response body extent')
    magic, ver, count, status = struct.unpack_from('<8s3Q', body)
    need(magic == b'HTORSP03' and ver == 3, 'unexpected response magic/version')
    if status:
        need(count == 0 and len(body) == 32, "unexpected record on refusal")
        raise NativeStatus(op, status)
    need(count == 1 and len(body) >= 64, "missing response record")
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
        body = self._read(size, end)
        self.frames += 1
        return parse(body, op)

    def call(self, *cmd):
        return self.exchange(request(cmd), cmd[0])


def lowering(shape, output_size):
    """Return the positive uniform bin strides, or refuse before launch."""
    if len(shape) != 4 or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError("expected positive integer [N,C,H,W] shape")
    if math.prod(shape) > MAX_WORDS or max(shape[-2:]) > MAX_SPATIAL:
        raise ValueError("input shape exceeds bounded scope")
    if len(output_size) != 2 or any(type(x) is not int or x < 1 for x in output_size):
        raise ValueError("expected positive integer [OH,OW] output size")
    if any(i % o for i, o in zip(shape[-2:], output_size)):
        raise ValueError("each spatial extent must divide evenly into its output extent")
    return tuple(i//o for i, o in zip(shape[-2:], output_size))

def contiguous_strides(shape):
    return [math.prod(shape[i+1:]) for i in range(len(shape))]

def factory(session, shape, raw, grad):
    need(len(raw) == 4 * math.prod(shape), "F32 input size mismatch")
    config = struct.pack("<8Q", 4, int(grad), *shape, 160, len(raw))
    return session.exchange(request((97, 0, 96), config + raw), 97)[0]

def read_tensor(session, handle, wanted_shape, want_grad):
    rank, count, grad, shape, strides, storage, version_id, version, offset, size = unpack_meta(
        session.call(13, handle)[1])
    need((rank, count, grad, shape, strides) ==
         (4, math.prod(wanted_shape), int(want_grad), list(wanted_shape), contiguous_strides(wanted_shape)),
         "output shape, stride or gradient flag mismatch")
    need(storage and version_id and version == 0 and offset == 0 and size == count * 4,
         "independent owned F32 extent and version required")
    actual_count, data = session.call(98, handle, 0, count)
    need(actual_count == count and len(data) == size, "F32 raw result extent mismatch")
    return (handle,storage,version_id), data

def adaptive_avg_pool2d(binary, shape, raw, output_size, *, seed=None):
    """Run contiguous positive NCHW F32 bytes; optional owned backward seed.

    Returns native forward bytes, optional native input gradient bytes, and
    the count of actual verified wire frames. Never computes values in Python.
    """
    shape = tuple(shape)
    output_size = tuple(output_size)
    kh, kw = lowering(shape, output_size)
    wanted = (*shape[:2], *output_size)
    if not isinstance(raw, bytes) or len(raw) != 4 * math.prod(shape):
        raise ValueError("expected complete bytes of little-endian F32 input")
    if seed is not None and (not isinstance(seed, bytes) or len(seed) != 4 * math.prod(wanted)):
        raise ValueError("expected complete bytes of little-endian F32 seed")
    if any(word & 0x7f800000 == 0x7f800000 for data in (raw, seed or b"")
           for (word,) in struct.iter_unpack("<I", data)):
        raise ValueError("nonfinite F32 input or seed is outside the scoped API")
    return _run(binary, ELF_SHA, shape, raw, wanted, (kh, kw), (kh, kw), (0, 0), False, seed)

def _run(binary, digest, shape, raw, wanted, kernel, stride, padding, include, seed, *, refusal=None):
    def pair(values):
        return values[0] << 32 | values[1]
    with tempfile.TemporaryDirectory(prefix="heptatorch-adaptive-pool-") as root:
        exe = stage_executable(root, pinned_bytes(binary, digest, ELF_CAP))
        with Session(exe) as session:
            x = factory(session, shape, raw, seed is not None)
            original = read_tensor(session, x, shape, seed is not None)
            need(original[1] == raw, "input constructor did not preserve bytes")
            try:
                y, _ = session.call(23, 23, x, pair(kernel), pair(stride), pair(padding), int(include))
            except NativeStatus as exc:
                if refusal is None or (exc.opcode, exc.status) != (23, refusal):
                    raise
                need(read_tensor(session, x, shape, False) == original, "parent refusal changed input")
                session.call(41, x)
                need(session.call(53)[0] == 1, "parent refusal owner audit failed")
                return exc.status, None, session.frames
            need(refusal is None, "expected exact parent refusal not observed")
            output = read_tensor(session, y, wanted, seed is not None)
            need(all(a != b for a,b in zip(original[0],output[0])), "fresh output aliases input")
            grad_bytes = None
            if seed is not None:
                u = factory(session, wanted, seed, False)
                u0 = read_tensor(session, u, wanted, False)
                need(u0[1] == seed, "seed constructor did not preserve bytes")
                need(all(len({original[0][i],output[0][i],u0[0][i]}) == 3 for i in range(3)),
                     "backward seed aliases a live tensor")
                session.call(29,y,u)
                size, grad_bytes = session.call(11,x)
                need(size == math.prod(shape) and len(grad_bytes) == 4*size, "input gradient extent mismatch")
                need(read_tensor(session,u,wanted,False) == u0, "backward seed changed")
                session.call(41,u)
            need(read_tensor(session,x,shape,seed is not None) == original, "input values/owner changed")
            session.call(41,x)
            need(read_tensor(session,y,wanted,seed is not None) == output, "output changed after input release")
            session.call(41,y)
            need(session.call(53)[0] == 1, "native owner audit failed")
            return output[1], grad_bytes, session.frames
