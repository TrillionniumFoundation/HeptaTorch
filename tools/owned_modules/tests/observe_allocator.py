"""Isolated stdlib observer of allocator bytes actually emitted by native HIA.

This is a qualification harness for a private backend ABI, not a public owner
pointer API and not part of the produced application's runtime.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import mmap
import resource
import sys

directory = Path(sys.argv[1])
mode = sys.argv[2]
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
resource.setrlimit(resource.RLIMIT_AS, (128 << 20, 128 << 20))
manifest = json.loads((directory / 'inputs.json').read_text())
sha = lambda raw: hashlib.sha256(raw).hexdigest()
libc = ctypes.CDLL(None)
libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.mprotect.restype = ctypes.c_int
maps = []


def executable(name):
    raw = (directory / (name + '.bin')).read_bytes()
    assert sha(raw) == manifest[name]['sha256']
    size = (len(raw) + mmap.PAGESIZE - 1) & -mmap.PAGESIZE
    mapping = mmap.mmap(-1, size, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    mapping[:len(raw)] = raw
    buffer = (ctypes.c_char * size).from_buffer(mapping)
    address = ctypes.addressof(buffer)
    assert libc.mprotect(address, size, mmap.PROT_READ | mmap.PROT_EXEC) == 0
    maps.append((mapping, buffer))
    return address


allocate_at = executable('calloc')
free_at = executable('free')
allocate = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64)(allocate_at)
release = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(free_at)
if mode == 'free-error':
    pointer = allocate(1, 17)
    assert pointer
    # Deliberate PRIVATE-ABI fault injection: native HIA cannot address this
    # hidden header or manufacture this corrupted owner. Linux munmap with
    # length zero returns EINVAL; the provider must take its declared UD2 path.
    ctypes.c_uint64.from_address(pointer - 16).value = 0
    release(pointer)
    raise AssertionError('munmap error was silently ignored')

rows = []
for count, size, cause in [(0, 91, 'zero-count'), (7, 0, 'zero-size'),
                          (1 << 63, 2, 'multiply-overflow'),
                          ((1 << 64)-1, 2, 'multiply-overflow-high'),
                          ((1 << 64)-1, 1, 'header-add-overflow'),
                          ((1 << 64)-8, 1, 'header-add-overflow-near-limit'),
                          ((1 << 63)-8, 1, 'signed-mapping-length-bound'),
                          (1, 1 << 62, 'actual-kernel-allocation-failure')]:
    assert allocate(count, size) is None, (count, size, cause)
    rows.append({'case': cause, 'count': count, 'size': size, 'result': 'NULL', 'status': 'PASS'})
for count, size in [(7, 13), (1, 2097169)]:
    pointer = allocate(count, size)
    assert pointer and pointer % 16 == 0
    requested = count * size
    assert ctypes.c_uint64.from_address(pointer - 16).value == requested + 16
    assert ctypes.string_at(pointer, requested) == bytes(requested)
    ctypes.c_ubyte.from_address(pointer).value = 93
    ctypes.c_ubyte.from_address(pointer + requested - 1).value = 173
    release(pointer)
    rows.append({'case': 'zero-filled-allocation-and-real-release', 'count': count, 'size': size,
                 'requested_bytes': requested, 'mapping_length': requested + 16,
                 'alignment': 16, 'status': 'PASS'})
release(None)
rows.append({'case': 'release-null-noop', 'status': 'PASS'})
probe_at = executable('abi_probe')
probe = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(probe_at)
assert probe(allocate_at, free_at) == 1
rows.append({'case': 'callee-saved-rbx-rbp-r12-r15-and-rsp-preserved-across-both-calls', 'status': 'PASS'})
report = {'status': 'PASS', 'cases': len(rows), 'rows': rows,
          'forbidden_modules': [name for name in sys.modules if name == 'torch' or name.startswith(('hepta.', 'tools.'))],
          'allocator_is_actual_native_hia_object_bytes': True,
          'observer_libc_use': 'mprotect only, to change observer code mappings from RW to RX',
          'allocator_libc_dependency': False}
assert not report['forbidden_modules']
print(json.dumps(report))
