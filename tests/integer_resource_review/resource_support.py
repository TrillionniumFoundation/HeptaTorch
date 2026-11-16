"""Source-bound transport and limited raw resource inspection for I64 tests.

This is an external qualification observer, not a production numerical runtime
or an admission checker for arbitrary caller-authored native owner memory.
"""
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
import struct

SOURCE = '46cba6e9951de45374f4ba0a7567811cd9e03a16db0e5b9e67ba7512559b9bb3'
IMAGE = 'b701b4c0b6f82f928150a9ed33f8805d0783fc652ca3d9f1c3d5f9ad19478bdd'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
REFERENCE = '4c454e443c4147c740ea400cb1195bcbcb80e16e5b50debce54c068fa9ae712b'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
PROBE = '54f81a65652e2c2ba9eb3f04d608e89b71c92c5ad33ff7061e9a03b89f45f9ea'
BASE = 139264
MASK64 = (1 << 64) - 1


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


class Custody:
    def __init__(self, output):
        self.output = Path(output)
        self.inputs = {}

    def bind(self, path, expected=None, copy=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = sha(raw)
        assert expected is None or digest == expected, (str(path), digest, expected)
        self.inputs[str(path)] = {'bytes': len(raw), 'sha256': digest,
                                  'mode': path.stat().st_mode & 0o777}
        if copy is not None:
            dest = self.output / 'inputs' / copy
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            dest.chmod(path.stat().st_mode & 0o777)
        return raw

    def finish(self):
        for name, row in self.inputs.items():
            p = Path(name)
            raw = p.read_bytes()
            assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
            assert p.stat().st_mode & 0o777 == row['mode']
        return self.inputs


def load(custody, build, protocol, reference=None, expected_source=SOURCE, expected_image=IMAGE):
    build = Path(build).resolve()
    protocol = Path(protocol).resolve()
    custody.bind(Path(__file__), copy='resource_support.py')
    custody.bind(build / 'runtime.hepta', expected_source, 'native/runtime.hepta')
    image = build / 'tensor-integer-session'
    custody.bind(image, expected_image, 'native/tensor-integer-session')
    info = json.loads(custody.bind(build / 'build.json', copy='native/build.json'))
    assert info['status'] == 'PASS' and info['source_sha256'] == expected_source and info['native_sha256'] == expected_image
    assert info['descriptor_bytes'] == 640 and info['state_format'] == 'HTINT002'
    assert len(info['units']) == 11
    for unit in info['units']:
        assert unit['compiler_exit'] == 0
        for suffix, field in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
            name = unit['name'] + suffix
            custody.bind(build / name, unit[field], 'native/' + name)
    custody.bind(protocol, PROTOCOL, 'wire/protocol.py')
    custody.bind(protocol.with_name('owner_client.py'), CLIENT, 'wire/owner_client.py')
    spec = importlib.util.spec_from_file_location('integer_resource_wire', protocol)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    observed = None
    if reference:
        observed = json.loads(custody.bind(reference, REFERENCE, 'reference54.json'))
        assert observed['status'] == 'OBSERVED' and observed['cases'] == 54
        assert observed['probe_error_count'] == 0
        checks = observed['oracle_provenance']['checks']
        assert observed['oracle_provenance']['status'] == 'PASS'
        assert set(checks) == {'python_matches', 'platform_matches', 'version_backend_matches',
            'git_matches_oracle_build_pin', 'distribution_version_matches', 'direct_url_matches',
            'direct_url_archive_sha256_matches', 'pip_report_url_matches',
            'pip_report_archive_sha256_matches', 'pip_report_version_matches', 'pip_report_requested'}
        assert all(value is True for value in checks.values())
        for digest, state in observed['states'].items():
            raw = bytes.fromhex(state['hex'])
            assert state['bytes'] == len(raw) == 5056 and sha(raw) == digest
    return wire, image, info, observed


def create_packet(dtype, shape, raw=b'', requires=0):
    assert len(shape) <= 4
    config = struct.pack('<8Q', len(shape), requires, *shape, *([0] * (4 - len(shape))), 160, len(raw))
    result = b'HTORCH03' + struct.pack('<3Q', 1, 160 + len(raw), 0)
    result += struct.pack('<8Q', 97, dtype, 96, 0, 0, 0, 0, 0) + config + raw
    assert len(result) <= 1 << 20
    return result


def create(wire, session, dtype, shape, raw=b'', requires=0):
    status, rows = wire.parse(session.exchange(create_packet(dtype, shape, raw, requires)))
    assert status == 0 and len(rows) == 1 and rows[0]['opcode'] == 97, ('create', status, rows)
    return rows[0]['value']


def integer_raw(values):
    return b''.join(struct.pack('<Q', value & MASK64) for value in values)


def dump_raw(out, name, raw):
    path = Path(out) / (name + '.xz')
    with lzma.open(path, 'wb', preset=1) as f:
        f.write(raw)
    restored = lzma.decompress(path.read_bytes())
    assert restored == raw
    return {'path': path.name, 'bytes': len(raw), 'sha256': sha(raw),
            'archive_bytes': path.stat().st_size, 'archive_sha256': sha(path.read_bytes())}


class State:
    """Decode table/descriptor/extent fields, without broad graph certification."""
    def __init__(self, raw):
        self.raw = raw
        assert raw[64:72] == b'HTINT002' and raw[BASE:BASE+8] == b'HTOINT02'
        self.slots = self.q(BASE + 16)
        self.pool = self.q(BASE + 24)
        self.data = BASE + 144 + 552 * self.slots
        assert len(raw) == self.data + self.pool
        self.next = self.q(8)

    def q(self, offset):
        assert 0 <= offset <= len(self.raw) - 8
        return struct.unpack_from('<Q', self.raw, offset)[0]

    def row(self, handle, kind=None, live=True):
        k = (handle >> 24) & 255
        slot = handle & 0xffffff
        assert 1 <= k <= 8 and (kind is None or k == kind) and 1 <= slot <= self.slots
        at = BASE + 128 + ((k - 1) * self.slots + slot - 1) * 64
        row = struct.unpack_from('<8Q', self.raw, at)
        assert row[0] == handle >> 32
        if live:
            assert row[1] == 1
        return row

    def table(self, kind):
        result = {}
        for slot in range(1, self.slots + 1):
            at = BASE + 128 + ((kind - 1) * self.slots + slot - 1) * 64
            row = struct.unpack_from('<8Q', self.raw, at)
            if row[1]:
                assert row[1] == 1 and row[0] != 0
                result[(row[0] << 32) | (kind << 24) | slot] = row
        return result

    def counts(self):
        return [len(self.table(kind)) for kind in range(1, 9)]

    def node(self, node):
        assert 0 < node < self.next
        page = self.q(4096 + (node >> 8) * 8)
        assert page >> 63 == 1
        at = self.data + (page & ((1 << 63) - 1)) + (node & 255) * 640
        return struct.unpack_from('<80Q', self.raw, at)

    def tensor_node(self, tid):
        node = self.row(tid, 1)[6]
        desc = self.node(node)
        assert desc[192 // 8] == tid and desc[304 // 8] == 1
        return node, desc

    def storage_bytes(self, sid):
        row = self.row(sid, 2)
        assert row[5] != 0 and row[7] == 0 and row[2] + row[3] <= self.pool
        return self.raw[self.data + row[2]:self.data + row[2] + row[3]]

    def rng(self, generator):
        hold = self.row(generator, 7)[2]
        sid = self.row(hold, 6)[2]
        raw = self.storage_bytes(sid)
        assert len(raw) == 5056
        return raw

    def free(self):
        start = BASE + 128 + 512 * self.slots
        count = self.q(BASE + 32)
        assert count <= self.slots + 1
        spans = [struct.unpack_from('<2Q', self.raw, start + 16 * i) for i in range(count)]
        assert all(size != 0 and offset + size <= self.pool for offset, size in spans)
        assert all(spans[i][0] + spans[i][1] < spans[i+1][0] for i in range(len(spans)-1))
        return spans

    def partition(self):
        spans = [(offset, size, 'free') for offset, size in self.free()]
        for sid, row in self.table(2).items():
            if row[5]:
                assert row[7] == 0 and row[4] == (row[3] + 15) & ~15
                if row[4]:
                    spans.append((row[2], row[4], str(sid)))
            else:
                # The real producer retains logical byte_size in a tombstone.
                assert row[2] == row[4] == 0 and row[7] == 1 and row[6] > 0
        at = 0
        for offset, size, label in sorted(spans):
            assert offset == at, ('extent partition', offset, at, label)
            at += size
        assert at == self.pool
        return len(spans)

    def final_pages_only(self):
        counts = self.counts()
        assert all(counts[k-1] == 0 for k in (1, 3, 4, 5, 7, 8)), counts
        pages = []
        for i in range(256):
            page, hold = self.q(4096 + 8*i), self.q(6144 + 8*i)
            if page:
                assert hold and self.row(hold, 6)[3] == 1
                sid = self.row(hold, 6)[2]
                storage = self.row(sid, 2)
                assert storage[3] == storage[4] == 256*640 and storage[5] == storage[6] == 1
                assert page == (1 << 63) | storage[2]
                pages.append(hold)
            else:
                assert hold == 0
        assert counts[1] == counts[5] == len(pages)
        self.partition()
        return {'live_counts_8_kinds': counts, 'descriptor_pages': len(pages),
                'all_remaining_storage_and_holds_are_actual_descriptor_pages': True}
