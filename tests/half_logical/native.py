"""Native Half observations and lossless offline replay of the same predicates.

Only the HIA child calculates reductions. Expected tensor bytes are copied from
the independently executed immutable Torch reference. Python transports bytes,
decodes public metadata and checks ownership, resource and protocol behavior.
"""

import hashlib
import importlib.util
import json
import lzma
import math
import os
from pathlib import Path
import re
import resource
import struct

from . import reference

BASE = 139264
MASK = (1 << 64) - 1
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
TRANSPORT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
DTYPE = {1:'torch.bool', 5:'torch.float16', 6:'torch.bfloat16'}
WIDTH = {1:1, 5:2, 6:2}
# scalar_previous_typed_descriptor clears all 672 bytes before writing the
# dense metadata. Neither raw factories nor logical_reduce add history or
# view state to these no-grad inputs/results. Keep every remaining word zero.
DENSE_ZERO_WORDS = tuple(offset // 8 for offset in (
    24,32,40,48,56,72,80,88,160,168,176,184,208,248,256,264,272,280,288,296,312,
    328,336,344,352,360,368,376,384,392,400,408,416,424,432,440,448,456,464,472,
    480,488,496,504,512,520,528,536,544,552,560,568,584,592,600,608,616,624,632,
    640,648,656,664))
require, sha = reference.require, reference.sha


def dump(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def load_wire(repository, bind):
    path = repository / 'tests/differentiable_view_native_review/protocol.py'
    bind(path, PROTOCOL)
    bind(path.with_name('owner_client.py'), TRANSPORT)
    spec = importlib.util.spec_from_file_location('half264_fixed_wire', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Memory:
    """Decode the complete 84-word Tensor and actual Storage/Version objects."""
    def __init__(self, raw):
        self.raw = raw
        require(len(raw) >= BASE + 144 and raw[64:72] == b'HTSINT02'
                and raw[BASE:BASE + 8] == b'HTOSIN02', 'actual current complete owner format')
        self.slots, self.pool = self.q(BASE + 16), self.q(BASE + 24)
        require(self.q(BASE + 8) == 2 and self.q(BASE + 48) == 1
                and 4 <= self.slots <= 65536 and self.slots & (self.slots - 1) == 0
                and 16 <= self.pool <= 1 << 30 and self.pool % 16 == 0,
                'current so_state_header and so_state_size constraints')
        self.start = BASE + 144 + 552 * self.slots
        require(self.start + self.pool == len(raw), 'actual complete owner extent')

    def q(self, offset):
        require(0 <= offset <= len(self.raw) - 8, 'owner word bounds')
        return struct.unpack_from('<Q', self.raw, offset)[0]

    def row(self, kind, token):
        slot = token & 0xffffff
        require(1 <= kind <= 8 and token >> 24 & 255 == kind and 0 < slot <= self.slots,
                'typed live handle extent')
        row = struct.unpack_from('<8Q', self.raw, BASE + 128 + ((kind - 1) * self.slots + slot - 1) * 64)
        require(row[0] == token >> 32 > 0 and row[1] == 1, 'typed live handle generation')
        return row

    def node(self, token):
        index = self.row(1, token)[6]
        require(0 < index < self.q(8), 'public_node descriptor index extent')
        address = self.q(4096 + (index >> 8) * 8)
        require(address >> 63 == 1, 'relative descriptor page address')
        at = self.start + (address & (MASK >> 1)) + (index & 255) * 672
        require(at <= len(self.raw) - 672, 'complete 84-word descriptor')
        values = struct.unpack_from('<84Q', self.raw, at)
        require(values[24] == token and values[25] == 1 and values[30] == 1 and values[38] > 0,
                'public Tensor descriptor bound to live owner handle')
        return values

    def live_counts(self):
        return [sum(self.q(BASE + 128 + ((kind - 1) * self.slots + slot) * 64 + 8)
                    for slot in range(self.slots)) for kind in range(1, 9)]


def split_frames(raw, magic):
    frames, at = [], 0
    while at < len(raw):
        require(at + 16 <= len(raw), 'complete physical frame header')
        observed, size = struct.unpack_from('<8sQ', raw, at)
        require(observed == magic and 32 <= size <= 1 << 20 and at + 16 + size <= len(raw),
                'physical frame magic and extent')
        frames.append(raw[at + 16:at + 16 + size])
        at += 16 + size
    return frames


class Live:
    replay = False
    def __init__(self, binary, wire, directory, bind):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory, self.wire, self.bind = directory, wire, bind
        self.s = wire.Session(binary, directory=directory / 'raw')
        self.owners = []
        self.maps_reads = []
        self.pid = self.s.pid
        self.limit = None
        self.resource_record = None

    def exchange(self, request):
        return self.s.exchange(request)

    def owner(self):
        observed = self.s.owner()
        raw = observed['bytes']
        digest = sha(raw)
        path = self.directory / (digest + '.owner.xz')
        if not path.exists():
            path.write_bytes(lzma.compress(raw, preset=1))
        self.owners.append(dict(frame=self.s.frames, sha256=digest, pointer=observed['pointer'],
                               length=observed['length'], live=observed['live']))
        return observed

    def lower_limit(self, owner):
        statm = Path('/proc/%d/statm' % self.pid).read_bytes()
        limits = Path('/proc/%d/limits' % self.pid).read_bytes()
        pages = int(statm.split()[0])
        virtual = pages * os.sysconf('SC_PAGE_SIZE')
        previous = resource.prlimit(self.pid, resource.RLIMIT_AS)
        page = lambda size: (size + 4095) & ~4095
        limit = virtual + page(owner['length'] + 16) + page(5056 + 16) + 65536
        require(previous[1] == resource.RLIM_INFINITY or limit <= previous[1], 'original hard limit permits bounded probe')
        (self.directory / 'before-limit.statm').write_bytes(statm)
        (self.directory / 'before-limit.limits').write_bytes(limits)
        self.limit = previous
        resource.prlimit(self.pid, resource.RLIMIT_AS, (limit, previous[1]))
        actual = resource.prlimit(self.pid, resource.RLIMIT_AS)
        require(actual == (limit, previous[1]), 'actual child RLIMIT_AS applied')
        self.resource_record = dict(pid=self.pid, page_size=os.sysconf('SC_PAGE_SIZE'),
            owner_bytes=owner['length'], old_limit=list(previous), applied_limit=list(actual),
            before_statm_sha256=sha(statm), before_limits_sha256=sha(limits), restored_limit=None)

    def maps(self):
        raw = Path('/proc/%d/maps' % self.pid).read_bytes()
        name = 'maps-%03d.txt' % len(self.maps_reads)
        (self.directory / name).write_bytes(raw)
        self.maps_reads.append(dict(frame=self.s.frames, path=name, bytes=len(raw), sha256=sha(raw)))
        return raw

    def restore_limit(self):
        require(self.limit is not None, 'a bounded resource probe is active')
        resource.prlimit(self.pid, resource.RLIMIT_AS, self.limit)
        restored = resource.prlimit(self.pid, resource.RLIMIT_AS)
        require(restored == self.limit, 'actual original child resource limit restored')
        self.resource_record['restored_limit'] = list(restored)
        self.limit = None
        dump(self.directory / 'resource.json', self.resource_record)

    def finish(self):
        if self.limit is not None and self.s.poll() is None:
            self.restore_limit()
        try:
            self.s.finish()
        finally:
            dump(self.directory / 'owners.json', self.owners)
            dump(self.directory / 'maps.json', self.maps_reads)
        transport = json.loads((self.directory / 'raw/transport.json').read_bytes())
        require(transport['exit_code'] == 0, 'actual native exit zero')
        # The pinned transport's normal finish reads both physical EOFs.
        dump(self.directory / 'finish.json', dict(exit_code=0, stdout_EOF=True, stderr_EOF=True,
             executable_sha256=transport['executable_sha256'], frames=transport['frames'],
             owner_reads=len(self.owners), mapping_reads=len(self.maps_reads), forced_cleanup=False))
        return dict(frames=transport['frames'], owner_reads=len(self.owners), mapping_reads=len(self.maps_reads))


class Replay:
    """Consume complete recorded requests, responses and owner reads without execution."""
    replay = True
    def __init__(self, binary, wire, directory, bind):
        self.directory, self.wire, self.bind = directory, wire, bind
        self.requests = split_frames(bind(directory / 'raw/requests.frames.bin'), b'HTOWNR01')
        self.responses = split_frames(bind(directory / 'raw/responses.frames.bin'), b'HTOWNS01')
        self.owners = json.loads(bind(directory / 'owners.json'))
        maps_path = directory / 'maps.json'
        self.maps_reads = json.loads(bind(maps_path)) if maps_path.exists() else []
        self.transport = json.loads(bind(directory / 'raw/transport.json'))
        self.finish_record = json.loads(bind(directory / 'finish.json'))
        self.pid = self.transport['pid']
        self.at, self.owner_at = 0, 0
        self.maps_at = 0
        self.limit = None
        require(sha(bind(binary)) == self.transport['executable_sha256'] == self.finish_record['executable_sha256'],
                'actual native image retained in recorded transport')
        require(self.transport['exit_code'] == self.finish_record['exit_code'] == 0
                and self.finish_record['stdout_EOF'] is True and self.finish_record['stderr_EOF'] is True
                and self.finish_record['forced_cleanup'] is False
                and self.transport['responses_before_stdin_eof'] is True,
                'actual normal native exit and complete physical EOFs')
        require(len(self.requests) == len(self.responses) == len(self.transport['rows'])
                == self.transport['frames'] == self.finish_record['frames'], 'complete physical frame census')

    def exchange(self, request):
        require(self.at < len(self.requests) and self.requests[self.at] == request,
                'independently reconstructed request equals original physical request')
        row = self.transport['rows'][self.at]
        response = self.responses[self.at]
        require(row['frame'] == self.at + 1 and row['pid'] == self.pid and row['stdin_open'] is True
                and row['response_bytes'] == len(response), 'actual request-response ordering and PID')
        self.at += 1
        return response

    def owner(self):
        require(self.owner_at < len(self.owners), 'every requested owner read was captured')
        row = self.owners[self.owner_at]
        require(row['frame'] == self.at, 'owner capture at exact original frame boundary')
        decoder = lzma.LZMADecompressor()
        raw = decoder.decompress(self.bind(self.directory / (row['sha256'] + '.owner.xz')), max_length=(64 << 20) + 1)
        require(decoder.eof and not decoder.unused_data and len(raw) == row['length'] <= 64 << 20
                and sha(raw) == row['sha256'], 'complete raw owner, not an observer projection')
        self.owner_at += 1
        return dict(pointer=row['pointer'], length=row['length'], bytes=raw, live=row['live'])

    def lower_limit(self, owner):
        record = json.loads(self.bind(self.directory / 'resource.json'))
        statm = self.bind(self.directory / 'before-limit.statm', record['before_statm_sha256'])
        limits = self.bind(self.directory / 'before-limit.limits', record['before_limits_sha256'])
        line = next(line for line in limits.decode().splitlines() if line.startswith('Max address space'))
        values = line[len('Max address space'):].split()
        previous = [-1 if text == 'unlimited' else int(text) for text in values[:2]]
        page = lambda size: (size + 4095) & ~4095
        expected = int(statm.split()[0]) * record['page_size'] + page(owner['length'] + 16) + page(5072) + 65536
        require(record['pid'] == self.pid and record['page_size'] == 4096
                and record['owner_bytes'] == owner['length'] and record['old_limit'] == previous
                and record['applied_limit'] == [expected, previous[1]] and values[2] == 'bytes'
                and (previous[1] == -1 or expected <= previous[1]), 'original actual resource observations and fixed budget')
        self.limit = record

    def maps(self):
        require(self.maps_at < len(self.maps_reads), 'each actual complete mapping observation is retained')
        row = self.maps_reads[self.maps_at]
        require(row['frame'] == self.at and row['path'] == 'maps-%03d.txt' % self.maps_at,
                'actual mapping observation at exact original frame boundary')
        raw = self.bind(self.directory / row['path'], row['sha256'])
        require(len(raw) == row['bytes'], 'complete actual mapping observation')
        self.maps_at += 1
        return raw

    def restore_limit(self):
        require(self.limit is not None and self.limit['restored_limit'] == self.limit['old_limit'],
                'actual original resource limit was restored before retry')
        self.limit = None

    def finish(self):
        require(self.at == len(self.requests) and self.owner_at == len(self.owners)
                == self.finish_record['owner_reads'] and self.limit is None
                and self.maps_at == len(self.maps_reads) == self.finish_record.get('mapping_reads', 0),
                'every original frame, complete owner and resource operation consumed')
        result = dict(frames=self.at, owner_reads=self.owner_at)
        if 'mapping_reads' in self.finish_record:
            result['mapping_reads'] = self.maps_at
        return result


class Observer:
    def __init__(self, session, wire, reference_data):
        self.s, self.wire, self.reference = session, wire, reference_data
        self.checks = 0

    def check(self, label, actual, expected):
        self.checks += 1
        require(actual == expected, label)

    def one(self, *command):
        status, records = self.wire.parse(self.s.exchange(self.wire.packet([command])))
        self.check('single-command success', status, 0)
        self.check('one matching response', [r['opcode'] for r in records], [command[0]])
        return records[0]

    def create(self, spec):
        raw = reference.input_bytes(self.reference, spec)
        shape = spec['shape']
        cfg = struct.pack('<8Q', len(shape), 0, *shape, *([0] * (4 - len(shape))), 160, len(raw))
        packet = b'HTORCH03' + struct.pack('<3Q', 1, 160 + len(raw), 0)
        packet += struct.pack('<8Q', 97, spec['native_dtype'], 96, 0, 0, 0, 0, 0) + cfg + raw
        status, records = self.wire.parse(self.s.exchange(packet))
        self.check('literal typed factory status', status, 0)
        self.check('literal typed factory response', [r['opcode'] for r in records], [97])
        return records[0]['value']

    def command(self, spec, source):
        axes = spec['axes']
        payload = 0 if not axes else self.wire.Payload(struct.pack('<' + 'q' * len(axes), *axes))
        return (spec['opcode'], source, MASK if axes is None else len(axes), payload,
                int(spec['keepdim']), 1, 0, 0)

    def issue(self, request):
        before = self.s.owner()
        status, records = self.wire.parse(self.s.exchange(request))
        after = self.s.owner()
        if status:
            self.check('failed request complete owner pointer/length/bytes rollback', after, before)
            self.check('failed request returns no partial records', records, [])
        return status, records, before, after

    def tensor(self, token):
        meta = self.wire.meta(self.one(13, token))
        dtype, width, incarnation, leaf = struct.unpack('<4Q', self.one(96, token, 0)['payload'])
        record = self.one(98, token, 0, meta['numel'])
        raw = self.s.owner()['bytes']
        memory = Memory(raw)
        node = memory.node(token)
        tensor_row = memory.row(1, token)
        storage, version = memory.row(2, node[27]), memory.row(3, node[28])
        require(dtype in DTYPE and node[0] <= 4 and node[4] in (0, 1)
                and leaf in (0, 1) and node[45] in (0, 1), 'Half/Bool observation metadata domains')
        # so_tensor_live / so_storage_live / so_version_live use these rows;
        # the descriptor's handles and byte offset must bind the same objects.
        self.check('actual Tensor row binds descriptor Storage/Version/byte offset',
                   list(tensor_row[2:5]), list(node[27:30]))
        require(storage[5] > 0 and storage[6] > 0 and storage[7] == 0 and version[3] > 0,
                'actual Storage and Version objects are live')
        require(node[29] % width == 0 and tensor_row[7] <= 3, 'aligned byte offset and valid retained flags')
        # gradient_present reads descriptor+360. gradient_binding validates
        # descriptor+384/+392, whereas retained_enabled reads Tensor row+56.
        self.check('absent gradient has no private gradient binding',
                   [node[48] == 0, node[49] == 0], [node[45] == 0] * 2)
        self.check('fresh no-grad Tensor has no retain or temporary-root flags', tensor_row[7], 0)
        self.check('all 63 unused dense descriptor words are zero',
                   [node[word] for word in DENSE_ZERO_WORDS], [0] * 63)
        self.check('public dtype and width reflect actual descriptor', [dtype, width, incarnation], [node[72], WIDTH[dtype], node[40]])
        start = memory.start + storage[2]
        require(start <= len(raw) and storage[3] <= len(raw) - start, 'actual full Storage bounds')
        physical = raw[start:start + storage[3]]
        rank, count = node[:2]
        shape, strides = list(node[12:12 + rank]), list(node[16:16 + rank])
        logical = bytearray()
        for i in range(count):
            remaining, offset = i, node[29]
            for axis in range(rank - 1, -1, -1):
                offset += (remaining % shape[axis]) * strides[axis] * width
                remaining //= shape[axis]
            require(0 <= offset <= len(physical) - width, 'actual logical address inside Storage')
            logical.extend(physical[offset:offset + width])
        self.check('public raw bytes reflect actual owned Storage', record['payload'], bytes(logical))
        self.check('public raw element count', record['value'], count)
        self.check('complete public geometry reflects actual owner',
            [meta['rank'], meta['numel'], meta['shape'], meta['strides'], meta['storage_id'],
             meta['version_id'], meta['storage_offset'], meta['storage_bytes'], meta['version'], meta['requires_grad'], leaf],
            [rank, count, list(node[12:16]), list(node[16:20]), node[27], node[28], node[29], storage[3],
             version[2], node[4], int(node[5] == 0)])
        self.check('complete source element census', count, math.prod(shape))
        result = dict(dtype=DTYPE[dtype], element_size=width, shape=shape, strides=strides, numel=count,
            storage_bytes=storage[3], storage_offset=node[29] // width, version=version[2],
            requires_grad=bool(node[4]), is_leaf=node[5] == 0, is_view=bool(node[51]),
            grad_fn=None if node[5] == 0 else 'NATIVE_HISTORY_PRESENT', grad_is_none=node[45] == 0,
            retains_grad=bool(tensor_row[7] & 1),
            base_tensor_identity=None if node[83] == 0 else 'NATIVE_VIEW_BASE_PRESENT',
            logical=bytes(logical), storage=physical)
        return result, node

    def expected(self, value):
        return {key: (reference.blob(self.reference, item) if key in ('logical', 'storage') else item)
                for key, item in value.items() if key not in ('tensor_identity', 'storage_identity')}

    def begin(self):
        generator = self.one(60, 1729)['value']
        rng = self.one(63, generator)['payload']
        self.check('complete original Generator state extent', len(rng), 5056)
        return generator, rng

    def end(self, generator, rng):
        self.check('complete Generator bytes unchanged', self.one(63, generator)['payload'], rng)
        self.one(66, generator)
        self.check('final native owner audit', self.one(53)['value'], 1)
        counts = Memory(self.s.owner()['bytes']).live_counts()
        self.check('all public Tensors released', counts[0], 0)
        self.check('all public Generators released', counts[6], 0)


def expected_status(row):
    if row['accepted']:
        return 0
    error = row['error']
    if error['type'] == 'IndexError' and error['message'].startswith('Dimension out of range'):
        return 2
    if error['type'] == 'RuntimeError' and re.fullmatch(r'dim [0-9]+ appears multiple times in the list of dims', error['message']):
        return 7
    raise ValueError('unmapped original Torch error')


def compare_case(n, row):
    spec = row['spec']
    generator, rng = n.begin()
    source = n.create(spec)
    original, source_node = n.tensor(source)
    n.check('all original input metadata and logical/physical bytes', original, n.expected(row['input']))
    status, records, before, after = n.issue(n.wire.packet([n.command(spec, source)]))
    n.check('actual Torch acceptance/error class', status, expected_status(row))
    n.check('all 84 original input descriptor words unchanged', Memory(after['bytes']).node(source), Memory(before['bytes']).node(source))
    n.check('complete input preserved', n.tensor(source)[0], original)
    if status == 0:
        n.check('one actual reduction response', [r['opcode'] for r in records], [spec['opcode']])
        result = records[0]['value']
        observed, result_node = n.tensor(result)
        n.check('complete independently observed Torch result', observed, n.expected(row['result']))
        n.check('distinct actual Tensor/Storage/Version identities',
                [result != source, result_node[27] != source_node[27], result_node[28] != source_node[28]], [True] * 3)
        n.check('no autograd, Saved, view or source edges',
                [result_node[o // 8] for o in (32,40,48,56,72,80,88,176,184,248,280,288,296,312,360,384,392,400,408,584,592,600,608,616,624,640,648,656,664)], [0] * 29)
        n.check('actual result Tensor row has no retained gradient', observed['retains_grad'], False)
        n.one(41, source)
        n.check('result remains complete after releasing input', n.tensor(result)[0], observed)
        n.one(41, result)
    else:
        n.one(41, source)
    n.end(generator, rng)
    return dict(index=row['index'], name=spec['name'], accepted=status == 0, reference_error_status=expected_status(row))


def packet_controls(n, rows, dtype):
    row = next(r for r in rows if r['spec']['native_dtype'] == dtype and r['spec']['name'].endswith('/signed-zeros-all'))
    generator, rng = n.begin()
    source = n.create(row['spec'])
    original = n.tensor(source)[0]
    controls = 0
    for opcode in (115, 118, 119):
        valid = [opcode, source, MASK, 0, 0, 1, 0, 0]
        mutations = [('reserved48',6,1,7), ('reserved56',7,1,7), ('invalid-keepdim',4,2,7),
                     ('too-many-axes',2,5,7), ('invalid-source',1,1,1)]
        if opcode != 115:
            mutations.append(('invalid-output-dtype',5,0,7))
        for label, position, value, expected in mutations:
            bad = valid.copy(); bad[position] = value
            status, _, _, _ = n.issue(n.wire.packet([bad]))
            n.check('packet/' + label, status, expected); controls += 1
        bad = valid.copy(); bad[2], bad[3] = 1, 4096
        status, _, _, _ = n.issue(n.wire.packet([bad]))
        n.check('packet/missing-axis-payload', status, 2); controls += 1
        bad = valid.copy(); bad[6] = 1
        status, _, _, _ = n.issue(n.wire.packet([valid, bad]))
        n.check('packet/successful-allocation-then-invalid-suffix', status, 7); controls += 1
    n.check('packet controls preserve all input metadata/storage', n.tensor(source)[0], original)
    n.one(41, source)
    for opcode in (115, 118, 119):
        status, _, _, _ = n.issue(n.wire.packet([(opcode, source, MASK, 0, 0, 1, 0, 0)]))
        n.check('packet/stale-source', status, 1); controls += 1
    n.end(generator, rng)
    return dict(dtype=dtype, controls=controls)


def allocation_control(n, row):
    generator, rng = n.begin()
    source = n.create(row['spec'])
    before = n.tensor(source)[0]
    n.check('OOM literal input equals actual Torch input', before, n.expected(row['input']))
    owner = n.s.owner()
    n.s.lower_limit(owner)
    live = []
    attempts = 0
    command = n.wire.packet([n.command(row['spec'], source)])
    for attempt in range(32):
        status, records, _, _ = n.issue(command)
        attempts += 1
        if status:
            n.check('actual RLIMIT allocation refusal', status, 3)
            break
        live.append(records[0]['value'])
    else:
        raise ValueError('did not reach a real allocation refusal in the fixed 32-attempt budget')
    n.check('OOM full RNG preserved', n.one(63, generator)['payload'], rng)
    n.s.restore_limit()
    status, records, _, _ = n.issue(command)
    n.check('identical request succeeds after actual limit restoration', status, 0)
    result = records[0]['value']; live.append(result)
    n.check('actual retry result equals independent Torch observation', n.tensor(result)[0], n.expected(row['result']))
    n.check('OOM source remains complete', n.tensor(source)[0], before)
    for token in live:
        n.one(41, token)
    n.one(41, source)
    n.end(generator, rng)
    return dict(reference_index=row['index'], attempts=attempts, allocation_refusals=1,
                complete_owner_rollback=True, identical_request_succeeded_after_limit_restore=True)


def factory_transaction(n, row, invalid_suffix=False):
    spec = row['spec']; raw = reference.input_bytes(n.reference, spec)
    require(spec['shape'] == [2] and spec['axes'] is None and spec['keepdim'] is False,
            'fixed signed-zero transaction fixture')
    count = 3 if invalid_suffix else 2
    end = 32 + 64 * count
    body = b'HTORCH03' + struct.pack('<3Q', count, end + 64 + len(raw), 0)
    body += struct.pack('<8Q', 97, spec['native_dtype'], end, 0, 0, 0, 0, 0)
    body += struct.pack('<8Q', spec['opcode'], n.wire.reference(1), MASK, 0, 0, 1, 0, 0)
    if invalid_suffix:
        body += struct.pack('<8Q', spec['opcode'], n.wire.reference(1), MASK, 0, 0, 1, 1, 0)
    body += struct.pack('<8Q', 1, 0, 2, 0, 0, 0, end + 64, len(raw)) + raw
    return body


def transaction_control(n, row, failed):
    generator, rng = n.begin()
    if not failed:
        source = n.create(row['spec'])
        before = n.tensor(source)[0]
        status, records, _, _ = n.issue(n.wire.packet([n.command(row['spec'], source)]))
        n.check('formerly unsupported direct Half reduction now succeeds', status, 0)
        n.check('new direct result independently observed by Torch', n.tensor(records[0]['value'])[0], n.expected(row['result']))
        n.check('direct input unchanged', n.tensor(source)[0], before)
        n.one(41, records[0]['value']); n.one(41, source)
    else:
        prefix = factory_transaction(n, row)
        status, records, _, _ = n.issue(prefix)
        n.check('complete same-frame factory/reduction prefix succeeds', status, 0)
        n.check('same-frame prefix has exactly the expected records', [r['opcode'] for r in records], [97, row['spec']['opcode']])
        n.check('same-frame prefix result equals independent reference', n.tensor(records[1]['value'])[0], n.expected(row['result']))
        n.one(41, records[1]['value']); n.one(41, records[0]['value'])
        maps_before = n.s.maps()
        status, records, _, _ = n.issue(factory_transaction(n, row, True))
        n.check('invalid suffix rejects the whole successful-prefix transaction', status, 7)
        n.check('failed full transaction exposes no records', records, [])
        n.check('failed full transaction restores all actual process mappings', n.s.maps(), maps_before)
    n.end(generator, rng)
    return dict(reference_index=row['index'], dtype=row['spec']['native_dtype'], opcode=row['spec']['opcode'],
                group='failed-FE-transaction' if failed else 'direct-success',
                reference_result_preserved=True, historical_refusal_report_modified=False)


def recipes(reference_data, group):
    rows = reference_data['report']['rows']
    if group == 'half_logical264':
        result = [('case-%03d' % r['index'], compare_case, (r,)) for r in rows]
        result += [('packet-dtype%d' % dtype, packet_controls, (rows, dtype)) for dtype in (5, 6)]
        result += [('allocation-%03d' % r['index'], allocation_control, (r,)) for r in rows[:6]]
        return result
    require(group == 'half_transactions12', 'known Half qualification group')
    selected = [r for r in rows if r['spec']['name'].endswith('/signed-zeros-all')]
    require(len(selected) == 6, 'six independent original transaction reference cases')
    return [('%s-%03d' % ('failed-FE' if failed else 'direct', r['index']), transaction_control, (r, failed))
            for failed in (False, True) for r in selected]
