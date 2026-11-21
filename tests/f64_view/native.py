"""F64 view observations and the same comparisons over complete recorded bytes."""
import math
import struct

from numeric_predicates.native import Observer, Live, Replay, Memory, load_wire
from .cases import raw_input, MASK, ALLOCATION_SPEC, ALLOCATION_SHA256, KNOWN_INCOMPATIBILITY_INDEX


def require(value, message):
    if not value:
        raise ValueError(message)


def view_config(operation):
    shape, strides = operation['shape'], operation['strides']
    require(len(shape) == len(strides) <= 4, 'exact current view geometry arity')
    words = [len(shape), *shape, *([0] * (4 - len(shape))),
             *strides, *([0] * (4 - len(strides))), operation['offset']]
    return struct.pack('<10Q', *(word & MASK for word in words))


def view_packet(wire, source, operation):
    return wire.packet([(111, source, wire.Payload(view_config(operation)), 80)])


def factory_view_packet(wire, spec, invalid_suffix=False):
    raw = raw_input(spec['base'])
    shape = spec['base']['shape']
    operation = spec['operations'][0]
    require(operation['operation'] == 'view' and operation['source'] == 'root', 'one exact first view')
    count = 2 + int(invalid_suffix)
    end = 32 + 64 * count
    data = end + 64
    view_at = data + len(raw)
    require(view_at % 8 == 0, 'typed F64 wire alignment')
    commands = struct.pack('<8Q', 97, 4, end, 0, 0, 0, 0, 0)
    commands += struct.pack('<8Q', 111, wire.reference(1), view_at, 80, 0, 0, 0, 0)
    if invalid_suffix:
        commands += struct.pack('<8Q', 125, wire.reference(2), 0, 0, 0, 0, 0, 0)
    config = struct.pack('<8Q', len(shape), 0, *shape, *([0] * (4 - len(shape))), data, len(raw))
    return b'HTORCH03' + struct.pack('<3Q', count, view_at + 80, 0) + commands + config + raw + view_config(operation)


def indexed_node(memory, index):
    require(0 < index < memory.q(8), 'retained descriptor index inside complete owner')
    page = memory.q(4096 + (index >> 8) * 8)
    require(page >> 63 == 1, 'relative descriptor page')
    at = memory.start + (page & (MASK >> 1)) + (index & 255) * 672
    require(at + 672 <= len(memory.raw), 'complete retained descriptor extent')
    return struct.unpack_from('<84Q', memory.raw, at)


def live_rows(memory, kind):
    result = {}
    for slot in range(memory.slots):
        at = 139264 + 128 + ((kind - 1) * memory.slots + slot) * 64
        row = struct.unpack_from('<8Q', memory.raw, at)
        require(row[1] in (0, 1), 'complete owner row state domain')
        if row[1]:
            token = (row[0] << 32) | (kind << 24) | (slot + 1)
            result[token] = row
    return result


def descriptor_only_contract(raw, root_index, root_storage, root_version):
    memory = Memory(raw)
    pages = (memory.q(8) + 255) // 256
    require(memory.live_counts() == [0, pages, 0, 0, 0, pages, 0, 0], 'only descriptor pages remain')
    storages, buffers = set(), set()
    for index in range(pages):
        buffer = memory.q(6144 + index * 8)
        buffer_row = memory.row(6, buffer)
        storage = buffer_row[2]
        storage_row = memory.row(2, storage)
        page = memory.q(4096 + index * 8)
        require(page >> 63 == 1 and list(storage_row[2:]) == [page & (MASK >> 1), 172032, 172032, 1, 1, 0]
                and buffer_row[3] == 1 and list(buffer_row[4:]) == [0, 0, 0, 0],
                'exact descriptor-page Buffer and Storage geometry')
        storages.add(storage)
        buffers.add(buffer)
    require(len(storages) == len(buffers) == pages and set(live_rows(memory, 2)) == storages
            and set(live_rows(memory, 6)) == buffers,
            'all remaining live rows are exact unique descriptor-page pairs')
    root = indexed_node(memory, root_index)
    require(root[38] == root[75] == 0 and root_storage not in live_rows(memory, 2)
            and root_version not in live_rows(memory, 3), 'root, all root edges, original Storage and Version released')
    return True


class OwnerModel:
    """All84 words of every live descriptor and all live rows of all8 tables.

Opaque capability tokens and relative allocations are observed identities.
Their relationships, complete descriptor fields and reference counts are
derived from the unchanged HIA constructors/root-retain/alias/release bodies.
No expected tensor bytes are computed here.
"""
    def __init__(self, n, root, reference_root, previous_counter, *, extra_views=0):
        memory = Memory(n.s.owner()['bytes'])
        row = memory.row(1, root)
        node = memory.node(root)
        n.check('factory publishes exactly the next native incarnation', node[40], previous_counter + 1)
        n.check('factory and any same-frame view consume exact constructor incarnations',
                memory.q(120), previous_counter + 1 + extra_views)
        self.root = root
        self.root_index, self.root_incarnation = row[6], node[40]
        self.storage, self.version = node[27:29]
        self.storage_geometry = memory.row(2, self.storage)[2:5]
        self.base = reference_root
        self.known = {'root': dict(handle=root, index=self.root_index, incarnation=self.root_incarnation,
                                   geometry=reference_root, created_version=reference_root['version'])}
        self.support = {kind: live_rows(memory, kind) for kind in range(1, 9)}
        self.support[1] = {}
        self.support[2].pop(self.storage)
        self.support[3].pop(self.version)

    def register(self, n, name, token, expected, previous_counter, *, remaining_views=0):
        memory = Memory(n.s.owner()['bytes'])
        row = memory.row(1, token)
        node = memory.node(token)
        n.check('each view publishes exactly the next incarnation', node[40], previous_counter + 1)
        n.check('every view constructor increments the complete native incarnation counter once',
                memory.q(120), previous_counter + 1 + remaining_views)
        n.check('new view capability differs from every previous Tensor identity',
                token not in [item['handle'] for item in self.known.values()], True)
        self.known[name] = dict(handle=token, index=row[6], incarnation=node[40], geometry=expected,
                                created_version=expected['version'])

    def observe(self, n, handles, stage):
        # Audit before metadata reads, so a read cannot normalize an invalid flag.
        n.check('actual command53 audit before any public metadata refresh', n.one(53)['value'], 1)
        n.check('all declared live public Tensor names', set(handles), set(stage['tensors']))
        for name, token in handles.items():
            actual, _ = n.tensor(token)
            n.check('complete actual Torch bits/geometry/version/flags: ' + name,
                    actual, n.expected(stage['tensors'][name]))
        memory = Memory(n.s.owner()['bytes'])
        liveness = stage['actual_Torch_Tensor_liveness']
        n.check('all created Tensor lifetimes have actual Torch weak-reference observations', set(liveness), set(self.known))
        active = {}
        for name, item in self.known.items():
            node = indexed_node(memory, item['index'])
            alive = node[38] > 0 and node[40] == item['incarnation']
            n.check('actual Tensor lifetime and incarnation match Torch: ' + name, alive, liveness[name])
            if alive:
                active[name] = item
        children = len(active) - int('root' in active)
        n.check('every surviving view keeps the original root alive', bool(children) <= ('root' in active), True)
        expected_rows = {kind: dict(rows) for kind, rows in self.support.items()}
        # Descriptor pages are permanent runtime bookkeeping. A larger alias
        # family may allocate another page; its complete rows are constrained
        # by the unchanged page allocator, not copied from arbitrary live rows.
        for page_index in range((memory.q(8) + 255) // 256):
            page = memory.q(4096 + page_index * 8)
            buffer = memory.q(6144 + page_index * 8)
            require(page >> 63 == 1, 'complete native descriptor-page table')
            storage = memory.row(6, buffer)[2]
            page_storage = (storage >> 32, 1, page & (MASK >> 1), 172032, 172032, 1, 1, 0)
            page_buffer = (buffer >> 32, 1, storage, 1, 0, 0, 0, 0)
            require(storage != self.storage and storage not in expected_rows[3]
                    and buffer not in expected_rows[1], 'descriptor pages do not alias the Tensor family')
            if storage in expected_rows[2]:
                n.check('prior descriptor-page Storage stays exact', expected_rows[2][storage], page_storage)
            if buffer in expected_rows[6]:
                n.check('prior descriptor-page Buffer stays exact', expected_rows[6][buffer], page_buffer)
            expected_rows[2][storage] = page_storage
            expected_rows[6][buffer] = page_buffer
        if active:
            require('root' in active and handles, 'live alias family has one retained true root')
            observed_version = next(iter(stage['tensors'].values()))['version']
            n.check('all independent Torch aliases expose the same version',
                    {value['version'] for value in stage['tensors'].values()}, {observed_version})
            storage_row = memory.row(2, self.storage)
            version_row = memory.row(3, self.version)
            expected_rows[2][self.storage] = (self.storage >> 32, 1, *self.storage_geometry, len(active), 1, 0)
            expected_rows[3][self.version] = (self.version >> 32, 1, observed_version, len(active), 0, 0, 0, 0)
            n.check('all original Storage geometry and strong/weak counts', storage_row, expected_rows[2][self.storage])
            n.check('complete shared Version object and strong count', version_row, expected_rows[3][self.version])
            for name, item in active.items():
                geometry = self.base if name == 'root' else stage['tensors'][name]
                shape, strides = geometry['shape'], geometry['strides']
                offset = geometry['storage_offset'] * 8
                token = item['handle']
                expected = [0] * 84
                expected[0:3] = [len(shape), geometry['numel'], (1 << 63) + self.storage_geometry[0] + offset]
                expected[8] = self.root_index
                expected[12:12 + len(shape)] = shape
                expected[16:16 + len(strides)] = strides
                expected[24:31] = [token, int(name in handles), children if name == 'root' else 0,
                                   self.storage, self.version, offset, 1]
                expected[38] = 1
                expected[40] = item['incarnation']
                expected[72] = 4
                if name == 'root':
                    expected[75] = children
                else:
                    expected[51] = 1
                    expected[80:84] = [self.root_index, self.root_incarnation, item['created_version'], 1]
                n.check('all84 independently derived no-AD descriptor words: ' + name,
                        list(indexed_node(memory, item['index'])), expected)
                expected_rows[1][token] = (token >> 32, 1, self.storage, self.version, offset,
                    max(self.base['storage_bytes'] - offset, 0), item['index'], 0)
        for kind in range(1, 9):
            n.check('entire live inventory of native owner table%d' % kind,
                    live_rows(memory, kind), expected_rows[kind])
        n.check('each actual Torch alias has the retained original Storage and true root',
                all(value['shared_original_storage'] and value['original_root_base'] for value in stage['aliases'].values()), True)

    def finish(self, n):
        n.check('all original family capabilities and owner references are released',
                descriptor_only_contract(n.s.owner()['bytes'], self.root_index, self.storage, self.version), True)


def compare(n, row):
    if row['index'] == KNOWN_INCOMPATIBILITY_INDEX:
        return known_incompatibility(n, row)
    spec = row['spec']
    generator, rng = n.begin()
    before_counter = Memory(n.s.owner()['bytes']).q(120)
    stages = row['stages']
    operations = spec['operations']
    start = 0
    if spec.get('legacy_same_frame'):
        status, records, _, _ = n.issue(factory_view_packet(n.wire, spec))
        n.check('unchanged original F64 factory plus FE view request succeeds', status, 0)
        n.check('both original same-frame records are retained', [item['opcode'] for item in records], [97, 111])
        root, first = [item['value'] for item in records]
        n.check('original FE factory bytes/metadata equal actual Torch before the view',
                n.tensor(root)[0], n.expected(stages[0]['tensors']['root']))
        model = OwnerModel(n, root, stages[0]['tensors']['root'], before_counter, extra_views=1)
        model.register(n, operations[0]['name'], first, stages[1]['tensors'][operations[0]['name']], before_counter + 1)
        handles = {'root': root, operations[0]['name']: first}
        model.observe(n, handles, stages[1])
        start = 1
    else:
        root = n.create(4, spec['base']['shape'], raw_input(spec['base']))
        handles = {'root': root}
        model = OwnerModel(n, root, stages[0]['tensors']['root'], before_counter)
        model.observe(n, handles, stages[0])
    for at in range(start, len(operations)):
        operation = operations[at]
        name = operation['name']
        if operation['operation'] == 'view':
            counter = Memory(n.s.owner()['bytes']).q(120)
            status, records, _, _ = n.issue(view_packet(n.wire, handles[operation['source']], operation))
            if not row['accepted']:
                n.check('native geometry error class matches the independently observed Torch refusal', status, spec['native_error'])
                n.check('failed view publishes no partial capability', records, [])
                model.observe(n, handles, stages[0])
                n.one(41, handles.pop('root'))
                n.end(generator, rng)
                model.finish(n)
                return dict(index=row['index'], accepted=False, original_owner_rollback=True,
                            actual_Torch_error=row['error']['type'], native_error=status)
            n.check('direct F64 view is an actual native success', status, 0)
            n.check('one actual F64 view result', [item['opcode'] for item in records], [111])
            handles[name] = records[0]['value']
            model.register(n, name, handles[name], stages[at + 1]['tensors'][name], counter)
        elif operation['operation'] == 'write':
            n.one(12, handles[name], operation['index'], operation['word'])
        else:
            require(operation['operation'] == 'release', 'literal declared operation kind')
            n.one(41, handles.pop(name))
        n.check('all original reference stages retain exact operation order',
                stages[at + 1]['label'], '%02d-%s-%s' % (at, operation['operation'], name))
        model.observe(n, handles, stages[at + 1])
    n.check('complete original operation and stage inventory', len(stages), len(operations) + 1)
    n.check('all public family handles explicitly released', handles, {})
    n.end(generator, rng)
    model.finish(n)
    return dict(index=row['index'], accepted=True, stages=len(stages),
                all84_live_descriptor_words_checked=True, all8_live_owner_table_inventories_checked=True,
                real_Torch_root_lifetimes_checked=True, direct_F64_alias_and_shared_version=True)


def known_incompatibility(n, row):
    spec = row['spec']
    require(row['index'] == KNOWN_INCOMPATIBILITY_INDEX and row['accepted'] is True
            and row['known_incompatibility'] == spec['known_incompatibility'] == 'negative_dimension_on_empty_view',
            'one exact genuinely accepted Torch request remains unimplemented by HIA')
    generator, rng = n.begin()
    counter = Memory(n.s.owner()['bytes']).q(120)
    root = n.create(4, spec['base']['shape'], raw_input(spec['base']))
    model = OwnerModel(n, root, row['stages'][0]['tensors']['root'], counter)
    handles = {'root': root}
    model.observe(n, handles, row['stages'][0])
    mappings = n.s.maps()
    status, records, before, after = n.issue(view_packet(n.wire, root, spec['operations'][0]))
    n.check('known incompatibility: HIA rejects the actual Torch-accepted negative empty shape', status, 3)
    n.check('known incompatible request returns no native Tensor', records, [])
    n.check('known incompatible refusal preserves complete owner bytes', after, before)
    n.check('known incompatible refusal preserves complete mappings', n.s.maps(), mappings)
    model.observe(n, handles, row['stages'][0])
    n.one(41, handles.pop('root'))
    n.end(generator, rng)
    model.finish(n)
    return dict(index=row['index'], compatible=False, known_incompatibility=spec['known_incompatibility'],
                actual_Torch_accepted=True, native_accepted=False, native_error=status,
                actual_Torch_result=row['boundary_result'],
                complete_owner_rollback=True, complete_mapping_rollback=True,
                counted_as_compatible_case=False, counted_as_Torch_error=False)


def packet_controls(n, row):
    generator, rng = n.begin()
    spec = row['spec']
    root = n.create(4, spec['base']['shape'], raw_input(spec['base']))
    before = n.tensor(root)[0]
    n.check('protocol-control input equals actual Torch', before, n.expected(row['stages'][0]['tensors']['root']))
    operation = spec['operations'][0]
    controls = 0

    def refused(request, status, label):
        nonlocal controls
        actual, records, _, _ = n.issue(request)
        n.check(label, actual, status)
        n.check('no rejected view capability is published', records, [])
        controls += 1

    for invalid in (0, 1, generator):
        refused(view_packet(n.wire, invalid, operation), 1, 'zero/raw/wrong-kind source capability')
    stale = n.create(4, [1], b'\0' * 8)
    n.one(41, stale)
    refused(view_packet(n.wire, stale, operation), 1, 'stale source capability')
    words = list(struct.unpack('<10Q', view_config(operation)))
    for position, value, expected in ((0, 5, 2), (2, 1, 2), (6, 1, 2),
                                     (1, 262145, 3), (9, 134217729, 3)):
        bad = words.copy()
        bad[position] = value
        refused(n.wire.packet([(111, root, n.wire.Payload(struct.pack('<10Q', *bad)), 80)]),
                expected, 'exact unchanged rank/tail/element/offset capacity refusal')
    malformed = bytearray(view_packet(n.wire, root, operation))
    struct.pack_into('<Q', malformed, 48, 97)
    refused(bytes(malformed), 100, 'unaligned or incomplete view payload')
    malformed = bytearray(view_packet(n.wire, root, operation))
    struct.pack_into('<Q', malformed, 48, len(malformed) - 72)
    refused(bytes(malformed), 100, 'view payload extent shorter than eighty bytes')
    refused(n.wire.packet([(69, root, n.wire.Payload(view_config(operation)), 80)]),
            7, 'original public F64 command69 refusal retained')
    for dtype in (5, 6):
        reduced = n.create(dtype, [1], b'\0\0')
        refused(view_packet(n.wire, reduced, dict(shape=[1], strides=[1], offset=0)),
                7, 'original Half/BF16 direct-view refusal retained')
        n.one(41, reduced)
    parent = n.create(0, [1], struct.pack('<I', 0x3f800000), True)
    graph = n.one(114, parent, 0x3ff0000000000000)['value']
    wrapped = n.one(107, graph, 1)['value']
    refused(view_packet(n.wire, wrapped, dict(shape=[], strides=[], offset=0)),
            7, 'original private scalar3 direct-view refusal retained')
    for token in (wrapped, graph, parent):
        n.one(41, token)
    cfg = struct.pack('<8Q', 1, 1, 2, 0, 0, 0, 160, 16)
    refused(n.wire.packet([(97, 4, n.wire.Payload(cfg + raw_input(spec['base'])))]),
            5, 'original public F64 requires-grad refusal retained')
    maps = n.s.maps()
    refused(n.wire.packet([(111, root, n.wire.Payload(view_config(operation)), 80), (125, root)]),
            4, 'allocated direct view and invalid suffix roll back complete root retention')
    n.check('all mappings restored after direct-view transaction failure', n.s.maps(), maps)
    maps = n.s.maps()
    refused(factory_view_packet(n.wire, spec, True), 4, 'factory plus FE view and invalid suffix roll back')
    n.check('all mappings restored after factory/FE transaction failure', n.s.maps(), maps)
    n.check('all protocol failures preserve complete original input', n.tensor(root)[0], before)
    n.one(41, root)
    refused(view_packet(n.wire, root, operation), 1, 'released original public root is not reusable')
    n.end(generator, rng)
    n.check('all nineteen exact native packet controls executed', controls, 19)
    return dict(controls=controls, complete_owner_rollback=True, complete_mapping_rollback=True,
                F64_command69_and_AD_refusals_retained=True, private3_Half_BF16_refusals_retained=True,
                actual_Torch_geometry_errors_recounted_as_controls=False)


def allocation_control(n, observation):
    spec = observation['spec']
    require(spec == ALLOCATION_SPEC and observation['spec_sha256'] == ALLOCATION_SHA256,
            'one exact independently observed allocation alias family')
    generator, rng = n.begin()
    counter = Memory(n.s.owner()['bytes']).q(120)
    root = n.create(4, spec['base']['shape'], raw_input(spec['base']))
    stages = observation['stages']
    model = OwnerModel(n, root, stages[0]['tensors']['root'], counter)
    handles = {'root': root}
    model.observe(n, handles, stages[0])
    request = n.wire.packet([(111, root, n.wire.Payload(view_config(spec)), 80)] * spec['views'])
    before = n.s.owner()
    counter = Memory(before['bytes']).q(120)
    n.s.lower_limit(before)
    mappings = n.s.maps()
    status, records, _, _ = n.issue(request)
    n.check('actual bounded native alias allocation fails', status, 3)
    n.check('failed alias batch publishes no partial capabilities', records, [])
    n.check('failed allocation restores every original mapping', n.s.maps(), mappings)
    n.check('failed allocation preserves complete Generator bytes', n.one(63, generator)['payload'], rng)
    n.s.restore_limit()
    status, records, _, _ = n.issue(request)
    n.check('the identical original alias request succeeds after actual limit restoration', status, 0)
    n.check('all original alias results retained after identical recovery',
            [item['opcode'] for item in records], [111] * spec['views'])
    for index, record in enumerate(records):
        name = 'alias-%03d' % index
        handles[name] = record['value']
        model.register(n, name, record['value'], stages[1]['tensors'][name], counter + index,
                       remaining_views=spec['views'] - index - 1)
    model.observe(n, handles, stages[1])
    n.one(41, handles.pop('root'))
    model.observe(n, handles, stages[2])
    for name in list(handles):
        n.one(41, handles.pop(name))
    model.observe(n, handles, stages[3])
    n.end(generator, rng)
    model.finish(n)
    return dict(actual_allocation_refusals=1, identical_request_recoveries=1,
                reference_spec_sha256=ALLOCATION_SHA256, recovered_aliases=spec['views'],
                complete_owner_rollback=True, complete_mapping_rollback=True,
                all84_live_descriptor_words_checked=True, all8_live_owner_table_inventories_checked=True,
                released_public_root_retained_until_last_alias=True)


FAULTS = (
    ('orphan-view-one', 'root', 408, 1, False),
    ('orphan-view-two', 'root', 408, 2, False),
    ('orphan-view-max', 'root', 408, MASK, False),
    ('view-mark-cleared', 'view', 408, 0, True),
    ('no-grad-view-flag-three', 'view', 664, 3, True),
    ('root-count-too-large', 'root', 600, 2, True),
    ('root-incarnation-too-large', 'view', 648, 2, True),
    ('view-version-in-future', 'view', 656, 1, True),
)


def injected_word(n, token, field, value):
    """One explicit diagnostic word in this newly spawned child's own buffer."""
    before = n.s.owner()
    memory = Memory(before['bytes'])
    index = memory.row(1, token)[6]
    page = memory.q(4096 + (index >> 8) * 8)
    require(page >> 63 == 1 and field % 8 == 0 and 0 <= field < 672, 'one complete owned descriptor word')
    offset = memory.start + (page & (MASK >> 1)) + (index & 255) * 672 + field
    original = before['bytes'][offset:offset + 8]
    modified = struct.pack('<Q', value)
    require(len(original) == 8 and modified != original, 'diagnostic mutation changes exactly one word')
    if not n.s.replay:
        with open('/proc/%d/mem' % n.s.pid, 'r+b', buffering=0) as stream:
            stream.seek(before['pointer'] + offset)
            require(stream.read(8) == original, 'exact newly spawned child word before diagnostic mutation')
            stream.seek(before['pointer'] + offset)
            require(stream.write(modified) == 8, 'complete diagnostic word write')
    after = n.s.owner()
    n.check('diagnostic changes exactly the one declared owned word', after,
            dict(before, bytes=before['bytes'][:offset] + modified + before['bytes'][offset + 8:]))
    return struct.unpack('<Q', original)[0], before


def audit_fault_control(n, row, fault):
    label, target, field, bad, with_view = fault
    spec = row['spec']
    generator, rng = n.begin()
    counter = Memory(n.s.owner()['bytes']).q(120)
    root = n.create(4, spec['base']['shape'], raw_input(spec['base']))
    model = OwnerModel(n, root, row['stages'][0]['tensors']['root'], counter)
    handles = {'root': root}
    stage = row['stages'][0]
    if with_view:
        token = n.one(111, root, n.wire.Payload(view_config(spec['operations'][0])), 80)['value']
        handles['view'] = token
        stage = row['stages'][1]
        model.register(n, 'view', token, stage['tensors']['view'], counter + 1)
    model.observe(n, handles, stage)
    original, before = injected_word(n, handles[target], field, bad)
    status = None
    try:
        status, records, _, _ = n.issue(n.wire.packet([(53,)]))
        n.check('actual live native audit refuses the original invalid state: ' + label, status, 9)
        n.check('refused actual audit publishes no result', records, [])
    finally:
        replaced, _ = injected_word(n, handles[target], field, original)
        n.check('restore the exact original diagnostic word', replaced, bad)
        n.check('all original owner bytes and identity restored', n.s.owner(), before)
    model.observe(n, handles, stage)
    for name in reversed(list(handles)):
        n.one(41, handles.pop(name))
    n.end(generator, rng)
    model.finish(n)
    return dict(label=label, actual_status=status, target=target, descriptor_field=field,
                injected_word=bad, original_word=original, exactly_one_owned_word_changed=True,
                input_kind='explicit synthetic diagnostic descriptor corruption',
                candidate_binary_modified=False, exact_original_owner_restored=True,
                complete_refused_request_owner_rollback=True, clean_recovery_compared_to_actual_Torch=True)


def recipes(reference):
    rows = reference['report']['rows']
    result = [('case-%03d' % row['index'], compare, (row,)) for row in rows]
    original = [row for row in rows if row['spec'].get('legacy_label') == 'unsupported-F64-view-111']
    require(len(original) == 1, 'one exact old single-request source')
    result.append(('packet-controls', packet_controls, (original[0],)))
    result.append(('allocation-control', allocation_control, (reference['report']['allocation_observation'],)))
    result.extend(('audit-fault-' + fault[0], audit_fault_control, (original[0], fault)) for fault in FAULTS)
    return result
