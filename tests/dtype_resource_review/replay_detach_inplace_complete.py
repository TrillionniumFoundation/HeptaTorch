#!/usr/bin/env python3
"""Independent native detach_ contract replay against the frozen actual 26 rows.

This is a test driver, not a Tensor/RNG numerical implementation. The 26 detach_
operations compare actual observations. Both later witness writes are replayed:
float32 add_ uses native command95 and bool logical_not_ uses native command102.
The original 26 observations and every old numerical expectation are unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import traceback

OBSERVATIONS_SHA = '121cdf20f84369b80c9e76c81a3eaa3d5cf86d79311f851b50b0ce968b28d573'
PROBE_SHA = '1b830c886dcad57436252384318962598d30e05a7f38f87a16166b0c9e3f5901'
PROTOCOL_SHA = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT_SHA = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
SOURCE_SHA = '7b8f597f6734722e38679163e2aea1058ae0d04c46c876bd682a19a319f76f26'
NATIVE_SHA = '165fe622d9ead60c9f94a01f7c5d09784cc096cb96c12dd008ce148e643ef96d'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches', 'direct_url_matches',
          'direct_url_archive_sha256_matches', 'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested'}
STATE_TAG, BACKING_TAG, DESCRIPTOR_BYTES = 'HTDTYP03', 'HTODTY03', 640


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def expected_plan():
    rows = []
    for dtype in ('float32', 'bool'):
        for construction in ('direct', 'clone', 'detach_alias', 'view', 'transpose'):
            for empty in (False, True):
                rows.append((f'{dtype}-{construction}-{"empty" if empty else "six"}',
                             'view-admission', dtype, construction, empty))
    for construction in ('requires_leaf_existing_grad', 'nonleaf_no_retain', 'nonleaf_retained_existing_grad',
                         'nonleaf_retained_no_grad', 'current_leaf_gradient', 'current_assigned_strided_gradient'):
        rows.append((construction.replace('_', '-'), 'autograd-identity', 'float32', construction, False))
    return rows


def check_reference(raw):
    assert sha(raw) == OBSERVATIONS_SHA
    value = json.loads(raw)
    assert value['schema'] == 'heptatorch.pinned-torch-detach-inplace-probe.v1'
    assert value['probe_sha256'] == PROBE_SHA
    assert value['status'] == 'OBSERVED' and value['torch_executed'] is True
    assert value['native_executed'] is False and value['native_values_compared'] == 0
    assert value['cases'] == value['expected_cases'] == len(value['rows']) == 26
    assert value['probe_errors'] == [] and value['probe_error_count'] == 0
    assert value['source_files_unchanged'] is True
    assert value['oracle_provenance']['status'] == 'PASS'
    checks = value['oracle_provenance']['checks']
    assert set(checks) == CHECKS and all(checks[name] is True for name in CHECKS)
    plans = expected_plan()
    for i, (row, plan) in enumerate(zip(value['rows'], plans, strict=True)):
        assert row['index'] == i and row['status'] == 'OBSERVED'
        assert tuple(row[k] for k in ('name', 'family', 'dtype', 'construction', 'empty')) == plan
        assert row['whole_case_rng_unchanged'] is True
        observation = row['observation']
        assert observation['rng_unchanged_by_detach'] is True
        for key in ('complete_state_before', 'complete_state_after'):
            state = value['states'][row[key]]
            payload = bytes.fromhex(state['hex'])
            assert state['bytes'] == len(payload) == 5056 and sha(payload) == row[key]
    return value


class Replay:
    def __init__(self, wire, args, directory, reference, row):
        self.wire, self.args, self.directory = wire, args, directory
        self.reference, self.row = reference, row
        self.session = wire.Session(args.binary.resolve(), directory=directory / 'transport')
        self.trace, self.events = [], []
        self.assertions = self.exact_words = self.rollback_bytes = self.rng_bytes = 0
        self.tensor_ids, self.storage_ids = {}, {}
        self.saved_owners = []
        self.phase = 'construction'

    def eq(self, got, expected, label):
        self.assertions += 1
        if got != expected:
            def diagnostic(value):
                return {'bytes': len(value), 'sha256': sha(value), 'first_64_hex': value[:64].hex()} if isinstance(value, bytes) else value
            raise AssertionError({'phase': self.phase, 'label': label, 'got': diagnostic(got), 'expected': diagnostic(expected)})

    def identity(self, table, actual, native, label):
        assert isinstance(actual, str), (label, 'reference native addresses are decimal strings')
        if actual in table:
            self.eq(native, table[actual], label + ' remains identical')
        else:
            self.eq(native not in table.values(), True, label + ' is distinct')
            table[actual] = native

    def one(self, *command):
        raw = self.wire.packet([command])
        status, records = self.wire.parse(self.session.exchange(raw))
        record = {'phase': self.phase, 'command': list(command), 'frame_status': status}
        self.trace.append(record)
        self.eq(status, 0, 'successful wire frame')
        self.eq(len(records), 1, 'one result')
        result = records[0]
        self.eq(result['opcode'], command[0], 'result opcode')
        record.update(value=result['value'], payload_hex=result['payload'].hex())
        return result

    def make(self, *command):
        return self.one(*command)['value']

    def create(self, dtype, shape, raw, requires=False):
        config = struct.pack('<8Q', len(shape), int(requires), *shape, *([0] * (4 - len(shape))), 160, len(raw))
        body = (b'HTORCH03' + struct.pack('<3Q', 1, 160 + len(raw), 0)
                + struct.pack('<8Q', 97, dtype, 96, 0, 0, 0, 0, 0) + config + raw)
        status, records = self.wire.parse(self.session.exchange(body))
        self.trace.append({'phase': self.phase, 'typed_create': {'dtype': dtype, 'shape': shape,
                          'requires_grad': bool(requires), 'raw_hex': raw.hex()}, 'frame_status': status,
                          'results': [{'value': r['value'], 'payload_hex': r['payload'].hex()} for r in records]})
        self.eq(status, 0, 'typed creation succeeds')
        self.eq(len(records), 1, 'typed creation one result')
        self.eq(records[0]['opcode'], 97, 'typed creation opcode')
        return records[0]['value']

    def base(self, dtype, shape, requires=False):
        # These are the literal input recipes in frozen probe 1b830c, not values
        # synthesized from a candidate's outputs or expected gradient model.
        count = 1
        for dim in shape:
            count *= dim
        values = [0.25, -0.5, 1.0, 2.0, -4.0, 8.0] if dtype == 0 else [0, 1, 1, 0, 1, 0]
        assert count in (0, 6)
        raw = struct.pack('<6f', *values) if dtype == 0 and count else bytes(values) if count else b''
        return self.create(dtype, shape, raw, requires)

    def back_sum(self, t, retain=False):
        total = self.make(4, 7, t)
        result = self.one(58, total, 0, int(retain))
        self.eq(result['value'], 0, 'construction backward succeeds')
        self.eq(struct.unpack('<4Q', result['payload'])[0], 0, 'construction backward no runtime error')
        self.one(41, total)

    def construct(self):
        row, holder = self.row, None
        form, dtype = row['construction'], int(row['dtype'] == 'bool')
        shape = [2, 0, 3] if row['empty'] else [2, 3]
        if row['family'] == 'view-admission':
            if form == 'transpose':
                source = self.base(dtype, [3, 0, 2] if row['empty'] else [3, 2])
                target = self.make(6, source, 0, 2 if row['empty'] else 1)
            else:
                source = self.base(dtype, shape)
                if form == 'direct':
                    target = source
                elif form == 'clone':
                    target = self.make(18, source)
                elif form == 'detach_alias':
                    target = self.make(19, source)
                elif form == 'view':
                    target = self.make(17, source, len(shape), *shape, *([0] * (4 - len(shape))))
                else:
                    raise AssertionError(form)
        else:
            source = self.base(0, [2, 3], True)
            if form == 'requires_leaf_existing_grad':
                self.back_sum(source)
                target = source
            elif form in ('nonleaf_no_retain', 'nonleaf_retained_existing_grad', 'nonleaf_retained_no_grad'):
                scalar = self.create(0, [], struct.pack('<f', 2.0))
                target = self.make(2, 3, source, scalar)
                self.one(41, scalar)
                if form != 'nonleaf_no_retain':
                    self.eq(self.make(93, target), 1, 'construction retain_grad succeeds')
                if form == 'nonleaf_retained_existing_grad':
                    self.back_sum(target, True)
            elif form == 'current_leaf_gradient':
                product = self.make(2, 3, source, source)
                self.back_sum(product)
                self.one(41, product)
                holder, target = source, self.make(91, source)
            elif form == 'current_assigned_strided_gradient':
                base = self.base(0, [3, 2])
                assigned = self.make(6, base, 0, 1)
                self.one(92, source, assigned)
                holder, target = source, self.make(91, source)
                self.eq(target, assigned, 'getter returns the assigned Tensor')
                # The reference retains this actual view's base via _base. Keep
                # the native construction base public as an explicit identity
                # witness, without claiming Python-wrapper lifetime parity.
            else:
                raise AssertionError(form)
        return target, source, holder

    def raw_node(self, tid):
        raw = self.session.owner()['bytes']
        def u(at):
            assert 0 <= at <= len(raw) - 8, ('owner read bounds', at, len(raw))
            return struct.unpack_from('<Q', raw, at)[0]
        self.eq(raw[64:72].decode('ascii'), STATE_TAG, 'state format')
        base = 139264
        self.eq(raw[base:base + 8].decode('ascii'), BACKING_TAG, 'backing format')
        slots = u(base + 16)
        self.eq((tid >> 24) & 255, 1, 'Tensor handle kind')
        slot = tid & 0xffffff
        assert 0 < slot <= slots
        row = base + 128 + (slot - 1) * 64
        self.eq(u(row), tid >> 32, 'Tensor generation')
        self.eq(u(row + 8), 1, 'Tensor row is live')
        node = u(row + 48)
        assert 0 < node < u(8)
        data = base + 144 + 552 * slots
        page = u(4096 + (node >> 8) * 8)
        self.eq(page >> 63, 1, 'descriptor page is pool-relative')
        at = data + (page & ((1 << 63) - 1)) + (node & 255) * DESCRIPTOR_BYTES
        self.eq(u(at + 304), 1, 'descriptor live')
        self.eq(u(at + 192), tid, 'actual public token')
        self.eq(u(at + 200), 1, 'public token ownership')
        return {'node': node, 'address_in_owner': at, 'retained_row_flag': u(row + 56),
                'fields': {str(field): u(at + field) for field in
                           (32, 40, 192, 200, 216, 224, 232, 304, 320, 360, 384, 392,
                            400, 408, 576, 584, 592, 600, 608, 624, 632)}}

    def tensor(self, tid, ref, label):
        assert ref is not None
        meta_record = self.one(13, tid)
        meta = self.wire.meta(meta_record)
        info_record = self.one(96, tid, 0)
        self.eq(len(info_record['payload']), 32, label + ' typed info bytes')
        dtype, width, incarnation, is_leaf = struct.unpack('<4Q', info_record['payload'])
        expected_dtype = {'torch.float32': 0, 'torch.bool': 1}[ref['dtype']]
        self.eq(dtype, expected_dtype, label + ' actual dtype')
        self.eq(width, ref['element_size'], label + ' element width')
        self.eq(info_record['value'], dtype, label + ' typed result')
        rank = len(ref['shape'])
        self.eq(meta['rank'], rank, label + ' rank')
        self.eq(meta['numel'], len(ref['binary_words']), label + ' element count')
        self.eq(meta['requires_grad'], int(ref['requires_grad']), label + ' requires_grad')
        self.eq(meta['shape'][:rank], ref['shape'], label + ' shape')
        self.eq(meta['strides'][:rank], ref['strides'], label + ' strides')
        self.eq(meta['storage_bytes'], ref['storage_bytes'], label + ' actual Storage extent')
        self.eq(meta['storage_offset'], ref['storage_offset_elements'] * width, label + ' byte offset')
        self.eq(meta['version'], ref['version'], label + ' exact Version counter')
        self.eq(bool(is_leaf), ref['is_leaf'], label + ' leaf role')
        flag = self.one(94, tid)
        self.eq(flag['payload'], b'', label + ' retain flag payload')
        self.eq(bool(flag['value']), ref['retains_grad'], label + ' retained flag')
        node = self.raw_node(tid)
        fields = node['fields']
        self.eq(fields['320'], incarnation, label + ' descriptor matches exposed Tensor incarnation')
        self.eq(fields['216'], meta['storage_id'], label + ' actual Storage identity')
        self.eq(fields['224'], meta['version_id'], label + ' actual Version identity')
        self.eq(fields['576'], dtype, label + ' actual descriptor dtype')
        self.eq(fields['40'] == 0, ref['grad_fn'] is None, label + ' history presence')
        self.eq(fields['408'] != 0, ref['is_view'], label + ' actual view discriminator')
        self.eq(node['retained_row_flag'] & 1, int(ref['retains_grad']), label + ' actual retained bit')
        self.identity(self.tensor_ids, ref['tensor_identity'], incarnation, label + ' Tensor')
        self.identity(self.storage_ids, ref['storage_identity'], meta['storage_id'], label + ' Storage')
        values = self.one(98, tid, 0, len(ref['binary_words']))
        self.eq(values['value'], len(ref['binary_words']), label + ' typed read count')
        self.eq(len(values['payload']), width * len(ref['binary_words']), label + ' typed read byte extent')
        got = list(values['payload']) if dtype == 1 else list(struct.unpack('<' + 'I' * len(ref['binary_words']), values['payload']))
        self.eq(got, ref['binary_words'], label + ' exact logical words')
        self.exact_words += len(got)
        return {'public_token': tid, 'incarnation': incarnation, 'metadata': meta, 'dtype': dtype,
                'raw_words': got, 'is_leaf': bool(is_leaf), 'retains_grad': bool(flag['value']), 'node': node,
                'reference_tensor_identity': ref['tensor_identity'], 'reference_storage_identity': ref['storage_identity'],
                'reference_python_identity_diagnostic_only': ref['python_identity'],
                'reference_base_identity_diagnostic_only': ref['base_tensor_identity'],
                'reference_grad_fn_class_diagnostic_only': ref['grad_fn'],
                'reference_output_nr_diagnostic_only': ref['output_nr']}

    def snapshot(self, tid, ref, label):
        if tid is None:
            self.eq(ref, None, label + ' absent holder')
            return None
        result = {'tensor': self.tensor(tid, ref['tensor'], label)}
        g = self.make(91, tid)
        self.eq(g == 0, ref['gradient'] is None, label + ' .grad None')
        self.eq(self.make(59, tid), int(ref['gradient'] is not None), label + ' grad presence')
        result['gradient_public_token'] = g
        result['gradient'] = None if g == 0 else self.tensor(g, ref['gradient'], label + '.grad')
        result['reference_gradient_warning_diagnostic_only'] = ref['gradient_read']
        return result

    def save_owner(self, name, owner):
        path = self.directory / (name + '.owner.bin')
        path.write_bytes(owner['bytes'])
        item = {'path': path.name, 'bytes': owner['length'], 'sha256': sha(owner['bytes']),
                'pointer': owner['pointer'], 'live': owner['live']}
        self.saved_owners.append(item)
        return item

    def refused(self, commands, name):
        before = self.session.owner()
        before_receipt = self.save_owner(name + '-before', before)
        status, records = self.wire.parse(self.session.exchange(self.wire.packet(commands)))
        after = self.session.owner()
        after_receipt = self.save_owner(name + '-after', after)
        event = {'name': name, 'commands': [list(c) for c in commands], 'frame_status': status,
                 'before': before_receipt, 'after': after_receipt}
        self.events.append(event)
        self.eq(status, 5, name + ' precise view refusal')
        self.eq(records, [], name + ' no failed-frame records')
        self.eq(after['pointer'], before['pointer'], name + ' committed owner pointer retained')
        self.eq(after['length'], before['length'], name + ' committed owner length retained')
        self.eq(after['bytes'], before['bytes'], name + ' entire owner byte equality')
        self.eq(after['live'], 1, name + ' exactly one live returned owner')
        self.rollback_bytes += before['length']
        return event

    def rng(self, generator, key, label):
        raw = self.one(63, generator)['payload']
        expected = bytes.fromhex(self.reference['states'][key]['hex'])
        self.eq(len(raw), 5056, label + ' full state byte count')
        self.eq(raw, expected, label + ' exact full MT/cache state')
        self.rng_bytes += len(raw)
        return {'bytes': len(raw), 'sha256': sha(raw)}

    def run(self):
        observation = self.row['observation']
        generator = self.make(60, 1729)
        initial_rng = self.rng(generator, self.row['complete_state_before'], 'initial Generator')
        target, source, holder = self.construct()
        # Match the strong identity witnesses of the actual recipe. Getter
        # publication exposes the existing Tensor; it does not synthesize values.
        held_gradient = self.make(91, target)
        held_holder_gradient = 0 if holder is None else self.make(91, holder)
        witness = self.make(19, target)
        self.phase = 'before_detach'
        before = self.snapshot(target, observation['before'], 'target')
        aliases_before = self.snapshot(source, observation['aliases_before']['construction_source'], 'source')
        holder_before = self.snapshot(holder, observation['holder_before'], 'holder')
        witness_before = self.tensor(witness, observation['version_sharing_observation']['witness_before_detach'], 'witness')
        rng_before = self.rng(generator, observation['rng_before_detach'], 'before detach Generator')
        self.eq(self.make(53), 1, 'audit before detach')
        self.phase = 'detach'
        owner_before = self.session.owner()
        self.save_owner('detach-before', owner_before)
        if observation['operation']['accepted']:
            result = self.one(101, target)
            self.eq(result['value'], target, 'detach_ returns same public Tensor token')
            self.eq(result['payload'], b'', 'detach_ has no synthetic data result')
            self.eq(observation['returned_is_target_object'], True, 'actual self result identity')
            self.identity(self.tensor_ids, observation['returned_tensor_identity'], before['tensor']['incarnation'], 'result Tensor')
        else:
            self.refused([(101, target)], 'view-refusal')
        owner_after = self.session.owner()
        self.save_owner('detach-after', owner_after)
        self.phase = 'after_detach'
        after = self.snapshot(target, observation['after_detach'], 'target')
        aliases_after = self.snapshot(source, observation['aliases_after_detach']['construction_source'], 'source')
        holder_after = self.snapshot(holder, observation['holder_after_detach'], 'holder')
        self.eq(after['tensor']['incarnation'], before['tensor']['incarnation'], 'detach exact Tensor identity')
        for key in ('storage_id', 'version_id', 'version'):
            self.eq(after['tensor']['metadata'][key], before['tensor']['metadata'][key], 'detach exact ' + key)
        self.eq(after['gradient_public_token'], held_gradient, 'existing held G stays the same public Tensor')
        if held_gradient:
            self.tensor(held_gradient, observation['held_gradient_after_detach'], 'held_gradient')
        else:
            self.eq(observation['held_gradient_after_detach'], None, 'held gradient remains None')
        if holder is not None:
            self.eq(holder_after['gradient_public_token'], held_holder_gradient, 'holder keeps original actual G Tensor')
        self.eq(after['tensor']['metadata']['version_id'], witness_before['metadata']['version_id'], 'prior witness shares real Version object')
        self.eq(after['tensor']['metadata']['storage_id'], witness_before['metadata']['storage_id'], 'prior witness shares real Storage')
        rng_after = self.rng(generator, observation['rng_after_detach'], 'after detach Generator')
        self.eq(self.make(53), 1, 'audit after detach')
        record = {'before': before, 'aliases_before': aliases_before, 'holder_before': holder_before,
                  'after_detach': after, 'aliases_after_detach': aliases_after, 'holder_after_detach': holder_after,
                  'witness_before': witness_before, 'initial_rng': initial_rng, 'rng_before': rng_before, 'rng_after': rng_after,
                  'detach_contract_status': 'PASS'}
        # Additional native transaction control. It is not counted as another
        # actual Torch case and never changes the frozen observation expectation.
        if not observation['operation']['accepted']:
            self.phase = 'native_prefix_rollback_control'
            prefix = (95, witness, self.wire.bits(0.25)) if self.row['dtype'] == 'float32' else (100, witness, 1, 0)
            suffix = (95, witness, self.wire.bits(1.0)) if self.row['dtype'] == 'float32' else (100, witness, 0, 0)
            self.refused([prefix, (101, target), suffix], 'view-refusal-with-mutating-prefix')
            record['native_prefix_rollback_control'] = 'PASS'
        sharing = observation['version_sharing_observation']
        if self.row['dtype'] == 'float32':
            self.phase = 'original_float32_post_witness_add'
            self.eq(sharing['operation_after_detach']['accepted'], True, 'actual original post-write succeeded')
            self.one(95, witness, self.wire.bits(0.25))
            post = {'witness': self.tensor(witness, sharing['witness_after_mutation'], 'post witness'),
                    'target': self.snapshot(target, sharing['target_after_witness_mutation'], 'post target'),
                    'source': self.snapshot(source, sharing['aliases_after_witness_mutation']['construction_source'], 'post source'),
                    'holder': self.snapshot(holder, sharing['holder_after_witness_mutation'], 'post holder')}
            self.eq(post['target']['tensor']['metadata']['version_id'], witness_before['metadata']['version_id'], 'original post-write same Version object')
            record['original_post_witness'] = {'status': 'PASS', 'operation': 'float32 add_(0.25)', 'observed': post}
        else:
            self.phase = 'original_bool_post_witness_logical_not'
            self.eq(sharing['operation_after_detach']['accepted'], True, 'actual original bool post-write succeeded')
            result = self.one(102, witness)
            self.eq(result['value'], witness, 'logical_not_ returns same Tensor token')
            self.eq(result['payload'], b'', 'logical_not_ has no synthetic data payload')
            post = {'witness': self.tensor(witness, sharing['witness_after_mutation'], 'post witness'),
                    'target': self.snapshot(target, sharing['target_after_witness_mutation'], 'post target'),
                    'source': self.snapshot(source, sharing['aliases_after_witness_mutation']['construction_source'], 'post source'),
                    'holder': self.snapshot(holder, sharing['holder_after_witness_mutation'], 'post holder')}
            self.eq(post['target']['tensor']['metadata']['version_id'], witness_before['metadata']['version_id'], 'original bool post-write same Version object')
            record['original_post_witness'] = {'status': 'PASS', 'operation': 'bool logical_not_()', 'observed': post}
        self.phase = 'final'
        record['final_rng'] = self.rng(generator, self.row['complete_state_after'], 'final Generator')
        self.eq(self.make(53), 1, 'final native invariant audit')
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'build-report', 'protocol', 'observations', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-source-sha256', default=SOURCE_SHA)
    parser.add_argument('--expected-native-sha256', default=NATIVE_SHA)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = (Path(__file__), args.binary, args.source, args.build_report, args.protocol,
             args.protocol.with_name('owner_client.py'), args.observations)
    inputs = {path.resolve(): path.read_bytes() for path in paths}
    report = {'schema': 'heptatorch.independent-detach-inplace-native-replay.v1', 'status': 'RUNNING',
              'native_executed': False, 'Torch_executed_here': False, 'case_count': 26, 'rows': [],
              'source_sha256': sha(inputs[args.source.resolve()]), 'native_sha256': sha(inputs[args.binary.resolve()]),
              'observations_sha256': sha(inputs[args.observations.resolve()]), 'probe_sha256': PROBE_SHA,
              'scope': {'detach_contract': '26 fixed actual cases, exact values/metadata/identity relations and view admission',
                        'float32_original_postwrite': '16 original add_(0.25) observations',
                        'bool_original_postwrite': '10 original logical_not_ observations use native102; no replacement operation',
                        'python_objects': 'Python object addresses, _base identity, grad_fn class names, warnings and output_nr diagnostic only',
                        'native_prefix_rollback': '9 added exact owner transaction controls, not new actual Torch cases',
                        'whole_reference_observation_fully_mapped': False},
              'whole_pytorch_gate': 'HOLD', 'pytorch_api_denominator': 2590,
              'full_pytorch_rewrite_complete': False, 'full_pytorch_parity': False}
    try:
        assert report['source_sha256'] == args.expected_source_sha256
        assert report['native_sha256'] == args.expected_native_sha256
        assert sha(inputs[args.protocol.resolve()]) == PROTOCOL_SHA
        assert sha(inputs[args.protocol.with_name('owner_client.py').resolve()]) == CLIENT_SHA
        reference = check_reference(inputs[args.observations.resolve()])
        report['oracle_provenance'] = reference['oracle_provenance']
        build = json.loads(inputs[args.build_report.resolve()])
        assert build['status'] == 'PASS' and build['link_exit'] == 0
        assert build['source_sha256'] == report['source_sha256'] and build['native_sha256'] == report['native_sha256']
        assert build['descriptor_bytes'] == DESCRIPTOR_BYTES
        assert build['state_format'] == STATE_TAG and build['backing_format'] == BACKING_TAG
        unit_rows = []
        for unit in build['units']:
            assert unit['compiler_exit'] == 0
            for extension, key in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
                path = (args.build_report.parent / (unit['name'] + extension)).resolve()
                payload = path.read_bytes()
                assert sha(payload) == unit[key]
                inputs[path] = payload
                unit_rows.append({'path': str(path), 'bytes': len(payload), 'sha256': sha(payload)})
        report['build_units_verified'] = unit_rows
        report['producer_runtime_qualified_field_is_not_reclassified'] = build['runtime_qualified']
        spec = importlib.util.spec_from_file_location('detach_native_wire', args.protocol.resolve())
        wire = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wire)
        (out / 'executed-test.py').write_bytes(inputs[Path(__file__).resolve()])
        for key, filename in ((args.source.resolve(), 'runtime.hepta'), (args.build_report.resolve(), 'build.json'),
                              (args.observations.resolve(), 'actual-detach-inplace.json'),
                              (args.protocol.resolve(), 'protocol.py'),
                              (args.protocol.with_name('owner_client.py').resolve(), 'owner_client.py')):
            (out / filename).write_bytes(inputs[key])
        report['native_executed'] = True
        for reference_row in reference['rows']:
            index = reference_row['index']
            directory = out / f'case-{index:02d}'
            directory.mkdir()
            replay = Replay(wire, args, directory, reference, reference_row)
            row = {'index': index, 'name': reference_row['name'], 'reference_json_pointer': f'/rows/{index}', 'status': 'RUNNING'}
            try:
                row['observation'] = replay.run()
                row['status'] = 'PASS'
            except BaseException as error:
                row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
            finally:
                row.update(frames_before_EOF=replay.session.frames, exact_assertions=replay.assertions,
                           exact_logical_words=replay.exact_words, rollback_bytes=replay.rollback_bytes,
                           complete_rng_bytes=replay.rng_bytes, events=replay.events, owner_snapshots=replay.saved_owners,
                           identity_maps={'Tensor': replay.tensor_ids, 'Storage': replay.storage_ids}, native_trace=replay.trace)
                try:
                    replay.session.finish()
                    row['transport_exit_code'] = replay.session.exit_code
                except BaseException as error:
                    row.update(status='FAIL', transport_error=repr(error))
                save_json(directory / 'report.json', row)
                report['rows'].append(row)
                print(json.dumps({'index': index, 'name': row['name'], 'status': row['status'],
                                  'frames': row['frames_before_EOF'], 'error': row.get('error')}), flush=True)
        report['passed_cases'] = sum(row['status'] == 'PASS' for row in report['rows'])
        report['failed_cases'] = sum(row['status'] == 'FAIL' for row in report['rows'])
        for field in ('frames_before_EOF', 'exact_assertions', 'exact_logical_words', 'rollback_bytes', 'complete_rng_bytes'):
            report[field] = sum(row[field] for row in report['rows'])
        report['original_post_witness_passed'] = sum(row.get('observation', {}).get('original_post_witness', {}).get('status') == 'PASS' for row in report['rows'])
        report['original_post_witness_unmapped'] = sum(row.get('observation', {}).get('original_post_witness', {}).get('status') == 'UNMAPPED' for row in report['rows'])
        report['native_prefix_rollback_passed'] = sum(row.get('observation', {}).get('native_prefix_rollback_control') == 'PASS' for row in report['rows'])
        report['status'] = 'PASS' if (report['passed_cases'] == 26 and report['original_post_witness_passed'] == 26
                                    and report['original_post_witness_unmapped'] == 0
                                    and report['native_prefix_rollback_passed'] == 9) else 'FAIL'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    finally:
        report['inputs'] = [{'path': str(path), 'bytes': len(raw), 'sha256': sha(raw),
                             'unchanged_after': path.read_bytes() == raw} for path, raw in inputs.items()]
        report['source_files_unchanged'] = all(item['unchanged_after'] for item in report['inputs'])
        if not report['source_files_unchanged']:
            report['status'] = 'FAIL'
        save_json(out / 'report.json', report)
    print(json.dumps({key: report.get(key) for key in ('status', 'passed_cases', 'failed_cases', 'frames_before_EOF',
                     'exact_assertions', 'exact_logical_words', 'original_post_witness_passed',
                     'original_post_witness_unmapped', 'native_prefix_rollback_passed')}), flush=True)
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    sys.exit(main())
