#!/usr/bin/env python3
"""Strict independent replay of the twelve frozen real-Torch retain_grad variants.

The only imported helper serializes the public wire and observes this script's
own child process. No production authoring, numerical implementation, or
native-generated expected result is imported. All compared tensor values,
layouts, identities, Versions, None states and exception effects come from the
unchanged ca8b5d3a reference. Python warning/wrapper/class spelling is recorded
as diagnostic data rather than claimed as a native API.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import struct
import traceback

ACTUAL_SHA = 'ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3'
PROBE_SHA = '7cdb53f7a95c48acd1b85d93d90c693d4caa97ada72a54a4021bb616d098fcba'
PROTOCOL_SHA = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
TRANSPORT_SHA = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
CHECKS = ('python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested')
PLAN = (
    ('admission-leaf-noop-and-idempotence', 'default'),
    ('retained-gradient-replacement-clear-and-old-saved-alias', 'default'),
    ('transpose-incoming-and-assigned-gradient-layout', 'default'),
    ('empty-retained-gradient-replacement', 'default'),
    ('retained-nonleaf-owned-optimizer-state', 'SGD'),
    ('retained-nonleaf-owned-optimizer-state', 'Adam'),
    ('retained-nonleaf-owned-optimizer-state', 'AdamW'),
    ('retained-hook-before-own-saved-version-failure', 'default'),
    ('retained-hook-before-own-released-saved-failure', 'default'),
    ('weak-retention-versus-optimizer-strong-root', 'never-materialized'),
    ('weak-retention-versus-optimizer-strong-root', 'materialized-and-held'),
    ('weak-retention-versus-optimizer-strong-root', 'optimizer-is-the-only-strong-parameter-root'),
)
# This is the existing frozen Adam-object tolerance, not a new relaxation.
ADAM_RTOL, ADAM_ATOL = 4e-5, 5e-6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_wire(path):
    assert digest(path) == PROTOCOL_SHA
    observer = path.with_name('owner_client.py')
    assert digest(observer) == TRANSPORT_SHA
    spec = importlib.util.spec_from_file_location('retained_nonleaf_public_wire', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def words(record):
    raw = record['payload']
    assert len(raw) % 8 == 0
    return list(struct.unpack('<' + 'Q' * (len(raw) // 8), raw))


def number(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def actual_variants(reference):
    assert reference['status'] == 'OBSERVED' and reference['torch_executed'] is True
    assert reference['probe_sha256'] == PROBE_SHA
    assert reference['cases'] == reference['expected_cases'] == len(reference['rows']) == 8
    assert reference['expected_variants'] == 12 and reference['probe_errors'] == 0
    assert reference['source_files_unchanged'] is True
    assert reference['native_executed'] is False and reference['native_values_compared'] == 0
    checks = reference['oracle_provenance']['checks']
    assert set(checks) == set(CHECKS) and all(checks[k] is True for k in CHECKS)
    result = []
    for i, row in enumerate(reference['rows']):
        assert row['status'] == 'OBSERVED'
        variants = row['observation'].get('variants', [row['observation']])
        assert len(variants) == row['planned_variants']
        for j, variant in enumerate(variants):
            pointer = f'/rows/{i}/observation'
            if 'variants' in row['observation']:
                pointer += f'/variants/{j}'
            result.append((row['case'], variant.get('variant', 'default'), variant, pointer))
    assert [(c, v) for c, v, _, _ in result] == list(PLAN)
    return result


class Replay:
    def __init__(self, wire, args, directory):
        self.wire, self.args = wire, args
        self.session = wire.Session(args.binary, directory=directory / 'transport')
        self.assertions = 0
        self.exact_words = 0
        self.adam_words = 0
        self.rollback_bytes = 0
        self.tensor_ids, self.storage_ids = {}, {}
        self.trace, self.bit_differences = [], []
        self.native_calls = []
        self.retired_tensor_identities = []
        self.current_label = 'setup'

    def eq(self, actual, expected, label):
        self.assertions += 1
        assert actual == expected, (self.current_label, label, actual, expected)

    def identity(self, table, actual_identity, native_identity, label):
        actual_identity = str(actual_identity)
        if actual_identity in table:
            self.eq(native_identity, table[actual_identity], label + ' stable identity')
        else:
            self.eq(native_identity not in table.values(), True, label + ' distinct live identity')
            table[actual_identity] = native_identity

    def retire_tensor(self, reference):
        # Only used where the fixed recipe actually drops the last Tensor root.
        # Torch addresses may be reused after death; native incarnations cannot.
        key = reference['tensor_identity']
        old = self.tensor_ids.pop(key, None)
        self.retired_tensor_identities.append({'reference_identity': key, 'native_incarnation': old})

    def one(self, *command, status=0):
        before = self.session.owner() if status else None
        got_status, records = self.wire.parse(self.session.exchange(self.wire.packet([command])))
        self.eq(got_status, status, 'frame status')
        item = {'command': [int(v) if not isinstance(v, self.wire.Payload) else
                            {'payload_bytes': len(v.data), 'sha256': hashlib.sha256(v.data).hexdigest()}
                            for v in command], 'status': got_status}
        self.native_calls.append(item)
        if status:
            self.eq(records, [], 'failed frame has no records')
            after = self.session.owner()
            self.eq(after, before, 'full committed owner unchanged after admission failure')
            self.rollback_bytes += before['length']
            item.update(owner_bytes_checked=before['length'], owner_sha256=hashlib.sha256(before['bytes']).hexdigest())
            return {'value': None, 'payload': b'', 'opcode': command[0], 'frame_error': status}
        self.eq(len(records), 1, 'one command produces one record')
        rec = records[0]
        self.eq(rec['opcode'], command[0], 'result opcode')
        item.update(value=rec['value'], payload_hex=rec['payload'].hex())
        return rec

    def make(self, *command):
        return self.one(*command)['value']

    def blob(self, values):
        return self.wire.Payload(struct.pack('<' + 'Q' * len(values), *values))

    def new(self, shape, values, requires=False):
        return self.make(*self.wire.tensor(shape, values, int(requires)))

    def full(self, shape, value=0., requires=False):
        return self.make(*self.wire.full(shape, value, int(requires)))

    def close_tensor(self, tensor):
        self.one(41, tensor)

    def scalar_binary(self, tensor, scalar, operation):
        other = self.new([], [scalar])
        result = self.make(2, operation, tensor, other)
        self.close_tensor(other)
        return result

    def transpose_temporary(self, shape, values, axis0, axis1):
        base = self.new(shape, values)
        value = self.make(6, base, axis0, axis1)
        self.close_tensor(base)
        return value

    def retain(self, tensor, status=0):
        rec = self.one(93, tensor, status=status)
        if not status:
            self.eq(rec['value'], 1, 'retain command success')
            self.eq(rec['payload'], b'', 'retain has no payload')
        return rec

    def back(self, tensor, seed=0, retain=0, runtime=0):
        rec = self.one(58, tensor, seed, retain)
        self.eq(rec['value'], runtime, 'structured backward error')
        result = words(rec)
        self.eq(len(result), 4, 'structured backward payload width')
        self.eq(result[0], runtime, 'structured error repeated in payload')
        # This probe did not install hooks, so no Torch expectation is invented
        # for the remaining native completed-node/leaf/operation diagnostics.
        return rec

    def raw_node(self, tid=None, node=None):
        raw = self.session.owner()['bytes']
        def u(offset):
            assert 0 <= offset <= len(raw) - 8
            return struct.unpack_from('<Q', raw, offset)[0]
        self.eq(raw[64:72], self.args.state_tag.encode(), 'state format')
        base = 139264
        self.eq(raw[base:base + 8], self.args.backing_tag.encode(), 'backing format')
        slots = u(base + 16)
        data = base + 144 + 552 * slots
        if tid is not None:
            self.eq((tid >> 24) & 255, 1, 'public Tensor kind')
            slot = tid & 0xffffff
            assert 0 < slot <= slots
            row = base + 128 + (slot - 1) * 64
            self.eq(u(row), tid >> 32, 'public generation')
            self.eq(u(row + 8), 1, 'public Tensor row live')
            node = u(row + 48)
        assert node is not None and 0 < node < u(8)
        page = u(4096 + (node >> 8) * 8)
        assert page >> 63 == 1
        at = data + (page & ((1 << 63) - 1)) + (node & 255) * 640
        self.eq(u(at + 304), 1, 'node metadata live')
        if tid is not None:
            self.eq(u(at + 192), tid, 'node current public token')
            self.eq(u(at + 200), 1, 'node public flag')
        return {k: u(at + k) for k in (32, 40, 192, 200, 240, 304, 312, 320, 360, 384, 392, 400)}

    def tensor(self, tensor, reference, label, adam_numeric=False):
        meta = words(self.one(13, tensor))
        rank = len(reference['shape'])
        expected = reference['binary32_words']
        self.eq(len(meta), 16, label + ' metadata width')
        self.eq(meta[0], rank, label + ' rank')
        self.eq(meta[1], len(expected), label + ' numel')
        self.eq(meta[2], int(reference['requires_grad']), label + ' requires flag')
        self.eq(meta[3:3 + rank], reference['shape'], label + ' shape')
        self.eq(meta[7:7 + rank], reference['strides'], label + ' strides')
        self.eq(meta[13], reference['version'], label + ' exact Version')
        self.eq(meta[14], reference['storage_offset_bytes'], label + ' storage offset')
        self.eq(meta[15], reference['storage_bytes'], label + ' storage size')
        self.eq(reference['dtype'], 'torch.float32', label + ' admitted dtype')
        node = self.raw_node(tid=tensor)
        self.eq(node[40] == 0, reference['is_leaf'], label + ' leaf role')
        self.eq(node[40] == 0, reference['grad_fn_type'] is None, label + ' grad function present')
        flag = self.one(94, tensor)
        self.eq(flag['value'], int(reference['retains_grad']), label + ' retains flag')
        self.eq(flag['payload'], b'', label + ' retains query payload')
        self.identity(self.tensor_ids, reference['tensor_identity'], node[320], label + ' Tensor')
        self.identity(self.storage_ids, reference['storage_identity'], meta[11], label + ' Storage')
        rec = self.one(10, tensor)
        self.eq(rec['value'], len(expected), label + ' value count')
        self.eq(len(rec['payload']), 4 * len(expected), label + ' value bytes')
        got = list(struct.unpack('<' + 'I' * len(expected), rec['payload']))
        if adam_numeric:
            for i, (observed, wanted) in enumerate(zip(got, expected, strict=True)):
                self.adam_words += 1
                if observed != wanted:
                    a, b = number(observed), number(wanted)
                    difference = {'label': label, 'index': i, 'native_word': observed,
                                  'actual_word': wanted, 'rtol': ADAM_RTOL, 'atol': ADAM_ATOL,
                                  'within_original_tolerance': math.isfinite(a) and math.isfinite(b)
                                      and abs(a - b) <= ADAM_ATOL + ADAM_RTOL * abs(b)}
                    self.bit_differences.append(difference)
                    self.eq(difference['within_original_tolerance'], True, label + ' original Adam tolerance')
                else:
                    self.assertions += 1
        else:
            self.eq(got, expected, label + ' exact binary32 words')
            self.exact_words += len(expected)
        return {'public_token': tensor, 'incarnation': node[320], 'metadata': meta,
                'binary32_words': got, 'is_leaf': node[40] == 0,
                'retains_grad': bool(flag['value']),
                'actual_tensor_identity': reference['tensor_identity'],
                'actual_storage_identity': reference['storage_identity'],
                'actual_grad_fn_type_diagnostic_only': reference['grad_fn_type']}

    def gradient(self, tensor, reference, label):
        node = self.raw_node(tid=tensor)
        already_public = bool(node[384] and self.raw_node(node=node[384])[200])
        gradient = self.make(91, tensor)
        self.eq(gradient == 0, reference is None, label + ' None')
        self.eq(self.make(59, tensor), int(reference is not None), label + ' presence')
        result = None
        if reference is not None:
            result = self.tensor(gradient, reference, label)
            if not already_public:
                self.close_tensor(gradient)
        return result

    def opt(self, parameter, kind='SGD', status=0):
        b = self.wire.bits
        algorithm = {'SGD': 0, 'Adam': 1, 'AdamW': 2}[kind]
        config = [b(.125), b(.5), 0, b(.25), 0, 0] if kind == 'SGD' else [
            b(.125), b(.5), b(.75), b(.01), b(.25), 0, 0, 0]
        groups = [1, 1, 0, *([0] * len(config)), parameter]
        rec = self.one(71, algorithm, self.blob(config), self.blob(groups), status=status)
        if status:
            return rec
        self.eq(words(rec), [1, 1, 1, 0], 'Optimizer creation inventory')
        return rec['value']

    def optimizer_snapshot(self, opt, reference):
        kind = reference['optimizer_type']
        self.eq(len(reference['optimizer_groups']), 1, 'actual one group recipe')
        self.eq(len(reference['optimizer_state']), 1, 'actual one parameter recipe')
        group, state = reference['optimizer_groups'][0], reference['optimizer_state'][0]
        rec = self.one(74, opt, 0)
        self.eq(rec['value'], 1, 'native group occurrences')
        group_words = words(rec)
        width = 8 if kind == 'SGD' else 10
        self.eq(len(group_words), width + 2, 'group metadata width')
        self.eq(group_words[:2], [0, 1], 'group ordinal/count')
        c, b = group['config'], self.wire.bits
        if kind == 'SGD':
            cfg = [b(c['lr']), b(c['momentum']), b(c['dampening']), b(c['weight_decay']),
                   int(c['nesterov']), int(c['maximize'])]
        else:
            cfg = [b(c['lr']), b(c['betas'][0]), b(c['betas'][1]), b(c['eps']),
                   b(c['weight_decay']), int(c['amsgrad']), int(c['maximize']), 0]
            self.eq(c['capturable'], False, 'admitted noncapturable profile')
            self.eq(c['decoupled_weight_decay'], kind == 'AdamW', 'actual Adam mode')
        for flag in ('foreach', 'differentiable', 'fused'):
            self.eq(c[flag], False, 'admitted ' + flag + ' mode')
        self.eq(group_words[2:width], cfg, 'native merged config')
        self.eq(len(group['parameter_identities']), 1, 'actual membership width')
        self.identity(self.tensor_ids, group['parameter_identities'][0], group_words[width], 'Optimizer member')
        self.eq(state['parameter_identity'], group['parameter_identities'][0], 'actual state key identity')
        rec = self.one(76, opt, 0, 0)
        inventory = words(rec)
        self.eq(len(inventory), 4 if kind == 'SGD' else 8, 'state inventory width')
        self.eq(inventory[0], group_words[width], 'state opaque parameter key')
        self.eq(rec['value'], int(state['entry_exists']), 'state entry presence')
        self.eq(reference['optimizer_state_entries'], int(state['entry_exists']), 'state dictionary size')
        fields = [('momentum_buffer', 0, 2)] if kind == 'SGD' else [
            ('step', 1, 2), ('exp_avg', 2, 4), ('exp_avg_sq', 3, 6)]
        expected_names = {name for name, _, _ in fields} if state['entry_exists'] else set()
        self.eq(set(state['fields']), expected_names, 'actual state field set')
        self.eq(inventory[1], (1 if kind == 'SGD' else 14) if state['entry_exists'] else 0, 'state field mask')
        result = {'kind': kind, 'group_words': group_words, 'inventory': inventory, 'fields': {}}
        for name, selector, index in fields:
            if not state['entry_exists']:
                self.eq(inventory[index:index + 2], [0, 0], 'missing state identity/token')
                self.eq(self.make(77, opt, 0, 0, selector), 0, 'missing state getter does not create')
                continue
            already_public = inventory[index + 1] != 0
            token = self.make(77, opt, 0, 0, selector)
            if already_public:
                self.eq(token, inventory[index + 1], 'state current public token')
            observed = self.tensor(token, state['fields'][name], 'state.' + name,
                                   adam_numeric=kind != 'SGD' and name != 'step')
            self.eq(observed['incarnation'], inventory[index], 'state stable incarnation')
            result['fields'][name] = observed
            if not already_public:
                self.close_tensor(token)
        return result

    def snapshot(self, reference, named, opt=None):
        self.eq(set(named), set(reference['tensors']), 'snapshot Tensor names')
        self.eq(set(named), set(reference['gradients']), 'snapshot gradient names')
        self.eq(set(named), set(reference['gradient_read_warnings']), 'snapshot warning labels')
        result = {'tensors': {}, 'gradients': {},
                  'actual_gradient_read_warnings_diagnostic_only': reference['gradient_read_warnings']}
        tolerant_ids = set()
        if opt is not None and reference['optimizer_type'] in ('Adam', 'AdamW'):
            for group in reference['optimizer_groups']:
                tolerant_ids.update(group['parameter_identities'])
        for name, tensor in named.items():
            result['tensors'][name] = self.tensor(tensor, reference['tensors'][name], name,
                adam_numeric=reference['tensors'][name]['tensor_identity'] in tolerant_ids)
            result['gradients'][name] = self.gradient(tensor, reference['gradients'][name], name + '.grad')
        if opt is not None:
            result['optimizer'] = self.optimizer_snapshot(opt, reference)
        else:
            self.eq('optimizer_type' in reference, False, 'no hidden actual Optimizer')
        return result

    def event(self, reference, action, named, opt=None, status=0, runtime=0):
        self.current_label = reference['action']
        entry = {'action': reference['action'], 'actual_outcome': reference['outcome'],
                 'expected_native_header_status': status, 'expected_native_runtime_status': runtime,
                 'actual_warning_objects_diagnostic_only': reference.get('warnings', [])}
        self.trace.append(entry)
        entry['before'] = self.snapshot(reference['before'], named, opt)
        self.eq(reference['outcome'], 'RAISED' if status or runtime else 'SUCCEEDED', 'actual outcome')
        if runtime:
            self.eq(reference['exception_type'], 'RuntimeError', 'actual backward exception class')
            if runtime == 6:
                self.eq('modified by an inplace operation' in reference['message'], True, 'actual saved-version exception')
            elif runtime == 10:
                self.eq('second time' in reference['message'], True, 'actual released-Saved exception')
        action()
        entry['after'] = self.snapshot(reference['after'], named, opt)

    def plain_outcome(self, reference, action, status=0):
        self.current_label = reference['action']
        self.eq(reference['outcome'], 'RAISED' if status else 'SUCCEEDED', 'actual plain outcome')
        result = action()
        self.trace.append({'action': reference['action'], 'actual_outcome': reference['outcome'],
                           'expected_native_header_status': status,
                           'actual_warnings_diagnostic_only': reference.get('warnings', [])})
        return result

    def weak(self, tensor):
        return self.make(42, tensor)

    def expired(self, weak, expected, label):
        rec = self.one(43, weak, status=6 if expected else 0)
        if not expected:
            self.one(45, rec['value'])
        self.trace.append({'weak_check': label, 'expired': bool(expected)})

    def weak_snapshot(self, reference, weak_handles):
        expected_keys = {key for key in reference if key.endswith('_storage_expired')}
        self.eq(set(weak_handles), expected_keys, 'weak Storage observation names')
        for name, handle in weak_handles.items():
            self.expired(handle, reference[name], name)
        self.trace.append({'python_wrapper_weak_observations_diagnostic_only': {
            k: v for k, v in reference.items() if k.startswith('python_')}})

    def case_admission(self, r):
        x = self.new([2], [1., -2.], True)
        q = self.scalar_binary(x, 2., 3)
        inert = self.new([1], [3.])
        named = {'leaf': x, 'nonleaf': q, 'inert': inert}
        events = r['events']
        self.eq(len(events), 6, 'admission event count')
        self.event(events[0], lambda: self.retain(x), named)
        self.event(events[1], lambda: self.retain(inert, status=5), named, status=5)
        self.eq(events[2]['exception_type'], 'ValueError', 'actual nonleaf optimizer refusal')
        self.plain_outcome(events[2], lambda: self.opt(q, status=5), status=5)
        self.event(events[3], lambda: self.retain(q), named)
        self.event(events[4], lambda: self.retain(q), named)
        opt = self.plain_outcome(events[5], lambda: self.opt(q))
        self.trace.append({'final': self.snapshot(r['after'], named, opt)})

    def case_replacement(self, r):
        x = self.new([2], [1., -2.], True)
        q = self.scalar_binary(x, 2., 3)
        self.retain(q)
        seeds = [self.new([2], v) for v in ([-0., 2.], [1., -2.], [-0., -0.])]
        named = {'x': x, 'q': q, 'seed1': seeds[0], 'seed2': seeds[1], 'seed3': seeds[2]}
        events = r['events']
        self.eq(len(events), 5, 'replacement event count')
        self.event(events[0], lambda: self.back(q, seeds[0], 1), named)
        old1 = self.make(91, q)
        z = self.new([2], [1., 1.], True)
        product = self.make(2, 3, old1, z)
        guard = self.make(4, 7, product)
        self.close_tensor(product)
        named.update(held_gradient1=old1, saved_guard_parameter=z)
        self.event(events[1], lambda: self.back(q, seeds[1], 1), named)
        old2 = self.make(91, q)
        named['held_gradient2'] = old2
        self.event(events[2], lambda: self.back(guard), named)
        self.event(events[3], lambda: self.one(92, q, 0), named)
        self.event(events[4], lambda: self.back(q, seeds[2]), named)
        self.trace.append({'final': self.snapshot(r['after'], named)})

    def case_strided(self, r):
        x = self.new([2, 3], [1., 2., 3., 4., 5., 6.], True)
        q = self.make(6, x, 0, 1)
        self.retain(q)
        seed = self.transpose_temporary([2, 3], [1., 3., 5., 2., 4., 6.], 0, 1)
        named = {'x': x, 'q': q, 'held_strided_seed': seed}
        events = r['events']
        self.eq(len(events), 3, 'strided event count')
        self.event(events[0], lambda: self.back(q, seed, 1), named)
        old = self.make(91, q)
        assigned = self.transpose_temporary([2, 3], [.25, .75, 1.25, .5, 1., 1.5], 0, 1)
        named.update(held_first_gradient=old, assigned_gradient=assigned)
        self.event(events[1], lambda: self.one(92, q, assigned), named)
        self.event(events[2], lambda: self.back(q, seed), named)
        self.trace.append({'final': self.snapshot(r['after'], named)})

    def case_empty(self, r):
        x = self.full([2, 0, 3], requires=True)
        q = self.make(6, x, 0, 2)
        self.retain(q)
        base = self.full([3, 0, 2])
        seed = self.make(69, base, self.blob([3, 3, 0, 2, 0, 100, 17, 1, 0, 0]), 80)
        self.close_tensor(base)
        named = {'x': x, 'q': q, 'held_empty_seed': seed}
        events = r['events']
        self.eq(len(events), 2, 'empty event count')
        self.event(events[0], lambda: self.back(q, seed, 1), named)
        named['held_empty_gradient'] = self.make(91, q)
        self.event(events[1], lambda: self.back(q, seed), named)
        self.trace.append({'final': self.snapshot(r['after'], named)})

    def case_optimizer(self, r):
        kind = r['variant']
        x = self.new([2], [1., -2.], True)
        q = self.scalar_binary(x, 2., 3)
        self.retain(q)
        opt = self.opt(q, kind)
        seed1, seed2 = self.new([2], [2., -1.]), self.new([2], [-.5, .25])
        named = {'x': x, 'q': q, 'seed1': seed1, 'seed2': seed2}
        events = r['events']
        self.eq(len(events), 9, 'optimizer event count')
        self.event(events[0], lambda: self.one(72, opt), named, opt)
        self.event(events[1], lambda: self.back(q, seed1, 1), named, opt)
        self.event(events[2], lambda: self.one(72, opt), named, opt)
        named['held_gradient'] = self.make(91, q)
        self.event(events[3], lambda: self.one(73, opt, 0), named, opt)
        self.event(events[4], lambda: self.one(72, opt), named, opt)
        self.event(events[5], lambda: self.one(73, opt, 1), named, opt)
        self.event(events[6], lambda: self.one(72, opt), named, opt)
        self.event(events[7], lambda: self.back(q, seed2), named, opt)
        self.event(events[8], lambda: self.one(72, opt), named, opt)
        self.trace.append({'final': self.snapshot(r['after'], named, opt)})

    def case_version(self, r):
        x, coefficient = self.new([2], [1., -2.], True), self.new([], [2.])
        q = self.make(2, 3, x, coefficient)
        self.retain(q)
        seed = self.new([2], [1., 1.])
        named = {'x': x, 'coefficient': coefficient, 'q': q, 'seed': seed}
        self.one(95, coefficient, self.wire.bits(1.))
        events = r['events']
        self.eq(len(events), 2, 'saved-version event count')
        self.event(events[0], lambda: self.back(q, seed, runtime=6), named, runtime=6)
        named['held_gradient_after_failure'] = self.make(91, q)
        self.event(events[1], lambda: self.back(q, seed, runtime=6), named, runtime=6)
        self.trace.append({'final': self.snapshot(r['after'], named)})

    def case_released(self, r):
        x = self.new([2], [1., -2.], True)
        q = self.make(2, 3, x, x)
        self.retain(q)
        seed = self.new([2], [2., -.5])
        named = {'x': x, 'q': q, 'seed': seed}
        events = r['events']
        self.eq(len(events), 2, 'released-Saved event count')
        self.event(events[0], lambda: self.back(q, seed), named)
        named['held_gradient_before_retry'] = self.make(91, q)
        self.event(events[1], lambda: self.back(q, seed, runtime=10), named, runtime=10)
        self.trace.append({'final': self.snapshot(r['after'], named)})

    def case_weak(self, r):
        materialize = r['variant'] == 'materialized-and-held'
        x = self.new([2], [1., -2.], True)
        q = self.scalar_binary(x, 2., 3)
        self.retain(q)
        seed = self.new([2], [2., -1.])
        held = None
        if materialize:
            self.back(q, seed, 1)
            held = self.make(91, q)
        child = self.scalar_binary(q, 1., 1)
        weak_q = self.weak(q)
        weak_g = self.weak(held) if held is not None else None
        self.trace.append({'before_drop': self.snapshot(r['before_drop'],
            {'x': x, 'q': q, 'child': child, 'seed': seed})})
        self.close_tensor(q)
        self.retire_tensor(r['before_drop']['tensors']['q'])
        self.expired(weak_q, r['after_drop']['q_storage_expired'], 'q after public drop')
        if weak_g is not None:
            self.expired(weak_g, r['after_drop']['held_gradient_storage_expired'], 'held old G after q drop')
        else:
            self.eq(r['after_drop']['held_gradient_storage_expired'], None, 'no held G in this recipe')
        self.trace.append({'actual_python_wrapper_alive_diagnostic_only': r['after_drop']['python_q_wrapper_alive']})
        self.plain_outcome(r['backward'], lambda: self.back(child, seed))
        named = {'x': x, 'child': child, 'seed': seed}
        if held is not None:
            named['held_old_gradient'] = held
        self.trace.append({'after_backward': self.snapshot(r['after_backward'], named)})
        self.expired(weak_q, r['q_storage_expired_after_backward'], 'q after child backward')
        self.one(44, weak_q)
        if weak_g is not None:
            self.one(44, weak_g)

    def case_private(self, r):
        x = self.new([2], [1., -2.], True)
        q = self.scalar_binary(x, 2., 3)
        self.retain(q)
        temporary_seed = self.new([2], [2., -1.])
        self.back(q, temporary_seed, 1)
        self.close_tensor(temporary_seed)
        held = self.make(91, q)
        opt = self.opt(q)
        xg = self.make(91, x)
        weak = {'x_storage_expired': self.weak(x), 'q_storage_expired': self.weak(q),
                'x_gradient_storage_expired': self.weak(xg), 'held_gradient_storage_expired': self.weak(held)}
        self.close_tensor(xg)
        self.trace.append({'before_drop': self.snapshot(r['before_drop'],
            {'x': x, 'q': q, 'held_gradient': held}, opt)})
        self.close_tensor(x)
        self.close_tensor(q)
        self.weak_snapshot(r['after_public_drop'], weak)
        self.plain_outcome(r['private_step'], lambda: self.one(72, opt))
        private = self.make(75, opt, 0, 0)
        self.trace.append({'after_private_step': self.snapshot(r['after_private_step'],
            {'private_parameter': private, 'held_gradient': held}, opt)})
        self.close_tensor(private)
        self.one(78, opt)
        self.weak_snapshot(r['after_optimizer_drop'], weak)
        self.trace.append({'held_gradient_after_optimizer_drop': self.tensor(
            held, r['held_gradient_after_optimizer_drop'], 'held old G after optimizer drop')})
        self.close_tensor(held)
        self.weak_snapshot(r['after_final_gradient_drop'], weak)
        for handle in weak.values():
            self.one(44, handle)

    def execute(self, index, observation):
        methods = (self.case_admission, self.case_replacement, self.case_strided, self.case_empty,
                   self.case_optimizer, self.case_optimizer, self.case_optimizer,
                   self.case_version, self.case_released, self.case_weak, self.case_weak, self.case_private)
        methods[index](observation)
        self.current_label = 'final native audit'
        self.eq(self.make(53), 1, 'complete native ownership audit')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--binary', type=Path)
    p.add_argument('--source', type=Path)
    p.add_argument('--expected-sha256')
    p.add_argument('--expected-source-sha256')
    p.add_argument('--state-tag', default='HTRET001')
    p.add_argument('--backing-tag', default='HTORET01')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    for name in ('observations', 'probe', 'protocol', 'output', 'binary', 'source'):
        value = getattr(a, name)
        if value is not None:
            setattr(a, name, value.resolve())
    before = {'runner': digest(__file__), 'observations': digest(a.observations),
              'probe': digest(a.probe), 'protocol': digest(a.protocol),
              'transport': digest(a.protocol.with_name('owner_client.py'))}
    assert before['observations'] == ACTUAL_SHA and before['probe'] == PROBE_SHA
    assert before['protocol'] == PROTOCOL_SHA and before['transport'] == TRANSPORT_SHA
    reference = json.loads(a.observations.read_bytes())
    variants = actual_variants(reference)
    assert len(a.state_tag.encode()) == len(a.backing_tag.encode()) == 8
    a.output.mkdir(parents=True, exist_ok=False)
    report = {
        'schema': 'heptatorch.retained-nonleaf-independent-native-replay.v1',
        'status': 'PREPARED_NOT_RUN' if a.prepare_only else 'RUNNING',
        'inputs': before, 'expected_families': 8, 'expected_variants': 12,
        'reference_provenance_checks': reference['oracle_provenance']['checks'],
        'reference_source_files_unchanged': reference['source_files_unchanged'],
        'native_executed': False, 'torch_executed_by_this_runner': False,
        'producer_code_imported': False, 'runtime_numeric_host_fallback': False,
        'state_tag': a.state_tag, 'backing_tag': a.backing_tag, 'descriptor_bytes': 640,
        'exact': ['None', 'logical binary32 gradients', 'SGD values', 'step Tensor values',
                  'shape', 'stride', 'offset', 'Storage size', 'Version', 'requires_grad',
                  'leaf role', 'retains_grad', 'live opaque identity equality relations',
                  'optimizer state membership and fields', 'Storage weak expiry', 'runtime exception category'],
        'adam_parameter_moment_tolerance': {'rtol': ADAM_RTOL, 'atol': ADAM_ATOL,
            'inherited_unchanged': True, 'raw_words_always_recorded': True},
        'diagnostic_only': ['Python warning objects', 'Python wrapper weakrefs',
                            'Python grad_fn class spelling', 'Python exception message spelling',
                            'native completed-node/leaf diagnostics not observed by this Torch probe'],
        'observer_policy': 'Fresh public G/state/parameter getter tokens used only for primitive snapshots are closed immediately; explicitly held Tensor aliases are kept. Native memory is read only to observe actual incarnation, public-root state and leaf role.',
        'full_rewrite_complete': False, 'pytorch_api_denominator': 2590,
        'rows': [],
    }
    # Freeze exactly what was executed/prepared; output cannot overwrite history.
    for name, path in (('runner.py', Path(__file__)), ('protocol.py', a.protocol),
                       ('owner_client.py', a.protocol.with_name('owner_client.py'))):
        (a.output / name).write_bytes(path.read_bytes())
    if a.prepare_only:
        report['rows'] = [{'case': c, 'variant': v, 'reference_json_pointer': ptr, 'status': 'NOT_RUN'}
                          for c, v, _, ptr in variants]
        emit(a.output / 'report.json', report)
        print(json.dumps({'status': report['status'], 'variants': 12, 'runner_sha256': before['runner']}))
        return 0
    assert a.binary and a.source and a.expected_sha256 and a.expected_source_sha256
    before['binary'], before['source'] = digest(a.binary), digest(a.source)
    assert before['binary'] == a.expected_sha256 and before['source'] == a.expected_source_sha256
    report['binary_sha256'], report['source_sha256'] = before['binary'], before['source']
    wire = load_wire(a.protocol)
    totals = {'frames': 0, 'assertions': 0, 'exact_binary32_words': 0, 'adam_numeric_words': 0,
              'raw_bit_differences': 0, 'full_owner_rollback_bytes_checked': 0}
    for index, (case, variant, observation, pointer) in enumerate(variants):
        directory = a.output / f'{index + 1:02d}'
        directory.mkdir()
        row = {'case': case, 'variant': variant, 'reference_json_pointer': pointer, 'status': 'RUNNING'}
        replay = None
        try:
            replay = Replay(wire, a, directory)
            report['native_executed'] = True
            replay.execute(index, observation)
            row['status'] = 'QUALIFIED_EXPOSED_SCOPE'
        except BaseException as error:
            row.update(status='FAIL', exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        finally:
            if replay is not None:
                try:
                    replay.session.finish()
                except BaseException as error:
                    row.update(status='FAIL', close_exception_type=type(error).__name__, close_message=str(error),
                               close_traceback=traceback.format_exc())
                row.update(frames=replay.session.frames, assertions=replay.assertions,
                           exact_binary32_words=replay.exact_words, adam_numeric_words=replay.adam_words,
                           raw_bit_differences=replay.bit_differences,
                           full_owner_rollback_bytes_checked=replay.rollback_bytes,
                           retired_tensor_address_identities=replay.retired_tensor_identities,
                           responses_before_stdin_eof=all(x['stdin_open'] for x in replay.session.rows),
                           exit_code=replay.session.exit_code)
                emit(directory / 'observed.json', {'trace': replay.trace, 'native_calls': replay.native_calls})
                totals['frames'] += replay.session.frames
                totals['assertions'] += replay.assertions
                totals['exact_binary32_words'] += replay.exact_words
                totals['adam_numeric_words'] += replay.adam_words
                totals['raw_bit_differences'] += len(replay.bit_differences)
                totals['full_owner_rollback_bytes_checked'] += replay.rollback_bytes
            emit(directory / 'result.json', row)
            report['rows'].append(row)
            emit(a.output / 'report.json', report)
    after = {'runner': digest(__file__), 'observations': digest(a.observations), 'probe': digest(a.probe),
             'protocol': digest(a.protocol), 'transport': digest(a.protocol.with_name('owner_client.py')),
             'binary': digest(a.binary), 'source': digest(a.source)}
    assert after == before
    failures = sum(row['status'] == 'FAIL' for row in report['rows'])
    report.update(status='QUALIFIED_EXPOSED_SCOPE' if not failures else 'FAIL',
                  variants_executed=len(report['rows']), variants_qualified=12 - failures,
                  unexpected_failures=failures, all_inputs_unchanged=True, totals=totals)
    emit(a.output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'variants_qualified': 12 - failures,
                      'unexpected_failures': failures, **totals}))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
