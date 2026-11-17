#!/usr/bin/env python3
"""Strict original view/reshape reference26 replay.

No Torch or gradient model runs here. This successor maps the two original
wrapped-float64 scalar recipes through genuine command114. The original
24-PASS/2-NOT_RUN report remains immutable. Optional selected-case execution
is reported PARTIAL_PASS, never full26 qualification.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import struct
import sys

SUPPORT_READER = '52c3b3242a73fb412102c83c5ba47cf8d95a4eb3c01e56f41bfe593066506d05'
PROBE = '4dd374419e4b21a9c814c855d846481730a12ea57a30dcf229a6b962f07021c5'
REFERENCE = 'fecf1956790f78a9b1238f535c1c7cce9f1237b211c389ac0736d88687bbb9c9'
UNSUPPORTED_FAMILIES = ('nonleaf_mul', 'nonleaf_mul_transpose')
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_status(event, support):
    if event['accepted']:
        return 0
    error = event['error']
    assert error['type'] == 'RuntimeError', error
    message = error['message']
    shape_errors = {
        "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.",
        "cannot reshape tensor of 0 elements into shape [0, -1] because the unspecified dimension size -1 can be any value and is ambiguous",
        "only one dimension can be inferred",
        "invalid shape dimension -2 at index 0 of shape [-2, 3]",
        "shape '[5]' is invalid for input of size 6",
    }
    if event['label'] == 'target_view' and message in shape_errors:
        return 2
    return support.expected_status(event)


def case_class(support):
    class ShapeCase(support.Case):
        def node_info(self, index, depth=4):
            if not index:
                return None
            memory = self.mem()
            node = memory.node(index)
            if node[5] == 0:
                target = node[76] & 65535
                if not target:
                    target = index
                result = {'kind': 'AccumulateGrad', 'variable': memory.tensor(target)}
                if depth:
                    result['next_edges'] = []
                return result
            kinds = {**support.KINDS, 10: 'TBackward0', 16: 'ViewBackward0',
                     17: 'CloneBackward0', 45: 'ReshapeAliasBackward0', 46: 'UnsafeViewBackward0'}
            assert node[5] in kinds, ('unmapped_actual_native_node', index, node[5])
            result = {'kind': kinds[node[5]]}
            if depth:
                count = 2 if node[5] == 3 else 1
                result['next_edges'] = [
                    {'output_nr': 0, 'node': self.node_info(node[6+i], depth-1) if node[9+i] else None}
                    for i in range(count)]
            return result

        def saved_attributes(self, token, expected):
            node = self.mem().node(self.mem().public_node(token))
            if node[5] not in (16, 45, 46):
                return super().saved_attributes(token, expected)
            # Fixed generated view/reshape derivatives save input sym_sizes only.
            # Read the real captured parent descriptor, never the reference plan.
            self.check('Saved_attribute_names', ['_saved_self_sym_sizes'], list(expected))
            observed = {}
            if '_saved_self_sym_sizes' in expected:
                event = expected['_saved_self_sym_sizes']
                parent = self.mem().node(node[6])
                value = list(parent[12:12+parent[0]])
                self.check('_saved_self_sym_sizes/accepted', event['accepted'], True)
                self.check('_saved_self_sym_sizes/warnings', event['warnings'], [])
                self.check('_saved_self_sym_sizes/value', value, event['value'])
                observed['_saved_self_sym_sizes'] = support.event_normal(True, True, value)
            return observed

        def execute(self, event, event_index):
            label = event['label']
            self.snapshot(event['before'], 'event%02d/%s/before' % (event_index, label))
            self.current_path = 'event%02d/%s/operation' % (event_index, label)
            commands, token, publish, status, record = [], None, None, 0, None
            wanted = expected_status(event, support)
            self.check('operation_warning_surface', event['warnings'], [])
            family = self.row['family']
            if label == 'construct_backing':
                count = self.row['backing_elements']
                dtype = self.row['dtype']
                shape = [2, 3] if family in UNSUPPORTED_FAMILIES else [count]
                if dtype == 'float32':
                    data = struct.pack('<' + 'f' * count, *range(1, count+1))
                    code = 0
                elif dtype == 'bool':
                    data = bytes(i % 2 for i in range(count))
                    code = 1
                elif dtype == 'int64':
                    pattern = [0, 1, -1, 16777217, -16777219, (1 << 62) + 3, -(1 << 62) + 3, 9]
                    data = struct.pack('<' + 'q' * count, *(pattern[i % len(pattern)] for i in range(count)))
                    code = 2
                else:
                    raise AssertionError(('UNMAPPED_DTYPE', dtype))
                token = self.n.create(code, shape, data)
                publish = 'backing'
            elif label in ('construct_mul_input', 'construct_mul_middle'):
                assert family in UNSUPPORTED_FAMILIES
                self.check('wrapped_scalar_input_literal', event['accepted'], True, True)
                command = (114, self.live['backing'], 0x4000000000000000)
                status, record = self.issue(command, label)
                commands.append(command)
                token = record['value'] if status == 0 else None
                publish = 'input' if label == 'construct_mul_input' else 'middle'
            elif label == 'construct_transpose_input':
                assert family == 'nonleaf_mul_transpose'
                command = (6, self.live['middle'], 0, 1)
                status, record = self.issue(command, label)
                commands.append(command)
                token = record['value'] if status == 0 else None
                publish = 'input'
            elif label.startswith('construct_') and label[len('construct_'):] in (
                    'detached_input', 'own_requires_view', 'derived_requires_view'):
                self.check('input_family', label[len('construct_'):], family, True)
                command = self.view(self.live['backing'], self.row['input_size'],
                                    self.row['input_stride'], self.row['input_storage_offset'])
                status, record = self.issue(command, label + '/true_view')
                commands.append(command)
                token = record['value'] if status == 0 else None
                if status == 0 and family == 'detached_input':
                    temporary = token
                    command = (19, temporary)
                    status, record = self.issue(command, label + '/detach')
                    commands.append(command)
                    token = record['value'] if status == 0 else None
                    command = (41, temporary)
                    cleanup_status, _ = self.issue(command, label + '/drop_temporary')
                    commands.append(command)
                    self.check('temporary_view_release', cleanup_status, 0, True)
                publish = 'input'
            elif label in ('backing_requires_grad_true', 'input_requires_grad_true'):
                name = 'backing' if label.startswith('backing_') else 'input'
                command = (90, self.live[name], 1)
                status, record = self.issue(command, label)
                commands.append(command)
                token = self.live[name] if status == 0 else None
            elif label in ('target_view', 'target_reshape'):
                operation = self.row['operation']
                assert label == 'target_' + operation
                dims = self.row['target_shape']
                cfg = struct.pack('<5Q', len(dims), *(v & ((1 << 64)-1) for v in dims),
                                  *([0] * (4-len(dims))))
                command = (112 if operation == 'view' else 113,
                           self.live['input'], self.n.w.Payload(cfg), 40)
                status, record = self.issue(command, label)
                token = record['value'] if status == 0 else None
                publish = 'result'
                commands.append(command)
            elif label.startswith('retain_grad_'):
                command = (93, self.live[label[len('retain_grad_'):]])
                status, record = self.issue(command, label)
                commands.append(command)
            elif label.startswith('construct_seed_phase_'):
                index = int(label[len('construct_seed_phase_'):])
                phase = self.row['seed_phases'][index]
                # Result shape is only used to construct the actual explicit input
                # seed; no target gradient/layout/Version is computed.
                shape = event['returned']['tensor']['shape']
                pattern = phase['repeating_input_words']
                words = [pattern[i % len(pattern)] for i in range(math.prod(shape))]
                self.check('declared_seed_words', words, event['returned']['tensor']['binary_words'], True)
                token = self.n.create(0, shape, struct.pack('<' + 'I' * len(words), *words))
                publish = 'seed'
            elif label.startswith('backward_phase_'):
                match = re.fullmatch(r'backward_phase_([0-9]+)_(.+)', label)
                assert match
                index = int(match[1])
                assert self.row['seed_phases'][index]['name'] == match[2]
                retained = index + 1 < len(self.row['seed_phases'])
                command = (58, self.live['result'], self.live['seed'], int(retained))
                status, record = self.issue(command, label, partial=True)
                commands.append(command)
            elif label == 'backward_retry_after_default':
                assert family in UNSUPPORTED_FAMILIES
                command = (58, self.live['result'], self.live['seed'], 0)
                status, record = self.issue(command, label, partial=True)
                commands.append(command)
            elif label.startswith('hold_phase_'):
                match = re.fullmatch(r'hold_phase_([0-9]+)_(.+)_gradient', label)
                assert match
                index, name = int(match[1]), match[2]
                assert index < len(self.row['seed_phases'])
                command = (91, self.live[name])
                status, record = self.issue(command, label)
                token = record['value'] if status == 0 else None
                publish = 'held_phase_' + str(index) + '_' + name + '_grad'
                commands.append(command)
            elif label.startswith('clear_before_phase_'):
                match = re.fullmatch(r'clear_before_phase_([0-9]+)_(.+)', label)
                assert match and 0 < int(match[1]) < len(self.row['seed_phases'])
                command = (92, self.live[match[2]], 0)
                status, record = self.issue(command, label)
                commands.append(command)
            elif label == 'mutate_result_fill_nograd':
                dtype = self.row['dtype']
                value = (self.n.w.bits(42.) if dtype == 'float32' else
                         1 if dtype == 'bool' else ((1 << 64)-9) if dtype == 'int64' else None)
                assert value is not None
                command = (100, self.live['result'], value, 1)
                status, record = self.issue(command, label)
                token = self.live['result'] if status == 0 else None
                commands.append(command)
            elif label.startswith('drop_seed_before_phase_') or label.startswith('final_drop_'):
                name = ('seed' if label.startswith('drop_seed_before_phase_')
                        else label[len('final_drop_'):])
                drop_token = self.live[name]
                if sum(t == drop_token for t in self.live.values()) == 1:
                    command = (41, drop_token)
                    status, record = self.issue(command, label)
                    commands.append(command)
                del self.live[name]
            else:
                raise AssertionError(('UNMAPPED_NEW_EVENT', label))
            self.check('normalized_operation_status', status, wanted)
            operation = {'label': label, 'normalized_actual_status': status,
                         'normalized_reference_status': wanted, 'reference_event': event_index,
                         'commands': commands, 'record': record}
            if wanted == 0 and status != 0:
                self.operations.append(operation)
                raise support.RequiredOperationFailed(('native rejected accepted operation', label, status))
            if wanted != 0 and status == 0:
                self.operations.append(operation)
                raise support.RequiredOperationFailed(('native accepted rejected operation', label, wanted))
            if status == 0:
                returned = event.get('returned')
                if isinstance(returned, dict) and 'tensor' in returned:
                    self.check('returned_Tensor_present', bool(token), True, True)
                    actual = self.raw_tensor(token)
                    self.tensor(returned['tensor'], actual, 'returned_Tensor')
                    self.check('returned_is_existing', {name: t == token for name, t in self.live.items()},
                               event['returned_is_existing'])
                    operation['returned_tensor'] = actual
                else:
                    self.check('returned_primitive', returned, None)
                if publish is not None:
                    self.check('publish_new_name', publish in self.live, False, True)
                    weak_name = publish + '_storage'
                    if weak_name in self.weak:
                        self.check('replaced_Weak_is_seed_phase', label.startswith('construct_seed_phase_'), True, True)
                        self.check('replaced_Weak_later_phase', int(label[len('construct_seed_phase_'):]) > 0, True, True)
                        old_weak = self.weak[weak_name]
                        command = (42, token)
                        weak_status, weak_record = self.issue(command, label + '/new_Weak')
                        commands.append(command)
                        self.check('replacement_Weak_create_status', weak_status, 0, True)
                        command = (44, old_weak)
                        weak_status, _ = self.issue(command, label + '/release_previous_Weak')
                        commands.append(command)
                        self.check('previous_Weak_release_status', weak_status, 0, True)
                        self.live[publish] = token
                        self.weak[weak_name] = weak_record['value']
                    else:
                        self.published(publish, token)
            self.operations.append(operation)
            self.snapshot(event['after'], 'event%02d/%s/after' % (event_index, label))
    return ShapeCase


def admit_reference(raw, probe, expected_hash, support):
    assert sha(raw) == expected_hash == REFERENCE and sha(probe) == PROBE
    data = json.loads(raw)
    assert data['schema'] == 'heptatorch.concrete-view-reshape-reference26.v1'
    assert data['status'] == 'OBSERVED' and data['cases'] == len(data['rows']) == 26
    assert data['probe_error_count'] == 0 and not data['probe_errors']
    assert data['source_files_unchanged'] is True and data['probe_sha256'] == PROBE
    assert data['prior_reference14_sha256'] == support.REFERENCE
    assert data['existing_multi_axis72_sha256'] == '2512acdfeaf98f0aa8d8f44ce37dec48b926244ae274066e6e5b86e919a806e9'
    assert data['oracle_provenance']['status'] == 'PASS'
    checks = data['oracle_provenance']['checks']
    assert set(checks) == support.CHECKS and all(value is True for value in checks.values())
    assert data['num_threads'] == 1 and data['cpu_capability'] == 'AVX512'
    assert data['environment_before'] == data['environment_after']
    assert data['environment_before']['deterministic_algorithms'] is True
    assert data['environment_before']['deterministic_fill'] is True
    assert data['environment_restored'] and data['original_outer_rng_restored']
    assert support.SUPPORT in data['inputs_sha256'].values() and support.LOCK in data['inputs_sha256'].values()
    assert len(data['case_plan']) == 26
    for key, state in data['states'].items():
        rawstate = bytes.fromhex(state['hex'])
        assert len(rawstate) == 5056 and state['bytes'] == 5056 and sha(rawstate) == key
    totals = dict(attempted=0, succeeded=0, raised=0)
    blocked = []
    for i, (row, spec) in enumerate(zip(data['rows'], data['case_plan'])):
        assert row['index'] == spec['index'] == i and row['status'] == 'OBSERVED'
        assert all(row[key] == value for key, value in spec.items())
        snapshots = [row['before_final_drops']]
        if row['family'] in UNSUPPORTED_FAMILIES:
            blocked.append(i)
        for event in row['events']:
            totals['attempted'] += 1
            totals['succeeded' if event['accepted'] else 'raised'] += 1
            expected_status(event, support)
            snapshots.extend((event['before'], event['after']))
        for snapshot in snapshots:
            assert all(relation['same_Python_object'] == relation['same_Tensor']
                       for relation in snapshot['relations'].values())
    assert totals == data['operation_totals'] == {'attempted': 483, 'succeeded': 475, 'raised': 8}
    assert blocked == [22, 25]
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'build-report', 'protocol', 'observer',
                 'reference', 'probe', 'support-reader', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    for name in ('source', 'native', 'observer', 'reference'):
        ap.add_argument('--expected-' + name + '-sha256', required=True)
    ap.add_argument('--expected-functions', type=int, required=True)
    ap.add_argument('--expected-native-units', type=int, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--case-index', type=int, action='append', default=[])
    a = ap.parse_args()
    selected = a.case_index or list(range(26))
    assert len(selected) == len(set(selected)) and all(0 <= i < 26 for i in selected)
    paths = [Path(__file__).resolve(), a.binary, a.source, a.build_report,
             a.protocol, a.protocol.with_name('owner_client.py'), a.observer,
             a.reference, a.probe, a.support_reader]
    inputs = {p.resolve(): p.read_bytes() for p in paths}
    pins = {a.binary.resolve(): a.expected_native_sha256, a.source.resolve(): a.expected_source_sha256,
            a.observer.resolve(): a.expected_observer_sha256, a.support_reader.resolve(): SUPPORT_READER,
            a.protocol.resolve(): PROTOCOL, a.protocol.with_name('owner_client.py').resolve(): CLIENT}
    for path, pin in pins.items():
        assert sha(inputs[path]) == pin, (path, sha(inputs[path]), pin)
    support = load(a.support_reader, 'immutable_actual14_reader_support')
    data = admit_reference(inputs[a.reference.resolve()], inputs[a.probe.resolve()],
                           a.expected_reference_sha256, support)
    assert len(support.FUNCTIONS.findall(inputs[a.source.resolve()])) == a.expected_functions
    build = json.loads(inputs[a.build_report.resolve()])
    assert build['status'] == 'PASS' and build['source_sha256'] == a.expected_source_sha256
    assert build['native_sha256'] == a.expected_native_sha256 and build['descriptor_bytes'] == 672
    assert build['functions'] == a.expected_functions and len(build['units']) == a.expected_native_units
    for unit in build['units']:
        for suffix, key in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
            path = (a.build_report.parent / (unit['name'] + suffix)).resolve()
            raw = path.read_bytes()
            assert sha(raw) == unit[key], (path, sha(raw), unit[key])
            inputs[path] = raw
    a.output.mkdir(parents=True, exist_ok=False)
    common = {
        'schema': 'heptatorch.concrete-view-reshape-native26.v1',
        'reference_sha256': a.expected_reference_sha256, 'probe_sha256': PROBE,
        'support_reader_sha256': SUPPORT_READER, 'source_sha256': a.expected_source_sha256,
        'native_sha256': a.expected_native_sha256, 'observer_sha256': a.expected_observer_sha256,
        'descriptor_bytes': 672, 'functions': a.expected_functions, 'native_units': a.expected_native_units,
        'expected_cases': 26, 'reference_events': data['operation_totals']['attempted'],
        'reference_operation_totals': data['operation_totals'], 'torch_executed_here': False,
        'full_pytorch_rewrite_complete': False, 'closed_schema_count': 0, 'pytorch_api_denominator': 2590,
        'fixed_configuration': {'cpu_capability': 'AVX512', 'threads': 1,
                                'deterministic_algorithms': True, 'deterministic_fill': True},
        'comparison_scope': 'Original Tensor/Storage/identity/Version/gradient/Weak/whole-owner/RNG comparison methods are inherited unchanged. New decoding adds View16, ReshapeAlias45, UnsafeView46, Clone17 and bounded T10, plus actual saved input-size metadata. Event mapping is only this fixed reference26.',
        'seed_scope': 'Raw words and geometry are declared reference inputs; no gradient values are calculated by the observer.',
        'same_Python_object_scope': 'Every fixed reference26 relation must have same_Python_object == same_Tensor; no general wrapper-refcount equivalence.',
        'warnings_scope': 'Original grad-warning predicates are inherited; non-getter operation warnings outside the empty warning surface cause an explicit mismatch.',
        'partial_prefix_scope': 'All before/after gradient and Saved effects are compared; native58 counters are retained as raw diagnostics, not labeled Torch observations.',
        'input_setup_mapping': 'True111(as_strided), optional19(detach) plus temporary41; no fabricated view flag. True100 typed fill with no-grad flag1. New112/113 only actual concrete view/reshape.',
        'original_blocked_history': {'indices': [22, 25], 'reader_sha256': 'aebe22c0e0a93c0211d7bdc42c1223773f912a17aaff4af5d64e7e07acc89aad',
                                    'native_report_sha256': '148ecc661cb7c65f2196930ed3fc19c2a53b77d7204bb63305de6a8c63b1f20c'},
        'wrapped_scalar_mapping': '114 receives exact raw binary64 2.0 and retains a real float64 Scalar Tensor; no public F32 substitute is created by this reader.',
        'selected_case_indices': selected,
        'complete_original_overload_mapping': True,
    }
    if a.prepare_only:
        report = {**common, 'status': 'PREPARED_NOT_RUN', 'native_executed': False,
                  'inputs_sha256': {str(p): sha(raw) for p, raw in inputs.items()},
                  'all_inputs_unchanged': all(p.read_bytes() == raw for p, raw in inputs.items())}
        support.write_json(a.output / 'report.json', report)
        print(json.dumps({key: report[key] for key in ('status', 'expected_cases', 'reference_events')}))
        return 0
    copies = a.output / 'inputs'
    copies.mkdir()
    custody = {}
    for index, (path, raw) in enumerate(inputs.items()):
        target = copies / ('%03d-' % index + path.name)
        target.write_bytes(raw)
        custody[str(path)] = {'copy': str(target.relative_to(a.output)), 'bytes': len(raw), 'sha256': sha(raw)}
    support.write_json(a.output / 'input-custody.json', custody)
    observer = load(a.observer, 'shape26_native_observer')
    Case = case_class(support)
    rows = []
    for row in data['rows']:
        if row['index'] not in selected:
            continue
        result = Case(a, observer, data, row, a.output / ('case-%02d' % row['index'])).run()
        rows.append({key: result[key] for key in ('index', 'name', 'status', 'assertions', 'exact_words',
                    'full_rng_bytes', 'rollback_bytes', 'frames', 'exit_code', 'error', 'mismatches')})
        print(json.dumps({'case': result['index'], 'status': result['status'], 'frames': result['frames'],
                          'mismatches': len(result['mismatches']),
                          'error': result['error']['type'] if result['error'] else None}), flush=True)
    unchanged = all(p.read_bytes() == raw for p, raw in inputs.items())
    passed = all(row['status'] == 'PASS' for row in rows) and unchanged
    complete = len(rows) == 26 and set(row['index'] for row in rows) == set(range(26))
    report = {**common, 'status': ('PASS' if complete else 'PARTIAL_PASS') if passed else 'FAIL',
              'native_executed': True, 'cases': len(rows), 'rows': rows,
              'passed_cases': sum(row['status'] == 'PASS' for row in rows),
              'failed_cases': sum(row['status'] == 'FAIL' for row in rows),
              'blocked_cases': sum(row['status'] == 'NOT_RUN_UNSUPPORTED_NATIVE_INPUT' for row in rows),
              'executed_cases': sum(row['status'] != 'NOT_RUN_UNSUPPORTED_NATIVE_INPUT' for row in rows),
              'all_inputs_unchanged': unchanged, 'inputs_sha256': {str(p): sha(raw) for p, raw in inputs.items()}}
    for key in ('assertions', 'exact_words', 'full_rng_bytes', 'rollback_bytes', 'frames'):
        report[key] = sum(row[key] for row in rows)
    report['all_executed_sessions_eof_exit_zero'] = all(row['exit_code'] == 0 for row in rows if row['status'] != 'NOT_RUN_UNSUPPORTED_NATIVE_INPUT')
    report['all_reference_cases_executed'] = complete and report['blocked_cases'] == 0
    support.write_json(a.output / 'report.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'passed_cases', 'failed_cases',
                                                 'blocked_cases', 'frames', 'assertions', 'all_executed_sessions_eof_exit_zero')}))
    return 0 if report['status'] in ('PASS', 'PARTIAL_PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
