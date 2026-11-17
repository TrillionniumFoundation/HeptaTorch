#!/usr/bin/env python3
"""Strict native replay of fixed input-boundaries4 observations.

The immutable actual14 R3 reader supplies all Tensor/Storage/graph/gradient,
weak-lifetime, whole-owner and complete-RNG comparison methods unchanged.
This new reader adds only the new four recipes' wire event mapping and exact
reference admission. No Torch or gradient model executes in this program.
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
PROBE = 'cf569335f48624c1edf2f97a649200f2e792ebcef5b6f1df6bac73293ce02efc'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_class(support):
    class InputBoundaryCase(support.Case):
        def execute(self, event, event_index):
            label = event['label']
            self.snapshot(event['before'], 'event%02d/%s/before' % (event_index, label))
            self.current_path = 'event%02d/%s/operation' % (event_index, label)
            commands, token, publish, status, record = [], None, None, 0, None
            wanted = support.expected_status(event)
            self.check('operation_warning_surface', event['warnings'], [])
            family = self.row['family']
            if label == 'construct_backing':
                kind = self.row['backing_recipe']
                if kind == 'matrix':
                    shape, values = [2, 3], [1., 2., 3., 4., 5., 6.]
                elif re.fullmatch(r'flat[1-9][0-9]*', kind):
                    count = int(kind[4:])
                    shape, values = [count], list(range(1, count + 1))
                else:
                    raise AssertionError(('UNMAPPED_BACKING', kind))
                token = self.n.create(0, shape, struct.pack('<' + 'f' * len(values), *values))
                publish = 'backing'
            elif label == 'construct_detached_strided_input':
                # Reproduce actual as_strided(...).detach() using a real temporary
                # view, then detach, then release only that temporary public root.
                command = self.view(self.live['backing'], self.row['input_size'],
                                    self.row['input_stride'], self.row['input_storage_offset'])
                status, record = self.issue(command, label + '/temporary_true_view')
                commands.append(command)
                if status == 0:
                    temporary = record['value']
                    command = (19, temporary)
                    status, record = self.issue(command, label + '/detach')
                    commands.append(command)
                    token = record['value'] if status == 0 else None
                    command = (41, temporary)
                    cleanup_status, cleanup_record = self.issue(command, label + '/drop_temporary')
                    commands.append(command)
                    self.check('temporary_view_release', cleanup_status, 0, True)
                publish = 'input'
            elif label in ('backing_requires_grad_true', 'input_requires_grad_true'):
                name = 'backing' if label.startswith('backing_') else 'input'
                command = (90, self.live[name], 1)
                status, record = self.issue(command, label)
                commands.append(command)
                token = self.live[name] if status == 0 else None
            elif label.startswith('as_strided_'):
                publish = label[len('as_strided_'):]
                source = {'view': 'input', 'middle': 'backing', 'left': 'middle',
                          'right': 'middle'}[publish]
                command = self.view(self.live[source], self.row[publish + '_size'],
                                    self.row[publish + '_stride'],
                                    self.row[publish + '_storage_offset'])
                status, record = self.issue(command, label)
                token = record['value'] if status == 0 else None
                commands.append(command)
            elif label.startswith('retain_grad_'):
                command = (93, self.live[label[len('retain_grad_'):]])
                status, record = self.issue(command, label)
                commands.append(command)
            elif label == 'mul_distinct_sibling_views':
                command = (2, 3, self.live['left'], self.live['right'])
                status, record = self.issue(command, label)
                token = record['value'] if status == 0 else None
                publish = 'root'
                commands.append(command)
            elif label.startswith('construct_seed_phase_'):
                index = int(label[len('construct_seed_phase_'):])
                phase = self.row['seed_phases'][index]
                shape = self.row['left_size'] if family == 'sibling_fanout' else self.row['view_size']
                words = phase['words']
                assert math.prod(shape) == len(words)
                token = self.n.create(0, shape, struct.pack('<' + 'I' * len(words), *words))
                publish = 'seed'
            elif label.startswith('backward_phase_'):
                match = re.fullmatch(r'backward_phase_([0-9]+)_(.+)', label)
                assert match
                index = int(match[1])
                assert self.row['seed_phases'][index]['name'] == match[2]
                root = 'root' if family == 'sibling_fanout' else 'view'
                retained = index + 1 < len(self.row['seed_phases'])
                assert self.row['backward_schedule'][index] == {
                    'phase': match[2], 'retain_graph': retained}
                command = (58, self.live[root], self.live['seed'], int(retained))
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
                        # The fixed probe assigns a new StorageWeakRef to the
                        # existing key when the next seed phase is published.
                        # Create the new real Weak before releasing the old one;
                        # assignment preserves the original dictionary key order.
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
    return InputBoundaryCase


def admit_reference(raw, probe, expected_hash, support):
    assert sha(raw) == expected_hash and sha(probe) == PROBE
    data = json.loads(raw)
    assert data['schema'] == 'heptatorch.differentiable-view-input-boundaries4.v1'
    assert data['status'] == 'OBSERVED' and data['cases'] == len(data['rows']) == 4
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
    assert len(data['case_plan']) == 4
    for key, state in data['states'].items():
        rawstate = bytes.fromhex(state['hex'])
        assert len(rawstate) == 5056 and state['bytes'] == 5056 and sha(rawstate) == key
    totals = dict(attempted=0, succeeded=0, raised=0)
    for i, (row, spec) in enumerate(zip(data['rows'], data['case_plan'])):
        assert row['index'] == spec['index'] == i and row['status'] == 'OBSERVED'
        assert all(row[key] == value for key, value in spec.items())
        snapshots = [row['before_final_drops']]
        for event in row['events']:
            totals['attempted'] += 1
            totals['succeeded' if event['accepted'] else 'raised'] += 1
            support.expected_status(event)  # Unknown actual causes block, never become generic rejection.
            snapshots.extend((event['before'], event['after']))
        for snapshot in snapshots:
            assert all(relation['same_Python_object'] == relation['same_Tensor']
                       for relation in snapshot['relations'].values())
    assert totals == data['operation_totals']
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'build-report', 'protocol', 'observer',
                 'reference', 'probe', 'support-reader', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    for name in ('source', 'native', 'observer', 'reference'):
        ap.add_argument('--expected-' + name + '-sha256', required=True)
    ap.add_argument('--expected-functions', type=int, default=614)
    ap.add_argument('--expected-native-units', type=int, default=12)
    ap.add_argument('--prepare-only', action='store_true')
    a = ap.parse_args()
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
        'schema': 'heptatorch.differentiable-view-input4-native-replay.v1',
        'reference_sha256': a.expected_reference_sha256, 'probe_sha256': PROBE,
        'support_reader_sha256': SUPPORT_READER, 'source_sha256': a.expected_source_sha256,
        'native_sha256': a.expected_native_sha256, 'observer_sha256': a.expected_observer_sha256,
        'descriptor_bytes': 672, 'functions': a.expected_functions, 'native_units': a.expected_native_units,
        'expected_cases': 4, 'reference_events': data['operation_totals']['attempted'],
        'reference_operation_totals': data['operation_totals'], 'torch_executed_here': False,
        'full_pytorch_rewrite_complete': False, 'closed_schema_count': 0, 'pytorch_api_denominator': 2590,
        'fixed_configuration': {'cpu_capability': 'AVX512', 'threads': 1,
                                'deterministic_algorithms': True, 'deterministic_fill': True},
        'comparison_scope': 'The original R3 Case methods are inherited unchanged: full logical/physical bytes, geometry/Versions, T/S identity, immediate and captured edges, Saved, gradients/retain, Weak, full5056 RNG, audit and EOF. Only these four new event recipes are mapped here.',
        'seed_scope': 'Raw words and geometry are declared reference inputs; no gradient values are calculated by the observer.',
        'same_Python_object_scope': 'Every fixed input4 relation must have same_Python_object == same_Tensor; no general wrapper-refcount equivalence.',
        'warnings_scope': 'Original grad-warning predicates are inherited; non-getter operation warnings outside the empty warning surface cause an explicit mismatch.',
        'partial_prefix_scope': 'All before/after gradient and Saved effects are compared; native58 counters are retained as raw diagnostics, not labeled Torch observations.',
        'input_setup_mapping': 'True111(as_strided) followed by19(detach), then explicit temporary41; original reference compound operation is not mapped to a fabricated view flag.',
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
    observer = load(a.observer, 'input_boundaries4_native_observer')
    Case = case_class(support)
    rows = []
    for row in data['rows']:
        result = Case(a, observer, data, row, a.output / ('case-%02d' % row['index'])).run()
        rows.append({key: result[key] for key in ('index', 'name', 'status', 'assertions', 'exact_words',
                    'full_rng_bytes', 'rollback_bytes', 'frames', 'exit_code', 'error', 'mismatches')})
        print(json.dumps({'case': result['index'], 'status': result['status'], 'frames': result['frames'],
                          'mismatches': len(result['mismatches']),
                          'error': result['error']['type'] if result['error'] else None}), flush=True)
    unchanged = all(p.read_bytes() == raw for p, raw in inputs.items())
    report = {**common, 'status': 'PASS' if all(row['status'] == 'PASS' for row in rows) and unchanged else 'FAIL',
              'native_executed': True, 'cases': len(rows), 'rows': rows,
              'passed_cases': sum(row['status'] == 'PASS' for row in rows),
              'failed_cases': sum(row['status'] != 'PASS' for row in rows),
              'all_inputs_unchanged': unchanged, 'inputs_sha256': {str(p): sha(raw) for p, raw in inputs.items()}}
    for key in ('assertions', 'exact_words', 'full_rng_bytes', 'rollback_bytes', 'frames'):
        report[key] = sum(row[key] for row in rows)
    report['all_sessions_eof_exit_zero'] = all(row['exit_code'] == 0 for row in rows)
    support.write_json(a.output / 'report.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'passed_cases', 'failed_cases',
                                                 'frames', 'assertions', 'all_sessions_eof_exit_zero')}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
