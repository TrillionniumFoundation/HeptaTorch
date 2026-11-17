#!/usr/bin/env python3
"""Replay the immutable fixed-Torch wrapped-F64 scalar32 observations.

Only native commands and byte observers run here. No Torch, conversion formula,
or gradient model constructs expected output. This reader does not claim a
weak-expiration result: the reference32 has no such observation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback

REFERENCE = 'f3ecc9dd7db2ce8fbd0e424557ad5b95ac2e899a1c09cf38dad47e4fecc2b0c6'
PROBE = 'c0ddafbae6ea81bd98f6066bc867f71e0b6e8d4f73e6c5d7f15f7fe3d29569a3'
SUPPORT_READER = '52c3b3242a73fb412102c83c5ba47cf8d95a4eb3c01e56f41bfe593066506d05'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
LABELS = ('construct_input', 'input_requires_grad', 'multiply_python_float',
          'retain_output_gradient', 'construct_seed', 'backward_default_release')
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def admit_reference(raw, probe, support):
    assert sha(raw) == REFERENCE and sha(probe) == PROBE
    data = json.loads(raw)
    assert data['schema'] == 'heptatorch.wrapped-f64-multiply-reference32.v1'
    assert data['status'] == 'OBSERVED' and data['cases'] == len(data['rows']) == len(data['case_plan']) == 32
    assert not data['probe_errors'] and data['probe_sha256'] == PROBE
    assert data['torch_executed'] is True and data['native_executed'] is False
    assert data['source_files_unchanged'] and data['inputs_sha256'] == data['inputs_after_sha256']
    assert data['oracle_provenance']['status'] == 'PASS'
    checks = data['oracle_provenance']['checks']
    assert set(checks) == support.CHECKS and all(value is True for value in checks.values())
    assert support.SUPPORT in data['inputs_sha256'].values()
    assert support.LOCK in data['inputs_sha256'].values()
    env = data['environment_before']
    assert env == data['environment_after']
    assert env == {'deterministic_algorithms': True, 'deterministic_fill': True,
                   'grad_enabled': True, 'num_threads': 1, 'cpu_capability': 'AVX512',
                   'default_dtype': 'torch.float32'}
    assert data['environment_unchanged'] and data['outer_rng_restored']
    for key, state in data['full_rng_states'].items():
        bits = bytes.fromhex(state['hex'])
        assert len(bits) == state['bytes'] == 5056 and sha(bits) == key
    totals = {'attempted': 0, 'succeeded': 0, 'raised': 0}
    for index, (row, plan) in enumerate(zip(data['rows'], data['case_plan'])):
        assert row['index'] == plan['index'] == index and row['status'] == 'OBSERVED'
        assert all(row[key] == value for key, value in plan.items())
        assert row['shape'] == [4] and row['requires_grad'] is True
        assert row['retain_graph'] is False and row['retain_output_gradient'] is True
        word = int(row['scalar_binary64_hex'], 16)
        assert 0 <= word < 1 << 64
        assert word.to_bytes(8, 'little').hex() == row['scalar_little_endian_hex']
        scalar = {'little_endian_hex': row['scalar_little_endian_hex'],
                  'binary64_hex': row['scalar_binary64_hex'], 'python_type': 'float'}
        for name in ('python_scalar_after_unpack', 'python_scalar_after_multiply',
                     'python_scalar_after_operations'):
            assert row[name] == scalar
        assert tuple(event['label'] for event in row['events']) == LABELS
        snapshots = [row['final']]
        for event in row['events']:
            totals['attempted'] += 1
            totals['succeeded' if event['accepted'] else 'raised'] += 1
            assert event['accepted'] is True and event['outcome'] == 'SUCCEEDED'
            assert event['error'] is None and event['warnings'] == []
            assert event['python_scalar'] == scalar
            snapshots.extend((event['before'], event['after']))
        for snapshot in snapshots:
            assert snapshot['tensor_getter_order'] == list(snapshot['tensors'])
            assert snapshot['state'] in data['full_rng_states']
            for name, tensor in snapshot['tensors'].items():
                assert name in ('input', 'output', 'seed')
                assert tensor['raw']['dtype'] == 'torch.float32'
                for getter in ('grad_fn', 'is_leaf', 'retains_grad', 'gradient'):
                    assert tensor[getter]['accepted'] is True
                saved = tensor['grad_fn']['saved']
                for saved_name, event in saved.items():
                    assert saved_name in ('_saved_other', '_saved_self')
                    support.expected_status(event)
                    if event['accepted'] and isinstance(event.get('value'), dict):
                        original = event['value'].get('tensor')
                        if original is not None:
                            assert saved_name == '_saved_other'
                            assert original['dtype'] == 'torch.float64' and original['element_size'] == 8
                            assert original['shape'] == original['strides'] == []
                            assert original['numel'] == 1 and original['storage_bytes'] == 8
                            assert original['storage_offset_elements'] == original['storage_offset_bytes'] == 0
                            assert original['logical_bytes_hex'] == original['storage_raw_hex'] == row['scalar_little_endian_hex']
                            assert original['binary_words'] == [word]
                            assert original['version'] == 0 and original['requires_grad'] is False
                            assert original['is_view'] is False and original['base_tensor_identity'] is None
    assert totals == data['operation_totals'] == {'attempted': 192, 'succeeded': 192, 'raised': 0}
    return data


def case_class(support):
    class ScalarCase(support.Case):
        def __init__(self, args, observer, reference, row, directory):
            # The original byte/identity/RNG helpers use the key "states".
            # This is only a schema alias to the exact reference dictionary.
            super().__init__(args, observer, {**reference, 'states': reference['full_rng_states']}, row, directory)

        def snapshot(self, expected, label):
            self.current_path = label
            self.check('getter_order', list(self.live), expected['tensor_getter_order'], True)
            output = {'label': label, 'tensor_getter_order': list(self.live), 'tensors': {}}
            for name in expected['tensor_getter_order']:
                self.current_path = label + '/' + name
                token = self.live[name]
                e = expected['tensors'][name]
                raw = self.raw_tensor(token)
                self.tensor(e['raw'], raw, 'raw_before_getters')
                meta = self.n.w.meta(self.n.one(13, token))
                memory = self.mem()
                index = memory.public_node(token)
                node = memory.node(index)
                fn = self.node_info(index) if node[5] else None
                self.check('grad_fn_getter_accepted', e['grad_fn']['accepted'], True)
                self.check('grad_fn_getter_warnings', e['grad_fn']['warnings'], [])
                self.compare_node(e['grad_fn']['node'], fn, 'grad_fn_node')
                # Preserve the probe's exact getter order: Saved immediately
                # follows this Tensor's grad_fn, before is_leaf/retains_grad/G.
                saved = self.saved_attributes(token, e['grad_fn']['saved'])
                self.check('public_metadata_requires', bool(meta['requires_grad']), bool(node[4]))
                self.check('public_metadata_shape', meta['shape'][:meta['rank']], list(node[12:12+node[0]]))
                self.check('public_metadata_strides', meta['strides'][:meta['rank']], list(node[16:16+node[0]]))
                self.check('leaf_getter_accepted', e['is_leaf']['accepted'], True)
                self.check('leaf_getter_warnings', e['is_leaf']['warnings'], [])
                self.check('leaf_value', node[5] == 0, e['is_leaf']['value'])
                retained = bool(self.n.one(94, token)['value'])
                self.check('retains_grad_getter_accepted', e['retains_grad']['accepted'], True)
                self.check('retains_grad_getter_warnings', e['retains_grad']['warnings'], [])
                self.check('retains_grad_value', retained, e['retains_grad']['value'])
                gradient = self.gradient(token, e['gradient'])
                logical = self.n.one(98, token, 0, node[1])['payload']
                self.check('logical_public_vs_real_Storage', logical.hex(), self.raw_tensor(token)['logical_bytes_hex'])
                output['tensors'][name] = {'raw': raw, 'grad_fn': fn, 'saved': saved,
                                          'is_leaf': node[5] == 0, 'retains_grad': retained,
                                          'gradient': gradient}
            self.current_path = label
            rng = self.n.one(63, self.gen)['payload']
            wanted = bytes.fromhex(self.reference['states'][expected['state']]['hex'])
            self.check('complete_5056_MT', rng, wanted)
            self.rng_bytes += len(wanted)
            output['rng_sha256'] = sha(rng)
            self.check('native_audit', self.n.one(53)['value'], 1)
            owner = self.owner(label)
            output['owner_sha256'], output['frame'] = sha(owner['bytes']), self.n.s.frames
            self.snapshots.append(output)

        def execute(self, event, index):
            label = event['label']
            self.snapshot(event['before'], 'event%02d/%s/before' % (index, label))
            self.current_path = 'event%02d/%s/operation' % (index, label)
            commands, token, publish, record, status = [], None, None, None, 0
            self.check('operation_warning_surface', event['warnings'], [])
            if label in ('construct_input', 'construct_seed'):
                word_key = 'input_binary32_hex' if label == 'construct_input' else 'seed_binary32_hex'
                raw = b''.join(int(word, 16).to_bytes(4, 'little') for word in self.row[word_key])
                token = self.n.create(0, self.row['shape'], raw)
                publish = 'input' if label == 'construct_input' else 'seed'
                commands.append({'actual_native_create_dtype': 0, 'shape': self.row['shape'],
                                 'literal_input_bytes_hex': raw.hex()})
            elif label == 'input_requires_grad':
                token = self.live['input']
                command = (90, token, 1)
                commands.append(command)
                status, record = self.issue(command, label)
            elif label == 'multiply_python_float':
                command = (114, self.live['input'], int(self.row['scalar_binary64_hex'], 16))
                commands.append(command)
                status, record = self.issue(command, label)
                if status == 0:
                    token, publish = record['value'], 'output'
            elif label == 'retain_output_gradient':
                command = (93, self.live['output'])
                commands.append(command)
                status, record = self.issue(command, label)
            elif label == 'backward_default_release':
                command = (58, self.live['output'], self.live['seed'], 0)
                commands.append(command)
                status, record = self.issue(command, label, partial=True)
            else:
                raise AssertionError(('UNMAPPED_REFERENCE_EVENT', label))
            self.check('normalized_operation_status', status, support.expected_status(event))
            operation = {'label': label, 'commands': commands, 'normalized_status': status,
                         'record': record, 'reference_python_scalar': event['python_scalar']}
            if status == 0:
                returned = event.get('returned')
                if isinstance(returned, dict) and 'tensor' in returned:
                    self.check('returned_Tensor_present', bool(token), True, True)
                    actual = self.raw_tensor(token)
                    self.tensor(returned['tensor'], actual, 'returned_Tensor')
                    operation['returned_tensor'] = actual
                else:
                    self.check('returned_primitive', returned, None)
                if publish is not None:
                    self.check('publish_new_name', publish in self.live, False, True)
                    # No reference32 weak-lifetime claim. Native gradient
                    # observer Weak handles are cleanup-only and stay separate.
                    self.live[publish] = token
            self.operations.append(operation)
            self.snapshot(event['after'], 'event%02d/%s/after' % (index, label))

        def run(self):
            error = None
            try:
                for index, event in enumerate(self.row['events']):
                    self.execute(event, index)
                self.snapshot(self.row['final'], 'final_observed_snapshot')
                self.current_path = 'explicit_root_cleanup'
                expected = bytes.fromhex(self.reference['states'][self.row['state_after']]['hex'])
                state = self.n.one(63, self.gen)['payload']
                self.check('state_after_operations', state, expected)
                self.rng_bytes += len(expected)
                for token in list(self.live.values()):
                    self.n.one(41, token)
                self.live.clear()
                expected = bytes.fromhex(self.reference['states'][self.row['state_after_explicit_roots_cleared']]['hex'])
                state = self.n.one(63, self.gen)['payload']
                self.check('state_after_explicit_roots_cleared', state, expected)
                self.rng_bytes += len(expected)
                self.check('cleanup_native_audit', self.n.one(53)['value'], 1)
                self.owner('after_explicit_root_cleanup')
                for token in self.weak.values():
                    self.n.one(44, token)
                self.weak.clear()
                self.n.one(66, self.gen)
            except BaseException as failure:
                error = {'type': type(failure).__name__, 'message': str(failure),
                         'traceback': traceback.format_exc(), 'at': self.current_path}
                try:
                    self.owner('first_failure')
                except BaseException as nested:
                    error['owner_capture_error'] = repr(nested)
            finally:
                try:
                    self.n.finish()
                except BaseException as failure:
                    error = error or {'type': 'EOF_FAILURE', 'message': repr(failure), 'at': self.current_path}
            report = {'index': self.row['index'], 'name': self.row['name'],
                      'status': 'PASS' if error is None and not self.failures else 'FAIL',
                      'assertions': self.checks, 'exact_words': self.words,
                      'full_rng_bytes': self.rng_bytes, 'rollback_bytes': self.rollback_bytes,
                      'frames': self.n.s.frames, 'exit_code': self.n.s.exit_code,
                      'error': error, 'mismatches': self.failures, 'operations': self.operations,
                      'snapshots': self.snapshots, 'owner_records': self.owner_records,
                      'owner_blobs': self.owner_blobs, 'reference_event_count': len(self.row['events'])}
            support.write_json(self.directory / 'report.json', report)
            return report
    return ScalarCase


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'build-report', 'protocol', 'observer',
                 'reference', 'probe', 'support-reader', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    for name in ('source', 'native', 'observer'):
        ap.add_argument('--expected-' + name + '-sha256', required=True)
    ap.add_argument('--expected-functions', type=int, required=True)
    ap.add_argument('--expected-native-units', type=int, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    paths = [Path(__file__).resolve(), args.binary, args.source, args.build_report,
             args.protocol, args.protocol.with_name('owner_client.py'), args.observer,
             args.reference, args.probe, args.support_reader]
    inputs = {path.resolve(): path.read_bytes() for path in paths}
    pins = {args.binary.resolve(): args.expected_native_sha256,
            args.source.resolve(): args.expected_source_sha256,
            args.observer.resolve(): args.expected_observer_sha256,
            args.support_reader.resolve(): SUPPORT_READER,
            args.protocol.resolve(): PROTOCOL,
            args.protocol.with_name('owner_client.py').resolve(): CLIENT}
    for path, pin in pins.items():
        assert sha(inputs[path]) == pin, (path, sha(inputs[path]), pin)
    support = load(args.support_reader, 'immutable_actual14_scalar32_support')
    data = admit_reference(inputs[args.reference.resolve()], inputs[args.probe.resolve()], support)
    assert len(support.FUNCTIONS.findall(inputs[args.source.resolve()])) == args.expected_functions
    build = json.loads(inputs[args.build_report.resolve()])
    assert build['status'] == 'PASS' and build['source_sha256'] == args.expected_source_sha256
    assert build['native_sha256'] == args.expected_native_sha256 and build['descriptor_bytes'] == 672
    assert build['functions'] == args.expected_functions and len(build['units']) == args.expected_native_units
    for unit in build['units']:
        for suffix, key in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
            path = (args.build_report.parent / (unit['name'] + suffix)).resolve()
            raw = path.read_bytes()
            assert sha(raw) == unit[key], (path, sha(raw), unit[key])
            inputs[path] = raw
    args.output.mkdir(parents=True, exist_ok=False)
    common = {'schema': 'heptatorch.wrapped-f64-scalar-native32.v1',
              'reference_sha256': REFERENCE, 'probe_sha256': PROBE,
              'support_reader_sha256': SUPPORT_READER,
              'source_sha256': args.expected_source_sha256,
              'native_sha256': args.expected_native_sha256,
              'observer_sha256': args.expected_observer_sha256,
              'descriptor_bytes': 672, 'functions': args.expected_functions,
              'native_units': args.expected_native_units, 'expected_cases': 32,
              'reference_events': 192, 'reference_operation_totals': data['operation_totals'],
              'torch_executed_here': False, 'full_pytorch_rewrite_complete': False,
              'closed_schema_count': 0, 'pytorch_api_denominator': 2590,
              'fixed_configuration': data['environment_before'],
              'comparison_scope': 'Exact observed Tensor/Storage identity, physical/logical bytes, dtype/geometry/Version, original8B Saved scalar/status/default release, immediate AD edges, retained/leaf G, complete5056 MT, real EOF. All expected numeric words come from immutable actual32.',
              'weak_scope': 'Reference32 does not observe Weak expiry; none claimed. Weak handles used by the inherited gradient observer are cleanup-only.',
              'python_scalar_scope': 'Reference-side literal/unpack/post-operation Python-float words are admitted identical; native114 receives exactly those original8B bits. No Python wrapper or float constructor is claimed implemented.',
              'partial_prefix_scope': 'Native58 status and observed before/after gradients/Saved/Versions are compared. Additional native execution counters are raw diagnostics, not Torch-observed values.'}
    if args.prepare_only:
        report = {**common, 'status': 'PREPARED_NOT_RUN', 'native_executed': False,
                  'all_inputs_unchanged': all(path.read_bytes() == raw for path, raw in inputs.items()),
                  'inputs_sha256': {str(path): sha(raw) for path, raw in inputs.items()}}
        support.write_json(args.output / 'report.json', report)
        print(json.dumps({'status': report['status'], 'expected_cases': 32, 'reference_events': 192}))
        return 0
    copies = args.output / 'inputs'
    copies.mkdir()
    custody = {}
    for index, (path, raw) in enumerate(inputs.items()):
        dst = copies / ('%03d-' % index + path.name)
        dst.write_bytes(raw)
        custody[str(path)] = {'copy': str(dst.relative_to(args.output)), 'bytes': len(raw), 'sha256': sha(raw)}
    support.write_json(args.output / 'input-custody.json', custody)
    observer = load(args.observer, 'wrapped_scalar32_native_observer')
    Case = case_class(support)
    rows = []
    for row in data['rows']:
        result = Case(args, observer, data, row, args.output / ('case-%02d' % row['index'])).run()
        rows.append({key: result[key] for key in ('index', 'name', 'status', 'assertions', 'exact_words',
                    'full_rng_bytes', 'rollback_bytes', 'frames', 'exit_code', 'error', 'mismatches')})
        print(json.dumps({'case': result['index'], 'status': result['status'],
                          'frames': result['frames'], 'mismatches': len(result['mismatches']),
                          'error': None if result['error'] is None else result['error']['type']}), flush=True)
    unchanged = all(path.read_bytes() == raw for path, raw in inputs.items())
    report = {**common, 'status': 'PASS' if len(rows) == 32 and all(row['status'] == 'PASS' for row in rows) and unchanged else 'FAIL',
              'native_executed': True, 'cases': len(rows), 'rows': rows,
              'passed_cases': sum(row['status'] == 'PASS' for row in rows),
              'failed_cases': sum(row['status'] == 'FAIL' for row in rows),
              'all_inputs_unchanged': unchanged,
              'inputs_sha256': {str(path): sha(raw) for path, raw in inputs.items()}}
    for key in ('assertions', 'exact_words', 'full_rng_bytes', 'rollback_bytes', 'frames'):
        report[key] = sum(row[key] for row in rows)
    report['all_sessions_eof_exit_zero'] = all(row['exit_code'] == 0 for row in rows)
    support.write_json(args.output / 'report.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'passed_cases', 'failed_cases',
                                                 'frames', 'assertions', 'all_sessions_eof_exit_zero')}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
