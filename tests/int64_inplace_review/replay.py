#!/usr/bin/env python3
"""Replay the frozen actual 40-call observation against a source-pinned image.

Native wire status categories are compared, not Python exception text parity.
Every accepted-call raw value, complete Storage, Version and identity is exact.
Refusals must preserve all native owned bytes, handles and complete RNG state.
"""
import argparse
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
import struct
import traceback

REFERENCE = '107188393d611c7b53df948c9404a4aca6550f9dc94b91499cdc5fff559c5475'
OPS = {'add': 1, 'sub': 2, 'mul': 3, 'bitwise_and': 4,
       'bitwise_or': 5, 'bitwise_xor': 6, 'true_divide': 7}
STATUS = {
    "Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.": 7,
    'The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1': 2,
    "output with shape [1, 3] doesn't match the broadcast shape [2, 3]": 2,
    "result type Float can't be cast to the desired output type Long": 7,
    'unsupported operation: more than one element of the written-to tensor refers to a single memory location. Please clone() the tensor before performing the operation.': 2,
    'unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation.': 2,
}
FIELDS = ('dtype', 'element_size', 'shape', 'strides', 'numel', 'storage_offset_elements',
          'storage_bytes', 'version', 'requires_grad', 'is_leaf', 'is_view', 'grad_fn',
          'logical_bytes', 'logical_bytes_hex', 'storage_bytes_hex')
sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'source', 'protocol', 'observer', 'reference', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--expected-source-sha256', required=True)
    p.add_argument('--expected-native-sha256', required=True)
    a = p.parse_args()
    files = [Path(__file__), a.binary, a.source, a.protocol, a.protocol.with_name('owner_client.py'),
             a.observer, a.reference]
    inputs = {path.resolve(): path.read_bytes() for path in files}
    assert sha(inputs[a.source.resolve()]) == a.expected_source_sha256
    assert sha(inputs[a.binary.resolve()]) == a.expected_native_sha256
    assert sha(inputs[a.reference.resolve()]) == REFERENCE
    spec = importlib.util.spec_from_file_location('independent_int64_native_observer', a.observer)
    observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)
    actual = json.loads(inputs[a.reference.resolve()])
    assert actual['status'] == 'OBSERVED' and actual['cases'] == 40 and actual['probe_error_count'] == 0
    assert len(actual['oracle_provenance']['checks']) == 11 and all(v is True for v in actual['oracle_provenance']['checks'].values())
    assert {r['observation']['operation']['error']['message'] for r in actual['rows']
            if not r['observation']['operation']['accepted']} == set(STATUS)
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for ref in actual['rows']:
        out = a.output / f"case-{ref['index']:02d}"; out.mkdir()
        n = observer.Native(a.binary, a.source, a.protocol, out)
        row = {'index': ref['index'], 'name': ref['name'], 'status': 'RUNNING',
               'checks': 0, 'raw_words': 0, 'rng_bytes': 0, 'rollback_bytes': 0,
               'differences': [], 'owner_snapshots': [], 'native_executed': True}
        roots, identities, storages = {}, {}, {}
        obs = ref['observation']

        def brief(x):
            if isinstance(x, bytes):
                return {'bytes': len(x), 'sha256': sha(x), 'first64': x[:64].hex()}
            return x

        def eq(label, got, want):
            row['checks'] += 1
            if got != want:
                row['differences'].append({'field': label, 'actual': brief(got), 'expected': brief(want)})

        def identity(table, key, value, label):
            if key in table:
                eq(label + '.stable', value, table[key])
            else:
                eq(label + '.distinct', value not in table.values(), True)
                table[key] = value

        def snapshot(token, expected, label):
            value = n.tensor(token)
            for key in FIELDS:
                eq(label + '.' + key, value[key], expected[key])
            identity(identities, expected['tensor_identity'], value['native_incarnation'], label + '.Tensor')
            identity(storages, expected['storage_identity'], value['native_storage_id'], label + '.Storage')
            base = expected['base_tensor_identity']
            eq(label + '.base_Tensor', value['native_base_incarnation'], None if base is None else identities[base])
            row['raw_words'] += value['numel']

        def rng(key, label):
            state = actual['states'][key]; wanted = bytes.fromhex(state['hex'])
            assert state['bytes'] == len(wanted) == 5056 and sha(wanted) == key
            eq(label, n.one(63, gen)['payload'], wanted)
            row['rng_bytes'] += len(wanted)

        def save_owner(label, owner):
            raw = owner['bytes']; packed = lzma.compress(raw, preset=3)
            path = out / (label + '.bin.xz'); path.write_bytes(packed)
            decoder = lzma.LZMADecompressor(); restored = decoder.decompress(packed)
            assert restored == raw and decoder.eof and not decoder.unused_data
            row['owner_snapshots'].append({'path': path.name, 'bytes': len(raw),
                                          'sha256': sha(raw), 'archive_sha256': sha(packed)})

        try:
            gen = n.one(60, 1729)['value']; rng(ref['state_before'], 'initial.RNG')
            for construction in obs['constructions']:
                name, expected = construction['name'], construction['returned']
                if construction['kind'] == 'tensor':
                    code = {'torch.float32': 0, 'torch.bool': 1, 'torch.int64': 2}[expected['dtype']]
                    token = n.create(code, expected['shape'], bytes.fromhex(expected['logical_bytes_hex']))
                else:
                    assert construction['kind'] == 'as_strided'
                    token = n.view(roots[construction['parent']], construction['shape'],
                                   construction['strides'], construction['offset'])
                roots[name] = token
                snapshot(token, expected, 'construction.' + name)
            eq('root_names', list(roots), obs['all_strong_roots'])
            for name, expected in obs['before'].items():
                snapshot(roots[name], expected, 'before.' + name)
            rng(obs['state_before_operation'], 'before_operation.RNG')
            before_owner = n.s.owner()
            if obs['overload'] == 'Scalar':
                assert obs['op'] == 'add' and obs['scalar'] == 1
                command = (106, roots['x'], 1)
            else:
                assert obs['overload'] == 'Tensor' and obs['scalar'] is None
                command = (104, OPS[obs['op']], roots['x'], roots['y'])
            status, records = n.w.parse(n.s.exchange(n.w.packet([command])))
            expected_status = 0 if obs['operation']['accepted'] else STATUS[obs['operation']['error']['message']]
            row['expected_wire_status'] = expected_status
            row['observed_wire_status'] = status
            row['actual_reference_error'] = obs['operation']['error']
            eq('operation.status', status, expected_status)
            if status == 0:
                eq('operation.record_count', len(records), 1)
                eq('operation.returned_same_public_Tensor', records[0]['value'], roots['x'])
                if obs['returned'] is not None:
                    snapshot(records[0]['value'], obs['returned'], 'returned')
            else:
                eq('operation.failed_records', records, [])
                after_owner = n.s.owner()
                for key in ('pointer', 'length', 'live', 'bytes'):
                    eq('operation.failure_preserves_owner_' + key, after_owner[key], before_owner[key])
                row['rollback_bytes'] += len(before_owner['bytes'])
                save_owner('before-refusal', before_owner); save_owner('after-refusal', after_owner)
            rng(obs['state_after_operation'], 'after_operation.RNG')
            for name, expected in obs['after'].items():
                snapshot(roots[name], expected, 'after.' + name)
            rng(ref['state_after'], 'final.RNG')
            eq('audit_after_operation', n.one(53)['value'], 1)
            for token in roots.values():
                n.one(41, token)
            n.one(66, gen)
            eq('audit_after_drop', n.one(53)['value'], 1)
            statistics = struct.unpack('<12Q', n.one(54)['payload'])
            eq('remaining_owned_live_counts', list(statistics[5:11]), [0, 1, 0, 0, 0, 1])
            row['status'] = 'PASS' if not row['differences'] else 'FAIL'
        except BaseException as error:
            row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
        finally:
            row['native_frames'] = n.s.frames
            try:
                row['transport'] = n.finish()
            except BaseException as error:
                row.update(status='FAIL', transport_error=repr(error))
            (out / 'report.json').write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    report = {'schema': 'heptatorch.int64-inplace40-native-replay.v1',
              'status': 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL',
              'cases': len(rows), 'passed': sum(r['status'] == 'PASS' for r in rows), 'rows': rows,
              'native_frames': sum(r['native_frames'] for r in rows),
              'exact_checks': sum(r['checks'] for r in rows),
              'raw_words': sum(r['raw_words'] for r in rows), 'rng_bytes': sum(r['rng_bytes'] for r in rows),
              'rollback_bytes': sum(r['rollback_bytes'] for r in rows),
              'reference_sha256': REFERENCE, 'source_sha256': a.expected_source_sha256,
              'native_sha256': a.expected_native_sha256, 'native_executed': True,
              'Torch_executed_here': False, 'source_files_unchanged': True,
              'inputs_sha256': {str(path): sha(raw) for path, raw in inputs.items()},
              'exception_text_parity_claimed': False,
              'refusal_contract': 'Exact fixed Torch messages map to existing native status categories; detailed Python exception text and class frontend remain unimplemented.',
              'full_pytorch_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'passed', 'native_frames', 'exact_checks')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
