"""Replay immutable Torch view/base lifetime observations on the native owner.

Private base Tensor identity is read from owned bytes without publishing a
handle or creating an extra native root. Storage weakness uses native Weak.
The producer supplies all values, updates and reference-count transitions.
"""
import argparse
import json
import lzma
from pathlib import Path
import struct
import traceback
from native_observer import Native, sha

REFERENCE = '2a9b7300b86df6931c6a35b3c6d175071095ecc093b3bc54eefd583bf57afa0a'
FIELDS = ('dtype', 'element_size', 'shape', 'strides', 'numel', 'storage_bytes',
          'storage_offset_elements', 'version', 'requires_grad', 'is_leaf',
          'is_view', 'logical_bytes_hex')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'source', 'protocol', 'reference', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    paths = (a.binary, a.source, a.protocol, a.protocol.with_name('owner_client.py'),
             a.reference, Path(__file__), Path(__file__).with_name('native_observer.py'))
    inputs = {path.resolve(): path.read_bytes() for path in paths}
    assert sha(inputs[a.reference.resolve()]) == REFERENCE
    actual = json.loads(inputs[a.reference.resolve()])
    assert actual['status'] == 'OBSERVED' and actual['cases'] == 6 and actual['probe_error_count'] == 0
    checks = actual['oracle_provenance']['checks']
    assert len(checks) == 11 and all(value is True for value in checks.values())
    rows = []
    for ref in actual['rows']:
        directory = out / f"case-{ref['index']:02d}"; directory.mkdir()
        n = Native(a.binary, a.source, a.protocol, directory)
        row = {'index': ref['index'], 'name': ref['name'], 'status': 'RUNNING',
               'checks': 0, 'raw_words': 0, 'rng_bytes': 0, 'rollback_bytes': 0,
               'differences': [], 'snapshots': [], 'owner_snapshots': [], 'role_checks': 0}
        tensor_ids, storage_ids = {}, {}

        def brief(value):
            if isinstance(value, bytes):
                return {'bytes': len(value), 'sha256': sha(value), 'first64': value[:64].hex()}
            return value

        def eq(label, got, wanted):
            row['checks'] += 1
            if got != wanted:
                row['differences'].append({'field': label, 'actual': brief(got), 'expected': brief(wanted)})

        def identity(table, key, value, label):
            if key in table:
                eq(label + '.stable', value, table[key])
            else:
                eq(label + '.distinct', value not in table.values(), True)
                table[key] = value

        def snapshot(expected, label, *, token=None, node=None):
            assert (token is None) != (node is None)
            value = n.tensor(token) if token is not None else n.memory().tensor(node)
            for key in FIELDS:
                eq(label + '.' + key, value[key], expected[key])
            identity(tensor_ids, expected['tensor_identity'], value['native_incarnation'], label + '.Tensor')
            identity(storage_ids, expected['storage_identity'], value['native_storage_id'], label + '.Storage')
            base_identity = expected['base_tensor_identity']
            wanted_base = None if base_identity is None else tensor_ids[base_identity]
            eq(label + '.actual_base_Tensor', value['native_base_incarnation'], wanted_base)
            row['raw_words'] += value['numel']
            row['snapshots'].append({'label': label, 'private_observation': node is not None,
                                     'node': value['native_node'], 'incarnation': value['native_incarnation'],
                                     'storage_id': value['native_storage_id'], 'version_id': value['native_version_id'],
                                     'base_incarnation': value['native_base_incarnation'], 'version': value['version']})
            return value

        def save_owner(label, owner=None):
            owner = n.s.owner() if owner is None else owner
            raw = owner['bytes']; target = directory / f"owner-{len(row['owner_snapshots']):03d}.bin.xz"
            packed = lzma.compress(raw, preset=3); target.write_bytes(packed)
            assert lzma.decompress(target.read_bytes()) == raw
            row['owner_snapshots'].append({'label': label, 'path': target.name, 'bytes': len(raw),
                                           'sha256': sha(raw), 'archive_sha256': sha(packed),
                                           'pointer': owner['pointer'], 'length': owner['length']})
            return owner

        def refused(commands, status, label):
            before = save_owner(label + '.before')
            got, records = n.w.parse(n.s.exchange(n.w.packet(commands)))
            eq(label + '.status', got, status); eq(label + '.records', records, [])
            after = save_owner(label + '.after')
            for key in ('pointer', 'length', 'live', 'bytes'):
                eq(label + '.owner_' + key, after[key], before[key])
            row['rollback_bytes'] += len(before['bytes'])

        def live_weak(expected, label):
            if expected:
                refused([(43, weak)], 6, label + '.expired_lock')
                return
            hold = n.one(43, weak)['value']
            eq(label + '.Storage_expired', False, expected)
            n.one(45, hold)

        def roles(label, count, public):
            memory = n.memory(); d = memory.node(root_node)
            for key, got, want in (('mask_or_base_target_count600', d[75], count),
                                   ('node_metadata_refs208', d[26], count),
                                   ('public_live200', d[25], int(public)),
                                   ('Tensor_row_live240', d[30], 1),
                                   ('descriptor_live304', d[38], 1),
                                   ('incarnation320', d[40], root_inc),
                                   ('temporary_audit616', d[77], 0)):
                eq(label + '.' + key, got, want); row['role_checks'] += 1
            eq(label + '.audit', n.one(53)['value'], 1)
            save_owner(label)

        def rng(label, expected):
            value = n.one(63, gen)['payload']; wanted = bytes.fromhex(expected)
            assert len(value) == len(wanted) == 5056
            eq(label, value, wanted); row['rng_bytes'] += len(value)

        try:
            gen = n.one(60, 1729)['value']; rng('initial.RNG', ref['rng_before_hex'])
            initial = ref['initial_base']
            base = n.create(2, initial['shape'], bytes.fromhex(initial['logical_bytes_hex']))
            root = snapshot(initial, 'initial_base', token=base)
            root_node, root_inc = root['native_node'], root['native_incarnation']
            weak = n.one(42, base)['value']
            if ref['form'] == 'as_strided':
                first = n.view(base, ref['first']['shape'], ref['first']['strides'], ref['first']['storage_offset_elements'])
            elif ref['form'] == 'transpose':
                first = n.one(6, base, 0, 1)['value']
            elif ref['form'] == 'view':
                shape = ref['first']['shape']; first = n.one(17, base, len(shape), *shape, *([0] * (4-len(shape))))['value']
            else:
                raise AssertionError(ref['form'])
            first_node = snapshot(ref['first'], 'first', token=first)['native_node']
            second = n.view(first, ref['second']['shape'], ref['second']['strides'], ref['second']['storage_offset_elements'])
            second_node = snapshot(ref['second'], 'second', token=second)['native_node']
            snapshot(ref['second_base_before_drop'], 'second_base_before_drop', node=root_node)
            roles('two_view_roots', 2, True)
            n.one(41, base)
            event = ref['after_base_local_drop']
            snapshot(event['first'], 'after_base_local_drop.first', token=first)
            snapshot(event['second'], 'after_base_local_drop.second', token=second)
            snapshot(event['second_base'], 'after_base_local_drop.second_base', node=root_node)
            live_weak(event['Storage_expired'], 'after_base_local_drop')
            roles('base_public_dropped', 2, False)
            refused([(13, base)], 1, 'stale_base_public_token')
            n.one(41, first)
            event = ref['after_first_view_drop']
            snapshot(event['second'], 'after_first_view_drop.second', token=second)
            snapshot(event['second_base'], 'after_first_view_drop.second_base', node=root_node)
            live_weak(event['Storage_expired'], 'after_first_view_drop')
            roles('first_view_dropped', 1, False)
            eq('first_view_really_released', n.memory().node(first_node)[38], 0)
            snapshot(ref['view_detach_inplace_before'], 'view_detach_inplace_before', token=second)
            eq('view_detach_inplace.accepted', ref['view_detach_inplace']['accepted'], False)
            refused([(101, second)], 5, 'view_detach_inplace')
            snapshot(ref['view_detach_inplace_after'], 'view_detach_inplace_after', token=second)
            eq('view_scalar_add.accepted', ref['view_scalar_add']['accepted'], True)
            result = n.one(106, second, 1)['value']; eq('view_scalar_add.returned_same_token', result, second)
            snapshot(ref['view_scalar_add']['returned'], 'view_scalar_add.returned', token=result)
            snapshot(ref['after_view_mutation']['second'], 'after_view_mutation.second', token=second)
            snapshot(ref['after_view_mutation']['second_base'], 'after_view_mutation.second_base', node=root_node)
            roles('after_shared_mutation', 1, False)
            held = n.one(19, second)['value']
            snapshot(ref['detached_before_last_view_drop'], 'detached_before_last_view_drop', token=held)
            n.one(41, second)
            snapshot(ref['after_last_view_drop']['detached'], 'after_last_view_drop.detached', token=held)
            live_weak(ref['after_last_view_drop']['Storage_expired'], 'after_last_view_drop')
            memory = n.memory()
            for label, node in (('root', root_node), ('second', second_node)):
                d = memory.node(node)
                eq('after_last_view_drop.' + label + '.descriptor_released', d[38], 0)
                eq('after_last_view_drop.' + label + '.Tensor_row_released', d[30], 0)
                eq('after_last_view_drop.' + label + '.base_edge_cleared', list(d[73:75]), [0, 0])
            eq('after_last_view_drop.audit', n.one(53)['value'], 1); save_owner('after_last_view_drop')
            eq('detached_detach_inplace.accepted', ref['detached_detach_inplace']['accepted'], True)
            result = n.one(101, held)['value']; eq('detached_detach_inplace.returned_same_token', result, held)
            snapshot(ref['detached_detach_inplace']['returned'], 'detached_detach_inplace.returned', token=result)
            snapshot(ref['detached_after_detach_inplace'], 'detached_after_detach_inplace', token=held)
            n.one(41, held)
            live_weak(ref['after_all_strong_drop']['Storage_expired'], 'after_all_strong_drop')
            eq('after_all_strong_drop.audit', n.one(53)['value'], 1); save_owner('after_all_strong_drop')
            rng('final.RNG', ref['rng_after_hex'])
            n.one(44, weak); n.one(66, gen)
            eq('final.audit', n.one(53)['value'], 1)
            stats = struct.unpack('<12Q', n.one(54)['payload'])
            eq('final.live_counts', list(stats[5:11]), [0, 1, 0, 0, 0, 1])
            row['status'] = 'PASS' if not row['differences'] else 'FAIL'
        except BaseException as error:
            row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
        finally:
            row['native_frames'] = n.s.frames
            try:
                n.finish(); row['EOF_exit_code'] = n.s.exit_code
            except BaseException as error:
                row.update(status='FAIL', transport_error=repr(error))
            (directory/'report.json').write_text(json.dumps(row, indent=2)+'\n'); rows.append(row)
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    report = {'schema': 'heptatorch.int64-view-base6-native-replay.v1',
              'status': 'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
              'cases': len(rows), 'passed': sum(r['status']=='PASS' for r in rows), 'rows': rows,
              'native_frames': sum(r['native_frames'] for r in rows),
              'exact_checks': sum(r['checks'] for r in rows), 'raw_words': sum(r['raw_words'] for r in rows),
              'rng_bytes': sum(r['rng_bytes'] for r in rows), 'rollback_bytes': sum(r['rollback_bytes'] for r in rows),
              'native_role_checks': sum(r['role_checks'] for r in rows),
              'reference_sha256': REFERENCE, 'source_sha256': sha(inputs[a.source.resolve()]),
              'native_sha256': sha(inputs[a.binary.resolve()]), 'all_inputs_unchanged': True,
              'inputs_sha256': {str(path): sha(raw) for path, raw in inputs.items()},
              'native_executed': True, 'torch_executed_here': False,
              'private_base_observer_publishes_Tensor': False, 'native_owner_mutation_by_host': False,
              'Python_wrapper_weakref_identity_claim': False,
              'actual_observations_and_native_internal_role_checks_counted_separately': True,
              'full_pytorch_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('status','cases','passed','native_frames','exact_checks','raw_words','rollback_bytes')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
