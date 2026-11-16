"""Independent native private-capability, migration, and rollback controls."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import signal
import struct
import traceback
from review_native_objects import Review, sha, tensor_command, fbits


def words(record):
    assert len(record['payload']) % 8 == 0
    return struct.unpack('<'+'Q'*(len(record['payload'])//8), record['payload'])


def native_table(owner_bytes, kind, object_state_base):
    """Read own-child state as a debugger; no pointer or state is ever injected."""
    slots, = struct.unpack_from('<Q', owner_bytes, object_state_base+16)
    result = []
    for slot in range(1, slots+1):
        at = object_state_base+128+((kind-1)*slots+slot-1)*64
        fields = struct.unpack_from('<8Q', owner_bytes, at)
        if fields[1] == 1:
            result.append(((fields[0] << 32) | (kind << 24) | slot, fields))
    return result


def private_guards(review):
    r = review.exchange
    x, = r([tensor_command([2], [1, 2], True)], label='leaf x')
    x = x['value']
    y, = r([(2, 3, x, x)], label='square with two private Saved')
    y = y['value']
    loss, = r([(4, 7, y)], label='loss')
    loss = loss['value']
    saved, weak = r([(46, x), (42, x)], label='public Saved and Weak')
    saved, weak = saved['value'], weak['value']
    hold, = r([(43, weak)], label='public Hold')
    hold = hold['value']
    owner = review.session.owner()
    internal_saved = [handle for handle, fields in native_table(owner['bytes'], 5, review.object_state_base) if fields[7] == 0]
    assert len(internal_saved) == 2
    public_record, = [fields for handle, fields in native_table(owner['bytes'], 5, review.object_state_base) if handle == saved]
    layout_hold = public_record[7]
    private_holds = [handle for handle, fields in native_table(owner['bytes'], 6, review.object_state_base) if fields[3] == 1]
    assert layout_hold in private_holds and len(private_holds) >= 4
    tested_holds = sorted(set(private_holds[:3] + [layout_hold]))
    for handle in internal_saved:
        for command in [(47, handle), (48, handle), (49, handle, 0, 0)]:
            r([command], 1, 'confirmed private AD Saved is not a public Saved')
    for handle in tested_holds:
        for command in [(45, handle), (50, handle, 0, 0)]:
            r([command], 1, 'confirmed private Hold is not a public Hold')
    meta, = r([(13, x)], label='leaf metadata')
    info = words(meta)
    sid, vid = info[11:13]
    for handle in [0, 1, sid, vid, weak, saved, hold, x+(1 << 32), (x & ~0xffffff)|0xffffff]:
        r([(13, handle)], 1, 'raw/wrong-kind/stale/out-of-range Tensor handle')
    r([(7, loss)], label='backward after all guard refusals')
    gradient, = r([(11, x)], label='gradient remains usable')
    review.bits(gradient, [fbits(2), fbits(4)], 'guard-preserved gradient')
    r([(41, x)], label='drop public leaf but retain graph-owned Tensor row')
    # The row still exists because AccumulateGrad needs it, yet the public ID
    # must be revoked. This is stronger than merely passing a stale generation.
    owner = review.session.owner()
    assert any(handle == x for handle, fields in native_table(owner['bytes'], 1, review.object_state_base))
    for command in [(13, x), (10, x), (12, x, 0, fbits(7)), (41, x)]:
        r([command], 1, 'graph-retained leaf ID remains publicly revoked')
    audit, = r([(53,)], label='private-guard final audit')
    review.eq(audit['value'], 1, 'private-guard final audit')
    return {'confirmed_private_AD_Saved': len(internal_saved),
            'confirmed_private_Holds_tested': len(tested_holds),
            'graph_retained_leaf_row_observed_after_public_revocation': True}


def saved_versions(review):
    r = review.exchange
    x, = r([tensor_command([2, 2], [1, 2, 3, 4])], label='source')
    x = x['value']
    t, = r([(6, x, 0, 1)], label='transposed source')
    saved, = r([(46, t['value'])], label='public Saved layout')
    saved = saved['value']
    r([(12, x, 0, fbits(10))], label='shared Version change')
    for command in [(47, saved), (49, saved, 0, 0), (49, saved, 0, 4), (56, saved)]:
        r([command], 6, 'stale public Saved refusal including zero-length read')
    r([(48, saved)], label='stale Saved may still be dropped')
    r([(47, saved)], 1, 'dropped Saved handle does not become valid')
    audit, = r([(53,)], label='Saved-version final audit')
    review.eq(audit['value'], 1, 'Saved-version final audit')
    return {'stale_saved_checks': 4, 'zero_length_read_checks_version': True}


def migration(review):
    r = review.exchange
    x, = r([tensor_command([2, 3], [1, -2, 3, -4, 5, -6], True)], label='small gradient leaf')
    x = x['value']
    t, = r([(6, x, 0, 1)], label='transposed alias')
    t = t['value']
    square, = r([(2, 3, x, x)], label='saved-value graph')
    square = square['value']
    saved, weak = r([(46, t), (42, x)], label='layout snapshot and Weak')
    saved, weak = saved['value'], weak['value']
    hold, = r([(43, weak)], label='storage Hold')
    hold = hold['value']
    mx, mt, stats = r([(13, x), (13, t), (54,)], label='identities before growth')
    initial_meta, alias_meta, initial_stats = words(mx), words(mt), words(stats)
    ids = r([tensor_command([0], []) for _ in range(300)], label='300 simultaneous independent empty Tensor objects')
    assert len({record['value'] for record in ids}) == 300
    after_tables, = r([(54,)], label='table growth witness')
    table_stats = words(after_tables)
    assert table_stats[0] > initial_stats[0] and table_stats[0] >= 512
    big, = r([(38, 2, 0, 512, 1024, 0, 0, fbits('1/8'))], label='two-MiB data Storage')
    big = big['value']
    mx, mt, stats, tail = r([(13, x), (13, t), (54,), (39, big, 524285, 3)], label='identities after table and pool growth')
    final_stats = words(stats)
    review.eq(words(mx), initial_meta, 'leaf metadata remains exact after migration')
    review.eq(words(mt), alias_meta, 'alias metadata remains exact after migration')
    assert final_stats[1] > table_stats[1] and final_stats[3] > initial_stats[3]
    review.bits(tail, [fbits('1/8')]*3, 'two-MiB tail')
    logical, physical = r([(49, saved, 0, 6), (50, hold, 0, 6)], label='Saved and Hold across relocation')
    review.bits(logical, [fbits(v) for v in [1, -4, -2, 5, 3, -6]], 'Saved logical transpose after migration')
    review.bits(physical, [fbits(v) for v in [1, -2, 3, -4, 5, -6]], 'Hold physical data after migration')
    seed, = r([tensor_command([2, 3], [1, -2, 3, -4, '1/2', 2])], label='nonuniform seed')
    r([(29, square, seed['value']), (41, seed['value'])], label='nonuniform VJP after migration')
    gradient, = r([(11, x)], label='migrated leaf gradient')
    review.bits(gradient, [fbits(v) for v in [2, 8, 18, 32, 5, -24]], 'independent exact VJP')
    r([(9, x, fbits('1/4'))], label='SGD mutates migrated Storage and Version')
    updated, meta = r([(10, x), (13, x)], label='updated parameter')
    review.bits(updated, [fbits(v) for v in ['1/2', -4, '-3/2', -12, '15/4', 0]], 'SGD after migration')
    review.eq(words(meta)[13], 1, 'SGD absolute Version')
    r([(47, saved)], 6, 'pre-growth public Saved detects post-growth mutation')
    # 4096x4096 is within shape and numel limits. The process is deliberately
    # admitted under RLIMIT_AS=64MiB, so this real allocation cannot be satisfied.
    r([(12, x, 0, fbits(123)), (38, 2, 0, 4096, 4096, 0, 0, 0)], 3,
      'real address-limit allocation refusal rolls back earlier mutation')
    r([(38, 1, 0, 262145, 0, 0, 0, 0)], 3, 'shape refusal leaves migrated state unchanged')
    final_values, final_grad, audit = r([(10, x), (11, x), (53,)], label='usable state after allocation refusal')
    review.bits(final_values, [fbits(v) for v in ['1/2', -4, '-3/2', -12, '15/4', 0]], 'parameter after refused allocation')
    review.bits(final_grad, [fbits(v) for v in [2, 8, 18, 32, 5, -24]], 'gradient after refused allocation')
    review.eq(audit['value'], 1, 'migration final audit')
    return {'initial_stats': initial_stats, 'table_growth_stats': table_stats,
            'pool_growth_stats': final_stats, 'simultaneous_new_empty_Tensors': 300,
            'large_tensor_data_bytes': 2 << 20, 'address_space_limit_bytes': 64 << 20,
            'independent_storage_and_version_handles_preserved': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--transport-observer', type=Path, required=True)
    parser.add_argument('--object-state-base', type=int, default=8192,
                        help='Explicit debugger layout: 8192 for candidate06; 139264 for candidate08.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = sha(args.binary.read_bytes())
    assert before == args.expected_sha256
    root = Path(__file__).resolve().parent
    paths = [Path(__file__), root/'review_native_objects.py', root/'reference_model.py', args.transport_observer]
    sources = {path.name: sha(path.read_bytes()) for path in paths}
    executed = args.output/'executed-sources'
    executed.mkdir()
    for path in paths:
        (executed/path.name).write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location('object_boundary_transport', args.transport_observer)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    results = []
    for name, fn in [('private-capabilities', private_guards), ('public-Saved-Version-refusal', saved_versions),
                     ('table-pool-migration-and-allocation-rollback', migration)]:
        directory = args.output/name
        session = transport.OwnerSession(args.binary, directory=directory, address_limit=64 << 20)
        review = Review(session)
        review.object_state_base = args.object_state_base
        result = {'case': name, 'status': 'FAIL'}
        try:
            result['witness'] = fn(review)
            session.finish()
            result['status'] = 'PASS'
        except Exception as exc:
            result.update(error=repr(exc), traceback=traceback.format_exc())
            if session.poll() is None:
                os.kill(session.pid, signal.SIGKILL)
            try:
                session.finish(expected=-signal.SIGKILL)
            except Exception:
                pass
        result.update(frames=review.frames, exact_scalar_or_binary32_fields=review.compared_fields,
                      entire_owner_rollback_bytes=review.rollback_bytes)
        (directory/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        results.append(result)
    assert sha(args.binary.read_bytes()) == before
    assert sources == {path.name: sha(path.read_bytes()) for path in paths}
    report = {'schema': 'heptatorch.numeric-object-boundary-review.v1',
              'status': 'PASS' if all(row['status']=='PASS' for row in results) else 'FAIL',
              'native_executable_sha256': before, 'source_hashes': sources, 'rows': results,
              'debugger_object_state_base': args.object_state_base,
              'native_target_execution': True, 'pinned_pytorch_executed': False,
              'cases': len(results), 'frames': sum(len(row['frames']) for row in results),
              'exact_scalar_or_binary32_fields': sum(row['exact_scalar_or_binary32_fields'] for row in results),
              'entire_owner_rollback_bytes': sum(row['entire_owner_rollback_bytes'] for row in results),
              'observer_role': 'Own-child memory read only; no injected pointer, capability or modified state.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','cases','frames','exact_scalar_or_binary32_fields','entire_owner_rollback_bytes']}))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
