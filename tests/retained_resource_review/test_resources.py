#!/usr/bin/env python3
"""Independent native retained-gradient ownership and resource controls.

The host only frames fixed requests, compares exact dyadic words and reads its
own blocked child's committed owner. It imports no runtime/authoring module and
does not execute Torch. The separate fixed actual12 replay is not repeated.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import struct
import traceback


CONTROLS = '43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
OBSERVER = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
REFERENCE = 'ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path, expected):
    raw = path.read_bytes()
    assert sha(raw) == expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert path.read_bytes() == raw
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'controls', 'observer', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--expected-native-sha256', required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {p.resolve(): p.read_bytes() for p in (Path(__file__), args.binary, args.source, args.controls, args.observer, args.reference)}
    assert sha(args.binary.read_bytes()) == args.expected_native_sha256
    assert sha(args.source.read_bytes()) == args.expected_source_sha256
    assert sha(args.reference.read_bytes()) == REFERENCE
    reference = json.loads(args.reference.read_bytes())
    assert reference['status'] == 'OBSERVED' and reference['expected_variants'] == 12
    assert len(reference['oracle_provenance']['checks']) == 11
    assert all(value is True for value in reference['oracle_provenance']['checks'].values())
    codec = load('retained_resource_codec', args.controls.resolve(), CONTROLS)
    observer = load('retained_resource_observer', args.observer.resolve(), OBSERVER)
    report = {'schema': 'heptatorch.retained-gradient-independent-resources.v1', 'status': 'RUNNING',
              'source_sha256': args.expected_source_sha256, 'native_sha256': args.expected_native_sha256,
              'reference_sha256': REFERENCE, 'test_sha256': sha(Path(__file__).read_bytes()),
              'inputs_sha256': {str(p): sha(raw) for p, raw in inputs.items()},
              'native_executed': True, 'Torch_executed_here': False,
              'scope': 'Additional resource/ownership controls, not a repetition or expansion of the separate actual12 numerical qualification.',
              'state_format': 'HTRET001', 'backing_format': 'HTORET01', 'descriptor_bytes': 576,
              'cases': [], 'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def words(c, *command):
        return codec.words(c.one(*command))

    def full(c, shape, value, requires=False):
        return c.make(38, len(shape), int(requires), *shape, *([0] * (4 - len(shape))), codec.bits(value))

    def blob(values):
        return codec.Blob(struct.pack('<' + 'Q' * len(values), *values))

    def optimizer(c, parameter):
        return c.make(71, 0, blob([codec.bits(.125), codec.bits(.5), 0, 0, 0, 0]),
                      blob([1, 1, 0, 0, 0, 0, 0, 0, 0, parameter]))

    def snapshot(c):
        owner = c.session.owner()
        raw = owner['bytes']
        q = lambda at: struct.unpack_from('<Q', raw, at)[0]
        assert raw[64:72] == b'HTRET001' and raw[139264:139272] == b'HTORET01'
        assert all(q(at) == 0 for at in (112, 3968, 3976, 3984, 3992, 4000, 4008))
        slots = q(139264 + 16)
        pool = 139264 + 144 + 552 * slots
        live = [0] * 8
        tensors = {}
        for kind in range(1, 9):
            for slot in range(1, slots + 1):
                at = 139264 + 128 + ((kind - 1) * slots + slot - 1) * 64
                if q(at + 8) == 0:
                    continue
                assert q(at + 8) == 1
                live[kind - 1] += 1
                if kind == 1:
                    node = q(at + 48)
                    flags = q(at + 56)
                    assert flags in (0, 1), ('committed temporary retain root', slot, node, flags)
                    assert node not in tensors
                    tensors[node] = {'slot': slot, 'generation': q(at), 'flags': flags}
        nodes = {}
        for node in range(1, q(8)):
            page = q(4096 + (node >> 8) * 8)
            if not page:
                continue
            assert page >> 63
            at = pool + (page & ((1 << 63) - 1)) + (node & 255) * 576
            assert all(q(at + field) == 0 for field in range(448, 576, 8)), ('committed incoming/reserve residue', node)
            if q(at + 304) == 0:
                continue
            assert all(q(at + field) == 0 for field in (24, 248, 336, 368, 376, 416, 424, 432, 440)), ('committed GC/stage residue', node)
            row = {field: q(at + field) for field in (0, 8, 32, 40, 192, 200, 208, 216, 224, 240, 304, 312, 320, 360, 384, 392, 400)}
            assert bool(row[240]) == (node in tensors)
            assert (row[360] == 0) == (row[384] == 0)
            row['flags'] = tensors.get(node, {}).get('flags', None)
            nodes[node] = row
        for node, row in nodes.items():
            if row[384]:
                gradient = nodes[row[384]]
                assert gradient[320] == row[392] and gradient[400] > 0 and gradient[240] != 0
        c.comparisons += len(nodes) + len(tensors)
        return {'owner_bytes': owner['length'], 'owner_sha256': sha(raw), 'next_node': q(8),
                'slots': slots, 'live_by_kind': live, 'nodes': nodes, 'tensor_rows': tensors}

    def node_of(c, token):
        state = snapshot(c)
        matches = [n for n, row in state['nodes'].items() if row[192] == token and row[200] == 1]
        assert len(matches) == 1, (token, matches)
        return matches[0]

    def audit(c):
        c.audit()
        return snapshot(c)

    def backward(c, target, seed=0, retain=0):
        row = c.one(58, target, seed, retain)
        values = codec.words(row)
        assert row['value'] == 0 and len(values) == 4 and values[0] == 0
        return values

    def partial(c, commands, ordinal, error, completed, leaves, op):
        request = codec.encode(commands)
        raw = c.session.exchange(request)
        status, records = codec.decode(raw)
        assert status == 0 and len(records) == ordinal + 1
        assert [r['code'] for r in records] == [v[0] for v in commands[:ordinal + 1]]
        values = codec.words(records[-1])
        assert records[-1]['value'] == error and values == (error, completed, leaves, op), values
        c.frames.append({'frame': c.session.frames, 'label': 'structured runtime exception stops suffix',
                         'expected_status': 0, 'actual_status': 0, 'request_sha256': sha(request),
                         'response_sha256': sha(raw), 'structured_result': list(values),
                         'commands_submitted': len(commands), 'commands_returned': len(records)})
        c.comparisons += 4
        return list(values)

    def case(name, body):
        directory = out / name
        session = observer.OwnerSession(args.binary, directory=directory, address_limit=128 << 20)
        c = codec.Check(session)
        row = {'name': name, 'status': 'RUNNING'}
        report['cases'].append(row)
        save()
        try:
            row['details'] = body(c, directory)
            row['final_committed_state'] = audit(c)
            row['status'] = 'PASS'
        except Exception as error:
            row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
            try:
                state = session.owner()
                (directory / 'failure.owner.bin').write_bytes(state['bytes'])
                row['failure_owner'] = {'bytes': state['length'], 'sha256': sha(state['bytes'])}
            except Exception as observation_error:
                row['owner_observation_error'] = repr(observation_error)
        finally:
            try:
                row['transport'] = session.finish()
            except Exception as error:
                row.update(status='FAIL', close_error=repr(error))
            row.update(frames=session.frames, exact_checks=c.comparisons,
                       whole_owner_rollback_bytes=c.rollback_bytes, frame_records=c.frames)
            save()

    def weak(c, directory, held):
        x = c.new([2], [1., 2.], True)
        coefficient = c.new([], [2.])
        one = c.new([], [1.])
        q = c.make(2, 3, x, coefficient)
        child = c.make(2, 1, q, one)
        seed = c.new([2], [1., 1.])
        qn = node_of(c, q)
        before = snapshot(c)['nodes'][qn]
        metadata = words(c, 13, q)
        c.one(93, q)
        c.one(93, q)
        assert c.make(94, q) == 1
        after = snapshot(c)['nodes'][qn]
        assert [before[k] for k in (208, 312, 400)] == [after[k] for k in (208, 312, 400)]
        assert metadata == words(c, 13, q)
        aliases = [c.make(18, q), c.make(19, q), c.make(17, q, 1, 2, 0, 0, 0)]
        assert all(c.make(94, alias) == 0 for alias in aliases)
        c.run([(41, alias) for alias in aliases])
        held_gradient = None
        if held:
            backward(c, child, seed, 1)
            held_gradient = c.make(91, q)
            c.values(held_gradient, [1., 1.])
        weak_q = c.make(42, q)
        c.one(41, q)
        c.expiry(weak_q, True)
        state = audit(c)
        assert qn in state['nodes'] and state['nodes'][qn][240] == 0 and qn not in state['tensor_rows']
        backward(c, child, seed, 1)
        state = audit(c)
        assert qn in state['nodes'] and state['nodes'][qn][240] == 0 and state['nodes'][qn][384] == 0
        c.values(x, [4., 4.] if held else [2., 2.], gradient=True)
        c.one(94, q, status=1)
        if held_gradient:
            c.values(held_gradient, [1., 1.])
            assert words(c, 13, held_gradient)[13] == 0
            c.one(41, held_gradient)
        c.run([(41, child), (44, weak_q)])
        assert qn not in audit(c)['nodes']
        return {'weak_hook_did_not_add_strong_refs': True, 'fresh_alias_flags_are_clear': True,
                'expired_Tensor_not_recreated_from_live_metadata': True,
                'held_old_gradient_unchanged': held, 'registration_preserves_data_Version': True}

    case('weak-retain-never-materialized', lambda c, d: weak(c, d, False))
    case('weak-retain-with-held-old-gradient', lambda c, d: weak(c, d, True))

    def version_prefix(c, directory):
        x = c.new([2], [1., 2.], True)
        a = c.new([], [2.]); b = c.new([], [3.])
        ancestor = c.make(2, 3, x, a)
        q = c.make(2, 3, ancestor, b)
        seed = c.new([2], [1., 1.]); marker = c.new([], [0.])
        c.run([(93, ancestor), (93, q), (95, b, codec.bits(.5))])
        qn, an = node_of(c, q), node_of(c, ancestor)
        before = snapshot(c)
        c.one(91, ancestor)
        payload = partial(c, [(95, marker, codec.bits(.5)), (58, q, seed, 0), (95, marker, codec.bits(.5))], 1, 6, 0, 0, 3)
        c.values(marker, [.5]); assert words(c, 13, marker)[13] == 1
        after = audit(c); gn = after['nodes'][qn][384]
        assert gn > qn and set(after['nodes']) - set(before['nodes']) == {gn}
        assert after['nodes'][an][384] == 0 and c.make(91, x) == 0
        g = c.make(91, q); c.values(g, [1., 1.]); assert words(c, 13, g)[13] == 0
        weak_old = c.make(42, g); old_incarnation = node_of(c, g); c.one(41, g)
        partial(c, [(58, q, seed, 0), (95, marker, codec.bits(.5))], 0, 6, 0, 0, 3)
        c.expiry(weak_old, True); c.one(44, weak_old)
        g = c.make(91, q); c.values(g, [2., 2.]); assert words(c, 13, g)[13] == 0
        assert node_of(c, g) != old_incarnation
        c.one(41, g)
        stable = audit(c)['live_by_kind']
        for expected in (3., 4., 5.):
            partial(c, [(58, q, seed, 0)], 0, 6, 0, 0, 3)
            state = audit(c)
            assert state['live_by_kind'] == stable and state['nodes'][an][384] == 0
            g = c.make(91, q); c.values(g, [expected, expected]); assert words(c, 13, g)[13] == 0; c.one(41, g)
        c.values(marker, [.5])
        return {'first_structured_result': payload, 'only_visited_prehook_commits': True,
                'unvisited_fresh_nodes_above_output_released': True, 'five_error_retries_keep_live_counts_stable': True,
                'completed_node_count_excludes_prehook': True, 'suffix_skipped_and_prefix_committed': True}

    case('version-error-prefix-and-unvisited-reserves', version_prefix)

    def released_prefix(c, directory):
        x = c.new([2], [1., 2.], True); one = c.new([], [1.])
        ancestor = c.make(2, 1, x, one); q = c.make(2, 3, ancestor, ancestor)
        seed = c.new([2], [1., 1.])
        c.run([(93, ancestor), (93, q)])
        backward(c, q, seed, 0)
        an, qn, xn = node_of(c, ancestor), node_of(c, q), node_of(c, x)
        before = audit(c)
        ancestor_gradient = before['nodes'][an][384]
        leaf_gradient = before['nodes'][xn][384]
        partial(c, [(58, q, seed, 0)], 0, 10, 0, 0, 3)
        after = audit(c)
        assert after['nodes'][an][384] == ancestor_gradient and after['nodes'][xn][384] == leaf_gradient
        fresh = after['nodes'][qn][384]
        assert fresh > qn and set(after['nodes']) - set(before['nodes']) == {fresh}
        g = c.make(91, q); c.values(g, [2., 2.]); assert words(c, 13, g)[13] == 0; c.one(41, g)
        g = c.make(91, ancestor); c.values(g, [4., 6.]); c.one(41, g)
        c.values(x, [4., 6.], gradient=True)
        return {'released_Saved_error_after_prehook': 10, 'unvisited_ancestor_and_leaf_gradients_unchanged': True,
                'only_one_new_live_gradient': True, 'all_temporary_reservations_drained': True}

    case('released-error-prefix-and-unvisited-reserves', released_prefix)

    def atomic(c, directory):
        observed = []
        for entry in (7, 29, 57, 58):
            x = c.new([], [2.], True); factor = c.new([], [3.]); q = c.make(2, 3, x, factor); seed = c.new([], [1.])
            c.one(93, q)
            command = (7, q) if entry == 7 else (29, q, seed) if entry == 29 else (entry, q, seed, 1)
            c.run([command, (999,)], status=4, label='successful retained work followed by ordinary error rolls back')
            assert c.make(91, q) == 0 and c.make(91, x) == 0
            audit(c)
            c.one(*command)
            g = c.make(91, q); c.values(g, [1.]); assert words(c, 13, g)[13] == 0; c.one(41, g)
            c.values(x, [3.], gradient=True)
            observed.append(entry)
        x = c.new([], [2.], True); factor = c.new([], [3.]); q = c.make(2, 3, x, factor); seed = c.new([], [1.])
        c.one(93, q); c.one(95, factor, codec.bits(.5))
        c.one(57, q, seed, 0, status=6, label='ordinary saved-version refusal cannot commit retained prehook')
        assert c.make(91, q) == 0
        return {'ordinary_error_after_successful_backward_entries': observed,
                'all_pointer_length_and_owner_bytes_preserved': True, 'atomic_own_version_error_keeps_gradient_None': True}

    case('ordinary-whole-owner-rollback', atomic)

    def oom(c, directory):
        x = full(c, [64, 512], 2., True); one = c.new([], [1.]); q = c.make(2, 1, x, one)
        loss = c.make(4, 7, q); c.one(93, q)
        initial = words(c, 54)
        fillers = []
        for _ in range(12):
            stats = words(c, 54)
            free = stats[4]
            if free <= 8192:
                break
            elements = min(262144, ((free - 4096) // 4) & ~3)
            assert elements > 0
            fillers.append(full(c, [elements], 1.))
            assert words(c, 54)[1] == initial[1], 'filling only existing free extents must not grow the pool'
        assert words(c, 54)[4] <= 8192
        session = c.session
        owner = session.owner(); original_limit = resource.prlimit(session.pid, resource.RLIMIT_AS)
        fields = Path(f'/proc/{session.pid}/status').read_text().splitlines()
        current_vm = int(next(line.split()[1] for line in fields if line.startswith('VmSize:'))) * 1024
        page = lambda size: (size + 4095) & ~4095
        soft = current_vm + page(owner['length'] + 16) + page(5056 + 16) + 65536
        assert soft < original_limit[1]
        resource.prlimit(session.pid, resource.RLIMIT_AS, (soft, original_limit[1]))
        try:
            assert c.make(94, q) == 1, 'normal same-capacity owner clone must still fit the imposed limit'
            before = session.owner()
            maps_before = Path(f'/proc/{session.pid}/maps').read_bytes()
            (directory / 'before.owner.bin').write_bytes(before['bytes'])
            c.one(58, loss, 0, 0, status=3, label='real kernel allocation refusal during required owner growth')
            after = session.owner()
            (directory / 'after.owner.bin').write_bytes(after['bytes'])
            assert after == before
            assert Path(f'/proc/{session.pid}/maps').read_bytes() == maps_before
        finally:
            resource.prlimit(session.pid, resource.RLIMIT_AS, original_limit)
        assert c.make(91, q) == 0 and c.make(91, x) == 0
        audit(c)
        backward(c, loss)
        g = c.make(91, q)
        for start in range(0, 32768, 8192):
            row = c.one(39, g, start, 8192)
            assert row['payload'] == struct.pack('<I', codec.bits(1.)) * 8192
            c.comparisons += 8192
        assert words(c, 13, g)[13] == 0
        c.one(41, g)
        assert words(c, 54)[1] > initial[1]
        return {'original_limit': list(original_limit), 'temporary_soft_limit': soft,
                'normal_query_under_same_limit_succeeded': True, 'actual_growth_OOM_status': 3,
                'complete_owner_bytes_unchanged': before['length'], 'owner_sha256': sha(before['bytes']),
                'mapping_set_unchanged_after_failed_trial': True, 'recovered_after_restoring_limit': True,
                'exact_retained_gradient_words_after_recovery': 32768, 'filler_count': len(fillers)}

    case('real-ENOMEM-and-recovery', oom)

    def optimizer_root(c, directory):
        x = c.new([2], [1., 2.], True); coefficient = c.new([], [2.])
        q = c.make(2, 3, x, coefficient); seed = c.new([2], [1., 1.]); c.one(93, q)
        opt = optimizer(c, q); backward(c, q, seed, 1)
        qn = node_of(c, q); identity = snapshot(c)['nodes'][qn][320]
        weak_q = c.make(42, q); weak_x = c.make(42, x)
        c.run([(41, q), (41, x), (41, coefficient), (41, seed)])
        c.expiry(weak_q, False); c.expiry(weak_x, False)
        before = snapshot(c)
        assert before['nodes'][qn][312] > 0 and before['nodes'][qn]['flags'] == 1
        c.run([(38, 0, 0, 0, 0, 0, 0, codec.bits(0.)) for _ in range(270)])
        grown = audit(c)
        assert grown['slots'] > before['slots'] and grown['nodes'][qn]['flags'] == 1
        c.one(33, 0)
        renewed = c.make(75, opt, 0, 0)
        assert renewed != q and c.make(94, renewed) == 1
        assert snapshot(c)['nodes'][node_of(c, renewed)][320] == identity
        seed = c.new([2], [1., 1.]); backward(c, renewed, seed, 1)
        held = c.make(91, renewed); c.values(held, [2., 2.]); weak_g = c.make(42, held)
        c.one(72, opt); c.values(renewed, [1.75, 3.75]); assert words(c, 13, renewed)[13] == 1
        c.run([(78, opt), (41, renewed), (41, seed)])
        c.expiry(weak_q, True); c.expiry(weak_x, True); c.expiry(weak_g, False)
        c.values(held, [2., 2.]); assert words(c, 13, held)[13] == 0
        c.one(41, held); c.expiry(weak_g, True)
        c.run([(44, weak_q), (44, weak_x), (44, weak_g), (33, 0)])
        assert not audit(c)['nodes']
        return {'real_private_nonleaf_root_preserved_across_growth_and_reset': True,
                'same_Tensor_republication_preserves_retain_flag_and_incarnation': True,
                'last_optimizer_root_releases_parameter_parent_graph': True,
                'held_gradient_independently_outlives_parameter': True,
                'no_live_node_after_final_release': True}

    case('optimizer-strong-root-growth-reset-release', optimizer_root)

    def empty(c, directory):
        x = c.new([0, 3], [], True); one = c.new([], [1.]); q = c.make(2, 1, x, one)
        seed = c.new([0, 3], []); c.one(93, q)
        backward(c, q, seed, 1)
        qn = node_of(c, q); old = snapshot(c)['nodes'][qn][384]
        baseline = audit(c)['live_by_kind']
        for _ in range(4):
            g = c.make(91, q); weak_g = c.make(42, g); c.values(g, []); c.one(41, g)
            backward(c, q, seed, 1)
            c.expiry(weak_g, True); c.one(44, weak_g)
            state = audit(c); new = state['nodes'][qn][384]
            assert new != 0 and new != old and state['live_by_kind'] == baseline
            g = c.make(91, q); c.values(g, []); assert words(c, 13, g)[13] == 0; c.one(41, g)
            old = new
        return {'defined_empty_is_not_None': True, 'fresh_zero_length_Storage_identity_each_time': True,
                'repeated_empty_replacement_has_stable_live_counts': True, 'temporary_flags_and_incoming_fields_zero': True}

    case('empty-fresh-gradient-cleanup', empty)
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    report.update(status='PASS' if all(row['status'] == 'PASS' for row in report['cases']) else 'FAIL',
                  case_count=len(report['cases']), passed=sum(row['status'] == 'PASS' for row in report['cases']),
                  failed=sum(row['status'] == 'FAIL' for row in report['cases']),
                  frames=sum(row['frames'] for row in report['cases']),
                  exact_checks=sum(row['exact_checks'] for row in report['cases']),
                  whole_owner_rollback_bytes=sum(row['whole_owner_rollback_bytes'] for row in report['cases']),
                  input_custody_after=True)
    save()
    (out / 'executed-test.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in report.items() if key not in ('cases', 'inputs_sha256')}, indent=2))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
