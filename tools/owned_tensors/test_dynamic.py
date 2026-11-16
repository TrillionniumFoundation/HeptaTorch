"""Actual dynamic tensor allocation, reverse AD, optimizers and rollback.

Expected tensor values use exact dyadic arithmetic or explicit independent
binary32 rounding. Python supplies wire transport and test expectations only.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import struct
import time
import traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dynamic_tensor_wire', HERE / 'protocol.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', type=Path, required=True)
    parser.add_argument('--build-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    app = args.service.resolve(); identity = sha(app.read_bytes())
    build_bytes = args.build_report.read_bytes(); build = json.loads(build_bytes)
    assert build['executable_sha256'] == identity
    (out / 'build.json').write_bytes(build_bytes)
    active = []; rows = []; values = 0; rollback_bytes = 0

    def new(name, **kwargs):
        value = p.Session(app, directory=out / name, **kwargs)
        active.append(value); value.control_base()
        return value

    def case(name, fn):
        start = time.monotonic()
        try:
            row = fn() or {}
            rows.append({'name': name, 'pass': True, 'elapsed_seconds': time.monotonic() - start, **row})
            print(name + ': PASS', flush=True)
        except BaseException:
            rows.append({'name': name, 'pass': False, 'error': traceback.format_exc()})
            raise
        finally:
            (out / 'cases.json').write_text(json.dumps(rows, indent=2) + '\n')

    def first_failure():
        value = new('first-failure')
        empty = value.owner()
        for raw in (b'', b'HTORCH01' + bytes(24), b'HTORCH02' + struct.pack('<3Q', 1, 32, 0)):
            status, records = p.parse(value.exchange(raw))
            assert status == 100 and not records and value.owner() == empty
        record, = value.run([p.full([1], 2)])
        assert record['value'] == 1
        value.finish()
        return {'failures_preserved_startup_live_empty': 3}

    def versions():
        nonlocal values, rollback_bytes
        value = new('initial-version-and-alias')
        records = value.run([p.tensor([2, 2], [1, 2, 3, 4], 1), (13, 1), (18, 1), (13, 2),
            (6, 1, 0, 1), (13, 3), (19, 1), (13, 4)])
        for index in (1, 3, 5, 7): assert p.meta(records[index])['version'] == 0
        assert p.meta(records[3])['owner_id'] == 2
        assert p.meta(records[5])['owner_id'] == p.meta(records[7])['owner_id'] == 1
        records = value.run([(12, 4, 1, p.bits(9)), (13, 1), (13, 3), (13, 4), (13, 2), (10, 1)])
        assert all(p.meta(records[index])['version'] == 1 for index in (1, 2, 3))
        assert p.meta(records[4])['version'] == 0
        assert records[5]['payload'] == struct.pack('<4f', 1, 9, 3, 4); values += 4
        rollback_bytes += value.error([(7, 2)], 5)
        # A reachable old clone now detects its saved parent version mismatch.
        reduction, = value.run([(4, 7, 2)])
        rollback_bytes += value.error([(7, reduction['value'])], 6)
        value.finish()
        return {'constructor_versions_exactly_zero': 4, 'shared_alias_versions_after_write': 1}

    def large_ad():
        nonlocal values
        value = new('two-MiB-tensor-AD-SGD')
        records = value.run([p.full([512, 1024], 1, 1), (13, 1)])
        assert records[0]['value'] == 1
        assert p.meta(records[1])['numel'] == 524288 and p.meta(records[1])['version'] == 0
        first = value.owner(False)
        assert first['length'] == 8 * p.MIB
        records = value.run([(2, 3, 1, 1), (4, 7, 2), (7, 3), (10, 3)])
        assert records[3]['payload'] == struct.pack('<f', 524288)
        grown = value.owner(False)
        assert grown['length'] == 16 * p.MIB and grown['pointer'] != first['pointer']
        values += value.read_constant(1, 524288, 2, gradient=True)
        records = value.run([(9, 1, p.bits(.125)), (13, 1), (33, 1)])
        assert p.meta(records[1])['version'] == 1
        values += value.read_constant(1, 524288, .75)
        value.finish()
        return {'tensor_data_bytes': 2 * p.MIB, 'state_bytes_before_forward': first['length'],
            'state_bytes_after_forward': grown['length'], 'exact_values_and_gradients': 1048576,
            'storage_relocated': True, 'graph_reset_preserved_large_parameter': True}

    def descriptor_limit():
        nonlocal rollback_bytes
        value = new('descriptor-pages-exact-limit')
        made = 0
        while made < 65535:
            count = min(12000, 65535 - made)
            records = value.run([p.full([0], 0)] * count)
            assert [record['value'] for record in records] == list(range(made + 1, made + count + 1))
            made += count
        state = value.owner()
        words = struct.unpack_from('<9Q', state['bytes'])
        assert words[1] == 65536 and words[2] <= state['length']
        pages = struct.unpack_from('<256Q', state['bytes'], 4096)
        assert pages[0] == 8192 and all(pages[index] >= pages[index - 1] + 49152 for index in range(1, 256))
        records = value.run([(13, 255), (13, 256), (13, 257), (13, 65535), (33, 65535)])
        assert all(p.meta(record)['version'] == 0 and p.meta(record)['numel'] == 0 for record in records[:4])
        rollback_bytes += value.error([p.full([0], 0)], 3)
        rollback_bytes += value.error([(33, 65534)], 7)
        record, = value.run([(13, 65535)])
        assert p.meta(record)['owner_id'] == 65535
        value.finish()
        return {'allocated_descriptors': made, 'dynamic_descriptor_pages': 256,
            'state_bytes': state['length'], 'full_prefix_survives_reset': True,
            'next_allocation_refused_atomically': True}

    def grown_abort():
        nonlocal rollback_bytes
        value = new('grown-trial-abort')
        value.run([p.full([4], 1, 1)])
        before = value.owner()
        rollback_bytes += value.error([(12, 1, 0, p.bits(99)), p.full([512, 1024], 3), (99,)], 4)
        assert value.owner() == before
        records = value.run([p.full([1], 7), (13, 1), (10, 1)])
        assert records[0]['value'] == 2 and p.meta(records[1])['version'] == 0
        assert records[2]['payload'] == struct.pack('<4f', 1, 1, 1, 1)
        value.finish()
        return {'grown_uncommitted_owner_discarded': True, 'original_capacity': before['length'],
            'failure_published_no_trial_IDs': True}

    def reset_abort():
        nonlocal rollback_bytes
        value = new('reset-generation-rollback')
        value.run([p.full([4], 1, 1), (2, 3, 1, 1)])
        rollback_bytes += value.error([(33, 1), p.full([2], 7), (99,)], 4)
        record, = value.run([(10, 2)])
        assert record['payload'] == struct.pack('<4f', 1, 1, 1, 1)
        value.run([(33, 1)])
        rollback_bytes += value.error([(10, 2)], 1)
        record, = value.run([p.full([2], 7)])
        assert record['value'] == (1 << 16) | 2
        value.run([(33, 1)])
        rollback_bytes += value.error([(10, (1 << 16) | 2)], 1)
        record, = value.run([p.full([2], 9)])
        assert record['value'] == (2 << 16) | 2
        value.finish()
        return {'failed_reset_generation_not_committed': True, 'released_generation_handles_rejected': 2}

    def genuine_allocation_failure():
        nonlocal rollback_bytes
        value = new('real-ENOMEM-rollback', address_limit=64 * p.MIB)
        value.run([p.full([4], 1, 1)])
        rollback_bytes += value.error([(12, 1, 0, p.bits(99)), p.full([4096, 4096], 3)], 3)
        record, = value.run([(10, 1)])
        assert record['payload'] == struct.pack('<4f', 1, 1, 1, 1)
        value.finish()
        return {'RLIMIT_AS_bytes': 64 * p.MIB, 'required_trial_capacity_bytes': 128 * p.MIB,
            'native_buffer_try_new_allocation_failure': True}

    def element_limits():
        nonlocal rollback_bytes, values
        value = new('element-bounds')
        value.run([p.full([1], 1)])
        rollback_bytes += value.error([p.full([262145, 0], 0)], 3)
        rollback_bytes += value.error([p.full([4096, 4097], 0)], 3)
        record, = value.run([p.full([4096, 4096], .5)])
        assert record['value'] == 2
        owner = value.owner(False); assert owner['length'] == 128 * p.MIB
        records = value.run([(13, 2), (39, 2, 0, 4), (39, 2, 16777212, 4)])
        assert p.meta(records[0])['numel'] == 16777216 and p.meta(records[0])['version'] == 0
        assert records[1]['payload'] == records[2]['payload'] == struct.pack('<4f', *([.5] * 4)); values += 8
        value.finish()
        return {'exact_max_numel': 16777216, 'tensor_data_bytes': 64 * p.MIB,
            'owner_capacity_bytes': owner['length'], 'per_dimension_and_numel_refusal_distinguished_from_growth': True}

    def training(kind):
        nonlocal values, rollback_bytes
        value = new('training-' + kind)
        # Parameters/moments plus a real 2 MiB retained tensor and empty leaves
        # cross the first descriptor page before any transient graph exists.
        initial = [p.full([4096], 8, 1), p.full([4096], 0), p.full([4096], 0),
                   p.full([512, 1024], 1)] + [p.full([0], 0)] * 296
        records = value.run(initial)
        assert [record['value'] for record in records] == list(range(1, 301))
        expected = 8.0; max_capacity = value.owner(False)['length']
        for step in range(1, 33):
            next_id = ((step - 1) << 16) | 301
            commands = [(8, 1), (4, 7, 1), (7, next_id)]
            if kind == 'SGD':
                optimizer = (9, 1, p.bits(.125))
            else:
                config = p.Payload(struct.pack('<6Q', p.bits(.125), 0, 0, 0,
                    p.bits(.0625) if kind == 'AdamW' else 0, 1 if kind == 'AdamW' else 0))
                optimizer = (24, 1, 2, 3, step, config)
            if step == 13:
                rollback_bytes += value.error(commands + [optimizer, (99,)], 4)
            commands.append(optimizer)
            if step in (8, 16):
                commands.append(p.full([1024, 1024 if step == 8 else 2048], 3))
            commands += [(13, 1), (39, 1, 0, 4), (39, 1, 4092, 4), (33, 300)]
            records = value.run(commands)
            if kind == 'AdamW': expected = p.f32(expected * (127 / 128))
            expected = p.f32(expected - .125)
            metadata = next(record for record in records if record['opcode'] == 13)
            assert p.meta(metadata)['version'] == step
            reads = [record for record in records if record['opcode'] == 39]
            assert all(record['payload'] == struct.pack('<4f', *([expected] * 4)) for record in reads)
            values += 8
            max_capacity = max(max_capacity, value.owner(False)['length'])
        values += value.read_constant(1, 4096, expected)
        if kind != 'SGD':
            values += value.read_constant(2, 4096, 1)
            values += value.read_constant(3, 4096, 1)
            records = value.run([(13, 2), (13, 3)])
            assert all(p.meta(record)['version'] == 32 for record in records)
        value.finish()
        return {'steps': 32, 'retained_descriptors': 300, 'retained_large_tensor_bytes': 2 * p.MIB,
            'maximum_owned_capacity_bytes': max_capacity, 'final_parameter': expected,
            'gradients_and_optimizer_versions_preserved_across_resize': True}

    try:
        case('first-failure', first_failure)
        case('initial-version-and-alias', versions)
        case('two-MiB-tensor-AD-SGD', large_ad)
        case('descriptor-pages-exact-limit', descriptor_limit)
        case('grown-trial-abort', grown_abort)
        case('reset-generation-rollback', reset_abort)
        case('real-ENOMEM-rollback', genuine_allocation_failure)
        case('element-bounds', element_limits)
        for kind in ('SGD', 'Adam', 'AdamW'):
            case('training-' + kind, lambda kind=kind: training(kind))
        assert sha(app.read_bytes()) == identity and args.build_report.read_bytes() == build_bytes
        report = {'schema': 'heptatorch.dynamic-owned-tensor-qualification.v1', 'pass': True,
            'cases': len(rows), 'response_frames': sum(value.frames for value in active),
            'compared_binary32_values': values, 'rollback_compared_owned_bytes': rollback_bytes,
            'executable_sha256': identity, 'core_overlay_sha256': build['candidate_source_sha256'],
            'canonical_source_sha256': build['canonical_source_sha256'],
            'test_sha256': sha(Path(__file__).read_bytes()), 'wire_helper_sha256': sha((HERE / 'protocol.py').read_bytes()),
            'actual_native_owner_allocation': True, 'all_responses_before_stdin_EOF': True,
            'host_tensor_runtime_used': False, 'rows': rows}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        for value in active:
            if value.poll() is None:
                os.kill(value.pid, signal.SIGKILL); os.waitpid(value.pid, 0)


if __name__ == '__main__':
    main()
