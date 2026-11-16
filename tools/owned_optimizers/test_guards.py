"""Independent packet, exact capacity, and real allocation-failure controls."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import struct

SHA = lambda data: hashlib.sha256(data).hexdigest()
Q = lambda words: struct.pack('<' + 'Q' * len(words), *words)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'protocol', 'profile', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    snapshots = {p.resolve(): p.read_bytes() for p in (Path(__file__), a.binary, a.source, a.protocol, a.profile)}
    profile = json.loads(snapshots[a.profile.resolve()])
    assert SHA(snapshots[a.binary.resolve()]) == profile['native_sha256']
    assert SHA(snapshots[a.source.resolve()]) == profile['source_sha256']
    assert SHA(snapshots[a.protocol.resolve()]) == '5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec = importlib.util.spec_from_file_location('optimizer_guard_wire', a.protocol)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    config = [wire.bits(.125), wire.bits(.5), 0, 0, 0, 0]
    def group(ids, mask=0, words=None):
        return Q([len(ids), mask] + (words or [0] * 6) + list(ids))
    def create(groups, defaults=config, algorithm=0):
        return (71, algorithm, wire.Payload(Q(defaults)), wire.Payload(Q([len(groups)]) + b''.join(groups)))
    report = {'schema': 'heptatorch.optimizer-admission-capacity-controls.v1', 'status': 'RUNNING',
              'native_sha256': profile['native_sha256'], 'source_sha256': profile['source_sha256'],
              'native_executed': True, 'Torch_executed_here': False, 'cases': [], 'frames': 0,
              'rejections': 0, 'rollback_bytes': 0,
              'inputs_sha256': {str(p): SHA(raw) for p, raw in snapshots.items()}}
    def save():
        (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    def case(name, body):
        s = wire.Session(a.binary, directory=a.output / name, address_limit=64 << 20)
        row = {'case': name, 'status': 'RUNNING', 'rejections': 0, 'rollback_bytes': 0}
        report['cases'].append(row)
        save()
        def reject(commands, status):
            row['rollback_bytes'] += s.error(commands, status)
            row['rejections'] += 1
        row['details'] = body(s, reject)
        s.run([(53,)])
        s.finish()
        row.update(status='PASS', frames=s.frames, all_responses_before_stdin_eof=True)
        for key in ('frames', 'rejections', 'rollback_bytes'):
            report[key] += row[key]
        save()

    def malformed(s, reject):
        p = s.run([wire.tensor((2,), [1, -2], 1)])[0]['value']
        opt = s.run([create([group([p])])])[0]['value']
        nonleaf = s.run([(2, 1, p, p)])[0]['value']
        for field in (0, 1, 3):
            bad = list(config)
            bad[field] = wire.bits(-.125)
            reject([create([group([p])], bad)], 7)
        for bad in ([wire.bits(.125), 0, 0, 0, 1, 0],
                    [wire.bits(.125), wire.bits(.5), wire.bits(.25), 0, 1, 0],
                    [1 << 32, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0]):
            reject([create([group([p])], bad)], 7)
        reject([create([group([p])], algorithm=1)], 7)
        reject([create([])], 7)
        reject([create([group([p], 64)])], 7)
        reject([create([group([p], 0, [1, 0, 0, 0, 0, 0])])], 7)
        reject([create([group([p], 1, [1 << 32, 0, 0, 0, 0, 0])])], 7)
        reject([create([group([p]), group([p])])], 7)
        reject([create([group([nonleaf])])], 5)
        reject([create([group([opt])])], 1)
        reject([(90, nonleaf, 1)], 5)
        reject([(90, p, 2)], 7)
        reject([(90, p, 1, 1)], 7)
        reject([(73, opt, 2)], 7)
        reject([(77, opt, 0, 0, 1)], 7)
        reject([(74, opt, 1)], 7)
        reject([(75, opt, 0, 1)], 7)
        reject([(71, 0, 96, 32)], 100)
        reject([(71, 0, 97, 104)], 100)
        reject([(79, opt, 96)], 100)
        reject([(72, p)], 1)
        s.run([(78, opt)])
        reject([(72, opt)], 1)
        reject([(78, opt)], 1)
        # These predicates intentionally admit NaN/+infinity defaults. This is
        # admission-only; the native test does not invent a numerical reference.
        accepted = []
        for bits in (0x7fc00001, 0x7f800000):
            cfg = list(config)
            cfg[0] = bits
            accepted.append(s.run([create([group([])], cfg)])[0]['value'])
        cfg = list(config)
        cfg[1] = 0x7fc00001
        cfg[4] = 1
        accepted.append(s.run([create([group([])], cfg)])[0]['value'])
        for handle in accepted:
            s.run([(78, handle)])
        return {'invalid_defaults_payload_types_membership_and_reserved_fields': True,
                'ordered_NaN_default_predicates_not_blanket_finite_rejection': True}
    case('typed-payload-and-configuration-refusals', malformed)

    def boundaries(s, reject):
        p = s.run([wire.full((0,), 0)])[0]['value']
        duplicate = s.run([create([group([p] * 4096)])])[0]
        assert struct.unpack('<4Q', duplicate['payload']) == (1, 4096, 1, 4095)
        opt = duplicate['value']
        observed = s.run([(74, opt, 0)])[0]
        assert observed['value'] == 4096 and len(observed['payload']) == 64 + 4096 * 16
        pairs = struct.unpack('<' + 'Q' * (4096 * 2), observed['payload'][64:])
        assert len(set(pairs[::2])) == 1 and all(x == p for x in pairs[1::2])
        s.run([(72, opt)])
        assert s.run([(76, opt, 0, 4095)])[0]['value'] == 0
        reject([create([group([p] * 4097)])], 7)
        reject([create([group([p] * 4096), group([p])])], 3)
        reject([(79, opt, wire.Payload(group([p])))], 3)
        s.run([(78, opt)])
        groups = [group([]) for _ in range(1024)]
        empty = s.run([create(groups)])[0]
        assert struct.unpack('<4Q', empty['payload']) == (1024, 0, 0, 0)
        opt = empty['value']
        assert s.run([(74, opt, 1023)])[0]['value'] == 0
        s.run([(72, opt), (73, opt, 1)])
        reject([create(groups + [group([])])], 7)
        reject([(79, opt, wire.Payload(group([])))], 3)
        s.run([(78, opt)])
        return {'maximum_occurrences': 4096, 'maximum_groups': 1024,
                'exact_boundary_accepted_and_one_beyond_rejected': True}
    case('exact-group-and-occurrence-capacity', boundaries)

    def refs(s, reject):
        first, second, third, fourth = s.run([wire.full((), 1, 1),
            create([group([0xfe000001])]), (72, 0xfe000002), (76, 0xfe000002, 0, 0)])
        p, opt = first['value'], second['value']
        assert p >> 24 & 255 == 1 and opt >> 24 & 255 == 8
        assert third['value'] == 1 and fourth['value'] == 0
        assert s.run([(75, opt, 0, 0)])[0]['value'] == p
        reject([(72, opt), (90, 0xfe000001, 1)], 1)
        reject([create([group([0xfe000002])])], 1)
        reject([(79, opt, wire.Payload(group([0xfe000001])))], 1)
        return {'same_frame_real_Tensor_and_Optimizer_refs': True,
                'scalar_and_forward_refs_cannot_forge_Tensor': True}
    case('same-frame-reference-kinds', refs)

    def oom(s, reject):
        p = s.run([wire.tensor((2,), [1, -2], 1)])[0]['value']
        seed = s.run([wire.tensor((2,), [2, -1])])[0]['value']
        s.run([(58, p, seed, 0)])
        opt = s.run([create([group([p])])])[0]['value']
        s.run([(72, opt)])
        state = s.run([(77, opt, 0, 0, 0)])[0]['value']
        before = s.owner()
        assert s.run([(10, p)])[0]['payload'] == struct.pack('<2f', .75, -1.875)
        assert wire.meta(s.run([(13, state)])[0])['version'] == 0
        # This legal 16,777,216-element tensor alone needs 64 MiB of data.
        # Under a real 64 MiB RLIMIT_AS the owning allocation cannot coexist
        # with the running ELF/stack/state. The earlier actual SGD writes and
        # newly advanced Versions must roll back with the refused growth.
        reject([(72, opt), wire.full((262144, 64), 7)], 3)
        assert s.owner() == before
        assert s.run([(10, p)])[0]['payload'] == struct.pack('<2f', .75, -1.875)
        assert wire.meta(s.run([(13, state)])[0])['version'] == 0
        s.run([(72, opt)])
        assert s.run([(10, p)])[0]['payload'] == struct.pack('<2f', .375, -1.6875)
        assert s.run([(10, state)])[0]['payload'] == struct.pack('<2f', 3, -1.5)
        assert wire.meta(s.run([(13, state)])[0])['version'] == 2
        return {'RLIMIT_AS_bytes': 64 << 20, 'legal_requested_tensor_data_bytes': 64 << 20,
                'prior_real_SGD_and_Version_updates_rolled_back': True,
                'next_step_uses_original_state_and_succeeds': True}
    case('actual-address-limit-after-real-optimizer-write', oom)
    for path, raw in snapshots.items():
        assert path.read_bytes() == raw
    report.update(status='PASS', case_count=len(report['cases']))
    save()
    print(json.dumps({k: report[k] for k in ('status', 'case_count', 'frames', 'rejections', 'rollback_bytes')}))


if __name__ == '__main__':
    main()
