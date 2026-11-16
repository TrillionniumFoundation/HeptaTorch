#!/usr/bin/env python3
"""Small independent mixed-owner contracts for the actual joint native ELF.

Only the frozen packet/own-child transport is imported. Expected arithmetic uses
exact dyadic Fractions; RNG uses CPython's MT engine seeded by the standard
MT19937 recurrence, calibrated against unchanged hosted observations. Neither
the producer nor a native output supplies expected numerical values.
"""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import struct
import traceback

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
Q = lambda words: struct.pack('<' + 'Q' * len(words), *words)
SOURCE = '5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873'
IMAGE = '3fc59ab7ed9c32dd75ff440ef9bfbec31756f50edd77715d665bdac098cc3f22'
WIRE = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
DIST_REF = 'c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467'
ADAM_REF = '7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160'


class MTReference:
    def __init__(self, seed):
        self.seed_value = seed
        words = [seed & 0xffffffff]
        for i in range(1, 624):
            previous = words[-1]
            words.append((1812433253 * (previous ^ (previous >> 30)) + i) & 0xffffffff)
        self.engine = random.Random(0)
        self.engine.setstate((3, tuple(words + [624]), None))
        self.fresh = True
        self.count = 0

    def word(self):
        self.fresh = False
        self.count += 1
        return self.engine.getrandbits(32)

    def probability32(self, p):
        return int(Fraction(self.word() & 0xffffff, 1 << 24) < p)

    def dropout_half(self, n):
        result = []
        for _ in range(n):
            value = (self.word() << 32) | self.word()
            result.append(2 if (value & ((1 << 53) - 1)) < (1 << 52) else 0)
        return result

    def state(self):
        words = self.engine.getstate()[1]
        index = 0 if self.fresh else words[-1]
        out = bytearray(5056)
        struct.pack_into('<QIIQ', out, 0, self.seed_value, 1 if self.fresh else 625-index, 1, index)
        struct.pack_into('<624Q', out, 24, *words[:624])
        return bytes(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'protocol', 'distribution-observations', 'adam-observations', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    build, out = a.build.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    image, source = build / 'tensor-adam-distributions-session', build / 'runtime.hepta'
    owner_client = a.protocol.resolve().with_name('owner_client.py')
    paths = {image: IMAGE, source: SOURCE, a.protocol.resolve(): WIRE, owner_client: CLIENT,
             a.distribution_observations.resolve(): DIST_REF, a.adam_observations.resolve(): ADAM_REF}
    snapshots = {path: path.read_bytes() for path in paths}
    for path, expected in paths.items():
        assert SHA(snapshots[path]) == expected, str(path)
    script = Path(__file__).resolve()
    snapshots[script] = script.read_bytes()
    (out / 'executed-test.py').write_bytes(snapshots[script])
    spec = importlib.util.spec_from_file_location('joint_contract_wire', a.protocol.resolve())
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    dist = json.loads(snapshots[a.distribution_observations.resolve()])
    adam = json.loads(snapshots[a.adam_observations.resolve()])
    assert dist['status'] == adam['status'] == 'OBSERVED'
    assert dist['cases'] == 183 and dist['probe_error_count'] == 0
    assert adam['cases'] == 24 and adam['probe_errors'] == 0
    for ref in (dist, adam):
        assert len(ref['oracle_provenance']['checks']) == 11
        assert all(ref['oracle_provenance']['checks'].values())

    # Validate the test MT model against actual pre-existing Torch bytes, without
    # executing any native test or regenerating a hosted expected value.
    calibration = []
    used_counts = set()
    for row in dist['rows']:
        counts = row.get('matching_raw32_uniform_counts', [])
        if len(counts) != 1 or not 0 <= counts[0] <= 64 or counts[0] in used_counts:
            continue
        raw = bytes.fromhex(dist['states'][row['state_before']]['hex'])
        if struct.unpack_from('<Q', raw, 16)[0] != 0:
            continue
        ref = MTReference(struct.unpack_from('<Q', raw)[0])
        if ref.state() != raw:
            continue  # Cases with preloaded normal caches are outside this model.
        for _ in range(counts[0]):
            ref.word()
        expected = bytes.fromhex(dist['states'][row['state_after']]['hex'])
        assert ref.state() == expected, row['name']
        calibration.append({'actual_case': row['name'], 'raw32_draws': counts[0],
                            'state_before_sha256': SHA(raw), 'state_after_sha256': SHA(expected)})
        used_counts.add(counts[0])
        if len(calibration) == 5:
            break
    assert len(calibration) == 5 and max(used_counts) > 0
    report = {
        'schema': 'heptatorch.adam-distributions.independent-joint-contracts.v1',
        'status': 'RUNNING', 'runtime_source_sha256': SOURCE, 'executable_sha256': IMAGE,
        'native_executed': True, 'Torch_executed_here': False,
        'host_numerics_are_test_oracles_only': True, 'whole_PyTorch_complete': False,
        'frozen_references': {'distribution_183': DIST_REF, 'optimizer_24': ADAM_REF},
        'qualification_unit': 'new finite cross-feature native contracts, not new APIs or new hosted Torch cases',
        'independent_MT_calibration_against_actual_Torch': calibration,
        'cases': [], 'frames_before_EOF': 0, 'exact_assertions': 0,
        'exact_float32_words': 0, 'complete_MT_state_comparisons': 0,
        'complete_MT_bytes_compared': 0, 'rollback_bytes_compared': 0,
        'inputs_sha256': {str(p): SHA(b) for p, b in snapshots.items()},
    }
    active = None

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def eq(actual, expected, label):
        assert actual == expected, {'label': label, 'actual': repr(actual)[:1000], 'expected': repr(expected)[:1000]}
        active['exact_assertions'] += 1

    def one(s, *cmd):
        return s.run([cmd])[0]

    def meta(s, t):
        return wire.meta(one(s, 13, t))

    def fields(s, t):
        m = meta(s, t)
        return (m['storage_id'], m['version_id'])

    def check_words(s, t, expected, gradient=False, saved=False):
        expected_words = [wire.bits(float(v)) for v in expected]
        command = 49 if saved else (40 if gradient else 39)
        record = one(s, command, t, 0, len(expected_words))
        eq(record['value'], len(expected_words), 'logical value count')
        got = list(struct.unpack('<' + 'I' * len(expected_words), record['payload']))
        eq(got, expected_words, 'exact dyadic logical binary32 words')
        active['exact_float32_words'] += len(got)

    def states(s, opt):
        return tuple(r['value'] for r in s.run([(77, opt, 0, 0, selector) for selector in (1, 2, 3)]))

    def create_opt(s, p, algorithm=1, eps=0, decay=0):
        cfg = [wire.bits(x) for x in (.125, 0, 0, eps, decay)] + [0, 0, 0]
        groups = [1, 1, 0] + [0] * 8 + [p]
        result = one(s, 71, algorithm, wire.Payload(Q(cfg)), wire.Payload(Q(groups)))
        eq((result['value'] >> 24) & 255, 8, 'actual Optimizer typed kind')
        eq(struct.unpack('<4Q', result['payload']), (1, 1, 1, 0), 'one real retained parameter')
        return result['value']

    def seed(s, p, values):
        shape = tuple(meta(s, p)['shape'][:meta(s, p)['rank']])
        seed_id = s.run([wire.tensor(shape, list(values))])[0]['value']
        s.run([(57, p, seed_id, 0), (41, seed_id)])

    def generator(s, seed_value):
        g = one(s, 60, seed_value)['value']
        ref = MTReference(seed_value)
        check_rng(s, g, ref)
        return g, ref

    def check_rng(s, g, ref):
        raw = one(s, 63, g)['payload']
        eq(len(raw), 5056, 'whole Generator state width')
        eq(raw, ref.state(), 'complete independently advanced MT state')
        active['complete_MT_state_comparisons'] += 1
        active['complete_MT_bytes_compared'] += len(raw)

    def stats(s):
        return struct.unpack('<12Q', one(s, 54)['payload'])

    def failure(s, commands, expected):
        before = s.owner()
        status, records = wire.parse(s.exchange(wire.packet(commands)))
        after = s.owner()
        eq((status, records), (expected, []), 'ordinary frame refusal')
        eq(after, before, 'complete owner pointer/length/bytes unchanged on refusal')
        n = len(active['rollbacks']) + 1
        directory = out / active['name']
        (directory / ('rollback-%02d-before.bin' % n)).write_bytes(before['bytes'])
        (directory / ('rollback-%02d-after.bin' % n)).write_bytes(after['bytes'])
        active['rollbacks'].append({'status': expected, 'owner_bytes': before['length'],
                                    'before_sha256': SHA(before['bytes']), 'after_sha256': SHA(after['bytes']),
                                    'pointer_and_length_unchanged': True})
        active['rollback_bytes_compared'] += before['length']

    def partial(s, commands, code, expected, executed):
        status, records = wire.parse(s.exchange(wire.packet(commands)))
        eq(status, 0, 'explicit operation error has successful outer transaction')
        eq([r['opcode'] for r in records], [c[0] for c in commands[:executed]], 'exact committed prefix and stopped suffix')
        eq(records[-1]['opcode'], code, 'last executed partial opcode')
        eq(records[-1]['value'], expected[0], 'operation result is error code, not Tensor ID')
        eq(struct.unpack('<4Q', records[-1]['payload']), expected, 'operation error, actual ID, writes, draws')
        return records

    def live(s, weak, expected):
        if expected:
            hold = one(s, 43, weak)['value']
            eq((hold >> 24) & 255, 6, 'live weak locks to actual StrongHold')
            one(s, 45, hold)
        else:
            failure(s, [(43, weak)], 6)

    def versions(s, ids, expected):
        eq([meta(s, t)['version'] for t in ids], list(expected), 'absolute shared Version counters')

    def case(name, action):
        nonlocal active
        active = {'name': name, 'status': 'RUNNING', 'frames': 0, 'exact_assertions': 0,
                  'exact_float32_words': 0, 'complete_MT_state_comparisons': 0,
                  'complete_MT_bytes_compared': 0, 'rollback_bytes_compared': 0, 'rollbacks': []}
        report['cases'].append(active)
        directory = out / name
        directory.mkdir()
        s = wire.Session(image, directory=directory / 'transport')
        try:
            active['contract'] = action(s)
            eq(one(s, 53)['value'], 1, 'mixed-kind graph/storage/private-root audit')
            s.finish()
            active['status'] = 'PASS'
        except BaseException as error:
            active.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
            if s.poll() is None:
                os.kill(s.pid, signal.SIGKILL)
            try:
                s.finish(expected=-signal.SIGKILL)
            except BaseException as cleanup:
                active['cleanup_after_failure'] = repr(cleanup)
        active['frames'] = s.frames
        for key in ('exact_assertions', 'exact_float32_words', 'complete_MT_state_comparisons',
                    'complete_MT_bytes_compared', 'rollback_bytes_compared'):
            report[key] += active[key]
        report['frames_before_EOF'] += s.frames
        (directory / 'assertions.json').write_text(json.dumps(active, indent=2) + '\n')
        save()

    def graph41(s):
        p = s.run([wire.tensor((2,), [.25, .75], 1)])[0]['value']
        original_fields = fields(s, p)
        weak = one(s, 42, p)['value']
        g, rng = generator(s, 401)
        opt = create_opt(s, p, eps=1)
        before = stats(s)
        wanted = [rng.probability32(Fraction(1, 4)), rng.probability32(Fraction(3, 4))]
        y = one(s, 82, p, g, 0)['value']
        eq(stats(s)[9], before[9], 'Bernoulli graph adds no Saved value record')
        check_words(s, y, wanted)
        check_rng(s, g, rng)
        one(s, 41, p)
        failure(s, [(13, p)], 1)
        seed_id = s.run([wire.tensor((2,), [float('nan'), -float('inf')])])[0]['value']
        s.run([(57, y, seed_id, 0), (57, y, seed_id, 0)])
        q = one(s, 75, opt, 0, 0)['value']
        eq(q != p, True, 'Optimizer republishes a fresh public token')
        eq(fields(s, q), original_fields, 'same retained parameter S/V identities')
        eq(one(s, 59, q)['value'], 1, 'zero derivative materializes defined gradient')
        check_words(s, q, [0, 0], gradient=True)
        one(s, 72, opt)
        st = states(s, opt)
        identities = [fields(s, t) for t in st]
        versions(s, (q,) + st, (1, 1, 1, 2))
        check_words(s, q, [Fraction(1, 4), Fraction(3, 4)])
        check_words(s, st[0], [1]);check_words(s, st[1], [0, 0]);check_words(s, st[2], [0, 0])
        s.run([(41, q)] + [(41, t) for t in st] + [(33, 0)])
        failure(s, [(13, y)], 1)
        q2 = one(s, 75, opt, 0, 0)['value'];st2 = states(s, opt)
        eq(fields(s, q2), original_fields, 'reset keeps actual private parameter object')
        eq([fields(s, t) for t in st2], identities, 'reset keeps actual private Adam state objects')
        versions(s, (q2,) + st2, (1, 1, 1, 2))
        wanted2 = [rng.probability32(Fraction(1, 4)), rng.probability32(Fraction(3, 4))]
        y2 = one(s, 82, q2, g, 0)['value'];check_words(s, y2, wanted2)
        saved = one(s, 46, q2)['value']
        s.run([(95, q2, wire.bits(2)), (70, q2), (90, q2, 0)])
        fresh_seed = s.run([wire.full((2,), 1)])[0]['value']
        one(s, 57, y2, fresh_seed, 0)
        eq(one(s, 59, q2)['value'], 0, 'captured edge traversed but disabled leaf does not accumulate')
        s.run([(90, q2, 1), (57, y2, fresh_seed, 0)])
        check_words(s, q2, [0, 0], gradient=True)
        failure(s, [(47, saved)], 6)
        one(s, 48, saved)
        s.run([(41, q2)] + [(41, t) for t in st2] + [(78, opt)])
        live(s, weak, True)
        one(s, 41, y2)
        live(s, weak, False)
        check_rng(s, g, rng)
        s.run([(44, weak), (41, fresh_seed), (66, g)])
        return {'op41_has_no_Saved_probability_dependency': True,
                'nonfinite_incoming_still_positive_zero_gradient': True,
                'private_optimizer_and_AccumulateGrad_roots_are_independent': True,
                'graph_reset_revokes_public_tokens_preserving_real_private_T_S_V': True}

    def dropout_saved(s):
        p = s.run([wire.tensor((2,), [1, 2], 1)])[0]['value']
        weak_p = one(s, 42, p)['value']
        opt = create_opt(s, p, eps=2)
        g, rng = generator(s, 312)
        noise_values = rng.dropout_half(2)
        y = one(s, 83, p, g, 0x3fe0000000000000, 1, 0)['value']
        noise = one(s, 85, y)['value']
        eq(meta(s, noise)['version'], 2, 'dropout exposes actual Version2 Saved noise alias')
        check_words(s, noise, noise_values)
        noise_weak = one(s, 42, noise)['value'];noise_saved = one(s, 46, noise)['value']
        seed_id = s.run([wire.full((2,), 1)])[0]['value']
        one(s, 57, y, seed_id, 0)
        check_words(s, p, noise_values, gradient=True)
        one(s, 72, opt)
        st = states(s, opt);old_fields = [fields(s, t) for t in (p,) + st]
        avg_saved = one(s, 46, st[1])['value'];avg_weak = one(s, 42, st[1])['value']
        one(s, 95, st[1], wire.bits(.5))
        failure(s, [(47, avg_saved)], 6)
        eq(one(s, 47, noise_saved)['value'], 1, 'state mutation does not alias dropout-noise Version')
        check_words(s, noise_saved, noise_values, saved=True)
        s.run([(41, noise), (33, 0)])
        live(s, noise_weak, True)
        check_words(s, noise_saved, noise_values, saved=True)
        p2 = one(s, 75, opt, 0, 0)['value'];st2 = states(s, opt)
        eq([fields(s, t) for t in (p2,) + st2], old_fields, 'reset retains optimizer object identities')
        check_words(s, st2[1], [Fraction(x) + Fraction(1, 2) for x in noise_values])
        expected_p = [Fraction(x) - (Fraction(1, 16) if grad else 0) for x, grad in zip((1, 2), noise_values)]
        check_words(s, p2, expected_p)
        s.run([(41, p2), (41, st2[0]), (41, st2[2]), (78, opt)])
        live(s, weak_p, False)
        live(s, avg_weak, True)
        check_words(s, st2[1], [Fraction(x) + Fraction(1, 2) for x in noise_values])
        one(s, 41, st2[1]);live(s, avg_weak, True)
        one(s, 48, avg_saved);live(s, avg_weak, False)
        one(s, 48, noise_saved);live(s, noise_weak, False)
        check_rng(s, g, rng)
        s.run([(44, weak_p), (44, avg_weak), (44, noise_weak), (66, g)])
        return {'dropout_noise_saved_alias_and_Adam_state_have_independent_S_V': True,
                'public_Saved_survives_graph_reset_and_optimizer_drop': True,
                'state_alias_and_stale_Saved_retain_actual_payload_until_last_release': True}

    def partial86(s):
        initial = [Fraction(1), Fraction(-2), Fraction(1, 2), Fraction(-1, 2)]
        gradient = [2, -1, 4, -2]
        p = s.run([wire.tensor((4,), list(map(float, initial)), 1)])[0]['value']
        seed(s, p, gradient);opt = create_opt(s, p)
        one(s, 72, opt);st = states(s, opt)
        old_avg_saved = one(s, 46, st[1])['value']
        g, rng = generator(s, 719)
        prob = s.run([wire.tensor((4,), [1, 0, float('nan'), 1])])[0]['value']
        commands = [(72, opt), (46, st[1]), (67, g, 1), (86, st[1], g, prob, 1),
                    (95, st[0], wire.bits(.5)), (78, opt), (999,)]
        rng.word();rng.word();rng.word()
        records = partial(s, commands, 86, (7, st[1], 2, 2), 4)
        new_avg_saved = records[1]['value']
        versions(s, (p,) + st, (2, 2, 2, 4))
        expected_p = [x - Fraction(1 if grad > 0 else -1, 4) for x, grad in zip(initial, gradient)]
        check_words(s, p, expected_p);check_words(s, st[0], [2])
        check_words(s, st[1], [1, 0, 4, -2]);check_words(s, st[2], [4, 1, 16, 4])
        eq(one(s, 47, new_avg_saved)['value'], 1, 'partial state write does not invent Version increment')
        check_words(s, new_avg_saved, [1, 0, 4, -2], saved=True)
        failure(s, [(47, old_avg_saved)], 6)
        check_rng(s, g, rng)
        failure(s, [(72, opt), (67, g, 1), (86, st[1], g, prob, 2)], 7)
        good = s.run([wire.full((4,), .5)])[0]['value']
        failure(s, [(72, opt), (86, st[1], g, good, 1), (999,)], 4)
        versions(s, (p,) + st, (2, 2, 2, 4));check_rng(s, g, rng)
        one(s, 72, opt)
        check_words(s, st[1], gradient);check_words(s, st[0], [3])
        failure(s, [(47, new_avg_saved)], 6)
        s.run([(48, old_avg_saved), (48, new_avg_saved), (78, opt), (66, g)])
        return {'Adam72_and_actual_state_Tensor_prefix_writes_committed': True,
                'suffix_step_mutation_and_Optimizer_drop_stopped': True,
                'ordinary_admission_and_post_success_errors_restore_entire_owner': True,
                'Saved_sees_changed_values_without_unperformed_Version_bump': True}

    def partial87(s):
        initial = [Fraction(1), Fraction(-2), Fraction(1, 2), Fraction(-1, 2)]
        gradient = [2, -1, 4, -2]
        p = s.run([wire.tensor((4,), list(map(float, initial)), 1)])[0]['value']
        seed(s, p, gradient);opt = create_opt(s, p, algorithm=2, decay=.25)
        old_saved = one(s, 46, p)['value']
        g, rng = generator(s, 333)
        before = stats(s)
        commands = [(72, opt), (77, opt, 0, 0, 1), (77, opt, 0, 0, 2), (77, opt, 0, 0, 3),
                    (87, p, g, 0x3fe0000000000000, 1, 0), (95, p, wire.bits(1)), (78, opt)]
        for _ in range(8):rng.word()
        records = partial(s, commands, 87, (5, p, 0, 8), 5)
        st = tuple(r['value'] for r in records[1:4])
        eq(stats(s)[5], before[5] + 3, 'only three actual Adam states remain; failed temporary noise reclaimed')
        expected = [x * Fraction(31, 32) - Fraction(1 if grad > 0 else -1, 8) for x, grad in zip(initial, gradient)]
        versions(s, (p,) + st, (2, 1, 1, 2))
        check_words(s, p, expected);check_words(s, st[0], [1]);check_words(s, st[1], gradient)
        failure(s, [(47, old_saved)], 6);one(s, 48, old_saved)
        check_rng(s, g, rng)
        cfg = wire.Payload(Q([2, 2, 2, 0, 0, 0, 1, 0, 0, 0]))
        view = one(s, 69, st[1], cfg, 80)['value']
        eq(fields(s, view), fields(s, st[1]), 'expanded view is actual shared Adam state S/V')
        before = stats(s)
        for _ in range(8):rng.word()
        partial(s, [(72, opt), (87, view, g, 0x3fe0000000000000, 1, 1), (999,)],
                87, (2, view, 0, 8), 2)
        eq(stats(s)[5:11], before[5:11], 'second failed temporary noise leaves no extra live records')
        expected = [x * Fraction(31, 32) - Fraction(1 if grad > 0 else -1, 8) for x, grad in zip(expected, gradient)]
        check_words(s, p, expected);check_words(s, st[1], gradient);check_words(s, st[0], [2])
        versions(s, (p,) + st, (4, 2, 2, 4));eq(meta(s, view)['version'], 2, 'no failed-dropout extra state Version')
        check_rng(s, g, rng)
        identities = [fields(s, t) for t in (p,) + st]
        one(s, 33, 0)
        p2 = one(s, 75, opt, 0, 0)['value'];st2 = states(s, opt)
        eq([fields(s, t) for t in (p2,) + st2], identities, 'partial commits survive private-root reset')
        failure(s, [(72, opt), (87, p2, g, 0x3fe0000000000000, 1, 1), (999,)], 4)
        versions(s, (p2,) + st2, (4, 2, 2, 4));check_rng(s, g, rng);check_words(s, p2, expected)
        s.run([(78, opt), (66, g)])
        return {'AdamW_lazily_created_real_state_commits_before_late_leaf_error': True,
                'expanded_moment_alias_late_error_preserves_moment_and_consumes_noise_RNG': True,
                'temporary_noise_reclaimed_without_releasing_optimizer_states': True,
                'reset_and_later_ordinary_rollback_preserve_committed_partial_history': True}

    def training(s, algorithm):
        expected = [Fraction(1, 2), Fraction(-1, 2), Fraction(1), Fraction(-1)]
        p = s.run([wire.tensor((4,), list(map(float, expected)), 1)])[0]['value']
        opt = create_opt(s, p, algorithm, eps=2, decay=.25 if algorithm == 2 else 0)
        g, rng = generator(s, 2000 + algorithm)
        fixed_fields = None
        first_avg_saved = None
        for step in range(1, 5):
            one(s, 73, opt, 1)
            eq(one(s, 59, p)['value'], 0, 'actual gradient cleared to None before next draw')
            noise = rng.dropout_half(4)
            root = one(s, 83, p, g, 0x3fe0000000000000, 1, 0)['value']
            # Fractions intentionally have no signed zero. IEEE multiplication
            # must preserve the sign of a negative parameter times a zero mask.
            # These dyadic products are exact in binary64 before encoding f32.
            check_words(s, root, [float(x) * n for x, n in zip(expected, noise)])
            seed_id = s.run([wire.full((4,), 1)])[0]['value']
            one(s, 57, root, seed_id, 0)
            check_words(s, p, noise, gradient=True)
            one(s, 72, opt)
            st = states(s, opt)
            expected = [x * (Fraction(31, 32) if algorithm == 2 else 1)
                        - (Fraction(1, 16) if n else 0) for x, n in zip(expected, noise)]
            check_words(s, p, expected);check_words(s, st[0], [step])
            check_words(s, st[1], noise);check_words(s, st[2], [n * n for n in noise])
            versions(s, (p,) + st, (step * algorithm, step, step, step * 2))
            current_fields = [fields(s, t) for t in (p,) + st]
            if fixed_fields is None:
                fixed_fields = current_fields
                first_avg_saved = one(s, 46, st[1])['value']
            else:
                eq(current_fields, fixed_fields, 'same real optimizer state S/V across graph generations')
            one(s, 53)
            previous_p = p
            one(s, 33, 0)
            p = one(s, 75, opt, 0, 0)['value']
            eq(p != previous_p, True, 'reset produces fresh public parameter generation')
            eq(fields(s, p), fixed_fields[0], 'private parameter survives graph collection')
            check_rng(s, g, rng)
        failure(s, [(47, first_avg_saved)], 6)
        one(s, 48, first_avg_saved)
        check_words(s, p, expected)
        one(s, 73, opt, 1)
        st = states(s, opt)
        unchanged = [meta(s, t) for t in (p,) + st]
        one(s, 72, opt)
        eq([meta(s, t) for t in (p,) + st], unchanged, 'None skips parameter and all real state mutators after training')
        s.run([(78, opt), (66, g)])
        return {'algorithm': 'AdamW' if algorithm == 2 else 'Adam', 'updates': 4,
                'real_dropout_forward_backward_state_step_clear_reset_in_one_owner': True,
                'exact_math': 'beta1=beta2=0; eps=2; noise in {0,2}; update is 0 or 1/16; AdamW factor31/32',
                'source_model_contract_not_new_hosted_Torch_case': True}

    case('op41-private-optimizer-and-accumulator-roots', graph41)
    case('dropout-Saved-noise-and-real-state-alias-lifetimes', dropout_saved)
    case('Adam-step-then86-state-write-prefix-and-rollback', partial86)
    case('AdamW-step-then87-leaf-and-state-view-errors', partial87)
    case('Adam-dropout-clear-step-reset-four-updates', lambda s: training(s, 1))
    case('AdamW-dropout-clear-step-reset-four-updates', lambda s: training(s, 2))
    for path, raw in snapshots.items():
        assert path.read_bytes() == raw, str(path)
    report['input_custody_after'] = True
    report['case_count'] = len(report['cases'])
    report['status'] = 'PASS' if all(r['status'] == 'PASS' for r in report['cases']) else 'FAIL'
    save()
    print(json.dumps({k: report[k] for k in ('status', 'case_count', 'frames_before_EOF',
        'exact_assertions', 'exact_float32_words', 'complete_MT_state_comparisons', 'rollback_bytes_compared')}))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
