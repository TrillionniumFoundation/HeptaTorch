"""Independent native checks for physical Preserve-layout incoming storage.

All arithmetic, storage ownership, GC, allocation and failure handling execute
in the native program. This host only frames fixed requests, checks exact
dyadic derivatives and observes its own child's committed owner read-only.
It imports no implementation or authoring code and is not a Torch oracle.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import struct
import traceback

CONTROLS = '43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
OBSERVER = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path, pin):
    assert sha(path.read_bytes()) == pin
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'controls', 'observer', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-native-sha256', required=True)
    p.add_argument('--expected-source-sha256', required=True)
    p.add_argument('--state-format', default='HTGAD002')
    p.add_argument('--backing-format', default='HTOGAD02')
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {x.resolve(): x.read_bytes() for x in
              (Path(__file__), a.binary, a.source, a.controls, a.observer)}
    assert sha(a.binary.read_bytes()) == a.expected_native_sha256
    assert sha(a.source.read_bytes()) == a.expected_source_sha256
    (out / 'executed-test.py').write_bytes(Path(__file__).read_bytes())
    b = load('preserve_storage_codec', a.controls, CONTROLS)
    observer = load('preserve_storage_owner_observer', a.observer, OBSERVER)

    def full(c, shape, value, requires=False):
        return c.make(38, len(shape), int(requires), *shape,
                      *([0] * (4 - len(shape))), b.bits(value))

    def words(c, *command):
        return b.words(c.one(*command))

    def back(c, root, seed, retain=0):
        reply = c.one(58, root, seed, retain)
        data = b.words(reply)
        assert reply['value'] == 0 and data[0] == 0, data
        return data

    def invariant(c):
        owner = c.session.owner()['bytes']
        q = lambda at: struct.unpack_from('<Q', owner, at)[0]
        assert owner[64:72] == a.state_format.encode('ascii')
        assert owner[139264:139272] == a.backing_format.encode('ascii')
        assert all(q(x) == 0 for x in (112, 3968, 3976, 3984, 3992, 4000, 4008))
        pool = 139264 + 144 + 552 * q(139280)
        live = []
        for node in range(1, q(8)):
            page = q(4096 + (node >> 8) * 8)
            if not page:
                continue
            assert page >> 63
            at = pool + (page & ((1 << 63) - 1)) + (node & 255) * 640
            assert all(q(at + f) == 0 for f in range(448, 576, 8)), (
                'incoming temporary survived commit', node)
            if q(at + 304):
                assert all(q(at + f) == 0 for f in
                           (24, 248, 336, 368, 376, 416, 424, 432, 440))
                assert (q(at + 160) == 0) == (q(at + 256) == 0)
                assert (q(at + 360) == 0) == (q(at + 384) == 0)
                live.append(node)
        c.comparisons += len(live)
        c.audit()
        return len(live)

    def prime(c):
        x = c.new([], [0.])
        c.one(41, x)
        c.one(33, 0)
        return words(c, 54)

    def finish_graph(c, generator):
        c.one(33, 0)
        c.one(66, generator)
        c.one(33, 0)
        assert c.make(53) == 1
        return words(c, 54)

    def graph(c, *, big=False, zero_first=True):
        shape = [128, 512] if big else [2, 3]
        base = full(c, shape, .5)
        param = c.make(6, base, 0, 1)
        c.one(90, param, 1)
        c.one(41, base)
        logical = [shape[1], shape[0]]
        coefficient = full(c, logical, 2.) if big else c.new(logical, [1., 2., 3., 4., 5., 6.])
        generator = c.make(60, 17)
        if zero_first:
            product = c.make(2, 3, param, coefficient)
            bernoulli = c.make(82, param, generator, 0)
        else:
            bernoulli = c.make(82, param, generator, 0)
            product = c.make(2, 3, param, coefficient)
        root = c.make(2, 1, product, bernoulli)
        seed_base = full(c, shape, 1.)
        seed = c.make(6, seed_base, 0, 1)
        c.one(41, seed_base)
        return param, coefficient, generator, product, bernoulli, root, seed

    report = {
        'schema': 'heptatorch.preserve-incoming-storage-native-lifetimes.v1',
        'status': 'RUNNING',
        'native_sha256': a.expected_native_sha256,
        'source_sha256': a.expected_source_sha256,
        'test_sha256': sha(Path(__file__).read_bytes()),
        'inputs_sha256': {str(path): sha(data) for path, data in inputs.items()},
        'expectation': 'Exact derivative of p*c + Bernoulli(p) is c for all-one seed; all fresh incoming reserves are transient, and failed admissions preserve the entire owner.',
        'native_executed': True, 'pinned_Torch_executed_here': False,
        'full_rewrite_complete': False, 'pytorch_api_denominator': 2590,
        'rows': [],
    }

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def case(name, body, address_limit=128 << 20):
        directory = out / name
        s = observer.OwnerSession(a.binary.resolve(), directory=directory,
                                  address_limit=address_limit)
        c = b.Check(s)
        row = {'name': name, 'status': 'RUNNING', 'address_limit': address_limit}
        report['rows'].append(row)
        save()
        try:
            row['details'] = body(c, directory)
            row['live_nodes_at_final_check'] = invariant(c)
            row['status'] = 'PASS'
        except Exception as error:
            row.update(status='FAIL', exception_type=type(error).__name__,
                       message=str(error)[:4096], traceback=traceback.format_exc()[-12000:])
            try:
                observed = s.owner()
                (directory / 'failure.owner.bin').write_bytes(observed['bytes'])
                row['failure_owner'] = {'bytes': observed['length'], 'sha256': sha(observed['bytes'])}
            except Exception as observation_error:
                row['observation_exception'] = repr(observation_error)
        finally:
            try:
                row['transport'] = s.finish()
            except Exception as error:
                row.update(status='FAIL', transport_exception=repr(error))
            row.update(frames=s.frames, exact_checks=c.comparisons,
                       rollback_bytes=c.rollback_bytes, frame_records=c.frames)
            save()

    def partial_cleanup(c, directory):
        baseline = prime(c)
        param, coefficient, generator, product, bernoulli, root, seed = graph(c)
        weak = c.make(42, coefficient)
        alias = c.make(19, coefficient)
        c.one(41, coefficient)
        c.one(12, alias, 0, b.bits(8.))
        mt_before = c.one(63, generator)['payload']
        reply = c.one(58, root, seed, 0)
        fields = b.words(reply)
        assert reply['value'] == 6 and fields == (6, 2, 0, 3), fields
        assert c.make(59, param) == 0 and c.make(91, param) == 0
        c.expiry(weak, False)
        c.one(41, alias)
        c.expiry(weak, False)
        checked = invariant(c)
        assert c.one(63, generator)['payload'] == mt_before
        c.one(33, 0)
        c.expiry(weak, True)
        c.one(44, weak)
        after = finish_graph(c, generator)
        assert after[5:11] == baseline[5:11], (after[5:11], baseline[5:11])
        assert after[4] == baseline[4], (after[4], baseline[4])
        return {'runtime_error': list(fields), 'leaf_remained_None': True,
                'committed_nodes_checked_after_partial_error': checked,
                'no_backward_RNG_consumption': True,
                'Saved_coefficient_kept_until_graph_release': True,
                'all_object_counts_and_free_bytes_returned_to_primed_baseline': True}

    case('partial-error-clears-unready-physical-reserve-and-Saved-lifetime', partial_cleanup)

    def rollback(c, directory):
        param, coefficient, generator, product, bernoulli, root, seed = graph(c)
        before = c.session.owner()
        (directory / 'before.owner.bin').write_bytes(before['bytes'])
        c.run([(58, root, seed, 0), (999,)], status=4,
              label='a suffix admission failure rolls back successful strided incoming adoption')
        after = c.session.owner()
        (directory / 'after.owner.bin').write_bytes(after['bytes'])
        assert after == before
        assert c.make(59, param) == 0 and c.make(91, param) == 0
        invariant(c)
        back(c, root, seed)
        grad = c.make(91, param)
        c.values(grad, [1., 2., 3., 4., 5., 6.])
        assert words(c, 13, grad)[7:9] == (1, 3)
        invariant(c)
        return {'entire_owner_rollback_bytes': before['length'],
                'before_owner_sha256': sha(before['bytes']),
                'after_owner_sha256': sha(after['bytes']),
                'same_graph_recovered_exact_derivative': True,
                'native_gradient_retained_transpose_layout': True}

    case('suffix-failure-rolls-back-real-strided-gradient-adoption', rollback)

    def repeat(c, directory):
        prime(c)
        base = c.new([2, 3], [.5] * 6)
        param = c.make(6, base, 0, 1)
        c.one(90, param, 1)
        c.one(41, base)
        generator = c.make(60, 29)
        output = c.make(82, param, generator, 0)
        seedbase = full(c, [2, 3], 1.)
        seed = c.make(6, seedbase, 0, 1)
        c.one(41, seedbase)
        mt = c.one(63, generator)['payload']
        back(c, output, seed)
        c.one(70, param)
        baseline = words(c, 54)
        previous = []
        for _ in range(6):
            back(c, output, seed)
            g = c.make(91, param)
            alias = c.make(19, g)
            weak = c.make(42, g)
            meta = words(c, 13, g)
            assert meta[7:9] == (1, 3) and meta[13] == 0
            assert meta[11] not in previous
            previous.append(meta[11])
            c.values(g, [0.] * 6)
            c.one(70, param)
            c.one(41, g)
            c.expiry(weak, False)
            c.one(41, alias)
            c.expiry(weak, True)
            c.one(44, weak)
            invariant(c)
            after = words(c, 54)
            assert after[4:11] == baseline[4:11], (after[4:11], baseline[4:11])
        assert c.one(63, generator)['payload'] == mt
        finish_graph(c, generator)
        return {'repetitions': 6, 'fresh_storage_generations': previous,
                'held_gradient_alias_outlived_None': True,
                'last_alias_expired_each_storage': True,
                'same_live_counts_and_free_bytes_after_each_release': True,
                'full_Generator_state_unchanged': True}

    case('repeated-physical-zero-gradient-adoption-and-final-alias-release', repeat)

    def oom(c, directory):
        param, coefficient, generator, product, bernoulli, root, seed = graph(c, big=True)
        back(c, root, seed, 1)
        c.one(70, param)
        large = full(c, [1024, 1024], 1.)
        fillers = [full(c, [128, 1024], 1.) for _ in range(3)]
        before = c.session.owner()
        (directory / 'before.owner.bin').write_bytes(before['bytes'])
        c.one(58, root, seed, 1, status=3,
              label='real address-space failure in admission including Preserve reserve')
        after = c.session.owner()
        (directory / 'after.owner.bin').write_bytes(after['bytes'])
        assert after == before
        assert c.make(59, param) == 0 and c.make(91, param) == 0
        invariant(c)
        c.run([(41, t) for t in fillers])
        back(c, root, seed, 1)
        grad = c.make(91, param)
        for start in range(0, 65536, 8192):
            payload = c.one(39, grad, start, 8192)['payload']
            assert payload == struct.pack('<I', b.bits(2.)) * 8192, ('wrong recovered chunk', start)
            c.comparisons += 8192
        assert words(c, 13, grad)[7:9] == (1, 512)
        c.one(41, grad)
        c.one(70, param)
        invariant(c)
        return {'address_space_limit_bytes': 32 << 20,
                'entire_owner_rollback_bytes': before['length'],
                'before_owner_sha256': sha(before['bytes']),
                'after_owner_sha256': sha(after['bytes']),
                'unready_gradient_remained_None': True,
                'recovered_after_releasing_unrelated_extents': True,
                'exact_gradient_words_after_recovery': 65536,
                'large_unrelated_tensor_still_live': large}

    case('real-ENOMEM-preserves-committed-state-and-recovers', oom, 32 << 20)
    assert all(path.read_bytes() == data for path, data in inputs.items())
    report.update(status='PASS' if all(x['status'] == 'PASS' for x in report['rows']) else 'FAIL',
                  cases=len(report['rows']), passed=sum(x['status'] == 'PASS' for x in report['rows']),
                  frames=sum(x['frames'] for x in report['rows']),
                  exact_checks=sum(x['exact_checks'] for x in report['rows']),
                  rollback_bytes=sum(x['rollback_bytes'] for x in report['rows']))
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'inputs_sha256')}, indent=2))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
