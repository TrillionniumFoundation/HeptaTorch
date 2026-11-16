"""Replay the fixed 16-family/25-variant Torch materialization observations.

Numerical and metadata expectations come only from the frozen hosted report.
The three hook/create_graph variants remain explicit unrepresented API paths.
The source imports only frozen transport and independent observation helpers.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import struct
import traceback

REFERENCE_SHA = 'e4a83999c853f07d14a7db2fceafef664ca6d92151b5da4f5252972c690dc044'
PROBE_SHA = '8706090d731cf705e68a2e474693d10448b1fa00371ca8ddd5db36163500bd69'
BASE_REPLAY_SHA = '8a9b56c837e9c7b3af47ed401a91e932fb3359a4c109157df738e5aa63e64f85'
CONTROLS_SHA = '43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
OBSERVER_SHA = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path, pin):
    assert sha(path.read_bytes()) == pin, path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_class(base):
    class Replay(base.Replay):
        def __init__(self, check, codec, arguments):
            super().__init__(check, codec)
            self.args = arguments
            self.mismatches = []
            self.snapshots = []

        def eq(self, actual, expected, label):
            self.c.comparisons += 1
            if actual != expected:
                self.mismatches.append({'field': label, 'actual': actual, 'expected': expected})

        def identity(self, table, reference, native, label):
            if reference in table:
                self.eq(native, table[reference], label+' stable opaque relation')
            else:
                self.eq(native not in table.values(), True, label+' distinct opaque relation')
                table[reference] = native

        def raw_node(self, tid=None, node=None):
            owner = self.c.session.owner()
            raw = owner['bytes']
            def u(offset):
                assert 0 <= offset <= len(raw)-8
                return struct.unpack_from('<Q', raw, offset)[0]
            assert raw[64:72] == self.args.state_format.encode('ascii')
            assert raw[139264:139272] == self.args.backing_format.encode('ascii')
            slots = u(139264+16)
            poolbase = 139264+144+552*slots
            if tid is not None:
                assert (tid >> 24) & 255 == 1
                slot = tid & 0xffffff
                assert 0 < slot <= slots
                row = 139264+128+(slot-1)*64
                assert u(row) == tid >> 32 and u(row+8) == 1
                node = u(row+48)
            assert node is not None and 0 < node < u(8)
            page = u(4096+(node >> 8)*8)
            assert page >> 63 == 1
            at = poolbase+(page & ((1 << 63)-1))+(node & 255)*self.args.descriptor_bytes
            assert u(at+304) == 1
            if tid is not None:
                assert u(at+192) == tid and u(at+200) == 1
            return {offset: u(at+offset) for offset in
                    (32,40,192,200,208,240,312,320,360,384,392,400,408)}

        def snapshot_all(self, reference, params, output, seed, held):
            observed = self.snapshot(reference, params, held)
            observed['output'] = self.tensor(output, reference['output'])
            self.eq(seed is None, reference['held_seed'] is None, 'held seed presence')
            observed['held_seed'] = None
            if seed is not None and reference['held_seed'] is not None:
                observed['held_seed'] = self.tensor(seed, reference['held_seed'])
            self.snapshots.append(observed)
            return observed

        def relation(self, a, b):
            ma = self.b.words(self.c.one(13, a))
            mb = self.b.words(self.c.one(13, b))
            return {'same_Tensor': self.raw_node(tid=a)[320] == self.raw_node(tid=b)[320],
                    'same_Storage': ma[11] == mb[11],
                    'same_current_Version_value': ma[13] == mb[13]}

        def constant_mul(self, p, value):
            constant = self.c.new([], [value])
            result = self.c.make(2, 3, p, constant)
            self.c.one(41, constant)
            return result

        def sum(self, t):
            return self.c.make(4, 7, t)

        def construct(self, family, variant):
            held = {}
            seed = None
            inline = False
            if family == 0:
                p = self.c.new([], [2.], True)
                output = p
            elif family == 1:
                p = self.c.new([2], [1., -2.], True)
                output = p
                inline = variant == 'inline'
                if not inline:
                    seed = self.c.new([2], [-0., 2.])
            elif family == 2:
                parent = self.c.new([2, 3], [1., -2., 3., .5, 2., -1.])
                p = self.c.make(6, parent, 0, 1)
                self.c.one(90, p, 1)
                held['parameter_base'] = parent
                output = p
                if variant == 'matching':
                    seed_base = self.c.new([2, 3], [1., 3., -2., 2., -1., -3.])
                    seed = self.c.make(6, seed_base, 0, 1)
                    held['seed_base'] = seed_base
                else:
                    seed = self.c.new([3, 2], [1., 2., 3., -1., -2., -3.])
            elif family == 3:
                parent = self.c.new([4], [99., 1., 99., -2.])
                config = self.blob([1, 2, 0, 0, 0, 2, 0, 0, 0, 1])
                p = self.c.make(69, parent, config, 80)
                self.c.one(90, p, 1)
                output = p
                seed = self.c.new([2], [2., -1.])
                held['parameter_base'] = parent
            else:
                p = self.c.new([2], [1., -2.], True)
                if family == 4:
                    output = self.c.make(18, p)
                    seed = self.c.new([2], [-0., 2.])
                elif family == 5:
                    if variant == 'view':
                        output = self.c.make(17, p, 2, 1, 2, 0, 0)
                        seed = self.c.new([1, 2], [-0., 2.])
                    else:
                        self.c.one(41, p)
                        p = self.c.new([2, 2], [1., -2., .5, 2.], True)
                        output = self.c.make(6, p, 0, 1)
                        seed = self.c.new([2, 2], [1., 2., 3., 4.])
                elif family == 6:
                    factor = self.c.new([2], [2., 3.])
                    product = self.c.make(2, 3, p, factor)
                    output = self.sum(product)
                    self.c.one(41, product)
                    held['coefficient'] = factor
                elif family == 7:
                    q = self.c.new([2], [2., 3.], True)
                    product = self.c.make(2, 3, p, q)
                    output = self.sum(product)
                    self.c.one(41, product)
                    return {'p': p, 'q': q}, output, None, held, False
                elif family == 8:
                    product = self.c.make(2, 3, p, p)
                    output = self.sum(product)
                    self.c.one(41, product)
                elif family == 9:
                    left = self.constant_mul(p, 2.)
                    right = self.constant_mul(p, 3.)
                    added = self.c.make(2, 1, left, right)
                    output = self.sum(added)
                    self.c.one(41, added)
                    held.update(left=left, right=right)
                elif family == 10:
                    shared = self.constant_mul(p, 2.)
                    added = self.c.make(2, 1, shared, shared)
                    output = self.sum(added)
                    self.c.one(41, added)
                    held['shared'] = shared
                elif family == 11:
                    output = self.sum(p)
                    if variant == 'held':
                        seed = self.c.new([], [2.])
                elif family == 12:
                    output = self.c.make(3, 6 if variant == 'neg' else 5, p)
                    seed = self.c.new([2], [2., -1.])
                elif family == 13:
                    output = self.c.make(34, 0, self.blob([2, p, p]))
                    seed = self.c.new([4], [1., 2., 3., 4.])
                elif family == 14:
                    self.c.one(41, p)
                    p = self.c.new([3], [1., -2., 3.], True)
                    output = self.c.make(31, p, 0, self.blob([1, 3, 2, 0, 2]))
                    seed = self.c.new([3], [1., 2., 3.])
                else:
                    assert family == 15
                    product = self.c.make(2, 3, p, p)
                    output = self.sum(product)
                    self.c.one(41, product)
            return {'p': p}, output, seed, held, inline

        def observe(self, family, variant, reference):
            params, output, seed, held, inline = self.construct(family, variant)
            self.snapshot_all(reference['initial'], params, output, seed, held)
            first = reference['events'][0]
            self.snapshot_all(first['before'], params, output, seed, held)
            self.eq(first['outcome'], 'SUCCEEDED', 'actual first backward outcome')
            retain = int(reference['first_retain_graph'])
            if inline:
                records = self.c.run([self.b.tensor([2], [-0., 2.]),
                    (58, output, 0xfe000001, retain), (41, 0xfe000001)])
                self.eq(records[1]['value'], 0, 'native inline-seed backward success')
            else:
                result = self.c.one(58, output, seed or 0, retain)
                self.eq(result['value'], 0, 'native backward success')
            self.snapshot_all(first['after'], params, output, seed, held)
            for name, p in params.items():
                g = self.c.make(91, p)
                expected = reference['first_gradient_relationships'][name]
                self.eq(g == 0, expected is None, 'relationship gradient presence')
                if g and expected is not None:
                    if seed is not None:
                        self.eq(self.relation(g, seed), expected['versus_held_seed'],
                                name+' gradient versus held seed')
                    for key, value in held.items():
                        self.eq(self.relation(g, value), expected['versus_held_Tensors'][key],
                                name+' gradient versus held '+key)
                    self.c.one(41, g)
            if family == 15:
                old = self.c.make(91, params['p'])
                self.snapshot_all(reference['before_clear'], params, output, seed, held)
                self.c.one(92, params['p'], 0)
                if old:
                    held['old_gradient'] = old
                self.snapshot_all(reference['after_clear'], params, output, seed, held)
                second = reference['events'][1]
                self.snapshot_all(second['before'], params, output, seed, held)
                self.eq(second['outcome'], 'SUCCEEDED', 'actual repeated backward outcome')
                result = self.c.one(58, output, seed or 0, 0)
                self.eq(result['value'], 0, 'native repeated backward success')
                self.snapshot_all(second['after'], params, output, seed, held)
                new = self.c.make(91, params['p'])
                self.eq(self.relation(new, old), reference['new_vs_old_gradient'],
                        'recreated gradient versus retained old gradient')
            self.c.audit()
    return Replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','controls','observer','base-replay','observations','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--state-format', default='HTGRAD01')
    parser.add_argument('--backing-format', default='HTOGRAD1')
    parser.add_argument('--descriptor-bytes', type=int, default=448)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert len(args.state_format) == len(args.backing_format) == 8
    assert args.descriptor_bytes >= 448 and args.descriptor_bytes % 64 == 0
    binary = args.binary.read_bytes()
    source = args.source.read_bytes()
    raw = args.observations.read_bytes()
    assert sha(binary) == args.expected_sha256 and sha(source) == args.expected_source_sha256
    assert sha(raw) == REFERENCE_SHA
    reference = json.loads(raw)
    assert reference['status'] == 'OBSERVED' and reference['cases'] == 16 and reference['variants'] == 25
    assert reference['probe_errors'] == 0 and reference['probe_sha256'] == PROBE_SHA
    checks = reference['oracle_provenance']['checks']
    assert len(checks) == 11 and all(value is True for value in checks.values())
    codec = load('materialization_codec', args.controls, CONTROLS_SHA)
    observer = load('materialization_observer', args.observer, OBSERVER_SHA)
    base = load('materialization_base_replay', args.base_replay, BASE_REPLAY_SHA)
    Replay = replay_class(base)
    records = []
    report = {'schema': 'heptatorch.gradient-materialization-native-replay.v1',
              'status': 'RUNNING', 'source_sha256': sha(source), 'native_sha256': sha(binary),
              'runner_sha256': sha(Path(__file__).read_bytes()),
              'reference_sha256': REFERENCE_SHA, 'probe_sha256': PROBE_SHA,
              'base_replay_sha256': BASE_REPLAY_SHA, 'controls_sha256': CONTROLS_SHA,
              'observer_sha256': OBSERVER_SHA, 'oracle_provenance': reference['oracle_provenance'],
              'layout': {'state_format': args.state_format, 'backing_format': args.backing_format,
                         'descriptor_bytes': args.descriptor_bytes},
              'comparison_policy': 'Exact words, layouts, None, Versions and opaque Tensor/Storage relations. No tolerance, counter reset or expected-value substitution.',
              'known_mapping_boundaries': ['No Python hook or create_graph API is represented.',
                  'Enabled versus no_grad calls with create_graph=False use the same native first-order backward operation.',
                  'The inline seed uses one native frame containing construction, backward and public drop; no seed token is retained after the frame.',
                  'Equal current Version values do not prove shared Version identity.'],
              'records': records, 'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    for family, row in enumerate(reference['rows']):
        for variant_index, item in enumerate(row['observations']):
            variant = item['variant']
            result = {'family': family+1, 'case': row['case'], 'variant': variant}
            if (family == 6 and variant != 'no-hook') or (family == 15 and variant == 'create-graph'):
                result.update(status='UNREPRESENTED_API', reason='Python hook or create_graph API is not implemented by this native profile',
                              native_executed=False, comparisons_attempted=0, frames=0, mismatches=[])
                records.append(result)
                continue
            session = observer.OwnerSession(args.binary.resolve(),
                directory=output/f'family-{family+1:02d}-variant-{variant_index+1:02d}', address_limit=64 << 20)
            check = codec.Check(session)
            replay = Replay(check, codec, args)
            result.update(status='RUNNING', native_executed=True)
            try:
                replay.observe(family, variant, item)
                result['status'] = 'FAIL' if replay.mismatches else 'QUALIFIED_EXPOSED_SCOPE'
            except Exception as error:
                result.update(status='FAIL', exception=repr(error), traceback=traceback.format_exc())
            finally:
                try:
                    result['transport'] = session.finish()
                except Exception as error:
                    result.update(status='FAIL', transport_exception=repr(error))
                result.update(frames=session.frames, comparisons_attempted=check.comparisons,
                    rollback_bytes=check.rollback_bytes, mismatches=replay.mismatches,
                    snapshots=replay.snapshots, frame_records=check.frames)
                records.append(result)
                (output/'report.json').write_text(json.dumps(report, indent=2, default=repr)+'\n')
    report.update(reference_families=16, reference_variants=25, executed_variants=sum(r['native_executed'] for r in records),
        qualified_variants=sum(r['status']=='QUALIFIED_EXPOSED_SCOPE' for r in records),
        failed_variants=sum(r['status']=='FAIL' for r in records),
        unrepresented_variants=sum(r['status']=='UNREPRESENTED_API' for r in records),
        comparisons_attempted=sum(r['comparisons_attempted'] for r in records),
        frames=sum(r['frames'] for r in records),
        mismatch_fields=sum(len(r['mismatches']) for r in records))
    report['status'] = 'FAIL' if report['failed_variants'] else 'QUALIFIED_WITH_UNREPRESENTED_APIS'
    assert args.binary.read_bytes() == binary and args.source.read_bytes() == source and args.observations.read_bytes() == raw
    (output/'report.json').write_text(json.dumps(report, indent=2, default=repr)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('records','oracle_provenance')}, indent=2))
    return int(bool(report['failed_variants']))


if __name__ == '__main__':
    raise SystemExit(main())
