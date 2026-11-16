#!/usr/bin/env python3
"""Independent I64 resource controls on the actual native owned session.

Qualification covers the explicit cases below. The host only supplies wire
packets and observes its child's memory. Numeric cases are dyadic controls or
are bound to a previously frozen real PyTorch observation; no Python/NumPy
numerical implementation is substituted for the native target.
"""
import argparse
import json
from pathlib import Path
import struct
import traceback

from resource_support import (Custody, IMAGE, SOURCE, State, create, create_packet,
                              dump_raw, integer_raw, load, sha, write_json)


class Case:
    def __init__(self, wire, image, root, name, report, address_limit=None):
        self.w = wire
        self.out = root / name
        self.out.mkdir()
        self.row = {'name': name, 'status': 'RUNNING', 'checks': 0, 'rollback_bytes': 0,
                    'events': [], 'RLIMIT_AS_bytes': address_limit}
        report['cases'].append(self.row)
        self.s = wire.Session(image, directory=self.out / 'transport', address_limit=address_limit)

    def one(self, *args):
        return self.s.run([args])[0]

    def check(self, name, actual, expected):
        if actual != expected:
            def describe(v):
                return {'bytes': len(v), 'sha256': sha(v)} if isinstance(v, bytes) else v
            raise AssertionError((name, describe(actual), describe(expected)))
        self.row['checks'] += 1

    def state(self):
        return State(self.s.owner()['bytes'])

    def meta(self, tid):
        return self.w.meta(self.one(13, tid))

    def typed(self, dtype, shape, raw=b'', requires=0):
        return create(self.w, self.s, dtype, shape, raw, requires)

    def view(self, tid, shape, strides, offset=0):
        args = struct.pack('<10Q', len(shape), *shape, *([0]*(4-len(shape))),
                           *strides, *([0]*(4-len(strides))), offset)
        return self.one(69, tid, self.w.Payload(args), 80)['value']

    def raw(self, tid, count):
        return self.one(98, tid, 0, count)['payload']

    def refusal(self, name, commands, expected, generator=None, maps=False, packet=None):
        before = self.s.owner()
        before_state = State(before['bytes'])
        rng = before_state.rng(generator) if generator else None
        proc = Path(f'/proc/{self.s.pid}/maps')
        maps_before = proc.read_bytes() if maps else None
        raw = self.s.exchange(self.w.packet(commands) if packet is None else packet)
        status, rows = self.w.parse(raw)
        after = self.s.owner()
        maps_after = proc.read_bytes() if maps else None
        event = {'name': name, 'status': 'RUNNING', 'frame_status': status,
                 'expected_frame_status': expected, 'response_sha256': sha(raw),
                 'owner_before': dump_raw(self.out, name+'-owner-before', before['bytes']),
                 'owner_after': dump_raw(self.out, name+'-owner-after', after['bytes']),
                 'pointer_before': before['pointer'], 'pointer_after': after['pointer']}
        self.row['events'].append(event)
        self.check(name+' status', status, expected)
        self.check(name+' no public partial records', rows, [])
        self.check(name+' pointer/length', (after['pointer'], after['length']), (before['pointer'], before['length']))
        self.check(name+' full owner byte equality', after['bytes'], before['bytes'])
        if generator:
            after_rng = State(after['bytes']).rng(generator)
            self.check(name+' full 5056B RNG', after_rng, rng)
            event['rng_before'] = dump_raw(self.out, name+'-rng-before', rng)
            event['rng_after'] = dump_raw(self.out, name+'-rng-after', after_rng)
        if maps:
            self.check(name+' real mappings restored', maps_after, maps_before)
            (self.out/(name+'-maps-before.txt')).write_bytes(maps_before)
            (self.out/(name+'-maps-after.txt')).write_bytes(maps_after)
            event['maps_before_sha256'] = sha(maps_before)
            event['maps_after_sha256'] = sha(maps_after)
        self.row['rollback_bytes'] += len(before['bytes'])
        self.check(name+' committed audit', self.one(53)['value'], 1)
        event['status'] = 'PASS'

    def closed(self):
        self.check('final committed audit', self.one(53)['value'], 1)
        owner = self.s.owner()
        self.row['closed_state'] = State(owner['bytes']).final_pages_only()
        self.row['closed_owner'] = dump_raw(self.out, 'closed-owner', owner['bytes'])

    def finish(self):
        self.row['frames_before_EOF'] = self.s.frames
        self.s.finish()
        self.row['EOF_exit_code'] = self.s.exit_code
        self.row['all_responses_before_stdin_EOF'] = True


def saved_weak_reuse(c, observed):
    values = [-(1 << 63), (1 << 63)-1, -1, 0, 17]
    a = c.typed(2, [5], integer_raw(values))
    b = c.typed(2, [3], integer_raw([101, 102, 103]))
    d = c.typed(2, [4], integer_raw([201, 202, 203, 204]))
    meta = c.meta(a)
    sid = meta['storage_id']
    initial = c.state().row(sid, 2)
    c.check('I64 real requested/rounded extent', initial[3:5], (40, 48))
    c.check('I64 storage uses actual 8-byte raw', c.state().storage_bytes(sid), integer_raw(values))
    saved = c.one(46, a)['value']
    weak = c.one(42, a)['value']
    c.check('Saved is a real Storage strong owner', c.state().row(sid, 2)[5:7], (2, 2))
    c.one(41, a)
    hold = c.one(43, weak)['value']
    c.refusal('I64_Hold_reject_F32_read', [(50, hold, 0, 1)], 7)
    c.refusal('I64_Hold_reject_empty_F32_read', [(50, hold, 0, 0)], 7)
    c.one(45, hold)
    restored = c.one(56, saved)['value']
    rm = c.meta(restored)
    c.check('Saved unpack preserves real dtype', struct.unpack('<4Q', c.one(96, restored)['payload'])[:2], (2, 8))
    c.check('Saved shares same Storage and Version', (rm['storage_id'], rm['version_id']), (sid, meta['version_id']))
    c.check('Saved reads all 64-bit payload', c.raw(restored, 5), integer_raw(values))
    c.one(100, restored, 1 << 63)
    c.check('fill increments shared Version once', c.meta(restored)['version'], meta['version']+1)
    c.refusal('stale_Saved_unpack', [(56, saved)], 6)
    c.one(48, saved)
    c.one(41, restored)
    tomb = c.state().row(sid, 2)
    c.check('expired payload clears only physical extent', (tomb[2], tomb[3], tomb[4], tomb[5], tomb[6], tomb[7]),
            (0, 40, 0, 0, 1, 1))
    fresh = c.typed(2, [5], integer_raw([31, 32, 33, 34, 35]))
    fm = c.meta(fresh)
    fs = c.state().row(fm['storage_id'], 2)
    c.check('real interior extent reused', (fs[2], fs[4]), (initial[2], initial[4]))
    c.check('new storage identity despite reused bytes', fm['storage_id'] != sid, True)
    c.refusal('old_Weak_not_resurrected_by_extent_reuse', [(43, weak)], 6)
    c.check('left surviving allocation unchanged', c.raw(b, 3), integer_raw([101, 102, 103]))
    c.check('right surviving allocation unchanged', c.raw(d, 4), integer_raw([201, 202, 203, 204]))
    c.state().partition()
    c.one(44, weak)
    for tid in (b, d, fresh):
        c.one(41, tid)
    c.closed()


def empty_saved_growth(c, observed):
    base = c.typed(2, [0])
    view = c.view(base, [2, 0, 3], [17, 101, 2], 7)
    vm = c.meta(view)
    c.check('legal empty element offset becomes 8-byte offset', vm['storage_offset'], 56)
    saved = c.one(46, view)['value']
    weak = c.one(42, view)['value']
    gen = c.one(60, 98765)['value']
    before = c.state()
    rng = before.rng(gen)
    for _ in range(270):
        c.typed(2, [0])
    for n in range(3):
        t = c.typed(2, [2, 32768], integer_raw([(1 << 63) + n]) * 65536)
        c.check('grown 8-byte extent tail', c.one(98, t, 65535, 1)['payload'], integer_raw([(1 << 63)+n]))
        c.check('grown Storage byte width', c.meta(t)['storage_bytes'], 524288)
    grown = c.state()
    c.check('record table actually grew', grown.slots > before.slots, True)
    c.check('pool actually grew', grown.pool > before.pool, True)
    c.check('second actual descriptor page allocated', grown.q(4104) != 0, True)
    c.check('full RNG survives table/page/pool migration', grown.rng(gen), rng)
    c.row['migration'] = {'slots_before': before.slots, 'slots_after': grown.slots,
                          'pool_before': before.pool, 'pool_after': grown.pool}
    c.one(33, 0)
    c.check('reset drops every public Tensor while Saved is independent', c.state().counts()[0], 0)
    restored = c.one(56, saved)['value']
    rm = c.meta(restored)
    for key in ('shape', 'strides', 'storage_id', 'version_id', 'version', 'storage_offset', 'storage_bytes'):
        c.check('Saved empty migration '+key, rm[key], vm[key])
    c.check('Saved empty dtype/width', struct.unpack('<4Q', c.one(96, restored)['payload'])[:2], (2, 8))
    c.check('empty read is actual zero bytes', c.raw(restored, 0), b'')
    c.check('RNG also survives graph_reset', c.state().rng(gen), rng)
    c.one(41, restored)
    c.one(48, saved)
    c.refusal('empty_Weak_expires_after_last_Saved', [(43, weak)], 6, gen)
    c.one(44, weak)
    c.one(66, gen)
    c.closed()


def late_prefix(c, observed):
    ref = observed['rows'][46]
    assert ref['index'] == 46 and ref['name'] == 'dropout-dense-true-0.5'
    assert ref['observation']['operation']['outcome'] == 'RAISED'
    base = c.typed(2, [2, 3], bytes.fromhex(ref['observation']['before']['x']['storage_bytes_hex']))
    x = c.one(6, base, 0, 1)['value']
    sentinel = c.typed(2, [1], integer_raw([9]))
    gen = c.one(60, 1729)['value']
    rng_before = bytes.fromhex(observed['states'][ref['state_before']]['hex'])
    rng_after = bytes.fromhex(observed['states'][ref['state_after']]['hex'])
    c.check('fixed actual54 full initial RNG', c.state().rng(gen), rng_before)
    xm = c.meta(x)
    before = c.state()
    before_counts = before.counts()
    response = c.s.exchange(c.w.packet([(106, sentinel, 1),
        (105, x, gen, 0x3fe0000000000000, 1), (106, sentinel, 100), (999,)]))
    status, rows = c.w.parse(response)
    after = c.state()
    c.check('late cast normal header', status, 0)
    c.check('late cast stops both valid and invalid suffix', [r['opcode'] for r in rows], [106, 105])
    c.check('late cast explicit error and unpublished output/mask',
            (rows[1]['value'], rows[1]['payload']), (0, struct.pack('<2Q', 12, 0)))
    c.check('prefix scalar write commits once', c.raw(sentinel, 1), integer_raw([10]))
    c.check('prefix shared Version commits once', c.meta(sentinel)['version'], 1)
    c.check('late cast full actual PyTorch RNG', after.rng(gen), rng_after)
    c.check('input remains exact original 64-bit values', c.raw(x, 6),
            bytes.fromhex(ref['observation']['after']['x']['logical_bytes_hex']))
    c.check('input metadata unchanged', c.meta(x), xm)
    c.check('both unpublished temporaries actually allocated', after.next, before.next + 2)
    for node in range(before.next, after.next):
        desc = after.node(node)
        c.check('late cast dead temporary '+str(node),
                (desc[304//8], desc[200//8], desc[208//8], desc[600//8]), (0, 0, 0, 0))
    c.check('late cast exact live kind counts preserved', after.counts(), before_counts)
    c.check('late cast all actual free extents restored', after.free(), before.free())
    after.partition()
    c.row['late_before'] = dump_raw(c.out, 'late-before', before.raw)
    c.row['late_after'] = dump_raw(c.out, 'late-after', after.raw)
    c.row['full_state_reference_index'] = 46
    c.refusal('ordinary_bad_reserved_after_prefix_is_atomic',
              [(106, sentinel, 7), (105, x, gen, 0x3fe0000000000000, 1, 1)], 7, gen)
    for tid in (x, base, sentinel):
        c.one(41, tid)
    c.one(66, gen)
    c.closed()


def promotion_temporaries(c, observed):
    base = c.typed(2, [11], integer_raw([0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0]))
    view = c.view(base, [3], [4], 1)
    f = c.typed(0, [3], struct.pack('<3f', .5, .5, .5))
    saved = c.one(46, view)['value']
    weak = c.one(42, view)['value']
    vm = c.meta(view)
    before = c.state()
    y = c.one(103, 1, view, f)['value']
    after = c.state()
    c.check('actual CPU conversion and result each allocate a Tensor', after.next, before.next+2)
    temp = after.node(before.next)
    c.check('promoted temporary truly collected', (temp[304//8], temp[208//8], temp[600//8]), (0, 0, 0))
    c.check('sole surviving Tensor is public output', after.tensor_node(y)[0], before.next+1)
    expected_counts = before.counts()
    for index in (0, 1, 2):
        expected_counts[index] += 1
    c.check('no surviving temporary S/V/Hold', after.counts(), expected_counts)
    c.check('real promoted numerical result', c.raw(y, 3), struct.pack('<3f', 3.5, 3.5, 3.5))
    ym = c.meta(y)
    c.check('result is a distinct F32 Storage', (ym['storage_bytes'], ym['storage_id'] != vm['storage_id']), (12, True))
    c.check('only output rounded extent remains allocated',
            sum(s for _, s in before.free()) - sum(s for _, s in after.free()), 16)
    c.refusal('promotion_then_ordinary_error_rolls_everything_back', [(103, 1, view, f), (999,)], 4)
    c.one(41, view)
    c.one(41, base)
    restored = c.one(56, saved)['value']
    c.check('independent Saved still reads I64 strided source', c.raw(restored, 3), integer_raw([3, 3, 3]))
    c.one(41, restored)
    c.one(48, saved)
    c.refusal('source_Weak_expires_independently_of_F32_result', [(43, weak)], 6)
    c.check('F32 result owns independent data after I64 release', c.raw(y, 3), struct.pack('<3f', 3.5, 3.5, 3.5))
    c.one(44, weak)
    for tid in (y, f):
        c.one(41, tid)
    c.closed()


def ordinary_admission(c, observed):
    i = c.typed(2, [], integer_raw([3]))
    f = c.typed(0, [], struct.pack('<f', .5), 1)
    gen = c.one(60, 1729)['value']
    # New09 source-bound AD admission; old07 refusal is preserved separately.
    result = c.one(103, 1, f, i)['value']
    c.check('new mixed add is a differentiable real Tensor', c.meta(result)['requires_grad'], 1)
    c.check('new mixed add dyadic value', c.one(98, result, 0, 1)['payload'], struct.pack('<f', 3.5))
    c.check('add has no saved self', c.one(107, result, 0)['value'], 0)
    c.check('add has no saved other', c.one(107, result, 1)['value'], 0)
    c.one(7, result)
    c.check('new mixed add actually propagates F32 gradient', c.one(40, f, 0, 1)['payload'], struct.pack('<f', 1.))
    c.one(70, f)
    c.one(41, result)
    c.refusal('requires_grad_inplace_still_rejected', [(104, 1, i, f)], 5, gen)
    c.refusal('public_F32_grad_rejects_I64_binding', [(92, f, i)], 2, gen)
    c.refusal('I64_requires_grad_creation_rejected', [], 5, gen,
              packet=create_packet(2, [], integer_raw([1]), 1))
    c.refusal('I64_105_invalid_probability_is_atomic', [(105, i, gen, 0x7ff8000000000001, 1)], 7, gen)
    c.refusal('I64_105_invalid_generator_is_atomic', [(105, i, 1, 0x3fe0000000000000, 1)], 1, gen)
    c.check('failed binding leaves original grad None', c.one(59, f)['value'], 0)
    c.one(41, i)
    c.one(41, f)
    c.one(66, gen)
    c.closed()


def real_oom(c, observed):
    count = 65536
    i = c.typed(2, [2, 32768], integer_raw([3])*count)
    f = c.typed(0, [2, 32768], struct.pack('<f', .5)*count)
    gen = c.one(60, 1729)['value']
    fork = c.one(65, gen)['value']
    fillers = []
    while True:
        state = c.state()
        available = sum(size for _, size in state.free())
        if state.pool == 16 << 20 and 262144 <= available < 524288:
            break
        assert state.pool <= 16 << 20 and len(fillers) < 80, ('bounded setup', state.pool, available, len(fillers))
        fillers.append(c.typed(2, [2, 16384], integer_raw([0])*32768))
    c.row['setup'] = {'pool': state.pool, 'free': available, 'fillers': len(fillers),
                      'owner_bytes': len(state.raw), 'address_limit_bytes': 56 << 20}
    (c.out/'child-limits.txt').write_bytes(Path(f'/proc/{c.s.pid}/limits').read_bytes())
    (c.out/'child-status.txt').write_bytes(Path(f'/proc/{c.s.pid}/status').read_bytes())
    c.refusal('real_OOM_after_promoted_temporary', [(103, 1, i, f)], 3, gen, maps=True)
    c.refusal('real_OOM_after_mask_before_rng', [(105, i, gen, 0x3fe0000000000000, 1)], 3, gen, maps=True)
    for _ in range(5):
        c.one(41, fillers.pop())
    y = c.one(103, 1, i, f)['value']
    c.check('same mixed operation recovers without increasing pool', c.state().pool, 16 << 20)
    c.check('recovered mixed operation computes every F32 value', c.raw(y, count), struct.pack('<f', 3.5)*count)
    c.one(41, y)
    rng_before = c.state().rng(gen)
    status, rows = c.w.parse(c.s.exchange(c.w.packet([(105, i, gen, 0x3fe0000000000000, 1), (999,)])))
    c.check('same dropout allocation recovers and commits real late error',
            (status, len(rows), rows[0]['payload']), (0, 1, struct.pack('<2Q', 12, 0)))
    c.one(67, fork, count*2)
    c.check('recovered dropout full MT state equals independent raw32 advance', c.state().rng(gen), c.state().rng(fork))
    c.check('successful late error actually consumes state', c.state().rng(gen) != rng_before, True)
    c.check('late-error recovery remains in same pool', c.state().pool, 16 << 20)
    c.check('large source 8-byte values remain unchanged', c.raw(i, count), integer_raw([3])*count)
    for tid in fillers + [i, f]:
        c.one(41, tid)
    c.one(66, gen)
    c.one(66, fork)
    c.closed()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'protocol', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-source-sha256', default=SOURCE)
    parser.add_argument('--expected-native-sha256', default=IMAGE)
    parser.add_argument('--only', choices=['real_RLIMIT_AS_allocation_and_recovery'])
    parser.add_argument('--mixed-reference', type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    custody = Custody(out)
    report = {'schema': 'heptatorch.i64-independent-native-resources.v1', 'status': 'RUNNING',
              'source_sha256': a.expected_source_sha256, 'native_sha256': a.expected_native_sha256, 'native_executed': False,
              'Torch_executed_here': False, 'cases': [], 'whole_pytorch_gate': 'HOLD',
              'pytorch_api_denominator': 2590, 'full_pytorch_rewrite_complete': False,
              'observer_scope': 'limited resource/extent inspection of real producer states; not arbitrary machine-code or foreign owner certification'}
    try:
        custody.bind(Path(__file__), copy='executed-test.py')
        mixed = json.loads(custody.bind(a.mixed_reference,
            '98f24165401f9aadfd86147113454104ae1c8e969dcc105a16b6cb83d605bdfb',
            'reference-mixed36.json'))
        assert mixed['status'] == 'OBSERVED' and mixed['cases'] == 36 and mixed['probe_error_count'] == 0
        assert mixed['oracle_provenance']['status'] == 'PASS'
        assert len(mixed['oracle_provenance']['checks']) == 11 and all(x is True for x in mixed['oracle_provenance']['checks'].values())
        assert mixed['rows'][0]['name'] == 'baseline-add-float-left'
        wire, image, info, observed = load(custody, a.build, a.protocol, a.reference,
                                         a.expected_source_sha256, a.expected_native_sha256)
        report['module_count'] = len(info['units'])
        plan = [('saved_weak_8byte_interior_reuse', saved_weak_reuse, None),
                ('empty_saved_growth_reset', empty_saved_growth, None),
                ('late_cast_prefix_suffix_gc', late_prefix, None),
                ('real_CPU_promotion_temporary_ownership', promotion_temporaries, None),
                ('ordinary_admission_full_rollback', ordinary_admission, None),
                ('real_RLIMIT_AS_allocation_and_recovery', real_oom, 56 << 20)]
        report['declared_complete_plan'] = [name for name, _, _ in plan]
        if a.only:
            plan = [row for row in plan if row[0] == a.only]
        report['selected_cases'] = [name for name, _, _ in plan]
        report['complete_six_case_plan_selected'] = a.only is None
        for name, function, limit in plan:
            c = Case(wire, image, out, name, report, limit)
            report['native_executed'] = True
            try:
                function(c, observed)
                c.row['status'] = 'PASS'
            except BaseException as error:
                c.row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
            finally:
                try:
                    c.finish()
                except BaseException as error:
                    c.row.update(status='FAIL', transport_error=repr(error))
                write_json(c.out/'report.json', c.row)
                write_json(out/'report.json', report)
        report['status'] = 'PASS' if all(c['status'] == 'PASS' for c in report['cases']) else 'FAIL'
    except BaseException as error:
        report.update(status='FAIL', preparation_error=repr(error), traceback=traceback.format_exc())
    try:
        report['inputs'] = custody.finish()
        report['source_custody_after'] = True
    except BaseException as error:
        report.update(status='FAIL', custody_error=repr(error))
    report['case_count'] = len(report['cases'])
    report['passed'] = sum(c['status'] == 'PASS' for c in report['cases'])
    report['failed'] = sum(c['status'] != 'PASS' for c in report['cases'])
    report['frames_before_EOF'] = sum(c.get('frames_before_EOF', 0) for c in report['cases'])
    report['checks'] = sum(c['checks'] for c in report['cases'])
    report['complete_rollback_bytes'] = sum(c['rollback_bytes'] for c in report['cases'])
    write_json(out/'report.json', report)
    print(json.dumps({k: report.get(k) for k in ('status', 'case_count', 'passed', 'failed', 'frames_before_EOF', 'checks', 'preparation_error')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
