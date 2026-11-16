"""Native ownership and transaction controls for separated Tensor / AD history.

The immutable actual9 reference supplies the named runtime outcomes. Additional
controls check protocol atomicity and real owned storage, not new Torch parity.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'protocol', 'reference', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {f.resolve(): f.read_bytes() for f in (a.binary, a.source, a.protocol, a.reference, Path(__file__))}
    assert sha(inputs[a.reference.resolve()]) == 'de7db2f985873be797bd7e167fb374f82b5690021e0a0541d2fb4f7a36b7ad83'
    reference = json.loads(inputs[a.reference.resolve()])
    assert reference['cases'] == 9 and reference['probe_error_count'] == 0
    spec = importlib.util.spec_from_file_location('history_control_wire', a.protocol)
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    rows = []
    names = ('nonleaf-old-history-success', 'nested-history-default-release',
             'old-child-and-atomic-error', 'same-frame-partial-versus-rollback',
             'actual-strong-cycle-and-detach', 'retained-hook-before-failure',
             'captured-accumulator-after-detach-and-reenable', 'retained-old-node-weak-hook')
    for index, name in enumerate(names):
        directory = out / name
        directory.mkdir()
        s = w.Session(a.binary.resolve(), directory=directory/'raw')
        row = {'name': name, 'status': 'RUNNING', 'checks': 0, 'rollback_bytes': 0}
        def one(*command):
            return s.run([command])[0]
        def check(label, got, expected):
            row['checks'] += 1
            assert got == expected, (label, got, expected)
        def metadata(t):
            return w.meta(one(13, t))
        def empty(requires=1):
            base = one(*w.full([0], 0))['value']
            cfg = struct.pack('<10Q', 3, 2, 0, 3, 0, 17, 101, 2, 0, 7)
            t = one(69, base, w.Payload(cfg), 80)['value']
            one(41, base)
            if requires:
                one(90, t, 1)
            return t
        def nd(t):
            result = one(99, t, generator, 4602678819172646912, 1)
            check('same Tensor output', result['value'], t)
            return struct.unpack('<Q', result['payload'])[0]
        def backward(t, retain=0):
            return one(58, t, seed, retain)['value']
        def error(commands, expected):
            before = s.owner()
            status, records = w.parse(s.exchange(w.packet(commands)))
            after = s.owner()
            check('admission error', status, expected)
            check('empty refusal records', records, [])
            check('owner pointer', after['pointer'], before['pointer'])
            check('owner byte length', after['length'], before['length'])
            check('entire committed owner bytes', sha(after['bytes']), sha(before['bytes']))
            row['rollback_bytes'] += before['length']
        def live(weak, expected):
            if expected:
                h = one(43, weak)['value']
                one(45, h)
                check('live Storage handle', h != 0, True)
            else:
                error([(43, weak)], 6)
        try:
            x = empty()
            seed = one(*w.full([2, 0, 3], 1))['value']
            generator = one(60, 1729)['value']
            before_rng = one(63, generator)['payload']
            if index == 0:
                factor = one(*w.full([], 2))['value']
                q = one(2, 3, x, factor)['value']
                mask = nd(q)
                check('old nonleaf apply succeeds', backward(q), 0)
                g = one(91, x)['value']
                expected = reference['rows'][8]['observation']['backward']['after']['gradient']['tensor']
                check('actual leaf grad strides', metadata(g)['strides'][:3], expected['strides'])
                check('actual leaf grad version', metadata(g)['version'], expected['version'])
                check('default retry spent', backward(q), 10)
                for t in (g, q, mask, factor):
                    one(41, t)
            elif index == 1:
                mask1 = nd(x)
                mask2 = nd(x)
                weak1, weak2 = one(42, mask1)['value'], one(42, mask2)['value']
                one(41, mask1)
                one(41, mask2)
                live(weak1, True)
                live(weak2, True)
                check('both nodes apply before accumulator', backward(x), 11)
                live(weak1, False)
                live(weak2, False)
                check('spent current history', backward(x), 10)
                check('no fake .grad', one(59, x)['value'], 0)
                one(44, weak1)
                one(44, weak2)
                one(101, x)
            elif index == 2:
                factor = one(*w.full([], 2))['value']
                old = one(2, 3, x, factor)['value']
                mask = nd(x)
                check('old child still reaches captured accumulator', backward(old, 1), 11)
                error([(29, old, seed)], 11)
                error([(57, old, seed, 0)], 11)
                one(101, x)
                check('current false flag skips old leaf accumulation', backward(old, 1), 0)
                check('skip leaves None', one(59, x)['value'], 0)
                for t in (old, mask, factor):
                    one(41, t)
            elif index == 3:
                error([(99, x, generator, 4602678819172646912, 1), (999,)], 4)
                check('rollback kept leaf', struct.unpack('<4Q', one(96, x)['payload'])[3], 1)
                response = s.exchange(w.packet([(99, x, generator, 4602678819172646912, 1),
                                               (58, x, seed, 0), (999,)]))
                status, records = w.parse(response)
                check('partial header success', status, 0)
                check('runtime error stops suffix', len(records), 2)
                check('actual runtime interior', records[1]['value'], 11)
                mask = struct.unpack('<Q', records[0]['payload'])[0]
                check('same-frame committed history', struct.unpack('<4Q', one(96, x)['payload'])[3], 0)
                one(101, x)
                one(41, mask)
            elif index == 4:
                check('initial G materialization', backward(x), 0)
                g = one(91, x)['value']
                weak_x, weak_g = one(42, x)['value'], one(42, g)['value']
                one(41, g)
                mask = nd(x)
                before = metadata(x)
                check('detach returns same T', one(101, x)['value'], x)
                after = metadata(x)
                for key in ('storage_id', 'version_id', 'version', 'storage_offset', 'storage_bytes'):
                    check('detach keeps ' + key, after[key], before[key])
                check('detach disables requires', after['requires_grad'], 0)
                check('detach keeps real G', one(59, x)['value'], 1)
                one(41, x)
                x = 0
                live(weak_x, False)
                live(weak_g, False)
                one(41, mask)
                one(44, weak_x)
                one(44, weak_g)
            elif index == 5:
                mask = nd(x)
                one(93, x)
                held = []
                actual = reference['rows'][6]['observation']
                for key in ('retained_failure', 'default_failure', 'released_retry'):
                    expected = 10 if key == 'released_retry' else 11
                    check(key + ' runtime', backward(x, int(key == 'retained_failure')), expected)
                    g = one(91, x)['value']
                    expected_g = actual[key]['after']['gradient']['tensor']
                    check(key + ' G strides', metadata(g)['strides'][:3], expected_g['strides'])
                    check(key + ' G version', metadata(g)['version'], expected_g['version'])
                    check(key + ' fresh G Tensor', g not in held, True)
                    held.append(g)
                one(101, x)
                check('retain property survives detach', one(94, x)['value'], 1)
                for t in held + [mask]:
                    one(41, t)
            elif index == 6:
                factor = one(*w.full([], 2))['value']
                old = one(2, 3, x, factor)['value']
                mask = nd(x)
                one(101, x)
                one(90, x, 1)
                newer = one(2, 3, x, factor)['value']
                check('direct restored leaf accumulator', backward(x, 1), 0)
                check('old captured edge succeeds after genuine detach', backward(old, 1), 0)
                check('new edge captures same surviving accumulator', backward(newer, 1), 0)
                check('real G present', one(59, x)['value'], 1)
                for t in (old, newer, factor, mask):
                    one(41, t)
            else:
                # Resource control: a retained hook belongs to the old AD node.
                factor = one(*w.full([], 2))['value']
                q = one(2, 3, x, factor)['value']
                one(93, q)
                child = one(18, q)['value']
                one(101, q)
                check('detached q remains retained flag', one(94, q)['value'], 1)
                check('old retained node still executes', backward(child, 1), 0)
                check('old weak hook writes actual q G', one(59, q)['value'], 1)
                weak_q = one(42, q)['value']
                one(41, q)
                live(weak_q, False)
                check('expired weak hook skips callback safely', backward(child, 1), 0)
                for t in (child, factor):
                    one(41, t)
                one(44, weak_q)
            check('complete RNG unchanged', one(63, generator)['payload'], before_rng)
            check('committed object and edge audit', one(53)['value'], 1)
            if x:
                one(41, x)
            one(41, seed)
            one(66, generator)
            check('final object audit', one(53)['value'], 1)
            row['status'] = 'PASS'
        except BaseException as error:
            row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
        finally:
            row['frames'] = s.frames
            try:
                row['transport'] = s.finish()
            except BaseException as error:
                row.update(status='FAIL', transport_error=repr(error))
            (directory/'report.json').write_text(json.dumps(row, indent=2)+'\n')
            rows.append(row)
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    result = {'schema': 'heptatorch.typed-native-dropout-history-controls.v1',
              'status': 'PASS' if all(row['status'] == 'PASS' for row in rows) else 'FAIL',
              'source_sha256': sha(inputs[a.source.resolve()]), 'native_sha256': sha(inputs[a.binary.resolve()]),
              'reference_sha256': sha(inputs[a.reference.resolve()]),
              'inputs_sha256': {str(path): sha(raw) for path, raw in inputs.items()},
              'cases': len(rows), 'passed': sum(row['status'] == 'PASS' for row in rows),
              'frames': sum(row['frames'] for row in rows), 'checks': sum(row['checks'] for row in rows),
              'rollback_bytes': sum(row['rollback_bytes'] for row in rows), 'rows': rows,
              'Torch_executed_here': False, 'complete_actual9_replay': False,
              'full_pytorch_rewrite_complete': False}
    (out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'cases', 'passed', 'frames', 'checks', 'rollback_bytes')}))
    return int(result['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
