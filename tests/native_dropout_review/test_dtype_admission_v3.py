"""Bool is never silently consumed by legacy four-byte numeric entry points.

Every expected refusal follows two genuine mutations in the same frame. The
entire committed owner, its pointer/length, and complete Generator state must
remain unchanged. These are protocol/resource controls, not upstream API tests.
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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {p.resolve(): p.read_bytes() for p in (a.binary, a.source, a.protocol, Path(__file__))}
    (out/'executed-test.py').write_bytes(inputs[Path(__file__).resolve()])
    spec = importlib.util.spec_from_file_location('dtype_admission_wire', a.protocol)
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    s = w.Session(a.binary.resolve(), directory=out/'raw')
    report = {'schema': 'heptatorch.dtype-f32-admission-controls.v1', 'status': 'RUNNING',
              'source_sha256': sha(inputs[a.source.resolve()]), 'native_sha256': sha(inputs[a.binary.resolve()]),
              'inputs_sha256': {str(p): sha(raw) for p, raw in inputs.items()}, 'rows': [],
              'native_executed': True, 'Torch_executed_here': False, 'rollback_bytes': 0}
    def one(*args):
        return s.run([args])[0]
    def boolean(shape):
        n = 1
        for d in shape:
            n *= d
        config = struct.pack('<8Q', len(shape), 0, *shape, *([0]*(4-len(shape))), 160, n)
        frame = b'HTORCH03' + struct.pack('<3Q', 1, 160+n, 0) + struct.pack('<8Q', 97, 1, 96, 0, 0, 0, 0, 0) + config + bytes(n)
        status, records = w.parse(s.exchange(frame))
        assert status == 0 and len(records) == 1
        return records[0]['value']
    def cfg(*values):
        return w.Payload(struct.pack('<'+'Q'*len(values), *values))
    def list_cfg(*ids):
        return cfg(len(ids), *ids)
    def group(algorithm, parameter, include_count=True):
        words = ([1] if include_count else []) + [1, 0] + [0]*(6 if algorithm == 0 else 8) + [parameter]
        return cfg(*words)
    def defaults(algorithm):
        if algorithm == 0:
            return cfg(w.bits(.125), w.bits(.5), 0, 0, 0, 0)
        return cfg(w.bits(.125), w.bits(.5), w.bits(.75), w.bits(.001), 0, 0, 0, 0)
    try:
        b = boolean([2, 2]); bv = boolean([2]); b4 = boolean([1, 1, 2, 2]); be = boolean([0])
        mutation_target = boolean([1])
        bool_grad = one(18, b)['value']
        f = one(*w.full([2, 2], .25))['value']; p = one(*w.full([2, 2], .25, 1))['value']
        v = one(*w.full([2], 1))['value']; f4 = one(*w.full([1, 1, 2, 2], .25))['value']
        fkernel = one(*w.full([1, 1, 1, 1], 1))['value']; bkernel = boolean([1, 1, 1, 1])
        state1 = one(*w.full([2, 2], 0))['value']; state2 = one(*w.full([2, 2], 0))['value']
        gen = one(60, 1729)['value']; complete_rng = one(63, gen)['payload']
        weak = one(42, b)['value']; hold = one(43, weak)['value']; saved = one(46, b)['value']
        empty_weak = one(42, be)['value']; empty_hold = one(43, empty_weak)['value']; empty_saved = one(46, be)['value']
        opt = one(71, 0, defaults(0), group(0, p))['value']
        adam = cfg(w.bits(.1), w.bits(.9), w.bits(.99), w.bits(.001), 0, 0)
        momentum = cfg(w.bits(.1), w.bits(.9), 0, 0, 0)
        index2 = cfg(2, 2, 2, 0, 0, 1, 1)
        index1 = cfg(1, 2, 0, 1)
        unit2 = (1 << 32) | 1
        rows = []
        def add(name, command, expected=7):
            rows.append((name, command, expected))
        for op in (1, 2, 3, 4):
            add('binary-left-'+str(op), (2, op, b, f)); add('binary-right-'+str(op), (2, op, f, b))
        for op in (5, 6, 11, 12, 13, 14, 15):
            add('unary-'+str(op), (3, op, b))
        for op in (7, 8):
            add('full-reduction-'+str(op), (4, op, b))
        for op in (5, 32):
            add('matmul-left-'+str(op), (op, b, f)); add('matmul-right-'+str(op), (op, f, b))
        for position in range(3):
            args = [f, f, v]; args[position] = b if position < 2 else bv
            add('linear-position-'+str(position), (15, *args))
        add('mse-left', (16, b, f)); add('mse-right', (16, f, b))
        for op in (18, 19):
            add('axis-reduction-'+str(op), (20, op, b, 0, 0))
        for op in (20, 21):
            add('softmax-'+str(op), (21, op, b, 0))
        add('conv-input', (22, b4, fkernel, unit2, 0, unit2, 1))
        add('conv-weight', (22, f4, bkernel, unit2, 0, unit2, 1))
        for op in (23, 24):
            add('pool-'+str(op), (23, op, b4, unit2, unit2, 0, 0))
        add('layernorm-input', (26, b, v, v, w.bits(.001)))
        add('layernorm-weight', (26, f, bv, v, w.bits(.001)))
        add('layernorm-bias', (26, f, v, bv, w.bits(.001)))
        add('cross-entropy-input', (27, b, f)); add('cross-entropy-target', (27, f, b))
        for position in range(5):
            args = [f, v, v, v, v]; args[position] = b if position == 0 else bv
            add('batchnorm-position-'+str(position), (28, *args, w.bits(.001)))
        add('index-select', (30, b, 0, 2, cfg(0, 1)))
        add('gather', (31, b, 0, index2))
        for op in (34, 35):
            add('join-bool-'+str(op), (op, 0, list_cfg(f, b)))
            add('join-empty-bool-'+str(op), (op, 0, list_cfg(be, f)))
        add('scatter-self', (36, b, f, 0, index2)); add('scatter-source', (36, f, b, 0, index2))
        add('index-add-self', (37, b, f, 0, index1, w.bits(1)))
        add('index-add-source', (37, f, b, 0, index1, w.bits(1)))
        add('legacy-sgd-parameter', (9, b, w.bits(.1)))
        add('legacy-adam-parameter', (24, b, state1, state2, 1, adam))
        add('legacy-adam-first-moment', (24, p, b, state2, 1, adam))
        add('legacy-adam-second-moment', (24, p, state1, b, 1, adam))
        add('legacy-momentum-parameter', (25, b, state1, 0, 1, momentum))
        add('legacy-momentum-state', (25, p, b, 0, 1, momentum))
        for op in (10, 11, 39, 40):
            add('legacy-read-bool-'+str(op), (op, b, 0, 0))
            add('legacy-read-empty-bool-'+str(op), (op, be, 0, 0))
        add('public-saved-f32-read', (49, saved, 0, 0))
        add('public-saved-empty-f32-read', (49, empty_saved, 0, 0))
        add('public-hold-f32-read', (50, hold, 0, 0))
        add('public-hold-empty-f32-read', (50, empty_hold, 0, 0))
        add('requires-grad-bool', (90, b, 1), 5)
        add('gradient-bool-target-f32-source', (92, b, f), 2)
        add('gradient-bool-target-bool-source', (92, b, bool_grad), 5)
        add('gradient-bool-source', (92, p, b), 2)
        for op in (29, 57, 58):
            add('backward-bool-root-'+str(op), (op, b, 0, 0), 5)
            add('backward-bool-seed-'+str(op), (op, p, b, 0), 2)
        add('legacy-backward-bool-root', (7, b), 5)
        add('uniform-bool', (68, b, gen, 0, w.bits(1), 1))
        add('bernoulli-scalar-bool', (80, b, gen, 0x3fe0000000000000, 1))
        add('bernoulli-tensor-output-bool', (81, b, gen, f, 1))
        add('bernoulli-tensor-prob-bool', (81, f, gen, b, 1))
        add('bernoulli-functional-bool', (82, b, gen, 1))
        for op in (83, 84, 87):
            add('dropout-bool-'+str(op), (op, b, gen, 0x3fe0000000000000, 1, 1))
        add('dropout-saved-bool', (85, b))
        add('partial-bernoulli-output-bool', (86, b, gen, f, 1))
        add('partial-bernoulli-prob-bool', (86, f, gen, b, 1))
        add('scalar-add-f32-only', (95, b, w.bits(1)))
        add('native-dropout-bool', (99, b, gen, 0x3fe0000000000000, 1))
        add('bool-fill-noncanonical', (100, b, 2, 0))
        for algorithm in (0, 1, 2):
            add('owned-optimizer-bool-'+str(algorithm), (71, algorithm, defaults(algorithm), group(algorithm, b)))
        add('owned-addgroup-bool', (79, opt, group(0, b, False)))
        for name, command, expected in rows:
            before = s.owner()
            status, records = w.parse(s.exchange(w.packet([(102, mutation_target), (68, f, gen, 0, w.bits(1), 1), command])))
            after = s.owner()
            item = {'name': name, 'expected_status': expected, 'actual_status': status, 'status': 'PASS'}
            report['rows'].append(item)
            assert status == expected and not records, (name, status, expected)
            assert (after['pointer'], after['length']) == (before['pointer'], before['length']), name
            assert sha(after['bytes']) == sha(before['bytes']), (name, 'committed owner changed')
            report['rollback_bytes'] += before['length']
        assert one(63, gen)['payload'] == complete_rng
        assert one(53)['value'] == 1
        report.update(status='PASS', refused_cases=len(rows), complete_rng_bytes_checked=len(complete_rng))
    except BaseException as error:
        if report['rows']:
            report['rows'][-1]['status'] = 'FAIL'
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    finally:
        report['frames'] = s.frames
        try:
            report['transport'] = s.finish()
        except BaseException as error:
            report.update(status='FAIL', transport_error=repr(error))
        assert all(p.read_bytes() == raw for p, raw in inputs.items())
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'refused_cases', 'frames', 'rollback_bytes', 'error')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
