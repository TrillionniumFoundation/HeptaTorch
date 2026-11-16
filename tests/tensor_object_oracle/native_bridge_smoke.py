"""Execute only the transport adapter against native HIA; does not import Torch."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import struct

from bridge import FINAL_NATIVE_SHA, install, prepare_sources, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--owned-reference', type=Path, required=True)
    parser.add_argument('--session-reference', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--expected-native-sha256', default=FINAL_NATIVE_SHA)
    args = parser.parse_args()
    executable, output = args.executable.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = sha(executable.read_bytes())
    assert before == args.expected_native_sha256
    scope, definitions, registrations, owned, base, audit = prepare_sources(
        args.owned_reference, args.session_reference, args.reference)
    exec(compile(ast.fix_missing_locations(ast.Module(body=base, type_ignores=[])),
                 '<unchanged-frozen-wire-serializer>', 'exec'), scope)
    bits = scope['bits']
    install(scope, definitions, owned, scope['Program'], executable, output/'wire')

    def floats(record):
        assert record[1]*4 == len(record[2])
        return struct.unpack('<'+'f'*record[1], record[2])

    def arithmetic(s):
        p = s.program()
        x = p.full((2,), 2., True)
        square = p.binary(3, x, x)
        loss = p.reduce(7, square)
        p.backward(loss)
        xi, gi, li = p.read(x), p.read(x, True), p.read(loss)
        records = s.exchange(p)
        assert floats(records[xi]) == (2., 2.)
        assert floats(records[gi]) == (4., 4.)
        assert floats(records[li]) == (8.,)
        p = s.program(); p.sgd(x, .25); ri = p.read(x); mi = p.command(13, x)
        records = s.exchange(p)
        assert floats(records[ri]) == (1., 1.)
        assert struct.unpack('<13Q', records[mi][2])[12] == 1
        s.row['native_only_exact_float_checks'] = 7

    def rollback(s):
        p = s.program(); x = p.full((2,), 2., True)
        loss = p.reduce(7, p.binary(3, x, x)); s.exchange(p)
        before_map, before_next = s.bindings.copy(), s.next_id
        p = s.program(); provisional = p.clone(x)
        p.command(12, x, 0, bits(9.)); p.backward(loss)
        s.exchange(p, status=6)
        assert s.bindings == before_map and s.next_id == before_next
        p = s.program(); committed = p.clone(x); assert committed == provisional
        ri = p.read(committed); s.exchange(p)
        p = s.program(); ri = p.read(x); p.backward(loss); gi = p.read(x, True)
        records = s.exchange(p)
        assert floats(records[ri]) == (2., 2.)
        assert floats(records[gi]) == (4., 4.)
        s.row['native_only_exact_float_checks'] = 4
        s.row['failed_frame_binding_ledger_unchanged'] = True

    def reset(s):
        p = s.program(); x = p.full((1,), 3.); y = p.clone(x); s.exchange(p)
        old_actual = s.bindings[y]
        p = s.program(); p.reset_graph(1); s.exchange(p)
        p = s.program(); p.read(y); s.exchange(p, status=1)
        p = s.program(); fresh = p.full((1,), 4.); assert fresh == (1<<16)|2
        ri = p.read(fresh); records = s.exchange(p)
        assert floats(records[ri]) == (4.,)
        assert s.bindings[fresh] != old_actual
        s.row['native_only_exact_float_checks'] = 1

    def join_refs(s):
        p = s.program(); a = p.full((), 2.); b = p.full((), 3.)
        joined = p.join([a, b, a], 0, stack=True); ri = p.read(joined)
        records = s.exchange(p)
        assert floats(records[ri]) == (2., 3., 2.)
        s.row['native_only_exact_float_checks'] = 3

    def compound(s):
        p = s.program(); x = p.full((1,2), 2.); w = p.full((1,2), 3.); b = p.full((1,), 1.)
        output_id = p.linear(x, w, b); assert output_id == 6
        ci = p.clone(output_id); assert ci == 7
        ri, rc = p.read(output_id), p.read(ci); records = s.exchange(p)
        assert floats(records[ri]) == floats(records[rc]) == (13.,)
        assert s.bindings[output_id] != output_id and s.bindings[ci] != ci
        s.row['native_only_exact_float_checks'] = 2

    def scalar(s):
        p = s.program(); i = p.command(14, 0, bits(2.), bits(3.))
        records = s.exchange(p)
        assert records[i][1] == bits(5.) and not records[i][2]
        assert p.produced == {} and s.bindings == {}
        s.row['cmd14_scalar_is_neither_ID_operand_nor_Tensor_result'] = True
        s.row['native_only_exact_float_checks'] = 1

    for name, function in [('same-frame-native-arithmetic', arithmetic),
                           ('failed-frame-binding-rollback', rollback),
                           ('native-generation-after-reset', reset),
                           ('concat-payload-result-references', join_refs),
                           ('compound-command-hidden-node-ledger', compound),
                           ('scalar-cmd14-unmodified-bits', scalar)]:
        scope['case'](name, function, 0., 0.)
    rows = scope['ROWS']
    passed = len(rows) == 6 and all(r['status'] == 'PASS' for r in rows)
    assert before == sha(executable.read_bytes())
    report = {'schema':'heptatorch.v3-oracle-native-bridge-smoke.v1',
        'status':'PASS' if passed else 'FAIL', 'native_executable_sha256':before,
        'native_executed':True, 'pytorch_executed':False,
        'cases':len(rows), 'frames':sum(r['frames'] for r in rows),
        'native_only_exact_float_checks':sum(r.get('native_only_exact_float_checks', 0) for r in rows),
        'source_overlay_audit':audit, 'rows':rows,
        'source_files':{path.name:sha(path.read_bytes()) for path in (Path(__file__), Path(__file__).with_name('bridge.py'))},
        'full_rewrite_complete':False, 'pytorch_api_denominator':2590}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'], 'cases':len(rows), 'frames':report['frames'],
                      'failed':[{k:r[k] for k in ('case','error','traceback')} for r in rows if r['status'] != 'PASS']}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
