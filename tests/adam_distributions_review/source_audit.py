#!/usr/bin/env python3
"""Read-only, byte-bound audit of the real Adam/distribution source composition.

This does not import the producer or replace native compiler/linker admission.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
FUNCTION = re.compile(rb'^(fn (\w+)\([^\n]* entry\(\w+\)\n.*?^end\n)', re.M | re.S)
PINS = {
    'adam': '253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6',
    'distribution': '55ed0c6e16317add03974c01ffd85e38684fc535a3e5cb6423ed043f4217f05b',
    'source': '5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873',
    'binary': '3fc59ab7ed9c32dd75ff440ef9bfbec31756f50edd77715d665bdac098cc3f22',
}


def functions(raw):
    pairs = FUNCTION.findall(raw)
    found = {name.decode(): body for body, name in pairs}
    assert len(found) == len(pairs)
    return found


def literal(tag):
    return b'{"value":' + str(int.from_bytes(tag, 'little')).encode() + b'}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('adam-build', 'distribution-build', 'build', 'output'):
        ap.add_argument('--' + key, type=Path, required=True)
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = {'adam': a.adam_build / 'runtime.hepta',
             'distribution': a.distribution_build / 'runtime.hepta',
             'source': a.build / 'runtime.hepta',
             'binary': a.build / 'tensor-adam-distributions-session'}
    snapshots = {path.resolve(): path.read_bytes() for path in paths.values()}
    for key, path in paths.items():
        assert SHA(snapshots[path.resolve()]) == PINS[key], key
    before = functions(snapshots[paths['adam'].resolve()])
    dist = functions(snapshots[paths['distribution'].resolve()])
    now = functions(snapshots[paths['source'].resolve()])
    assert (len(before), len(dist), len(now)) == (352, 300, 378)
    changed = {name for name in before if before[name] != now[name]}
    tags = {'init_state', 'state_header', 'so_state_init', 'so_state_header'}
    assert changed == tags | {'entry', 'gradient_direct', 'partial_apply_node'}
    added = {name for name in now if name not in before}
    dist_names = {name for name in dist if name.startswith('dist_')}
    assert len(dist_names) == 25
    assert added == dist_names | {'partial_apply_node_adam'}
    assert all(now[name] == dist[name] for name in dist_names)
    assert now['partial_apply_node_adam'].replace(
        b'fn partial_apply_node_adam(', b'fn partial_apply_node(', 1
    ) == before['partial_apply_node']
    assert now['gradient_direct'] == dist['gradient_direct']
    bound = b'  v115:u64 = const() {"value":40}\n'
    assert before['gradient_direct'].count(bound) == 1
    assert before['gradient_direct'].replace(bound, bound.replace(b':40}', b':41}')) == now['gradient_direct']
    for name in tags:
        old, new = ((b'HTADAM01', b'HTADIST1') if name in ('init_state', 'state_header')
                    else (b'HTOADAM1', b'HTOADIS1'))
        assert before[name].count(literal(old)) == 1
        assert before[name].replace(literal(old), literal(new)) == now[name]

    # Inverse edit the exact producer authoring; no execution/import of it.
    old_path, new_path = a.adam_build / 'overlay-authoring.py', a.build / 'overlay-authoring.py'
    snapshots.update({p.resolve(): p.read_bytes() for p in (old_path, new_path)})
    old_entry = next(n for n in ast.parse(snapshots[old_path.resolve()]).body
                     if isinstance(n, ast.FunctionDef) and n.name == 'entry')
    new_ast = ast.parse(snapshots[new_path.resolve()])
    assert [n.name for n in new_ast.body] == ['entry', 'partial_apply_node']
    new_entry = ast.unparse(new_ast.body[0])
    dist_dispatch = '''        elif code >= 80 and code <= 89:
            if code == 86 or code == 87:
                next = dist_partial_command(s, o, w, scratch, 32 + i * 64, at)
            else:
                next = dist_command(s, o, w, scratch, 32 + i * 64, at)
'''
    assert new_entry.count(dist_dispatch) == 1
    restored = new_entry.replace(dist_dispatch, '')
    new_condition = 'if (code == 58 or code == 86 or code == 87) and getw(o, at + 16) != 0:'
    assert restored.count(new_condition) == 1
    restored = restored.replace(new_condition,
        'if getr(s, 32 + i * 64) == 58 and getw(o, at + 16) != 0:')
    assert restored == ast.unparse(old_entry)
    wrapper = '''def partial_apply_node(w: mutbuf, t: u64) -> u64:
    if getw(w, dptr(w, t) + 40) == 41:
        dist_backward_bernoulli(w, t)
        if getw(w, 0) != 0:
            return 0
        return 1
    return partial_apply_node_adam(w, t)
'''
    assert ast.dump(new_ast.body[1], include_attributes=False) == ast.dump(
        ast.parse(wrapper).body[0], include_attributes=False)

    # Verify actual complete production units contain these real bodies and real
    # external declarations. This is custody, not an alternative HIA checker.
    partition_path = a.build / 'partition.json'
    partition = json.loads(partition_path.read_bytes())
    snapshots[partition_path.resolve()] = partition_path.read_bytes()
    rng = functions((a.build / 'rng03.hepta').read_bytes())
    interfaces = {**now, **rng}
    defined = set()
    unit_rows = []
    for unit in partition:
        path = a.build / (unit['name'] + '.hepta')
        raw = path.read_bytes()
        snapshots[path.resolve()] = raw
        local = functions(raw)
        assert set(local) == set(unit['definitions'])
        assert not (defined & set(local))
        defined |= set(local)
        assert all(body == now[name] for name, body in local.items())
        imports = re.findall(rb'^fn (\w+)\([^\n]+ external\(\)$', raw, re.M)
        assert {x.decode() for x in imports} == set(unit['imports'])
        for name in unit['imports']:
            header = interfaces[name].split(b'\n', 1)[0]
            expected = re.sub(rb' entry\(\w+\)$', b' external()', header)
            assert raw.splitlines().count(expected) == 1
        unit_rows.append({'name': unit['name'], 'source_sha256': SHA(raw),
                          'real_definitions': len(local), 'real_imports': len(imports)})
    assert defined == set(now)
    for name in ('rng03.hepta', 'owners.hepta'):
        data = (a.build / name).read_bytes()
        assert data == (a.adam_build / name).read_bytes() == (a.distribution_build / name).read_bytes()
        snapshots[(a.build / name).resolve()] = data

    reviewed = ('record_body', 'record', 'gradient_parent_count', 'graph_parent_at',
                'graph_edge_enabled', 'graph_node_release', 'graph_collect_since',
                'graph_audit', 'gradient_parent_at', 'gadd', 'gradient_prepare',
                'gradient_finish', 'partial_execute', 'release_backward_saved',
                'optimizer_audit_record', 'owned_adam_step', 'owned_adam_drop',
                'private_tensor_retain', 'private_tensor_release', 'graph_reset')
    assert all(name in before and before[name] == now[name] for name in reviewed)
    diffs = ''.join(''.join(difflib.unified_diff(before[n].decode().splitlines(True),
                    now[n].decode().splitlines(True), fromfile='Adam02/'+n, tofile='joint02/'+n))
                    for n in sorted(changed))
    (out / 'focused-real-hia.diff').write_text(diffs)
    (out / 'executed-source-audit.py').write_bytes(Path(__file__).read_bytes())
    for path, raw in snapshots.items():
        assert path.read_bytes() == raw, str(path)
    report = {
        'schema': 'heptatorch.adam-distributions.independent-source-review.v1',
        'status': 'PASS_SOURCE_CUSTODY_AND_SCOPED_REVIEW', 'native_executed_here': False,
        'pins': PINS, 'old_functions': 352, 'unchanged_old_bodies': 345,
        'new_functions': 26, 'core_functions': 378,
        'changed': sorted(changed), 'added': sorted(added),
        'entry_inverse_restores_Adam_exact_AST_text': True,
        'old_dispatch_retained_as_real_alias_exact_bytes': True,
        'distribution_functions_copied_exact_bytes': 25,
        'gradient_direct_only_extends_upper_opcode_40_to_41': True,
        'native_units_actual_body_and_import_custody': unit_rows,
        'critical_unchanged_bodies': {n: SHA(now[n]) for n in reviewed},
        'review_findings': [
            'op41 uses the ordinary two-slot parent format with second parent zero; captured first-edge and AccumulateGrad references are retained by unchanged record.',
            'op41 has no Saved value/version requirement. Its derivative writes positive zero through staged gadd; current leaf requires_grad controls accumulation, while the captured edge controls traversal.',
            'Dropout remains op3 and its existing Saved noise marker at node+176. The marker does not overlap Optimizer private refs+312, incarnation+320, or accumulator+88.',
            '86/87 keep operation errors in normal record values; the entry stops their suffix while leaving top-level admission errors to the unchanged owned transaction.',
            'All kind8 audit, private-root release and growth helpers are unchanged; real mixed-lifetime execution is separately required.'
        ],
        'source_boundaries': ['No aggregate Core admission claimed.',
                              'No replacement of the actual native unit/ABI3 linker checker.',
                              'No all-schema or complete-autograd compatibility claim.'],
        'input_custody_after': True,
        'inputs_sha256': {str(p): SHA(b) for p, b in snapshots.items()},
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'core_functions', 'unchanged_old_bodies')}))


if __name__ == '__main__':
    main()
