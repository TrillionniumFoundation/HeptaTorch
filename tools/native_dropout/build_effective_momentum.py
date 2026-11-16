"""Author checked HIA dtype/storage and native_dropout over immutable retained446.

Host Python performs custody checks, AST-to-HIA emission and native tool calls.
The produced HIA performs every tensor, random, gradient and ownership action.
"""
import argparse
import ast
import json
import re
import struct
import subprocess
from pathlib import Path

from authoring_support import CALL, COMPILER, EXTERNAL, LINKER, PARENT, Emitter, functions, once, sha
from identity_overlay import compose_identity

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'repository', 'upstream', 'compactor', 'compiler', 'linker', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    compiler, linker = a.compiler.resolve(), a.linker.resolve()
    assert sha(compiler.read_bytes()) == COMPILER
    assert sha(linker.read_bytes()) == LINKER
    e = Emitter(a.parent.resolve(), a.repository.resolve(), a.upstream.resolve(), a.compactor.resolve())
    own = {name: (HERE / name).read_bytes() for name in
           ('build_effective_momentum.py', 'effective_momentum.py', 'effective-momentum-baseline.py', 'optimizer-history-baseline.py', 'logical_not.py', 'identity.py', 'identity_overlay.py', 'identity-baseline.py', 'authoring_support.py', 'baseline-authoring.py', 'baseline-authoring-custody.json',
            'typed.py', 'native_dropout.py', 'preserve-original.py', 'dtype-baseline-additions.py')}
    for name, raw in own.items():
        (out / name).write_bytes(raw)
    original = {n.name: ast.unparse(n) for n in ast.parse(own['baseline-authoring.py']).body}
    original.update({n.name: ast.unparse(n) for n in ast.parse(own['preserve-original.py']).body})
    original.update({n.name: ast.unparse(n) for n in ast.parse(own['dtype-baseline-additions.py']).body})
    original.update({n.name: ast.unparse(n) for n in ast.parse(own['identity-baseline.py']).body})
    original.update({n.name: ast.unparse(n) for n in ast.parse(own['optimizer-history-baseline.py']).body})
    original.update({n.name: ast.unparse(n) for n in ast.parse(own['effective-momentum-baseline.py']).body})
    for name, source in original.items():
        regenerated = functions(e.emit(source))
        assert regenerated == {name: e.parent[name]}, ('original authoring differs', name)

    combined = dict(e.parent)
    aliases = ('clone', 'inspect_command', 'saved_logical', 'leaf_requires_grad',
               'gradient_gc_enqueue', 'tensor_row_release', 'graph_node_release',
               'release_backward_saved', 'gradient_audit', 'partial_apply_node',
               'backward_execute', 'partial_admit', 'weak_create', 'weak_lock', 'object_command')
    inverse_aliases = {}
    for name in aliases:
        alias = 'dtype_original_' + name
        body = once(combined[name], ('fn ' + name + '(').encode(), ('fn ' + alias + '(').encode())
        assert once(body, ('fn ' + alias + '(').encode(), ('fn ' + name + '(').encode()) == combined[name]
        combined[alias] = body
        e.add_context(alias, body)
        inverse_aliases[name] = {'original_sha256': sha(combined[name]), 'alias_sha256': sha(body), 'inverse_exact': True}

    patches = {}
    descriptor = once(original['descriptor'], 'def descriptor(', 'def typed_descriptor(')
    descriptor = once(descriptor, 'own: u64)', 'own: u64, dtype: u64)')
    descriptor = once(descriptor, '    if rank > 4 or requires > 1:',
        '    width = dtype_width(dtype)\n    if width == 0:\n        return terror(w, 7)\n'
        '    if dtype == 1 and requires == 1:\n        return terror(w, 5)\n    if rank > 4 or requires > 1:')
    descriptor = once(descriptor, 'memory_fill(w, p, 576, 0)', 'memory_fill(w, p, 640, 0)')
    descriptor = once(descriptor, '    putw(w, p, rank)\n', '    putw(w, p, rank)\n    putw(w, p + 576, dtype)\n')
    descriptor = once(descriptor, 'data = alloc(w, n * 4)', 'data = alloc(w, n * width)')
    descriptor = once(descriptor, 'so_uput(w, tp + 40, n * 4)', 'so_uput(w, tp + 40, n * width)')
    patches['typed_descriptor'] = descriptor
    patches['bind_alias'] = once(original['bind_alias'], '    q = dptr(w, source)\n',
        '    q = dptr(w, source)\n    putw(w, p + 576, getw(w, q + 576))\n')
    patches['save_public'] = once(original['save_public'], '    putw(w, layout + 8, getw(w, p + 8))\n',
        '    putw(w, layout + 8, getw(w, p + 8))\n    putw(w, layout + 16, getw(w, p + 576))\n')
    patches['unpack_saved'] = once(original['unpack_saved'],
        't = descriptor(w, getw(w, layout), layout + 32, 0, 0)',
        't = typed_descriptor(w, getw(w, layout), layout + 32, 0, 0, getw(w, layout + 16))')
    view = original['rng_strided_view']
    view = once(view, '    size = so_uget(w, sp + 24) // 4\n',
        '    width = dtype_width(getw(w, p + 576))\n    size = so_uget(w, sp + 24) // width\n')
    view = once(view, 'if offset > 268435456:', 'if offset > 1073741824 // width:')
    assert view.count('offset * 4') == 3
    view = view.replace('offset * 4', 'offset * width')
    view = once(view, 'remaining = (size - offset) * 4', 'remaining = (size - offset) * width')
    patches['rng_strided_view'] = view
    patches['gradient_shape_matches'] = once(original['gradient_shape_matches'], '    rank = getw(w, p)\n',
        '    if getw(w, p + 576) != getw(w, q + 576):\n        return 0\n    rank = getw(w, p)\n')
    patches['gradient_assign'] = once(original['gradient_assign'], '    if gradient_binding(w, t) == g:\n',
        '    if tensor_dtype(w, t) != 0:\n        return terror(w, 5)\n    if gradient_binding(w, t) == g:\n')
    patches['entry'] = once(original['entry'], '        if code >= 91 and code <= 94:\n',
        '        if code >= 96 and code <= 100:\n            next = typed_command(s, o, w, scratch, 32 + i * 64, at, end)\n'
        '        elif code >= 91 and code <= 94:\n')
    patches['incoming_describe'] = once(original['incoming_describe'], '    if op == 41:\n',
        '    if op == 42:\n        return native_dropout_grad_layout(w, t, 3856)\n    if op == 41:\n')
    patches['incoming_mark_preserve_targets'] = once(original['incoming_mark_preserve_targets'],
        'and (getw(w, p + 40) == 41):', 'and (getw(w, p + 40) == 41 or getw(w, p + 40) == 42):')
    audit = original['graph_audit']
    audit = once(audit, 'getw(w, p + 376) + getw(w, p + 312) + getw(w, p + 400)',
        'getw(w, p + 376) + getw(w, p + 312) + getw(w, p + 400) + getw(w, p + 600)')
    patches['graph_audit'] = audit
    records = original['so_audit_records']
    records = once(records, '                    j = 24\n                    if kind == 6:\n',
        '                    j = 32\n                    if kind == 4 and so_uget(w, p + 24) > 1:\n'
        '                        return 9\n                    if kind == 6:\n')
    records = once(records, '                        j = 32\n                    while j < 64:\n',
        '                        if so_uget(w, p + 32) > 1 or (so_uget(w, p + 24) == 1 and so_uget(w, p + 32) != 0):\n'
        '                            return 9\n                        j = 40\n                    while j < 64:\n')
    patches['so_audit_records'] = records
    typed_author, dropout_author = compose_identity(original, patches, own, combined, e, inverse_aliases)
    typed_author = once(typed_author, 'if code < 96 or code > 101:', 'if code < 96 or code > 102:')
    typed_author = once(typed_author, 'if code == 101 and (b != 0 or c != 0):',
        'if (code == 101 or code == 102) and (b != 0 or c != 0):')
    typed_author = once(typed_author, '                if code == 101:\n',
        '                if code == 102:\n                    if bool_logical_not(w,t) != 0:\n                        result = publish_tensor(w,t)\n'
        '                if code == 101:\n')
    patches['entry'] = once(patches['entry'], 'code >= 96 and code <= 101', 'code >= 96 and code <= 102')
    for name in ('optimizer_audit_entry', 'owned_adam_audit_entry'):
        patches[name] = once(original[name],
            'if p == 0 or retained_parameter_blocker(w, node) != 0:',
            'if p == 0 or tensor_dtype(w,node) != 0:')
    for name in ('zero_grad', 'zero_grad_none'):
        helper = 'history_member_' + name
        body = once(original[name], 'def ' + name + '(', 'def ' + helper + '(')
        body = once(body, 'valid(w, t)', 'valid_f32(w,t)')
        body = once(body, '    if retained_parameter_blocker(w, t) != 0:\n        return terror(w, 5)\n', '')
        patches[helper] = body
    for name in ('optimizer_zero', 'owned_adam_zero'):
        body = once(original[name], 'zero_grad_none(w, node)', 'history_member_zero_grad_none(w,node)')
        patches[name] = once(body, 'zero_grad(w, node)', 'history_member_zero_grad(w,node)')
    layout = once(original['pointwise_layout'], 'def pointwise_layout(', 'def optimizer_effective_pointwise_layout(')
    layout = once(layout, 'if is_contiguous(w, a) != 0 and is_contiguous(w, b) != 0:\n            return 1',
                  'if is_contiguous(w, a) != 0 and is_contiguous(w, b) != 0:\n            return incoming_contiguous_layout(w,t,p+128)')
    layout = once(layout, 'if inverted != 0:\n        return 1',
                  'if inverted != 0:\n        return incoming_contiguous_layout(w,t,p+128)')
    patches['optimizer_effective_pointwise_layout'] = layout
    patches['optimizer_sgd_occurrence'] = once(original['optimizer_sgd_occurrence'],
        'optimizer_new_momentum(w, parameter)', 'optimizer_new_effective_momentum(w,parameter,gp)')
    admission = '''
def dtype_backward_admit(w:mutbuf, t:u64, seed:u64) -> u64:
    if valid(w,t) == 0:
        return 0
    if tensor_dtype(w,t) != 0:
        return terror(w,5)
    if seed != 0:
        if valid(w,seed) == 0:
            return 0
        if tensor_dtype(w,seed) != tensor_dtype(w,t):
            return terror(w,2)
    return 1

def backward_execute(w:mutbuf, t:u64, seed:u64, retain:u64) -> u64:
    if dtype_backward_admit(w,t,seed) == 0:
        return 0
    epoch = getw(w,24)+1
    if epoch == 0:
        return terror(w,8)
    root = history_backward_root(w,t)
    if root == 0 or partial_mark(w,root,epoch) == 0:
        return 0
    if root != t or history_has_rebased_node(w,epoch) != 0:
        if history_preflight_saved(w,epoch) == 0:
            return 0
        return history_atomic_execute(w,t,seed,retain)
    return dtype_original_backward_execute(w,t,seed,retain)

def partial_admit(w:mutbuf, t:u64, seed:u64) -> u64:
    if dtype_backward_admit(w,t,seed) == 0:
        return 0
    return dtype_original_partial_admit(w,t,seed)
'''
    authored = '\n\n'.join(patches.values()) + '\n\n' + typed_author + '\n\n' + dropout_author + '\n\n' + own['identity.py'].decode() + '\n\n' + own['logical_not.py'].decode() + '\n\n' + own['effective_momentum.py'].decode() + admission
    (out / 'overlay-authoring.py').write_text(authored)
    try:
        fragment = e.emit(authored)
    except Exception as error:
        (out / 'refusal.json').write_text(json.dumps({'stage': 'authoring', 'error': repr(error), 'native_attempted': False}, indent=2) + '\n')
        raise
    combined.update(functions(fragment))
    (out / 'overlay.hepta').write_bytes(fragment)

    byte_edits = {}
    def exact_edit(name, before, after):
        raw = combined[name]
        revised = once(raw, before, after)
        assert once(revised, after, before) == raw
        combined[name] = revised
        byte_edits.setdefault(name, []).append({'before_sha256': sha(raw), 'after_sha256': sha(revised),
                                               'before': before.decode(), 'after': after.decode(), 'inverse_exact': True})
    for name, before, after in (
        ('dptr', 576, 640), ('graph_reset', 576, 640),
        ('descriptor_page', 147456, 163840), ('state_header', 147456, 163840),
        ('gradient_direct', 41, 42)):
        exact_edit(name, ('{"value":' + str(before) + '}').encode(), ('{"value":' + str(after) + '}').encode())
    for name in ('init_state', 'state_header', 'so_state_init', 'so_state_header'):
        before = b'HTRET001' if name in ('init_state', 'state_header') else b'HTORET01'
        after = b'HTDTYP03' if name in ('init_state', 'state_header') else b'HTODTY03'
        exact_edit(name, ('{"value":' + str(int.from_bytes(before, 'little')) + '}').encode(),
                   ('{"value":' + str(int.from_bytes(after, 'little')) + '}').encode())

    guards = {}
    numeric_valid = ('linear', 'tgrad', 'binary', 'unary', 'reduce', 'mm', 'reduce_axis', 'softmax',
        'sgd', 'opt_parameter', 'opt_state', 'vector_shape', 'layernorm', 'cross_entropy_probs',
        'batchnorm_eval', 'conv2d', 'pool2d', 'matmul', 'index_select', 'gather', 'concat_shape', 'indexed_add')
    numeric_public = ('optimizer_command', 'rng_uniform_tensor', 'optimizer_fill_group', 'owned_adam_fill_group',
        'scalar_add_nograd', 'dist_bernoulli_scalar', 'dist_bernoulli_tensor', 'dist_bernoulli_functional',
        'dist_dropout', 'dist_saved_mask', 'dist_partial_bernoulli', 'dist_partial_dropout')
    for names, old, new in ((numeric_valid, 'valid', 'valid_f32'), (numeric_public, 'public_node', 'public_f32')):
        before = ('{"function":"' + old + '"}').encode()
        after = ('{"function":"' + new + '"}').encode()
        for name in names:
            raw = combined[name]
            count = raw.count(before)
            assert count > 0, name
            revised = raw.replace(before, after)
            assert revised.replace(after, before) == raw
            combined[name] = revised
            guards[name] = {'from': old, 'to': new, 'calls': count, 'inverse_exact': True,
                            'before_sha256': sha(raw), 'after_sha256': sha(revised)}

    raw = b'hia-asm.v1 hepta.hia.program.v8 native_dropout_typed_runtime\n' + b''.join(combined.values())
    (out / 'runtime.hepta').write_bytes(raw)
    (out / 'authoring-custody.json').write_text(json.dumps({
        'baseline_reemission_all_exact': True, 'original_functions': list(original),
        'alias_inverse': inverse_aliases, 'exact_native_edits': byte_edits,
        'numeric_dtype_admission': guards, 'old_gradient_temporary_end_576_preserved': True,
        'compactor_records': e.checks,
        'changes': {name: {'before': original.get(name), 'after': source} for name, source in patches.items()}
    }, indent=2) + '\n')
    (out / 'source-map.json').write_text(json.dumps([
        {'name': name, 'sha256': sha(body), 'parent_sha256': sha(e.parent[name]) if name in e.parent else None,
         'unchanged_parent': e.parent.get(name) == body} for name, body in combined.items()], indent=2) + '\n')

    ordered, done, active = [], set(), set()
    def visit(name):
        if name in EXTERNAL:
            return
        assert name in combined and name not in active, ('missing/recursive function', name)
        if name in done:
            return
        active.add(name)
        for target in sorted({n.decode() for n in CALL.findall(combined[name])}):
            visit(target)
        active.remove(name)
        done.add(name)
        ordered.append(name)
    for name in combined:
        visit(name)
    groups, group, instructions, blocks = [], [], 0, 0
    for name in ordered:
        body = combined[name]
        ni, nb = body.count(b' = '), len(re.findall(rb'^block ', body, re.M))
        if group and (instructions + ni > 3600 or blocks + nb > 1000 or len(group) >= 65):
            groups.append(group)
            group, instructions, blocks = [], 0, 0
        group.append(name)
        instructions += ni
        blocks += nb
    if group:
        groups.append(group)
    units = [('rng03', e.rng_raw)]
    for i, names in enumerate(groups):
        calls = {n.decode() for name in names for n in CALL.findall(combined[name])}
        imports = sorted(calls - set(names))
        assert len(names) + len(imports) <= 256
        declarations = []
        for name in imports:
            body = e.rng[name] if name in EXTERNAL else combined[name]
            declarations.append(re.sub(rb' entry\(\w+\)$', b' external()', body.split(b'\n')[0]) + b'\n')
        unit = f'dtype_dropout_{i:02d}'
        source = f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode() + b''.join(declarations) + b''.join(combined[n] for n in names)
        units.append((unit, source))
    units.append(('owners', e.owner_raw))
    report = {'schema': 'heptatorch.native-dropout-dtype-native-build.v1', 'status': 'BUILDING',
              'parent_source_sha256': PARENT, 'source_sha256': sha(raw), 'descriptor_bytes': 640,
              'state_format': 'HTDTYP03', 'backing_format': 'HTODTY03',
              'functions': len(combined), 'blocks': len(re.findall(rb'^block ', raw, re.M)),
              'instructions': raw.count(b' = '), 'compiler_sha256': COMPILER, 'linker_sha256': LINKER,
              'changed_parent_functions': [n for n in e.parent if combined[n] != e.parent[n]],
              'added_functions': [n for n in combined if n not in e.parent],
              'inputs_sha256': {str(p): sha(data) for p, data in e.inputs.items()},
              'authoring_inputs_sha256': {name: sha(data) for name, data in own.items()}, 'units': []}
    def save():
        (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    objects = []
    for name, source in units:
        (out / (name + '.hepta')).write_bytes(source)
        result = subprocess.run([str(compiler)], input=source, capture_output=True, env={'PATH': ''}, timeout=120)
        (out / (name + '.o')).write_bytes(result.stdout)
        (out / (name + '.stderr')).write_bytes(result.stderr)
        row = {'name': name, 'source_sha256': sha(source), 'object_sha256': sha(result.stdout),
               'functions': len(functions(source)), 'blocks': len(re.findall(rb'^block ', source, re.M)),
               'instructions': source.count(b' = '), 'compiler_exit': result.returncode}
        report['units'].append(row)
        save()
        if result.returncode != 0 or result.stderr or not result.stdout.startswith(b'\x7fELF'):
            report['status'] = 'NATIVE_COMPILER_REFUSAL'
            save()
            raise AssertionError(row)
        objects.append(result.stdout)
    request = b'HIASTKS1' + struct.pack('<3Q', len(objects), 4, 0) + b'step' + b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (out / 'link.request.bin').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=120)
    (out / 'link.stderr').write_bytes(result.stderr)
    report['link_exit'] = result.returncode
    save()
    if result.returncode != 0 or result.stderr or not result.stdout.startswith(b'\x7fELF'):
        (out / 'link.refusal.stdout').write_bytes(result.stdout)
        report['status'] = 'NATIVE_LINKER_REFUSAL'
        save()
        raise AssertionError(report['link_exit'])
    app = out / 'tensor-dtype-dropout-session'
    app.write_bytes(result.stdout)
    app.chmod(0o500)
    e.unchanged()
    for name, data in own.items():
        assert (HERE / name).read_bytes() == data
    assert sha(compiler.read_bytes()) == COMPILER and sha(linker.read_bytes()) == LINKER
    report.update(status='PASS', native_sha256=sha(result.stdout), native_bytes=len(result.stdout),
                  runtime_qualified=False, full_pytorch_rewrite_complete=False, pytorch_api_denominator=2590)
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ('inputs_sha256', 'authoring_inputs_sha256', 'units')}, indent=2))


if __name__ == '__main__':
    main()
