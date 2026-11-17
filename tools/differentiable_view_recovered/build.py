"""RecoveryR1 native build. New authoring identity; exact original HIA constructor."""
import argparse
import json
import re
import struct
import subprocess
from pathlib import Path
from authoring_support import CALL, COMPILER, EXTERNAL, LINKER, functions, sha
from differentiable_view_overlay import apply, PARENT09B, PARENT_IMAGE
from parent_emitter import ParentEmitter

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'repository', 'upstream', 'compactor', 'compiler', 'linker', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    compiler, linker = a.compiler.resolve(), a.linker.resolve()
    assert sha(compiler.read_bytes()) == COMPILER and sha(linker.read_bytes()) == LINKER
    own_names = ('build.py', 'parent_emitter.py', 'authoring_support.py',
                 'differentiable_view.py', 'differentiable_view_overlay.py',
                 'constructor-parent-authoring.py')
    own = {name: (HERE / name).read_bytes() for name in own_names}
    for name, raw in own.items():
        (out / name).write_bytes(raw)
    assert sha(own['differentiable_view.py']) == '934c039186eb794b2294cf3cc4713685c14a2b1071b13834779b39e16bba3021'
    e = ParentEmitter(a.previous.resolve(), a.repository.resolve(), a.upstream.resolve(), a.compactor.resolve())
    try:
        combined = apply(e, dict(e.parent), own['differentiable_view.py'],
                         own['constructor-parent-authoring.py'], a.previous.resolve(), out)
    except BaseException as error:
        (out / 'refusal.json').write_text(json.dumps({'stage': 'new-constructor-authoring',
            'error': repr(error), 'native_compile_started': False}, indent=2) + '\n')
        raise
    raw = b'hia-asm.v1 hepta.hia.program.v8 differentiable_view_runtime\n' + b''.join(combined.values())
    (out / 'runtime.hepta').write_bytes(raw)
    (out / 'source-map.json').write_text(json.dumps([
        {'name': name, 'sha256': sha(body), 'parent_sha256': sha(e.parent[name]),
         'unchanged_parent': e.parent[name] == body} for name, body in combined.items()], indent=2) + '\n')
    (out / 'emitter-custody.json').write_text(json.dumps({
        'orchestration_revision': 'RECOVERED_SOURCE_R1',
        'authoring_wrapper_sha256': sha(own['authoring_support.py']),
        'baseline_constructor_reemission_exact': True,
        'compactor_records': e.checks, 'compiler_limits_changed': False,
        'objects_and_link_image_reused': False,
        'cloud_attempt01_and01b': 'SIGKILL before native compilation; original files remain inaccessible, not relabeled.'
    }, indent=2) + '\n')
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
        unit = f'direct_view_{i:02d}'
        source = (f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()
                  + b''.join(declarations) + b''.join(combined[n] for n in names))
        units.append((unit, source))
    units.append(('owners', e.owner_raw))
    report = {
        'schema': 'heptatorch.differentiable-view-native-build.v1',
        'orchestration_revision': 'RECOVERED_SOURCE_R1', 'status': 'BUILDING',
        'parent_source_sha256': PARENT09B, 'parent_native_sha256': PARENT_IMAGE,
        'source_sha256': sha(raw), 'descriptor_bytes': 672,
        'state_format': 'HTVAD001', 'backing_format': 'HTOVAD01',
        'functions': len(combined), 'blocks': len(re.findall(rb'^block ', raw, re.M)),
        'instructions': raw.count(b' = '), 'compiler_sha256': COMPILER, 'linker_sha256': LINKER,
        'changed_parent_functions': [n for n in e.parent if combined[n] != e.parent[n]],
        'added_functions': [], 'inputs_sha256': {str(p): sha(data) for p, data in e.inputs.items()},
        'authoring_inputs_sha256': {name: sha(data) for name, data in own.items()}, 'units': [],
        'runtime_qualified': False, 'torch_executed': False,
        'full_pytorch_rewrite_complete': False, 'pytorch_api_denominator': 2590,
    }
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
    request = b'HIASTKS1' + struct.pack('<3Q', len(objects), 4, 0) + b'step' + b''.join(struct.pack('<Q', len(o)) + o for o in objects)
    (out / 'link.request.bin').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=120)
    (out / 'link.stderr').write_bytes(result.stderr)
    report['link_exit'] = result.returncode
    if result.returncode != 0 or result.stderr or not result.stdout.startswith(b'\x7fELF'):
        (out / 'link.refusal.stdout').write_bytes(result.stdout)
        report['status'] = 'NATIVE_LINKER_REFUSAL'
        save()
        raise AssertionError(result.returncode)
    app = out / 'tensor-differentiable-view-session'
    app.write_bytes(result.stdout)
    app.chmod(0o500)
    e.unchanged()
    assert all((HERE / name).read_bytes() == data for name, data in own.items())
    assert sha(compiler.read_bytes()) == COMPILER and sha(linker.read_bytes()) == LINKER
    report.update(status='PASS', native_sha256=sha(result.stdout), native_bytes=len(result.stdout))
    save()
    print(json.dumps({k: report[k] for k in ('status', 'source_sha256', 'native_sha256',
        'functions', 'blocks', 'instructions', 'descriptor_bytes', 'runtime_qualified')}, indent=2))


if __name__ == '__main__':
    main()
