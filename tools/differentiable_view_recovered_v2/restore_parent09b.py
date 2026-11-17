"""Reconstruct exact frozen09b units and image; never substitute metadata for bytes."""
import argparse
import json
import re
import struct
import subprocess
from pathlib import Path
from authoring_support import CALL, COMPILER, EXTERNAL, LINKER, RNG, OWNERS, functions, sha
from differentiable_view_overlay import PARENT09B, PARENT_IMAGE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-inputs', 'rng-source', 'owner-source', 'compiler', 'linker', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name('authoring_support.py'),
             Path(__file__).with_name('differentiable_view_overlay.py'),
             a.native_inputs / 'runtime.hepta', a.native_inputs / 'tensor-to-copy-session',
             a.native_inputs / 'build.json', a.rng_source, a.owner_source, a.compiler, a.linker]
    inputs = {p.resolve(): p.read_bytes() for p in paths}
    def data(p): return inputs[p.resolve()]
    raw = data(a.native_inputs / 'runtime.hepta')
    image = data(a.native_inputs / 'tensor-to-copy-session')
    original_build = data(a.native_inputs / 'build.json')
    build = json.loads(original_build)
    assert sha(raw) == PARENT09B and sha(image) == PARENT_IMAGE
    assert sha(original_build) == '9913ccb35c690f1a8795479ddd468531a966df71c375fcb91ed22d9f3c887557'
    assert sha(data(a.rng_source)) == RNG and sha(data(a.owner_source)) == OWNERS
    assert sha(data(a.compiler)) == COMPILER and sha(data(a.linker)) == LINKER
    assert build['status'] == 'PASS' and build['functions'] == 614 and len(build['units']) == 12
    combined = functions(raw)
    rng = functions(data(a.rng_source))
    assert len(combined) == 614 and len(rng) == 17
    assert raw == b'hia-asm.v1 hepta.hia.program.v8 tensor_to_copy_runtime\n' + b''.join(combined.values())
    ordered, done, active = [], set(), set()
    def visit(name):
        if name in EXTERNAL:
            return
        assert name in combined and name not in active
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
    units = [('rng03', data(a.rng_source))]
    for i, names in enumerate(groups):
        calls = {n.decode() for name in names for n in CALL.findall(combined[name])}
        imports = sorted(calls - set(names))
        assert len(names) + len(imports) <= 256
        declarations = []
        for name in imports:
            body = rng[name] if name in EXTERNAL else combined[name]
            declarations.append(re.sub(rb' entry\(\w+\)$', b' external()', body.split(b'\n')[0]) + b'\n')
        unit = f'to_copy_{i:02d}'
        source = (f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()
                  + b''.join(declarations) + b''.join(combined[n] for n in names))
        units.append((unit, source))
    units.append(('owners', data(a.owner_source)))
    assert [n for n, _ in units] == [u['name'] for u in build['units']]
    for (name, source), expected in zip(units, build['units']):
        (out / (name + '.hepta')).write_bytes(source)
        assert sha(source) == expected['source_sha256'], ('source-reconstruction', name, sha(source))
    receipt = {'schema': 'heptatorch.parent09b.reconstruction.v1', 'status': 'COMPILING',
               'source_sha256': PARENT09B, 'native_sha256': PARENT_IMAGE,
               'all12_source_hashes_match': True, 'units': [],
               'inputs_sha256': {str(p): sha(b) for p, b in inputs.items()},
               'new_tensor_runtime_qualified': False, 'numerical_suites_executed': False}
    def save():
        (out / 'reconstruction.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    objects = []
    for (name, source), expected in zip(units, build['units']):
        cp = subprocess.run([str(a.compiler.resolve())], input=source, capture_output=True, env={'PATH': ''}, timeout=120)
        (out / (name + '.o')).write_bytes(cp.stdout)
        (out / (name + '.stderr')).write_bytes(cp.stderr)
        row = {'name': name, 'compiler_exit': cp.returncode, 'object_sha256': sha(cp.stdout),
               'expected_object_sha256': expected['object_sha256']}
        receipt['units'].append(row)
        save()
        assert cp.returncode == 0 and not cp.stderr and cp.stdout.startswith(b'\x7fELF'), row
        assert sha(cp.stdout) == expected['object_sha256'], row
        objects.append(cp.stdout)
    request = b'HIASTKS1' + struct.pack('<3Q', len(objects), 4, 0) + b'step' + b''.join(struct.pack('<Q', len(o)) + o for o in objects)
    (out / 'link.request.bin').write_bytes(request)
    cp = subprocess.run([str(a.linker.resolve())], input=request, capture_output=True, env={'PATH': ''}, timeout=120)
    (out / 'link.stdout').write_bytes(cp.stdout)
    (out / 'link.stderr').write_bytes(cp.stderr)
    assert cp.returncode == 0 and not cp.stderr and cp.stdout == image
    (out / 'runtime.hepta').write_bytes(raw)
    app = out / 'tensor-to-copy-session'
    app.write_bytes(cp.stdout)
    app.chmod(0o500)
    (out / 'build.json').write_bytes(original_build)
    assert all(p.read_bytes() == b for p, b in inputs.items())
    receipt.update(status='PASS', all12_objects_byte_identical=True, linked_image_byte_identical=True,
                   original_build_report_bytes_preserved=True, all_inputs_unchanged=True)
    save()
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
