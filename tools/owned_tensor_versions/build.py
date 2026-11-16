"""Natively build the explicitly versioned Adam metadata correction.

The frozen candidate-03 source and qualification remain independently replayable.
This source organizer changes one call boundary, adds one HIA function, and
records every function body; it never executes target numerical operations.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHA = lambda raw: hashlib.sha256(raw).hexdigest()
FUNCTION = re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]*\n.*?^end\n)', re.M | re.S)
COMPILER = '0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73'
LINKER = '55a2b0a762c8db574a65d0d0e6a18e3cedc668c01d20080a11c26cf1ccb601b3'
BASE_SOURCE = '91e5cb7d80299e735dcac2dc3520bafbf3e272465ccdced6aa2f732800127b07'
BASE_IMAGE = '05c5544da2b7b34a8db9bd019a3ad0fa930f9ed5c18fd4b7582e7150f21eaeea'
CANONICAL = '264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233'


def functions(raw):
    matches = list(FUNCTION.finditer(raw))
    result = {m[2].decode(): m[1] for m in matches}
    assert len(result) == len(matches)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--linker', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--record-new-profile', type=Path,
                   help='Development only: record a new native profile; cannot overwrite an existing file.')
    a = p.parse_args()
    baseline = a.baseline.resolve()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    compiler, linker = a.compiler.resolve(), a.linker.resolve()
    assert SHA(compiler.read_bytes()) == COMPILER and SHA(linker.read_bytes()) == LINKER
    profile_path = HERE / 'profile.json'
    if a.record_new_profile:
        assert not a.record_new_profile.exists(), 'existing native qualification cannot be replaced'
        profile = None
    else:
        profile = json.loads(profile_path.read_text())
    inputs = [baseline / 'runtime.hepta', baseline / 'build.json', baseline / 'owned-tensor-session',
              baseline / 'units/owned_tensor_session.hepta', HERE / 'adam_versions.hepta',
              ROOT / 'src/heptatorch.hepta', ROOT / 'tools/modules/split_runtime.py',
              ROOT / 'tools/owned_tensors/profile.json', Path(__file__)]
    originals = {path: path.read_bytes() for path in inputs}
    source = originals[baseline / 'runtime.hepta']; assert SHA(source) == BASE_SOURCE
    assert SHA(originals[baseline / 'owned-tensor-session']) == BASE_IMAGE
    base_build = json.loads(originals[baseline / 'build.json'])
    assert base_build['pass'] and base_build['candidate_source_sha256'] == BASE_SOURCE
    assert base_build['executable_sha256'] == BASE_IMAGE
    canonical = originals[ROOT / 'src/heptatorch.hepta']; assert SHA(canonical) == CANONICAL
    base_functions = functions(source); assert len(base_functions) == 150
    body = base_functions['adam']
    changes = (
        (b'branch v355 b26(v341,v342,v343,v344,v346,v353,v354) b27(v341,v342,v343,v344)',
         b'branch v355 b26(v341,v342,v343,v344,v346,v353,v354) b27(v341,v342,v343,v344,v346)'),
        (b'block b27(v370:mutbuf,v371:u64,v372:u64,v373:u64)\n'
         b'  v387:u64 = call(v370,v371) {"function":"bump_version"}\n'
         b'  v388:u64 = call(v370,v372) {"function":"bump_version"}\n'
         b'  v389:u64 = call(v370,v373) {"function":"bump_version"}\n'
         b'  v390:u64 = const() {"value":1}\n'
         b'  ret(v390)\n',
         b'block b27(v370:mutbuf,v371:u64,v372:u64,v373:u64,v391:u64)\n'
         b'  v392:u64 = call(v370,v371,v372,v373,v391) {"function":"adam_versions"}\n'
         b'  ret(v392)\n'),
    )
    for old, new in changes:
        assert body.count(old) == 1
        body = body.replace(old, new, 1)
    helper = originals[HERE / 'adam_versions.hepta']
    assert list(functions(helper)) == ['adam_versions'] and helper == functions(helper)['adam_versions']
    core = source.replace(base_functions['adam'], body, 1) + helper
    result_functions = functions(core)
    assert len(result_functions) == 151
    assert [name for name in base_functions if base_functions[name] != result_functions[name]] == ['adam']
    (out / 'runtime.hepta').write_bytes(core)
    spec = importlib.util.spec_from_file_location('version_split', ROOT / 'tools/modules/split_runtime.py')
    split = importlib.util.module_from_spec(spec); spec.loader.exec_module(split)
    split.split(out / 'runtime.hepta', out / 'units')
    owner = originals[baseline / 'units/owned_tensor_session.hepta']
    needs = {name.decode() for name in re.findall(rb'^fn (\w+)\([^\n]+ external\(\)$', owner, re.M)}
    units = []
    for name in ('ht_kernel', 'ht_runtime'):
        path = out / 'units' / (name + '.hepta'); raw = path.read_bytes()
        local = set(functions(raw)); lines = raw.split(b'\n')
        exports = set(lines[1][8:-1].decode().split(',')) | (local & needs)
        lines[1] = ('exports(' + ','.join(key for key in result_functions if key in exports) + ')').encode()
        raw = b'\n'.join(lines); path.write_bytes(raw); units.append((name, raw))
    (out / 'units/owned_tensor_session.hepta').write_bytes(owner)
    units.append(('owned_tensor_session', owner))
    objects, unit_rows = [], []
    for name, raw in units:
        result = subprocess.run([str(compiler)], input=raw, capture_output=True,
                                env={'PATH': ''}, timeout=120)
        (out / (name + '.compiler.stderr')).write_bytes(result.stderr)
        (out / (name + '.compiler.stdout')).write_bytes(result.stdout)
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), name
        (out / (name + '.o')).write_bytes(result.stdout); objects.append(result.stdout)
        unit_rows.append({'name': name, 'source_sha256': SHA(raw), 'object_sha256': SHA(result.stdout),
                          'functions': len(functions(raw)), 'blocks': len(re.findall(rb'^block ', raw, re.M)),
                          'instructions': raw.count(b' = ')})
    request = b'HIAOWNS1' + struct.pack('<QQQ', len(objects), 4, 0) + b'step'
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (out / 'link.request.bin').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=120)
    (out / 'link.stderr').write_bytes(result.stderr); (out / 'link.stdout').write_bytes(result.stdout)
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    app = out / 'owned-tensor-session'; app.write_bytes(result.stdout); app.chmod(0o500)
    original_functions = functions(canonical)
    mapping = [{'name': name, 'canonical_sha256': SHA(raw), 'candidate_sha256': SHA(result_functions[name]),
                'byte_identical': raw == result_functions[name]} for name, raw in original_functions.items()]
    expected = {'schema': 'heptatorch.owned-tensor-version-profile.v1',
                'baseline_source_sha256': BASE_SOURCE, 'baseline_executable_sha256': BASE_IMAGE,
                'canonical_source_sha256': CANONICAL, 'candidate_source_sha256': SHA(core),
                'compiler_sha256': COMPILER, 'linker_sha256': LINKER,
                'executable_sha256': SHA(result.stdout), 'helper_sha256': SHA(helper),
                'builder_sha256': SHA(originals[Path(__file__)]), 'unit_rows': unit_rows,
                'canonical_function_count': 143, 'candidate_function_count': 151,
                'canonical_functions_byte_identical': sum(row['byte_identical'] for row in mapping),
                'baseline_functions_changed': ['adam'], 'function_source_mapping': mapping}
    if a.record_new_profile:
        a.record_new_profile.parent.mkdir(parents=True, exist_ok=True)
        a.record_new_profile.write_text(json.dumps(expected, indent=2) + '\n')
    else:
        assert profile == expected, 'native result does not reproduce the exact qualified profile'
    for path, raw in originals.items():
        assert path.read_bytes() == raw, path
    assert SHA(compiler.read_bytes()) == COMPILER and SHA(linker.read_bytes()) == LINKER
    report = dict(expected, pass_=True, native_compilation_and_linking=True,
                  inputs_unchanged=True, full_PyTorch_closure_claimed=False,
                  input_sources={str(path): SHA(raw) for path, raw in originals.items()})
    report['pass'] = report.pop('pass_')
    (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('schema', 'pass', 'candidate_source_sha256',
                     'executable_sha256', 'canonical_functions_byte_identical')}, indent=2))


if __name__ == '__main__':
    main()
