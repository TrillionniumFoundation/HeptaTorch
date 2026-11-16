"""Checked source overlay and real native modular build for owned tensors.

The canonical tensor HIA program remains the unique source of all numerical
functions. This organizer writes only the isolated output directory. It is not
a HIA compiler, a native linker, or a tensor numerical runtime.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = None
CORE_SHA = '264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233'
COMPILER_SHA = '0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73'
LINKER_SHA = '55a2b0a762c8db574a65d0d0e6a18e3cedc668c01d20080a11c26cf1ccb601b3'
FUNCTION = re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]*\n.*?^end\n)', re.M | re.S)
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def functions(raw):
    result = {}
    for match in FUNCTION.finditer(raw):
        body, name = match.groups(); name = name.decode()
        assert name not in result
        result[name] = body
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=HERE.parents[1])
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--linker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    global BASE
    BASE = args.repository.resolve()
    profile_bytes = (HERE / 'profile.json').read_bytes()
    profile = json.loads(profile_bytes)
    upstream = args.source_root.resolve()
    upstream_inputs = {upstream / name: (upstream / name).read_bytes() for name in profile['upstream_authoring_sources']}
    for name, digest in profile['upstream_authoring_sources'].items():
        assert sha(upstream_inputs[upstream / name]) == digest, name
    sys.path.insert(0, str(upstream))
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    source_path = BASE / 'src/heptatorch.hepta'
    source = source_path.read_bytes(); assert sha(source) == CORE_SHA
    original = functions(source); assert len(original) == 143
    compiler = args.compiler.resolve(); linker = args.linker.resolve()
    assert sha(compiler.read_bytes()) == COMPILER_SHA and sha(linker.read_bytes()) == LINKER_SHA
    inputs = {path: path.read_bytes() for path in [HERE / 'helpers.py', HERE / 'owners.hepta',
        HERE / 'authoring.py', HERE / 'command_base.py',
        HERE / 'matmul_base.py', HERE / 'gather_base.py',
        BASE / 'docs/functions.json', BASE / 'tools/modules/split_runtime.py']}
    spec = importlib.util.spec_from_file_location('owned_tensor_emit', HERE / 'authoring.py')
    for path in (BASE / 'docs/functions.json', BASE / 'tools/modules/split_runtime.py'):
        assert sha(inputs[path]) == profile['canonical_support_files'][str(path.relative_to(BASE))]
    emitter = importlib.util.module_from_spec(spec); spec.loader.exec_module(emitter)
    emitter.ROOT = BASE
    command = inputs[HERE / 'command_base.py'].decode()
    old_fragment, _ = emitter.emit(command)
    assert functions(old_fragment.encode())['command'] == original['command'], 'base command emission changed'
    needle = '                        tset(w, result, i, read4(s, source + 4 * i))'
    assert command.count(needle) == 1
    command = command.replace(needle, needle.replace('tset(', 'tensor_initialize('), 1)
    insertion = '''    if code == 38:
        recognized = 1
        putw(w,2048,c)
        putw(w,2056,getr(s,cp+32))
        putw(w,2064,getr(s,cp+40))
        putw(w,2072,getr(s,cp+48))
        result = full_tensor(w,a,b,2048,getr(s,cp+56))
    if code == 39 or code == 40:
        recognized = 1
        if valid(w,a) != 0:
            dp = dptr(w,a)
            n = getw(w,dp+8)
            if code == 40 and (getw(w,dp+40) != 0 or getw(w,dp+32) == 0):
                terror(w,5)
            if b > n or c > n-b:
                terror(w,2)
            if c > (length(o)-at-32)//4:
                terror(w,3)
            if getw(w,0) == 0:
                i = 0
                while i < c:
                    value = 0
                    if code == 39:
                        value = tget(w,a,b+i)
                    else:
                        value = tgrad(w,a,b+i)
                    put4(o,at+32+i*4,value)
                    i = i + 1
                result = c
                payload = c*4
'''
    needle = '    if recognized == 0:\n'
    assert command.count(needle) == 1
    command = command.replace(needle, insertion + needle, 1)
    matmul = inputs[HERE / 'matmul_base.py'].decode()
    gather = inputs[HERE / 'gather_base.py'].decode()
    for name, baseline in (('matmul', matmul), ('gather', gather)):
        emitted, _ = emitter.emit(baseline)
        assert functions(emitted.encode())[name] == original[name], name + ' baseline source differs'
    old = '''    n = 1
    axis = 0
    while axis < i:
        n = n*getw(w,128+axis*8)
        if n > 262144:
            return terror(w,3)
        axis = axis+1
'''
    new = '''    n = 1
    axis = 0
    while axis < i:
        if getw(w,128+axis*8) == 0:
            n = 0
        axis = axis+1
    if n != 0:
        axis = 0
        while axis < i:
            n = n*getw(w,128+axis*8)
            if n > 16777216:
                return terror(w,3)
            axis = axis+1
'''
    assert matmul.count(old) == 1; matmul = matmul.replace(old, new, 1)
    old = '''        count = count * dim
        if count > 262144:
            return terror(w,3)
        putw(w,2048+j*8,dim)
        j = j + 1
    offset = config+8+rank*8
'''
    new = '''        if dim == 0:
            count = 0
        putw(w,2048+j*8,dim)
        j = j + 1
    if count != 0:
        j = 0
        while j < rank:
            count = count * getw(w,2048+j*8)
            if count > 16777216:
                return terror(w,3)
            j = j + 1
    offset = config+8+rank*8
'''
    assert gather.count(old) == 1; gather = gather.replace(old, new, 1)
    authoring = inputs[HERE / 'helpers.py'].decode() + '\n' + command + '\n' + matmul + '\n' + gather
    (out / 'overlay-authoring.py').write_text(authoring)
    fragment, mapping = emitter.emit(authoring)
    replacements = functions(fragment.encode())
    # These three substitutions retain the rest of each canonical body exactly.
    for name in ('mark_graph', 'backward_seed'):
        old = original[name]
        needle = b'= const() {"value":255}'
        assert old.count(needle) == 1
        replacements[name] = old.replace(needle, b'= const() {"value":65535}', 1)
    old = original['constant_tensor']
    assert old.count(b'{"function":"tset"}') == 1
    replacements['constant_tensor'] = old.replace(b'{"function":"tset"}', b'{"function":"tensor_initialize"}', 1)
    # Literal-only changes to preflight allocation limits. Per-dimension bounds
    # remain 262144, and all numerical kernel instructions remain unchanged.
    for name, ordinal, count, old_value, new_value in (
        ('indexed_add_validate', 1, 2, 262144, 16777216),
        ('concat_shape', 1, 2, 262144, 16777216),
        ('tensor_join', 0, 1, 256, 65536),
    ):
        body = original[name]
        needle = ('= const() {"value":' + str(old_value) + '}').encode()
        positions = [match.start() for match in re.finditer(re.escape(needle), body)]
        assert len(positions) == count
        position = positions[ordinal]
        replacement = ('= const() {"value":' + str(new_value) + '}').encode()
        replacements[name] = body[:position] + replacement + body[position + len(needle):]
    core = source.split(b'\n', 1)[0] + b'\n'
    for name, body in original.items(): core += replacements.pop(name, body)
    for body in replacements.values(): core += body
    assert sha(core) == profile['core_overlay_sha256']
    (out / 'runtime.hepta').write_bytes(core)
    (out / 'overlay.hepta').write_text(fragment)
    (out / 'functions.json').write_text(json.dumps(mapping, indent=2) + '\n')
    split_spec = importlib.util.spec_from_file_location('owned_tensor_split', BASE / 'tools/modules/split_runtime.py')
    split = importlib.util.module_from_spec(split_spec); split_spec.loader.exec_module(split)
    split.split(out / 'runtime.hepta', out / 'units')
    all_functions = functions(core)
    owner = inputs[HERE / 'owners.hepta']
    needs = {name.decode() for name in re.findall(rb'^fn (\w+)\([^\n]+ external\(\)$', owner, re.M)}
    units = []
    for name in ('ht_kernel', 'ht_runtime'):
        path = out / 'units' / (name + '.hepta')
        raw = path.read_bytes(); local = set(functions(raw))
        lines = raw.split(b'\n'); exports = lines[1][8:-1].decode().split(',')
        exports = set(exports) | (local & needs)
        lines[1] = ('exports(' + ','.join(key for key in all_functions if key in exports) + ')').encode()
        raw = b'\n'.join(lines); path.write_bytes(raw)
        units.append((name, raw))
    (out / 'units' / 'owned_tensor_session.hepta').write_bytes(owner)
    units.append(('owned_tensor_session', owner))
    unit_rows = []; objects = []
    for name, raw in units:
        result = subprocess.run([str(compiler)], input=raw, capture_output=True, env={'PATH': ''}, timeout=120)
        (out / (name + '.compiler.stderr')).write_bytes(result.stderr)
        (out / (name + '.compiler.stdout')).write_bytes(result.stdout)
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), (name, result.returncode, result.stderr)
        (out / (name + '.o')).write_bytes(result.stdout); objects.append(result.stdout)
        unit_rows.append({'name': name, 'source_sha256': sha(raw), 'object_sha256': sha(result.stdout),
            'source_bytes': len(raw), 'object_bytes': len(result.stdout),
            'functions': len(functions(raw)), 'blocks': len(re.findall(rb'^block ', raw, re.M)),
            'instructions': raw.count(b' = ')})
    request = b'HIAOWNS1' + struct.pack('<QQQ', len(objects), 4, 0) + b'step'
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (out / 'link.request.bin').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=120)
    (out / 'link.stderr').write_bytes(result.stderr); (out / 'link.stdout').write_bytes(result.stdout)
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), (result.returncode, result.stderr)
    assert sha(result.stdout) == profile['executable_sha256']
    app = out / 'owned-tensor-session'; app.write_bytes(result.stdout); app.chmod(0o500)
    assert source_path.read_bytes() == source
    assert sha(compiler.read_bytes()) == COMPILER_SHA and sha(linker.read_bytes()) == LINKER_SHA
    for path, raw in inputs.items(): assert path.read_bytes() == raw
    for path, raw in upstream_inputs.items(): assert path.read_bytes() == raw
    assert (HERE / 'profile.json').read_bytes() == profile_bytes
    assert (out / 'runtime.hepta').read_bytes() == core
    for (name, raw), obj in zip(units, objects):
        assert (out / 'units' / (name + '.hepta')).read_bytes() == raw
        assert (out / (name + '.o')).read_bytes() == obj
        assert sha(raw) == profile['units'][name]['source_sha256']
        assert sha(obj) == profile['units'][name]['object_sha256']
    rows = [{'name': name, 'canonical_sha256': sha(body), 'candidate_sha256': sha(all_functions[name]),
        'byte_identical': body == all_functions[name]} for name, body in original.items()]
    report = {'schema': 'heptatorch.owned-tensor-native-build.v1', 'pass': True,
        'canonical_source_sha256': sha(source), 'canonical_function_count': len(original),
        'candidate_source_sha256': sha(core), 'candidate_function_count': len(all_functions),
        'canonical_functions_byte_identical': sum(row['byte_identical'] for row in rows),
        'compiler_sha256': COMPILER_SHA, 'linker_sha256': LINKER_SHA,
        'executable_sha256': sha(result.stdout), 'executable_bytes': len(result.stdout),
        'input_sources': {str(path): sha(raw) for path, raw in inputs.items()},
        'unit_rows': unit_rows, 'function_source_mapping': rows,
        'numerics_compiled_and_linked_natively': True, 'canonical_modified': False,
        'full_PyTorch_closure_claimed': False}
    (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'function_source_mapping'}, indent=2))


if __name__ == '__main__':
    main()
