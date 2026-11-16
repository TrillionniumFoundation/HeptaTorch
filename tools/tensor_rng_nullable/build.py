#!/usr/bin/env python3
"""Compose frozen native RNG and nullable-gradient HIA bodies, then link ABI3.

This is a source-only build adapter. It does not execute Tensor or RNG math.
All bodies come from hash-bound predecessors; the native compiler checks every
bounded unit and the native linker checks signatures, effects and stack budget.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
COMMON = '7447c169d3b2cad1e55cb7b1b8970d67f4103c5866f6db0a60c0b4d05d02cc0b'
MATERIALIZATION = '9feab47afea6723284e15a2ad6c9071dc8c5bdc27b572d2eef4ab4628ed1aea9'
NULLABLE = 'cebd8b474c2604c635bdc39a8a4e75721ff04a0544c5cbff9b706d5e5c85b126'
RNG_UNIT = 'a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3'
OWNERS = '9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FUNCTION = re.compile(rb'^(fn (\w+)\([^\n]* entry\(\w+\)\n.*?^end\n)', re.M | re.S)
CALL = re.compile(rb'\{"function":"(\w+)"\}')
EXTERNAL_RNG = ['rng_seed', 'rng_restore', 'rng_next', 'rng_next64', 'rng_unit32', 'rng_unit64']
CHANGED70 = {'descriptor', 'tgrad', 'zero_grad', 'init_state', 'state_header',
             'object_command', 'ensure_backward_work', 'release_backward_scratch'}
NEW70 = {'zero_grad_none'}
TAG_FUNCTIONS = {'init_state', 'state_header', 'so_state_init', 'so_state_header'}


def functions(raw):
    matches = FUNCTION.findall(raw)
    result = {name.decode(): body for body, name in matches}
    assert len(result) == len(matches), 'duplicate function'
    return result


def literal(tag):
    return b'{"value":' + str(int.from_bytes(tag, 'little')).encode() + b'}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('common-build', 'materialization-build', 'nullable-build', 'stack-build', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    common = a.common_build.resolve()
    materialization = a.materialization_build.resolve()
    nullable = a.nullable_build.resolve()
    compiler = a.stack_build.resolve() / 'native/stack-module-compiler'
    linker = a.stack_build.resolve() / 'native/stack-linker-session'
    pins = {common / 'runtime.hepta': COMMON, common / 'rng03.hepta': RNG_UNIT,
            common / 'owners.hepta': OWNERS, materialization / 'runtime.hepta': MATERIALIZATION,
            nullable / 'runtime.hepta': NULLABLE, compiler: COMPILER, linker: LINKER}
    inputs = {path: path.read_bytes() for path in pins}
    for path, pin in pins.items():
        assert SHA(inputs[path]) == pin, path
    own_source = Path(__file__).resolve()
    inputs[own_source] = own_source.read_bytes()
    before = functions(inputs[materialization / 'runtime.hepta'])
    rn = functions(inputs[common / 'runtime.hepta'])
    ng = functions(inputs[nullable / 'runtime.hepta'])
    rng = functions(inputs[common / 'rng03.hepta'])
    assert len(before) == 261 and len(ng) == 262 and len(rng) == 17
    assert set(ng) - set(before) == NEW70
    assert set(before) - set(ng) == set()
    changed70 = {name for name in before if before[name] != ng[name]}
    changed_rn = {name for name in before if before[name] != rn[name]}
    assert changed70 == CHANGED70
    assert changed70 & changed_rn == {'init_state', 'state_header'}
    for name in ('init_state', 'state_header'):
        old = literal(b'HTSTATE5')
        assert before[name].count(old) == 1
        assert before[name].replace(old, literal(b'HTSTATE6')) == ng[name]
    combined = dict(rn)
    for name in (CHANGED70 - {'init_state', 'state_header'}) | NEW70:
        if name in before:
            assert combined[name] == before[name], name
        combined[name] = ng[name]
    for name, old, new in (
        ('init_state', b'HTRNGP01', b'HTRNGN01'),
        ('state_header', b'HTRNGP01', b'HTRNGN01'),
        ('so_state_init', b'HTORNGP1', b'HTORNGN1'),
        ('so_state_header', b'HTORNGP1', b'HTORNGN1'),
    ):
        assert combined[name].count(literal(old)) == 1
        combined[name] = combined[name].replace(literal(old), literal(new))
    changed_common = {name for name in rn if rn[name] != combined[name]}
    assert changed_common == (CHANGED70 - {'init_state', 'state_header'}) | TAG_FUNCTIONS
    assert combined['entry'] == rn['entry'], 'command58 prefix commit entry changed'
    assert set(combined) - set(rn) == NEW70
    for name in (CHANGED70 - {'init_state', 'state_header'}) | NEW70:
        assert combined[name] == ng[name], name
    raw = b'hia-asm.v1 hepta.hia.program.v8 tensor_rng_nullable_runtime\n' + b''.join(combined.values())
    (out / 'runtime.hepta').write_bytes(raw)
    ordered, active, done = [], set(), set()

    def visit(name):
        if name in EXTERNAL_RNG:
            return
        assert name in combined, name
        if name in done:
            return
        assert name not in active, ('cycle', name)
        active.add(name)
        for target in sorted({s.decode() for s in CALL.findall(combined[name])}):
            visit(target)
        active.remove(name)
        done.add(name)
        ordered.append(name)

    for name in combined:
        visit(name)
    groups, group, ins, blocks = [], [], 0, 0
    for name in ordered:
        body = combined[name]
        ni, nb = body.count(b' = '), len(re.findall(rb'^block ', body, re.M))
        if group and (ins + ni > 3600 or blocks + nb > 1000 or len(group) >= 65):
            groups.append(group)
            group, ins, blocks = [], 0, 0
        group.append(name)
        ins += ni
        blocks += nb
    if group:
        groups.append(group)
    units = [('rng03', inputs[common / 'rng03.hepta'])]
    for index, group in enumerate(groups):
        local = set(group)
        imported = sorted({s.decode() for name in group for s in CALL.findall(combined[name])} - local)
        assert len(group) + len(imported) <= 256
        name = 'tensor_rng_' + str(index)
        source = ('hia-asm.v1 hepta.hia.program.v8 ' + name + '\nexports(' + ','.join(group) + ')\n').encode()
        for target in imported:
            body = rng[target] if target in EXTERNAL_RNG else combined[target]
            source += re.sub(rb' entry\(\w+\)$', b' external()', body.split(b'\n', 1)[0]) + b'\n'
        source += b''.join(combined[target] for target in group)
        units.append((name, source))
    units.append(('owners', inputs[common / 'owners.hepta']))
    rows, objects = [], []
    report = {
        'schema': 'heptatorch.tensor-rng-nullable-native-build.v1', 'status': 'BUILDING',
        'profile': 'heptatorch.tensor-rng.nullable.f32.seven-kind.abi3.v1',
        'common_source_sha256': COMMON, 'nullable_source_sha256': NULLABLE,
        'materialization_source_sha256': MATERIALIZATION,
        'runtime_source_sha256': SHA(raw), 'compiler_sha256': COMPILER, 'linker_sha256': LINKER,
        'core_functions': len(combined), 'core_blocks': len(re.findall(rb'^block ', raw, re.M)),
        'core_instructions': raw.count(b' = '),
        'changed_common_functions': sorted(changed_common), 'added_functions': sorted(NEW70),
        'common_functions_unchanged': len(rn) - len(changed_common),
        'nullable_functions_copied_byte_identical': sorted((CHANGED70 - {'init_state', 'state_header'}) | NEW70),
        'entry_byte_identical': True, 'rng_module_byte_identical': True, 'owner_module_byte_identical': True,
        'state': {'prefix_magic': 'HTRNGN01', 'backing_magic': 'HTORNGN1', 'kinds': 7,
                  'prefix_bytes': 139264, 'pool_base_formula': '144+488*slots',
                  'generator_kind': 7, 'gradient_presence': 360, 'reserved_temporaries': [336, 368, 376]},
        'input_sha256': {str(path): SHA(data) for path, data in inputs.items()},
        'common_function_body_map': {name: {'before': SHA(body), 'after': SHA(combined[name]),
                                         'unchanged': body == combined[name]} for name, body in rn.items()},
        'units': rows, 'host_production_numerical_runtime': False,
        'full_pytorch_rewrite_complete': False, 'optimizer_object_lifecycle_complete': False,
    }

    def save():
        (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    for name, source in units:
        (out / (name + '.hepta')).write_bytes(source)
        result = subprocess.run([str(compiler)], input=source, capture_output=True, env={'PATH': ''}, timeout=60)
        (out / (name + '.o')).write_bytes(result.stdout)
        (out / (name + '.compile.stderr')).write_bytes(result.stderr)
        row = {'name': name, 'source_sha256': SHA(source), 'source_bytes': len(source),
               'functions': len(re.findall(rb'^fn ', source, re.M)),
               'blocks': len(re.findall(rb'^block ', source, re.M)), 'instructions': source.count(b' = '),
               'compile_exit': result.returncode, 'object_sha256': SHA(result.stdout)}
        rows.append(row)
        save()
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), row
        objects.append(result.stdout)
    request = b'HIASTKS1' + struct.pack('<3Q', len(objects), 4, 0) + b'step'
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (out / 'link.request').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=90)
    (out / 'link.stderr').write_bytes(result.stderr)
    (out / 'tensor-rng-session').write_bytes(result.stdout)
    report['link_exit'] = result.returncode
    save()
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    (out / 'tensor-rng-session').chmod(0o500)
    for path, data in inputs.items():
        assert path.read_bytes() == data, path
    for (name, source), obj in zip(units, objects):
        assert (out / (name + '.hepta')).read_bytes() == source
        assert (out / (name + '.o')).read_bytes() == obj
    assert (out / 'runtime.hepta').read_bytes() == raw
    report.update(status='PASS', executable_sha256=SHA(result.stdout), executable_bytes=len(result.stdout))
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ('input_sha256', 'units', 'common_function_body_map')}, indent=2))


if __name__ == '__main__':
    main()
