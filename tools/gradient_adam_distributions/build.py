"""Compose pinned HIA bodies and compile the combined owned Tensor profile.

The host only verifies byte custody, selects existing actual HIA function
bodies, and transports them to the frozen native compiler and linker. It
executes no Tensor, gradient, optimizer, random or ownership operation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import subprocess

HERE = Path(__file__).resolve().parent
FUNCTION = re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)', re.M | re.S)
CALL = re.compile(rb'\{"function":"(\w+)"\}')
RNG_EXPORTS = {'rng_seed', 'rng_restore', 'rng_next', 'rng_next64', 'rng_unit32', 'rng_unit64'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def functions(raw):
    pairs = FUNCTION.findall(raw)
    result = {name.decode(): body for body, name in pairs}
    assert len(pairs) == len(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('gradient-build', 'adam-distribution-build', 'compiler', 'linker', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    gradient = a.gradient_build.resolve()
    adam_dist = a.adam_distribution_build.resolve()
    compiler, linker = a.compiler.resolve(), a.linker.resolve()
    paths = [Path(__file__).resolve(), HERE / 'profile.json', HERE / 'functions.json', HERE / 'overlay.hepta']
    for directory in (gradient, adam_dist):
        paths.extend(directory / name for name in ('runtime.hepta', 'rng03.hepta', 'owners.hepta', 'build.json'))
    inputs = {path: path.read_bytes() for path in paths}
    profile = json.loads(inputs[HERE / 'profile.json'])
    mapping = json.loads(inputs[HERE / 'functions.json'])
    assert sha(compiler.read_bytes()) == profile['compiler_sha256']
    assert sha(linker.read_bytes()) == profile['linker_sha256']
    for name, directory in (('gradient', gradient), ('adam_distribution', adam_dist)):
        pin = profile['parents'][name]
        receipt = json.loads(inputs[directory / 'build.json'])
        assert receipt['status'] == 'PASS' and receipt['schema'] in pin['accepted_receipt_schemas']
        assert receipt[pin['receipt_source_field']] == pin['source_sha256']
        assert receipt[pin['receipt_native_field']] == pin['native_sha256']
        assert sha(inputs[directory / 'runtime.hepta']) == pin['source_sha256']
        assert sha(inputs[directory / 'rng03.hepta']) == profile['rng_source_sha256']
        assert sha(inputs[directory / 'owners.hepta']) == profile['owners_source_sha256']
    assert sha(inputs[HERE / 'functions.json']) == profile['functions_map_sha256']
    assert sha(inputs[HERE / 'overlay.hepta']) == profile['overlay_sha256']
    sources = {'gradient': functions(inputs[gradient / 'runtime.hepta']),
               'adam_distribution': functions(inputs[adam_dist / 'runtime.hepta']),
               'overlay': functions(inputs[HERE / 'overlay.hepta'])}
    for name in ('gradient', 'adam_distribution'):
        assert len(sources[name]) == profile['parents'][name]['functions']
    assert len(sources['overlay']) == profile['overlay_functions']
    combined = {}
    for item in mapping:
        name, origin = item['name'], item['origin']
        assert name not in combined and origin in sources
        body = sources[origin][name]
        assert sha(body) == item['sha256']
        for parent in ('gradient', 'adam_distribution'):
            previous = sources[parent].get(name)
            assert item[parent + '_sha256'] == (sha(previous) if previous is not None else None)
        combined[name] = body
    assert set(sources['overlay']) == {item['name'] for item in mapping if item['origin'] == 'overlay'}
    raw = profile['runtime_header'].encode() + b''.join(combined.values())
    assert sha(raw) == profile['source_sha256']
    assert len(combined) == profile['functions']
    assert len(re.findall(rb'^block ', raw, re.M)) == profile['blocks']
    assert raw.count(b' = ') == profile['instructions']
    for name, body in combined.items():
        if name.startswith('dist_'):
            assert body == sources['adam_distribution'][name]
    assert sum(name.startswith('dist_') for name in combined) == 25
    assert combined['tgrad'] == sources['gradient']['tgrad']
    assert combined['optimizer_new_momentum'] == sources['gradient']['optimizer_new_momentum']
    assert combined['owned_optimizer_gradient_value'] == sources['adam_distribution']['owned_optimizer_gradient_value']
    assert combined['owned_adam_new_moment'] == sources['adam_distribution']['owned_adam_new_moment']
    (out / 'runtime.hepta').write_bytes(raw)
    custody = {'schema': 'heptatorch.gradient-adam-distributions-source-custody.v1',
               'source_sha256': sha(raw), 'functions_map_sha256': profile['functions_map_sha256'],
               'overlay_sha256': profile['overlay_sha256'], 'parents': profile['parents'],
               'actual_function_order_and_hashes_match': True,
               'all_25_distribution_bodies_unchanged': True,
               'gradient_binding_is_shared_by_SGD_and_Adam': True,
               'SGD_gradient_layout_and_Adam_parameter_layout_allocators_remain_distinct': True}
    (out / 'source-custody.json').write_text(json.dumps(custody, indent=2) + '\n')
    rng = functions(inputs[gradient / 'rng03.hepta'])
    assert len(rng) == 17
    ordered, done, active = [], set(), set()

    def visit(name):
        if name in RNG_EXPORTS:
            return
        assert name in combined and name not in active, ('missing or recursive function', name)
        if name in done:
            return
        active.add(name)
        for target in sorted({item.decode() for item in CALL.findall(combined[name])}):
            visit(target)
        active.remove(name)
        done.add(name)
        ordered.append(name)

    for name in combined:
        visit(name)
    groups, group = [], []
    instructions = blocks = 0
    for name in ordered:
        body = combined[name]
        ni, nb = body.count(b' = '), len(re.findall(rb'^block ', body, re.M))
        if group and (instructions + ni > 3600 or blocks + nb > 1000 or len(group) >= 65):
            groups.append(group)
            group = []
            instructions = blocks = 0
        group.append(name)
        instructions += ni
        blocks += nb
    if group:
        groups.append(group)
    units = [('rng03', inputs[gradient / 'rng03.hepta'])]
    for index, names in enumerate(groups):
        calls = {item.decode() for name in names for item in CALL.findall(combined[name])}
        imports = sorted(calls - set(names))
        declarations = []
        for name in imports:
            body = rng[name] if name in RNG_EXPORTS else combined[name]
            declarations.append(re.sub(rb' entry\(\w+\)$', b' external()', body.split(b'\n')[0]) + b'\n')
        assert len(names) + len(imports) <= 256
        unit = f'gradient_adam_distributions_{index:02d}'
        body = f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()
        body += b''.join(declarations) + b''.join(combined[name] for name in names)
        units.append((unit, body))
    units.append(('owners', inputs[gradient / 'owners.hepta']))
    assert len(units) == len(profile['units'])
    report = {'schema': 'heptatorch.gradient-adam-distributions-native-build.v1', 'status': 'BUILDING',
              'source_sha256': sha(raw), 'functions': profile['functions'], 'blocks': profile['blocks'],
              'instructions': profile['instructions'], 'state_format': profile['state_format'],
              'backing_format': profile['backing_format'], 'descriptor_bytes': 576,
              'compiler_sha256': profile['compiler_sha256'], 'linker_sha256': profile['linker_sha256'],
              'parents': profile['parents'], 'inputs_sha256': {str(path): sha(data) for path, data in inputs.items()},
              'units': [], 'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}

    def save():
        (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    objects = []
    for (name, body), pin in zip(units, profile['units'], strict=True):
        assert name == pin['name'] and sha(body) == pin['source_sha256']
        (out / (name + '.hepta')).write_bytes(body)
        result = subprocess.run([str(compiler)], input=body, capture_output=True,
                                env={'PATH': ''}, timeout=120)
        (out / (name + '.o')).write_bytes(result.stdout)
        (out / (name + '.stderr')).write_bytes(result.stderr)
        row = {'name': name, 'source_sha256': sha(body), 'object_sha256': sha(result.stdout),
               'functions': len(functions(body)), 'blocks': len(re.findall(rb'^block ', body, re.M)),
               'instructions': body.count(b' = '), 'compiler_exit': result.returncode}
        report['units'].append(row)
        save()
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        assert all(row[key] == pin[key] for key in ('object_sha256', 'functions', 'blocks', 'instructions'))
        objects.append(result.stdout)
    request = b'HIASTKS1' + struct.pack('<3Q', len(objects), 4, 0) + b'step'
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (out / 'link.request.bin').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True,
                            env={'PATH': ''}, timeout=120)
    (out / 'link.stderr').write_bytes(result.stderr)
    report['link_exit'] = result.returncode
    save()
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    assert sha(result.stdout) == profile['native_sha256'] and len(result.stdout) == profile['native_bytes']
    binary = out / 'tensor-gradient-adam-distributions-session'
    binary.write_bytes(result.stdout)
    binary.chmod(0o500)
    assert all(path.read_bytes() == data for path, data in inputs.items())
    assert sha(compiler.read_bytes()) == profile['compiler_sha256']
    assert sha(linker.read_bytes()) == profile['linker_sha256']
    report.update(status='PASS', native_sha256=sha(result.stdout), native_bytes=len(result.stdout),
                  module_count=len(units))
    save()
    print(json.dumps({key: report[key] for key in ('status', 'source_sha256', 'native_sha256',
                                                'functions', 'blocks', 'instructions', 'module_count')}))


if __name__ == '__main__':
    main()
