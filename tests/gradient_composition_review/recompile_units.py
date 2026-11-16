#!/usr/bin/env python3
"""Independently bind complete HIA bodies to units and reproduce native objects.

This validator uses no HIA host parser or producer authoring import. Its small
line scanner establishes byte custody; semantic admission is performed by the
pinned actual native compiler and ABI3 linker.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
RNG = 'a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3'
OWNERS = '9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884'


def definitions(raw):
    result = {}
    name = None
    pending = []
    for line in raw.splitlines(keepends=True):
        if line.startswith(b'fn '):
            assert name is None, 'nested or unterminated function'
            match = re.fullmatch(rb'fn ([a-zA-Z0-9_]+)\([^\n]*\) -> \([^\n]*\) effects\([^\n]*\) (entry\([a-zA-Z0-9_]+\)|external\(\))\n', line)
            assert match, line
            if match[2] == b'external()':
                continue
            name = match[1].decode()
            assert name not in result
            pending = [line]
        elif name is not None:
            pending.append(line)
            if line == b'end\n':
                result[name] = b''.join(pending)
                name = None
                pending = []
    assert name is None
    return result


def exports(raw):
    lines = [line for line in raw.splitlines() if line.startswith(b'exports(')]
    assert len(lines) == 1 and lines[0].endswith(b')')
    names = [name.decode() for name in lines[0][8:-1].split(b',')]
    assert len(names) == len(set(names))
    return set(names)


def static_elf(raw):
    assert raw[:7] == b'\x7fELF\x02\x01\x01'
    assert struct.unpack_from('<HH', raw, 16) == (2, 62)
    offset = struct.unpack_from('<Q', raw, 32)[0]
    width, count = struct.unpack_from('<HH', raw, 54)
    assert width == 56 and offset + width * count <= len(raw)
    types = [struct.unpack_from('<I', raw, offset + index * width)[0] for index in range(count)]
    assert 2 not in types and 3 not in types
    return {'machine': 'x86-64', 'type': 'ET_EXEC', 'program_header_types': types,
            'PT_INTERP_absent': True, 'PT_DYNAMIC_absent': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'compiler', 'linker', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-source', 'expected-image', 'image-name'):
        parser.add_argument('--' + name, required=True)
    a = parser.parse_args()
    a.build = a.build.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    compiler, linker = a.compiler.resolve(), a.linker.resolve()
    inputs = {path: path.read_bytes() for path in (
        Path(__file__).resolve(), compiler, linker, a.build / 'runtime.hepta',
        a.build / 'build.json', a.build / a.image_name)}
    assert SHA(inputs[compiler]) == COMPILER and SHA(inputs[linker]) == LINKER
    assert SHA(inputs[a.build / 'runtime.hepta']) == a.expected_source
    assert SHA(inputs[a.build / a.image_name]) == a.expected_image
    manifest = json.loads(inputs[a.build / 'build.json'])
    assert manifest['status'] == 'PASS'
    assert manifest.get('source_sha256', manifest.get('runtime_source_sha256')) == a.expected_source
    assert manifest.get('native_sha256', manifest.get('executable_sha256')) == a.expected_image
    complete = definitions(inputs[a.build / 'runtime.hepta'])
    seen, objects, rows = {}, [], []
    result = {'schema': 'heptatorch.gradient-composition.independent-native-recompile.v1',
              'status': 'RUNNING', 'source_sha256': a.expected_source, 'image_sha256': a.expected_image,
              'compiler_sha256': COMPILER, 'linker_sha256': LINKER,
              'host_producer_authoring_imported': False, 'host_HIA_parser_imported': False,
              'host_numerical_runtime': False, 'units': rows, 'complete_core_body_count': len(complete)}

    def save():
        (a.output / 'report.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')

    save()
    assert 1 <= len(manifest['units']) <= 64
    for row in manifest['units']:
        name = row['name']
        assert re.fullmatch(r'[a-zA-Z0-9_]+', name)
        source = a.build / (name + '.hepta')
        original_object = a.build / (name + '.o')
        inputs[source] = source.read_bytes()
        inputs[original_object] = original_object.read_bytes()
        raw = inputs[source]
        assert SHA(raw) == row['source_sha256'] and SHA(inputs[original_object]) == row['object_sha256']
        bodies = definitions(raw)
        public = exports(raw)
        assert public <= set(bodies)
        if name == 'rng03':
            assert SHA(raw) == RNG
            assert public == {'rng_seed', 'rng_restore', 'rng_next', 'rng_next64', 'rng_unit32', 'rng_unit64'}
            origin = 'unchanged pinned RNG unit'
        elif name == 'owners':
            assert SHA(raw) == OWNERS and public == {'step', 'close'}
            origin = 'unchanged pinned linear owned step/close unit'
        else:
            assert public == set(bodies), (name, sorted(public ^ set(bodies)))
            for function, body in bodies.items():
                assert function in complete and body == complete[function], (name, function)
                assert function not in seen
                seen[function] = {'unit': name, 'sha256': SHA(body), 'bytes': len(body)}
            origin = 'byte-identical complete runtime bodies'
        compiled = subprocess.run([str(compiler)], input=raw, capture_output=True, env={'PATH': ''}, timeout=60)
        (a.output / (name + '.o')).write_bytes(compiled.stdout)
        (a.output / (name + '.stderr')).write_bytes(compiled.stderr)
        record = {'name': name, 'source_sha256': SHA(raw), 'source_bytes': len(raw),
                  'origin': origin, 'definition_count': len(bodies), 'exports': sorted(public),
                  'compile_exit': compiled.returncode, 'object_sha256': SHA(compiled.stdout),
                  'object_byte_identical': compiled.stdout == inputs[original_object]}
        rows.append(record)
        save()
        assert compiled.returncode == 0 and not compiled.stderr and compiled.stdout == inputs[original_object]
        objects.append(compiled.stdout)
    assert set(seen) == set(complete), sorted(set(complete) - set(seen))
    request = b'HIASTKS1' + struct.pack('<3Q', len(objects), 4, 0) + b'step'
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (a.output / 'link.request').write_bytes(request)
    linked = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=60)
    (a.output / 'link.stderr').write_bytes(linked.stderr)
    (a.output / a.image_name).write_bytes(linked.stdout)
    (a.output / a.image_name).chmod(0o500)
    result['link_exit'] = linked.returncode
    save()
    assert linked.returncode == 0 and not linked.stderr and linked.stdout == inputs[a.build / a.image_name]
    elf = static_elf(linked.stdout)
    execution = subprocess.run([str((a.output / a.image_name).resolve())], input=b'',
                               capture_output=True, env={'PATH': ''}, timeout=10)
    (a.output / 'EOF.stdout').write_bytes(execution.stdout)
    (a.output / 'EOF.stderr').write_bytes(execution.stderr)
    assert execution.returncode == 0 and not execution.stdout and not execution.stderr
    for path, raw in inputs.items():
        assert path.read_bytes() == raw, path
    result.update(status='PASS', source_to_units_exact_once=True, core_body_bindings=seen,
                  all_object_and_image_bytes_identical=True, native_compiler_and_linker_executed=True,
                  actual_EOF_cleanup_exit=execution.returncode, ELF=elf, input_custody_after=True,
                  input_sha256={str(path): SHA(raw) for path, raw in inputs.items()},
                  full_pytorch_rewrite_complete=False)
    save()
    print(json.dumps({key: result[key] for key in ('status', 'source_sha256', 'image_sha256',
                                                  'complete_core_body_count', 'source_to_units_exact_once',
                                                  'all_object_and_image_bytes_identical')}))


if __name__ == '__main__':
    main()
