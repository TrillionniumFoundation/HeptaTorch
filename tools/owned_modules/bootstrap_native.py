"""Native bootstrap of the compiler as independent checked HIA source units.

Python transports sources, objects and explicit request frames. All target
compilation, interface admission, relocation and executable packing are HIA.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess

SPEC = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sources', type=Path, required=True)
parser.add_argument('--borrowed-tools', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = args.sources.resolve()
OLD = args.borrowed_tools.resolve()
OUT = args.output.resolve()
OUT.mkdir(exist_ok=False)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
profile = json.loads((SPEC / 'profile.json').read_text())
assert sha((OLD / 'module-compiler').read_bytes()) == 'e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59'
assert sha((OLD / 'linker-packer').read_bytes()) == '98f0181dd509ffeb09679c6602751c86c2b972e22c5f1b9c3fd26a1e677f972b'
for relative, expected in profile['generated_source_sha256'].items():
    assert sha((ROOT / relative).read_bytes()) == expected, relative


def native(executable, request, destination):
    result = subprocess.run([str(executable.resolve())], input=request, capture_output=True,
                            env={'PATH': ''}, timeout=30)
    destination.with_suffix(destination.suffix + '.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr and result.stdout, (
        executable, destination, result.returncode, result.stderr)
    destination.write_bytes(result.stdout)
    return result.stdout


linkers = {}
for label in ('abi1-w2m', 'abi2-w2m', 'abi2-w1m'):
    source = ROOT / ('linker-' + label + '-source/module-linker.hepta')
    image = native(OLD / 'module-compiler', source.read_bytes(), OUT / ('linker-' + label + '.image'))
    executable = OUT / ('linker-' + label)
    native(OLD / 'linker-packer', b'HIAFLT01' + struct.pack('<Q', len(image)) + image, executable)
    executable.chmod(0o700)
    linkers[label] = executable

source = (ROOT / 'compiler/module-compiler.hepta').read_bytes()
names = json.loads((ROOT / 'compiler/compiler-functions.json').read_text())
refused = subprocess.run([str(OLD / 'module-compiler')], input=source, capture_output=True,
                         env={'PATH': ''}, timeout=30)
assert refused.returncode == 2 and not refused.stdout and not refused.stderr
(OUT / 'aggregate-capacity-refusal.json').write_text(json.dumps({
    'status': 'PASS', 'source_sha256': sha(source),
    'instructions': 8229, 'per_unit_instruction_limit': 8192,
    'compiler_sha256': sha((OLD / 'module-compiler').read_bytes()),
    'exit': refused.returncode, 'stdout_bytes': len(refused.stdout)}, indent=2) + '\n')
matches = list(re.finditer(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]*\n.*?^end\n)', source, re.M | re.S))
functions = {match.group(2).decode(): match.group(1) for match in matches}
assert len(functions) == len(matches) == len(names)
assert source[:matches[0].start()] == b'hia-asm.v1 hepta.hia.program.v8 selfhost\n'
for left, right in zip(matches, matches[1:]):
    assert not source[left.end():right.start()].strip()
assert not source[matches[-1].end():].strip()
callees = {name: set(match.decode() for match in re.findall(rb'\{"function":"([A-Za-z_][A-Za-z_0-9]*)"\}', body))
           for name, body in functions.items()}
assert set().union(*callees.values()) <= set(functions)
kernel = {names[name] for name in ('operation_dispatch', 'owned_instruction', 'fix_all', 'exports_finish',
                                  'native_allocator_helpers', 'chain_drop_helper', 'cli_finish',
                                  'function_start', 'wrapper')}
while True:
    expanded = kernel | set().union(*(callees[name] for name in kernel))
    if expanded == kernel:
        break
    kernel = expanded
frontend = set(functions) - kernel
assert names['compile'] in frontend
imports = set().union(*(callees[name] for name in frontend)) & kernel
ordered = lambda selection: [name for name in functions if name in selection]
units = []
for label, local, external, exports in (('compiler_frontend', frontend, imports, {names['compile']}),
                                        ('compiler_backend', kernel, set(), imports)):
    raw = ('hia-asm.v1 hepta.hia.program.v8 ' + label + '\nexports(' + ','.join(ordered(exports)) + ')\n').encode()
    for name in ordered(external):
        header = functions[name].split(b'\n')[0]
        header, count = re.subn(rb'entry\([^)]*\)$', b'external()', header)
        assert count == 1 and b'return_loans' not in header
        raw += header + b'\n'
    raw += b''.join(functions[name] for name in ordered(local))
    path = OUT / (label + '.hepta')
    path.write_bytes(raw)
    assert sha(raw) == profile['compiler_partition_source_sha256'][label]
    counts = {'definitions': len(local), 'external_declarations': len(external),
              'functions_including_declarations': len(local) + len(external),
              'blocks': len(re.findall(rb'^block ', raw, re.M)), 'instructions': raw.count(b' = ')}
    assert counts['functions_including_declarations'] <= 256
    assert counts['blocks'] <= 2560 and counts['instructions'] <= 8192
    units.append({'name': label, 'source_sha256': sha(raw), 'source_bytes': len(raw), **counts,
                  'exports': ordered(exports), 'imports': ordered(external),
                  'unchanged_function_body_sha256': {name: sha(functions[name]) for name in ordered(local)}})
partition = {'source_sha256': sha(source), 'source_functions': len(functions), 'complete_partition': True,
             'per_unit_limits_unchanged': True, 'module_dag': [['compiler_frontend', 'compiler_backend']],
             'units': units}
(OUT / 'partition.json').write_text(json.dumps(partition, indent=2) + '\n')
print(json.dumps({**partition, 'units': [{k: v for k, v in row.items() if k not in ('exports', 'imports', 'unchanged_function_body_sha256')}
                                      for row in units]}, indent=2), flush=True)


def compile_and_link(compiler, linker, magic, generation):
    directory = OUT / generation
    directory.mkdir()
    objects = []
    for unit in units:
        raw = (OUT / (unit['name'] + '.hepta')).read_bytes()
        objects.append(native(compiler, raw, directory / (unit['name'] + '.o')))
    entry = names['compile'].encode()
    request = magic + struct.pack('<3Q', len(objects), len(entry), 0) + entry
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (directory / 'link-request.bin').write_bytes(request)
    executable = directory / 'compiler'
    native(linker, request, executable)
    executable.chmod(0o700)
    receipt = {'generation': generation, 'compiler_input_sha256': sha(compiler.read_bytes()),
               'linker_sha256': sha(linker.read_bytes()), 'request_magic': magic.decode(),
               'workspace_bytes': 2097152, 'objects_sha256': [sha(obj) for obj in objects],
               'link_request_sha256': sha(request), 'executable_sha256': sha(executable.read_bytes())}
    print(json.dumps(receipt), flush=True)
    return executable, receipt


compiler, first = compile_and_link(OLD / 'module-compiler', linkers['abi1-w2m'], b'HIACMP01', 'stage0-abi1')
generations = [first]
for index in range(1, 4):
    compiler, receipt = compile_and_link(compiler, linkers['abi2-w2m'], b'HIAOWNC2', 'stage' + str(index) + '-abi2')
    generations.append(receipt)
assert len({row['executable_sha256'] for row in generations[1:]}) == 1
assert generations[-1]['executable_sha256'] == profile['module_compiler_executable_sha256']
shutil.copy2(compiler, OUT / 'owned-module-compiler')
for label, linker in linkers.items():
    assert sha(linker.read_bytes()) == profile['linker_profiles'][label]['executable_sha256']
    image = native(compiler, (ROOT / ('linker-' + label + '-source/module-linker.hepta')).read_bytes(),
                   OUT / ('linker-' + label + '.rebuilt-image'))
    assert image == (OUT / ('linker-' + label + '.image')).read_bytes()
packer = native(compiler, (ROOT / 'linker-abi2-w1m-source/linker-packer.hepta').read_bytes(),
                OUT / 'linker-packer.rebuilt')
assert packer == (OLD / 'linker-packer').read_bytes()
for relative, expected in profile['generated_source_sha256'].items():
    assert sha((ROOT / relative).read_bytes()) == expected, relative
assert sha((OLD / 'module-compiler').read_bytes()) == 'e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59'
assert sha((OLD / 'linker-packer').read_bytes()) == '98f0181dd509ffeb09679c6602751c86c2b972e22c5f1b9c3fd26a1e677f972b'
report = {'status': 'PASS', 'native_compilation_and_linking': True, 'host_numerical_runtime': False,
          'source_sha256': sha(source), 'partition': partition, 'generations': generations,
          'successive_abi2_generations_byte_identical': True,
          'stage0_matches_abi2_observation': first['executable_sha256'] == generations[-1]['executable_sha256'],
          'all_linkers_rebuilt_by_final_compiler_byte_identical': True,
          'linker_packer_rebuilt_by_final_compiler_byte_identical': True,
          'runtime_path': 'empty',
          'linkers': {name: {'executable_sha256': sha(path.read_bytes()),
                             'source_sha256': sha((ROOT / ('linker-' + name + '-source/module-linker.hepta')).read_bytes())}
                      for name, path in linkers.items()}}
(OUT / 'reproduction.json').write_text(json.dumps(report, indent=2) + '\n')
