#!/usr/bin/env python3
"""Author the independent dual-entry owned-session native HIA linker.

The frozen ABI2 compiler/linkers remain unchanged. This adapter preserves all
object/signature/effect/module/relocation admission functions and replaces only
the public entry gate, startup emitter and explicit link-request identity.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import runpy
import struct
import sys

sys.dont_write_bytecode = True
SPEC = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--owned-build', type=Path, required=True)
parser.add_argument('--owned-profile', type=Path, required=True)
parser.add_argument('--startup-template', type=Path, required=True)
parser.add_argument('--qualification-template-sha256')
parser.add_argument('--qualification-prefix-sha256')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
UP = args.source_root.resolve()
BUILD = args.owned_build.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
sha = lambda raw: hashlib.sha256(raw).hexdigest()

profile_raw = args.owned_profile.read_bytes()
assert sha(profile_raw) == '79eed3cb308a1a38e8312ebdcc14bc8da4cca45a6119915f51a904bdfa792eb8'
profile = json.loads(profile_raw)
sources = BUILD / 'sources'
for relative, expected in profile['generated_source_sha256'].items():
    assert sha((sources / relative).read_bytes()) == expected, relative
baseline_path = sources / 'linker-abi2-w1m-source/linker-authoring.txt'
baseline = baseline_path.read_text()
baseline_nodes = ast.parse(baseline).body
baseline_functions = {node.name: node for node in baseline_nodes}
old_receipt = json.loads((sources / 'linker-abi2-w1m-source/linker-authoring.json').read_text())
closure = old_receipt['upstream_source_files_verified_before_and_after']
contract = (UP / 'docs/HIA_SELFHOST_CORE_SUPPORT.json').read_bytes()
assert sha(contract) == '8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c'
for relative, expected in closure.items():
    assert sha((UP / relative).read_bytes()) == expected, relative
sys.path.insert(0, str(UP))
from tools import generate_hia_elf_linker as g
from hepta.hia_agent_assembly import render_assembly, assemble_program
from tools.generate_hia_selfhost import compact_authoring_cfg, total_scalar_functions


def document(source, module):
    nodes = ast.parse(source).body
    names = {node.name: 'f' + str(index) for index, node in enumerate(nodes)}
    signatures = {node.name: ([arg.annotation.id for arg in node.args.args], node.returns.id) for node in nodes}
    total = total_scalar_functions(nodes)
    functions = []
    for node in nodes:
        builder = g.LinkerBuilder(node, names, signatures, total)
        builder.statements(node.body)
        functions.append({'name': names[node.name], 'params': [builder.param(v) for v in builder.params],
                          'returns': [builder.returns], 'effects': ['read', 'write', 'trap'],
                          'entry': 'b0', 'blocks': builder.blocks})
    doc = compact_authoring_cfg({'schema': 'hepta.hia.program.v8', 'module': module, 'functions': functions})
    raw = render_assembly(doc).encode()
    assert assemble_program(raw).as_dict() == doc
    return raw, names, doc


baseline_raw, _, _ = document(baseline, 'borrowed_module_linker')
assert sha(baseline_raw) == profile['linker_profiles']['abi2-w1m']['source_sha256']
template_path = args.startup_template.resolve()
template_bytes = template_path.read_bytes()
qualification = args.qualification_template_sha256 is not None
assert qualification == (args.qualification_prefix_sha256 is not None)
if qualification:
    assert len(args.qualification_template_sha256) == 64 and len(args.qualification_prefix_sha256) == 64
    assert sha(template_bytes) == args.qualification_template_sha256
else:
    assert sha(template_bytes) == 'ab66f4ca7308799df362a281f255709c2d3bb9ace61d97dd0edb7da0ee73438e'
namespace = runpy.run_path(str(template_path), run_name='owned_startup_authoring_only')
prefix, step_hole, close_hole, startup = namespace['owned_session_template']()
assert template_path.read_bytes() == template_bytes
assert sha(prefix) == (args.qualification_prefix_sha256 if qualification else
                       'f68e28de33e2d4120b1a8a4b4463dc9b1d63f095eb8920955d43647ab3fa7f8b')
assert startup['object_text_origin'] == 4096 and len(prefix) < 4096
assert len(prefix) == startup['prefix_bytes'] and sha(prefix) == startup['prefix_sha256']
assert (step_hole, close_hole) == (startup['step_call_hole'], startup['close_call_hole'])
assert step_hole != close_hole
for hole in (step_hole, close_hole):
    assert 176 < hole <= len(prefix) - 4 and prefix[hole - 1:hole + 4] == b'\xe8\0\0\0\0'
assert struct.unpack_from('<Q', prefix, 24)[0] == 0x400000 + startup['elf_entry_offset']

source = baseline
edits = []


def replace_once(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)
    edits.append({'old': old, 'new': new, 'matches': 1})


request_magic = b'HIAOWNT1' if qualification else b'HIAOWNS1'
replace_once(str(int.from_bytes(b'HIAOWN02', 'little')), str(int.from_bytes(request_magic, 'little')))
replace_once('    put(w, 40, 1024)\n', '    put(w, 40, 4096)\n')
replace_once('    fill_range(w, 0, 256, 0)\n',
             '    if name != 4 or uint(s, 32, 4) != ' + str(int.from_bytes(b'step', 'little')) + ':\n'
             '        return 4294967295\n    fill_range(w, 0, 256, 0)\n')


def replace_function(name, replacement):
    global source
    nodes = {node.name: node for node in ast.parse(source).body}
    node = nodes[name]
    lines = source.splitlines(keepends=True)
    previous = ''.join(lines[node.lineno - 1:node.end_lineno])
    source = ''.join(lines[:node.lineno - 1]) + replacement.rstrip() + '\n' + ''.join(lines[node.end_lineno:])
    edits.append({'function': name, 'old_sha256': sha(previous.encode()),
                  'new_sha256': sha(replacement.encode())})


header = ['def headers(o:mutbuf, w:mutbuf) -> u64:']
for offset in range(0, len(prefix), 8):
    chunk = prefix[offset:offset + 8]
    header.append(f'    write(o, {offset}, {int.from_bytes(chunk, "little")}, {len(chunk)})')
header += ['    put(o, 96, get(w, 64))', '    put(o, 104, get(w, 64))',
           f'    write(o, {step_hole}, get(w, 88) - {step_hole + 4}, 4)',
           f'    write(o, {close_hole}, get(w, 96) - {close_hole + 4}, 4)', '    return 0']
replace_function('headers', '\n'.join(header) + '\n')
entry_fragment = (SPEC / 'entry_additions.txt').read_text()
replace_function('module_entry', entry_fragment.replace('CLOSE_PREFIX', str(int.from_bytes(b'clos', 'little'))))
new_functions = {node.name: node for node in ast.parse(source).body}
assert set(new_functions) == set(baseline_functions)
preserved = []
for name in baseline_functions:
    if name not in ('link', 'headers', 'module_entry'):
        previous = ast.dump(baseline_functions[name], include_attributes=False)
        current = ast.dump(new_functions[name], include_attributes=False)
        assert previous == current, name
        preserved.append({'function': name, 'normalized_ast_sha256': sha(current.encode())})
raw, names, doc = document(source, 'owned_session_linker')
counts = {'functions': len(doc['functions']), 'blocks': sum(len(fn['blocks']) for fn in doc['functions']),
          'instructions': sum(len(block['ops']) for fn in doc['functions'] for block in fn['blocks']),
          'source_bytes': len(raw)}
assert counts['functions'] <= 256 and counts['blocks'] <= 2560 and counts['instructions'] <= 8192
assert len(raw) <= 1048576
(OUT / 'linker-authoring.txt').write_text(source)
(OUT / 'owned-session-linker.hepta').write_bytes(raw)
(OUT / 'linker-functions.json').write_text(json.dumps(names, indent=2) + '\n')
(OUT / 'startup-prefix.bin').write_bytes(prefix)
(OUT / 'startup.json').write_text(json.dumps(startup, indent=2) + '\n')
loaded = {}
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if not filename:
        continue
    path = Path(filename).resolve()
    try:
        relative = path.relative_to(UP).as_posix()
    except ValueError:
        continue
    actual = sha(path.read_bytes())
    assert closure.get(relative) == actual, relative
    loaded[relative] = actual
for relative, expected in closure.items():
    assert sha((UP / relative).read_bytes()) == expected, relative
for relative, expected in profile['generated_source_sha256'].items():
    assert sha((sources / relative).read_bytes()) == expected, relative
assert args.owned_profile.read_bytes() == profile_raw
assert template_path.read_bytes() == template_bytes
assert (SPEC / 'entry_additions.txt').read_text() == entry_fragment
report = {'schema': 'heptatorch.owned-session-linker-authoring.v1',
          'profile': ('heptatorch.hia.owned-session-linker-qualification.v1' if qualification else
                      'heptatorch.hia.owned-session-linker.v1'),
          'qualification_only': qualification, 'upstream_commit': profile['upstream_commit'],
          'owned_profile_sha256': sha(profile_raw), 'baseline_authoring_sha256': sha(baseline.encode()),
          'baseline_hia_sha256': sha(baseline_raw), 'source_sha256': sha(raw), 'source_structure': counts,
          'adapter_sha256': sha(Path(__file__).read_bytes()), 'entry_fragment_sha256': sha(entry_fragment.encode()),
          'startup_author_sha256': sha(template_bytes), 'startup_prefix_sha256': sha(prefix),
          'startup': startup, 'request_magic': request_magic.decode(), 'fixed_entry_name': 'step',
          'fixed_close_export': 'close', 'object_text_origin': 4096, 'interface_abi': 2,
          'preserved_function_asts': preserved, 'edits': edits,
          'upstream_source_files_verified_before_and_after': closure, 'upstream_files_actually_loaded': loaded,
          'native_compilation_performed': False, 'native_runtime_dependency': 'Linux syscalls only'}
(OUT / 'authoring.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items()
                  if key not in ('edits', 'upstream_source_files_verified_before_and_after',
                                 'upstream_files_actually_loaded', 'preserved_function_asts', 'startup')}, indent=2))
