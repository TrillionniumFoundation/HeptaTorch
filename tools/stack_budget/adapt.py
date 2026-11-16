#!/usr/bin/env python3
"""Author a separate ABI3 native stack-budget compiler and linkers."""
from pathlib import Path
from types import FunctionType
import argparse
import ast
import hashlib
import json
import sys

sys.dont_write_bytecode = True
SPEC = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--owned-build', type=Path, required=True)
parser.add_argument('--owned-profile', type=Path, required=True)
parser.add_argument('--session-build', type=Path, required=True)
parser.add_argument('--session-profile', type=Path, required=True)
parser.add_argument('--session-template', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
UP = args.source_root.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
own_snapshots = {path: path.read_bytes() for path in (Path(__file__), SPEC/'startup.py', SPEC/'compiler_additions.txt', SPEC/'linker_additions.txt', SPEC/'machine_additions.txt')}
owned_raw = args.owned_profile.read_bytes()
session_raw = args.session_profile.read_bytes()
assert sha(owned_raw) == '79eed3cb308a1a38e8312ebdcc14bc8da4cca45a6119915f51a904bdfa792eb8'
assert sha(session_raw) == 'c7389eeaa711491230dc047a553dbffa908ed7015b44136f1c84eb5ad0b08ce6'
owned, session = json.loads(owned_raw), json.loads(session_raw)
SOURCES = args.owned_build.resolve() / 'sources'
for relative, expected in owned['generated_source_sha256'].items():
    assert sha((SOURCES / relative).read_bytes()) == expected, relative
closure = json.loads((SOURCES / 'linker-abi2-w1m-source/linker-authoring.json').read_text())['upstream_source_files_verified_before_and_after']
assert sha((UP / 'docs/HIA_SELFHOST_CORE_SUPPORT.json').read_bytes()) == '8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c'
for relative, expected in closure.items():
    assert sha((UP / relative).read_bytes()) == expected, relative
sys.path.insert(0, str(UP))
sys.path.insert(0, str(SPEC))
from tools import generate_hia_selfhost as compiler_generator
from tools import generate_hia_elf_linker as linker_generator
from tools import generate_hia_native_filter as filter_generator
from tools.hia_selfhost_chain_elf import FRAGMENTS
from hepta.hia_agent_assembly import render_assembly, assemble_program
from startup import session_prefix, filter_prefix


def compiler_document(source):
    namespace = dict(compiler_generator.document.__globals__, source_program=lambda: source)
    return FunctionType(compiler_generator.document.__code__, namespace)(compact=True)


def linker_document(source, module):
    nodes = ast.parse(source).body
    names = {node.name: 'f' + str(index) for index, node in enumerate(nodes)}
    signatures = {node.name: ([arg.annotation.id for arg in node.args.args], node.returns.id) for node in nodes}
    total = compiler_generator.total_scalar_functions(nodes)
    functions = []
    for node in nodes:
        builder = linker_generator.LinkerBuilder(node, names, signatures, total)
        builder.statements(node.body)
        functions.append({'name': names[node.name], 'params': [builder.param(v) for v in builder.params],
                          'returns': [builder.returns], 'effects': ['read', 'write', 'trap'],
                          'entry': 'b0', 'blocks': builder.blocks})
    return compiler_generator.compact_authoring_cfg({'schema': 'hepta.hia.program.v8',
            'module': module, 'functions': functions}), names


def change(source, old, new, count=1):
    assert source.count(old) == count, (old, source.count(old), count)
    return source.replace(old, new)


def function_source(source, name):
    node = next(node for node in ast.parse(source).body if node.name == name)
    return ast.get_source_segment(source, node)


def replace_function(source, name, replacement):
    previous = function_source(source, name)
    return change(source, previous, replacement.rstrip())


def write_source(directory, source, doc, names, per_unit=True):
    directory.mkdir()
    raw = render_assembly(doc).encode()
    assert assemble_program(raw).as_dict() == doc
    count = {'functions': len(doc['functions']), 'blocks': sum(len(fn['blocks']) for fn in doc['functions']),
             'instructions': sum(len(block['ops']) for fn in doc['functions'] for block in fn['blocks']),
             'source_bytes': len(raw), 'maximum_block_parameters': max(len(block['params']) for fn in doc['functions'] for block in fn['blocks'])}
    assert count['functions'] <= 256 and count['blocks'] <= 2560 and len(raw) <= 1048576
    if per_unit:
        assert count['instructions'] <= 8192
        assert count['maximum_block_parameters'] <= 16, count
    (directory / 'authoring.txt').write_text(source)
    (directory / 'source.hepta').write_bytes(raw)
    (directory / 'functions.json').write_text(json.dumps(names, indent=2) + '\n')
    return {'source_sha256': sha(raw), 'authoring_sha256': sha(source.encode()), 'structure': count}


base_compiler = (SOURCES / 'compiler/compiler-authoring.txt').read_text()
base_doc, _ = compiler_document(base_compiler)
assert sha(render_assembly(base_doc).encode()) == owned['module_compiler_source_sha256']
source = base_compiler
finish = function_source(source, 'exports_finish')
finish = change(finish, '    while i < state_get32(w, 64):\n        if get(w, 4096 + i * 320 + 272) == 1:',
                '    while i < state_get32(w, 64):\n        put(w, 4096 + i * 320 + 288, 0)\n        if get(w, 4096 + i * 320 + 272) == 1:')
names = b'\0.text\0.symtab\0.strtab\0.shstrtab\0.note.GNU-stack\0.rela.text\0.hia.interface\0.hia.stack\0'
assert len(names) == 86 and names.index(b'.hia.stack') == 75
begin = finish.index('    names = state_get32(w, 56)\n') + len('    names = state_get32(w, 56)\n')
end = finish.index('    align8(o, w)\n    relocs =', begin)
finish = finish[:begin] + ''.join(f'    emit_bytes(o, w, {int.from_bytes(names[i:i+8], "little")}, {len(names[i:i+8])})\n'
                                for i in range(0, len(names), 8)) + finish[end:]
finish = change(finish, '    emit8(o, w, 3544414555398555976)\n    emit8(o, w, 2)',
                '    emit8(o, w, 3544414555398555976)\n    emit8(o, w, 3)')
finish = change(finish, '    headers = state_get32(w, 56)\n',
                '    stack = state_get32(w, 56)\n    stack_metadata(o, w)\n    headers = state_get32(w, 56)\n')
finish = change(finish, 'write4(o, 60, 262152)', 'write4(o, 60, 262153)')
finish = change(finish, 'names, 75,', 'names, 86,')
finish = change(finish, 'names + 75,', 'names + 86,')
finish = change(finish, '    return 0',
                '    return 0', count=2)
end_section = '    elf_section(o, w, 4294967356, 0, metadata, 64 + (count + imports) * 256, 0, 8, 0)\n'
finish = change(finish, end_section, end_section + '    elf_section(o, w, 4294967371, 0, stack, headers - stack, 0, 8, 0)\n')
finish = change(finish, 'module_interface(o, w, 4096 + get(w, record + 24) * 320, 0)',
                'module_interface(o, w, 4096 + get(w, record + 24) * 320, 0, get(w, record + 24))')
finish = change(finish, 'module_interface(o, w, fb, 1)', 'module_interface(o, w, fb, 1, i)')
source = replace_function(source, 'exports_finish', finish)
interface = function_source(source, 'module_interface')
interface = change(interface, 'imported: u64) -> u64:', 'imported: u64, ordinal: u64) -> u64:')
interface = change(interface, '    i = 0\n    while i < 8:\n',
                   '    emit8(o, w, ordinal)\n    i = 0\n    while i < 7:\n')
source = replace_function(source, 'module_interface', interface)
source += '\n' + (SPEC / 'compiler_additions.txt').read_text().replace('STACK_MAGIC', str(int.from_bytes(b'HIASTK03', 'little')))
doc, fns = compiler_document(source)
compiler_result = write_source(OUT / 'compiler', source, doc, fns, per_unit=False)
preserved = []
for node in ast.parse(base_compiler).body:
    if node.name not in ('exports_finish', 'module_interface'):
        before = ast.dump(node, include_attributes=False)
        after = ast.dump(ast.parse(function_source(source, node.name)).body[0], include_attributes=False)
        assert before == after, node.name
        preserved.append({'name': node.name, 'normalized_ast_sha256': sha(before.encode())})
compiler_result['preserved_functions'] = preserved
compiler_result['ordinary_codegen_and_all_owner_checks_unchanged'] = True

allocator = json.loads((SOURCES / 'compiler/compiler-adaptation.json').read_text())['allocator_helpers']
allocator_bytes = {name: bytes.fromhex(allocator[name]['hex']) for name in ('calloc', 'free')}
assert sha(allocator_bytes['calloc']) == '0d261348d23b4c9e27ffb4b3f5c417288ef75d3cbc6bc8b050987b40a632dfcc'
assert sha(allocator_bytes['free']) == '04c281c9c08baa1b6dccb4578984ade361ff81e78ed3dd2e38ccb03daa4e2550'
drop, relocations = FRAGMENTS['drop']
drop = bytearray(drop)
for relocation in relocations:
    offset = relocation['offset']
    assert relocation['symbol'] == 'free' and relocation['type'] == 4 and relocation['addend'] == -4
    assert drop[offset-1:offset+4] == b'\xe8\0\0\0\0'
    drop[offset:offset+4] = (len(drop) + len(allocator_bytes['calloc']) - offset - 4).to_bytes(4, 'little', signed=True)
drop = bytes(drop)
helpers = ['def stack_helper_bytes(s:readbuf, text:u64, size:u64, end:u64, flags:u64) -> u64:',
           '    expected = end', '    if flags == 3:', f'        expected = expected + {len(drop)}',
           '    if flags >= 2:', f'        expected = expected + {sum(map(len, allocator_bytes.values()))}',
           '    if expected != size:', '        return 0', '    at = text + end', '    if flags == 3:']
for offset in range(0, len(drop), 8):
    part = drop[offset:offset+8]
    helpers += [f'        if uint(s, at + {offset}, {len(part)}) != {int.from_bytes(part,"little")}:', '            return 0']
helpers += [f'        at = at + {len(drop)}', '    if flags >= 2:']
combined = allocator_bytes['calloc'] + allocator_bytes['free']
for offset in range(0, len(combined), 8):
    part = combined[offset:offset+8]
    helpers += [f'        if uint(s, at + {offset}, {len(part)}) != {int.from_bytes(part,"little")}:', '            return 0']
helpers += ['    return 1']
helper_source = '\n'.join(helpers) + '\n'
machine_source = (SPEC/'machine_additions.txt').read_text().replace('CHAIN_DROP_BYTES', str(len(drop))).replace('CALLOC_BYTES', str(len(allocator_bytes['calloc'])))

base_session = (args.session_build.resolve() / 'production-source/linker-authoring.txt').read_text()
doc, _ = linker_document(base_session, 'owned_session_linker')
assert sha(render_assembly(doc).encode()) == session['production']['source_sha256']
base_filter = (SOURCES / 'linker-abi2-w1m-source/linker-authoring.txt').read_text()
doc, _ = linker_document(base_filter, 'borrowed_module_linker')
assert sha(render_assembly(doc).encode()) == owned['linker_profiles']['abi2-w1m']['source_sha256']
results = {}
variants = [('session', 8388608, 0, b'HIASTKS1'), ('compiler', 8388608, 2097152, b'HIASTKC1'),
            ('linker', 8388608, 16777216, b'HIASTKL1'),
            ('runtime', 8388608, 1048576, b'HIASTKR1'), ('qualification', 65536, 0, b'HIASTKQ1'),
            ('qualification-runtime', 65536, 1048576, b'HIASTKQ2'),
            ('qualification-cleanup', 8388608, 0, b'HIASTKT1'),
            ('qualification-guard-low', 8388608, 0, b'HIASTKT2'),
            ('qualification-guard-high', 8388608, 0, b'HIASTKT3')]
for label, stack_bytes, workspace, magic in variants:
    source = base_filter if workspace else base_session
    old_magic = b'HIAOWN02' if workspace else b'HIAOWNS1'
    source = change(source, str(int.from_bytes(old_magic, 'little')), str(int.from_bytes(magic, 'little')))
    if workspace:
        source = change(source, '    put(w, 40, 1024)', '    put(w, 40, 4096)')
    admission = '    if module_contracts(s, w) == 0:\n        return 4294967295\n'
    source = change(source, admission, admission + '    if stack_contracts(s, w) == 0:\n        return 4294967295\n')
    layout = function_source(source, 'module_layout')
    layout = change(layout, 'record + 24) != 8', 'record + 24) != 9')
    section = '    if module_section(s, table + 448, 60, 1, 0, 8, 0) == 0:\n        return 0\n'
    layout = change(layout, section, section + '    if module_section(s, table + 512, 75, 1, 0, 8, 0) == 0:\n        return 0\n')
    layout = change(layout, '    while i < 8:', '    while i < 9:')
    layout = change(layout, '!= 75:', '!= 86:')
    layout = layout.replace('names + 75', 'names + 86')
    old_tail = '    if metadata_bytes < 64 or table != pos + metadata + metadata_bytes or get(w, record + 8) != metadata + metadata_bytes + 512:\n        return 0\n'
    new_tail = ('    stack = uint(s, table + 512 + 24, 8)\n'
                '    stack_bytes = uint(s, table + 512 + 32, 8)\n'
                '    if metadata_bytes < 64 or stack_bytes < 64 or stack != metadata + metadata_bytes or table != pos + stack + stack_bytes or get(w, record + 8) != stack + stack_bytes + 576:\n'
                '        return 0\n')
    layout = change(layout, old_tail, new_tail)
    layout = change(layout, 'uint(s, at + 8, 8) != 2', 'uint(s, at + 8, 8) != 3')
    layout = change(layout, '        offset = uint(s, relocation, 8)\n',
                    '        offset = uint(s, relocation, 8)\n        if i != 0 and offset <= uint(s, relocation - 24, 8):\n            return 0\n')
    source = replace_function(source, 'module_layout', layout)
    name_check = ['def module_names(s:readbuf, at:u64) -> u64:']
    for offset in range(0, len(names), 8):
        part = names[offset:offset+8]
        name_check += [f'    if uint(s, at + {offset}, {len(part)}) != {int.from_bytes(part, "little")}:', '        return 0']
    name_check += ['    return 1']
    source = replace_function(source, 'module_names', '\n'.join(name_check))
    signature = function_source(source, 'module_signature_valid')
    signature = change(signature, '    i = 192\n', '    i = 200\n')
    source = replace_function(source, 'module_signature_valid', signature)
    symbols = function_source(source, 'module_symbols')
    check = '        if module_signature_valid(s, contract) == 0:\n            return 0\n'
    symbols = change(symbols, check, check + '        if uint(s, contract + 192, 8) >= uint(s, metadata + 24, 8):\n            return 0\n')
    source = replace_function(source, 'module_symbols', symbols)
    if workspace:
        io_bytes = 8388608 if workspace == 16777216 else 1048576
        prefix, hole, startup, authoring = filter_prefix(filter_generator, stack_bytes, workspace, io_bytes)
        holes = [('entry', hole, 88)]
    else:
        observation = label.removeprefix('qualification-') if label in ('qualification-cleanup','qualification-guard-low','qualification-guard-high') else None
        prefix, step, close, startup, authoring = session_prefix(args.session_template, stack_bytes, observation)
        holes = [('step', step, 88), ('close', close, 96)]
    header = ['def headers(o:mutbuf, w:mutbuf) -> u64:']
    for offset in range(0, len(prefix), 8):
        part = prefix[offset:offset+8]
        header += [f'    write(o, {offset}, {int.from_bytes(part, "little")}, {len(part)})']
    header += ['    put(o, 96, get(w, 64))', '    put(o, 104, get(w, 64))']
    for _, hole, field in holes:
        header += [f'    write(o, {hole}, get(w, {field}) - {hole+4}, 4)']
    if not workspace:
        header += [f'    put(o, {startup["stack"]["budget_hole"]}, get(w, 104) + 4096)']
    header += ['    return 0']
    source = replace_function(source, 'headers', '\n'.join(header))
    source += '\n' + (SPEC / 'linker_additions.txt').read_text().replace('STACK_MAGIC', str(int.from_bytes(b'HIASTK03', 'little'))).replace('STACK_LIMIT', str(stack_bytes-4096))
    source += '\n' + machine_source + '\n' + helper_source
    doc, fns = linker_document(source, 'stack_budget_linker')
    directory = OUT / ('linker-' + label)
    result = write_source(directory, source, doc, fns)
    (directory / 'startup-prefix.bin').write_bytes(prefix)
    (directory / 'startup-authoring.txt').write_text(authoring)
    (directory / 'startup.json').write_text(json.dumps(startup, indent=2) + '\n')
    result.update(request_magic=magic.decode(), stack_bytes=stack_bytes, stack_budget=stack_bytes-4096,
                  workspace_bytes=workspace, startup_prefix_sha256=sha(prefix),
                  startup_authoring_sha256=sha(authoring.encode()), call_holes=holes,
                  qualification_only=label.startswith('qualification'))
    results[label] = result
for relative, expected in closure.items():
    assert sha((UP / relative).read_bytes()) == expected, relative
for relative, expected in owned['generated_source_sha256'].items():
    assert sha((SOURCES / relative).read_bytes()) == expected, relative
assert args.owned_profile.read_bytes() == owned_raw and args.session_profile.read_bytes() == session_raw
for path, snapshot in own_snapshots.items():
    assert path.read_bytes() == snapshot, str(path)
loaded = {}
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if filename:
        path = Path(filename).resolve()
        if path.is_relative_to(UP):
            relative = path.relative_to(UP).as_posix()
            assert closure.get(relative) == sha(path.read_bytes()), relative
            loaded[relative] = closure[relative]
report = {'schema':'heptatorch.stack-budget-adaptation.v1','profile':'heptatorch.hia.stack-budget-modules.v1',
          'interface_abi':3,'compiler':compiler_result,'linkers':results,
          'upstream_commit':owned['upstream_commit'],'upstream_source_files_verified_before_after':closure,
          'upstream_files_loaded':loaded,'authoring_sources_sha256':{path.name:sha(path.read_bytes()) for path in
            [Path(__file__),SPEC/'startup.py',SPEC/'compiler_additions.txt',SPEC/'linker_additions.txt',SPEC/'machine_additions.txt']},
          'machine_call_binding':{'sequential_instruction_boundary_decoder':True,'exact_reconstructed_matrix':True,
            'relocations_must_be_sorted_unique_and_at_decoded_calls':True,'private_branches_stay_at_in_function_instruction_boundaries':True,
            'closed_chain_drop_hex':drop.hex(),'closed_chain_drop_sha256':sha(drop),
            'closed_allocator_helpers':allocator,'maximum_object_text_bytes':1048576,
            'scratch_end':13394016},
          'owned_profile_sha256':sha(owned_raw),'session_profile_sha256':sha(session_raw),
          'native_execution_performed':False,'all_per_unit_limits_unchanged':True,
          'trust_boundary':'Pinned compiler production and byte custody; not authentication of foreign machine code.'}
(OUT / 'adaptation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'compiler':{k:v for k,v in compiler_result.items() if k!='preserved_functions'},'linkers':results},indent=2))
