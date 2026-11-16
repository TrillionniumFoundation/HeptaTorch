#!/usr/bin/env python3
"""Compose frozen owned Adam02 and CPU distributions03/partial01 in native HIA.

This host program only verifies source custody, emits HIA, partitions actual
function bodies, and invokes the pinned native compiler and native linker.
It is not a Tensor, Generator, optimizer, or distribution execution engine.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SHA = lambda raw: hashlib.sha256(raw).hexdigest()
ADAM = '253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6'
ADAM_AUTHORING = 'abe546944d7be6c1c9728ea67ac6ccb2a2cd8286a30989d170bba3e6ef92bcc3'
DISTRIBUTIONS = '55ed0c6e16317add03974c01ffd85e38684fc535a3e5cb6423ed043f4217f05b'
RNG = 'a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3'
OWNERS = '9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
CONTRACT = '8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c'
COMPACTOR = '36730b55b58c29b210d921a426c62eab50d9f579ad5d7a1e4199b38799218f37'
UPSTREAM_PINS = 'b83c71fdde9df73e18d233b4cb5a2edc5468cbedce06e6b0d3145d01c761bb90'
FUNCTION = re.compile(rb'^(fn (\w+)\([^\n]* entry\(\w+\)\n.*?^end\n)', re.M | re.S)
CALL = re.compile(rb'\{"function":"(\w+)"\}')
EXTERNAL_RNG = {'rng_seed', 'rng_restore', 'rng_next', 'rng_next64', 'rng_unit32', 'rng_unit64'}
TAG_FUNCTIONS = ('init_state', 'state_header', 'so_state_init', 'so_state_header')
NEW_RUNTIME_TAG = b'HTADIST1'
NEW_OBJECT_TAG = b'HTOADIS1'


def functions(raw):
    matches = FUNCTION.findall(raw)
    result = {name.decode(): body for body, name in matches}
    assert len(result) == len(matches)
    return result


def literal(tag):
    assert len(tag) == 8
    return b'{"value":' + str(int.from_bytes(tag, 'little')).encode() + b'}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('adam-build', 'distribution-build', 'upstream', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    adam_dir = args.adam_build.resolve()
    dist_dir = args.distribution_build.resolve()
    up = args.upstream.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    compiler = args.stack_build.resolve() / 'native/stack-module-compiler'
    linker = args.stack_build.resolve() / 'native/stack-linker-session'
    pins = {
        adam_dir / 'runtime.hepta': ADAM,
        adam_dir / 'overlay-authoring.py': ADAM_AUTHORING,
        adam_dir / 'rng03.hepta': RNG,
        adam_dir / 'owners.hepta': OWNERS,
        dist_dir / 'runtime.hepta': DISTRIBUTIONS,
        dist_dir / 'rng03.hepta': RNG,
        dist_dir / 'owners.hepta': OWNERS,
        compiler: COMPILER,
        linker: LINKER,
        up / 'docs/HIA_SELFHOST_CORE_SUPPORT.json': CONTRACT,
        HERE / 'compact_functions.py': COMPACTOR,
        HERE / 'upstream-pins.json': UPSTREAM_PINS,
    }
    inputs = {path: path.read_bytes() for path in pins}
    for path, expected in pins.items():
        assert SHA(inputs[path]) == expected, (str(path), SHA(inputs[path]), expected)
    inputs[Path(__file__).resolve()] = Path(__file__).read_bytes()

    # Verify the exact imported source closure before any upstream code executes.
    upstream_pins = json.loads(inputs[HERE / 'upstream-pins.json'])
    assert len(upstream_pins) == 30
    support_files = json.loads(inputs[up / 'docs/HIA_SELFHOST_CORE_SUPPORT.json'])['source_identity']['files']
    for name, expected in upstream_pins.items():
        assert support_files[name] == expected
        assert SHA((up / name).read_bytes()) == expected, name
    sys.path.insert(0, str(up))
    from tools.generate_hia_selfhost import Builder, total_scalar_functions
    from hepta.hia_agent_assembly import _parse_assembly_document, _render_assembly_document
    spec = importlib.util.spec_from_file_location('composition_body_compactor', HERE / 'compact_functions.py')
    compactor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compactor)

    adam = functions(inputs[adam_dir / 'runtime.hepta'])
    dist = functions(inputs[dist_dir / 'runtime.hepta'])
    rng = functions(inputs[adam_dir / 'rng03.hepta'])
    assert len(adam) == 352 and len(dist) == 300 and len(rng) == 17
    added_dist = {name: body for name, body in dist.items() if name.startswith('dist_')}
    assert len(added_dist) == 25 and not (set(added_dist) & set(adam))
    combined = dict(adam)
    combined.update(added_dist)
    combined['gradient_direct'] = dist['gradient_direct']

    # Retain the complete Adam dispatcher as a real helper, changing only its
    # declaration name. All old opcode branches and leaf accumulation are exact.
    old_partial = adam['partial_apply_node']
    assert b'{"function":"partial_apply_node"}' not in old_partial
    alias_name = 'partial_apply_node_adam'
    alias = old_partial.replace(b'fn partial_apply_node(', b'fn partial_apply_node_adam(', 1)
    assert alias.replace(b'fn partial_apply_node_adam(', b'fn partial_apply_node(', 1) == old_partial
    combined[alias_name] = alias

    adam_nodes = {node.name: node for node in ast.parse(inputs[adam_dir / 'overlay-authoring.py']).body
                  if isinstance(node, ast.FunctionDef)}
    original_entry = ast.unparse(adam_nodes['entry'])
    entry = original_entry
    needle = '        elif code >= 60 and code <= 69:'
    assert entry.count(needle) == 1
    entry = entry.replace(needle, '''        elif code >= 80 and code <= 89:
            if code == 86 or code == 87:
                next = dist_partial_command(s, o, w, scratch, 32 + i * 64, at)
            else:
                next = dist_command(s, o, w, scratch, 32 + i * 64, at)
        elif code >= 60 and code <= 69:''')
    needle = 'if getr(s, 32 + i * 64) == 58 and getw(o, at + 16) != 0:'
    assert entry.count(needle) == 1
    entry = entry.replace(needle, 'if (code == 58 or code == 86 or code == 87) and getw(o, at + 16) != 0:')
    wrapper = '''def partial_apply_node(w: mutbuf, t: u64) -> u64:
    if getw(w, dptr(w, t) + 40) == 41:
        dist_backward_bernoulli(w, t)
        if getw(w, 0) != 0:
            return 0
        return 1
    return partial_apply_node_adam(w, t)
'''
    authoring = entry + '\n\n' + wrapper
    (out / 'overlay-authoring.py').write_text(authoring)
    signatures = {}
    context = []
    for name, body in {**combined, **rng}.items():
        match = re.match(rb'fn \w+\(([^)]*)\) -> \(([^)]*)\)', body)
        signatures[name] = ([x.split(b':')[1].decode() for x in match[1].split(b',') if x], match[2].decode())
        context.extend(_parse_assembly_document(b'hia-asm.v1 hepta.hia.program.v8 composition_context\n' + body)['functions'])
    compaction_records = []

    def emit(source):
        nodes = ast.parse(source).body
        interfaces = dict(signatures)
        interfaces.update({node.name: ([arg.annotation.id for arg in node.args.args], node.returns.id) for node in nodes})
        names = {name: name for name in interfaces}
        total = total_scalar_functions(nodes)
        emitted = []
        for node in nodes:
            builder = Builder(node, names, interfaces, total)
            builder.statements(node.body)
            emitted.append({'name': node.name, 'params': [builder.param(v) for v in builder.params],
                            'returns': [builder.returns], 'effects': ['read', 'write', 'trap'],
                            'entry': 'b0', 'blocks': builder.blocks})
        selected = {node.name for node in nodes}
        complete = [function for function in context if function['name'] not in selected] + emitted
        changed, records = compactor.compact(complete, selected)
        compaction_records.append(records)
        return _render_assembly_document({'schema': 'hepta.hia.program.v8', 'module': 'composition_overlay',
                                          'functions': changed}).encode()

    original = functions(emit(original_entry))
    assert original == {'entry': adam['entry']}, 'the pinned Adam entry authoring must reproduce its exact native source'
    fragment = emit(authoring)
    emitted = functions(fragment)
    assert set(emitted) == {'entry', 'partial_apply_node'}
    combined.update(emitted)
    for name in TAG_FUNCTIONS:
        old_tag = b'HTADAM01' if name in TAG_FUNCTIONS[:2] else b'HTOADAM1'
        new_tag = NEW_RUNTIME_TAG if name in TAG_FUNCTIONS[:2] else NEW_OBJECT_TAG
        assert combined[name].count(literal(old_tag)) == 1
        combined[name] = combined[name].replace(literal(old_tag), literal(new_tag))
    changed_names = {name for name in adam if adam[name] != combined[name]}
    assert changed_names == set(TAG_FUNCTIONS) | {'entry', 'gradient_direct', 'partial_apply_node'}
    assert len(combined) == 378
    assert all(combined[name] == body for name, body in added_dist.items())
    (out / 'overlay.hepta').write_bytes(fragment)
    raw = b'hia-asm.v1 hepta.hia.program.v8 tensor_adam_distributions_runtime\n' + b''.join(combined.values())
    (out / 'runtime.hepta').write_bytes(raw)
    (out / 'body-compaction.json').write_text(json.dumps({
        'boundary': 'selected actual bodies checked against actual source interfaces before and after compaction',
        'aggregate_core_check': False, 'old_adam_entry_reproduced_exactly': True,
        'groups': compaction_records}, sort_keys=True, indent=2) + '\n')

    done, active, ordered = set(), set(), []

    def visit(name):
        if name in EXTERNAL_RNG:
            return
        assert name in combined, name
        if name in done:
            return
        assert name not in active, ('unexpected function cycle', name)
        active.add(name)
        for target in sorted({value.decode() for value in CALL.findall(combined[name])}):
            visit(target)
        active.remove(name)
        done.add(name)
        ordered.append(name)

    for name in combined:
        visit(name)
    groups, group, instructions, blocks = [], [], 0, 0
    for name in ordered:
        body = combined[name]
        ni = body.count(b' = ')
        nb = len(re.findall(rb'^block ', body, re.M))
        if group and (instructions + ni > 3600 or blocks + nb > 1000 or len(group) >= 65):
            groups.append(group)
            group, instructions, blocks = [], 0, 0
        group.append(name)
        instructions += ni
        blocks += nb
    if group:
        groups.append(group)
    units = [('rng03', inputs[adam_dir / 'rng03.hepta'])]
    partition = []
    for index, group in enumerate(groups):
        imports = sorted({value.decode() for name in group for value in CALL.findall(combined[name])} - set(group))
        assert len(group) + len(imports) <= 256
        name = 'adam_distributions_' + str(index)
        source = ('hia-asm.v1 hepta.hia.program.v8 ' + name + '\nexports(' + ','.join(group) + ')\n').encode()
        for target in imports:
            body = rng[target] if target in EXTERNAL_RNG else combined[target]
            source += re.sub(rb' entry\(\w+\)$', b' external()', body.split(b'\n', 1)[0]) + b'\n'
        source += b''.join(combined[target] for target in group)
        units.append((name, source))
        partition.append({'name': name, 'definitions': group, 'imports': imports})
    units.append(('owners', inputs[adam_dir / 'owners.hepta']))
    (out / 'partition.json').write_text(json.dumps(partition, sort_keys=True, indent=2) + '\n')
    rows, objects = [], []
    report = {
        'schema': 'heptatorch.adam-distributions-composition-native-build.v1', 'status': 'BUILDING',
        'profile': 'heptatorch.adam-distributions.f32.eight-kind.abi3.v1',
        'adam_source_sha256': ADAM, 'distribution_source_sha256': DISTRIBUTIONS,
        'runtime_source_sha256': SHA(raw), 'compiler_sha256': COMPILER, 'linker_sha256': LINKER,
        'runtime_state_tag': NEW_RUNTIME_TAG.decode(), 'object_state_tag': NEW_OBJECT_TAG.decode(),
        'core_functions': len(combined), 'core_blocks': len(re.findall(rb'^block ', raw, re.M)),
        'core_instructions': raw.count(b' = '), 'units': rows,
        'replaced_adam_functions': sorted(changed_names),
        'added_functions': sorted(set(combined) - set(adam)),
        'tag_functions': list(TAG_FUNCTIONS), 'old_adam_entry_reproduced_exactly': True,
        'host_check_boundary': 'each changed actual function before and after compaction; final modules admitted by native compiler and ABI3 linker',
        'aggregate_core_host_check': False,
        'adam_function_body_map': {name: {'before': SHA(body), 'after': SHA(combined[name]), 'unchanged': body == combined[name],
                                         **({'exact_renamed_helper': alias_name, 'helper_sha256': SHA(alias)} if name == 'partial_apply_node' else {})}
                                   for name, body in adam.items()},
        'distribution_function_body_map': {name: {'before': SHA(body), 'after': SHA(combined[name]), 'unchanged': body == combined[name]}
                                           for name, body in added_dist.items()},
        'gradient_direct_byte_identical_to_distribution': combined['gradient_direct'] == dist['gradient_direct'],
        'rng_module_byte_identical': True, 'owner_module_byte_identical': True,
        'all_distribution_functions_byte_identical': True,
        'eight_kind_layout_and_optimizer_gc_unchanged': True,
        'graph_capture_and_audit_unchanged': all(combined[name] == adam[name] for name in (
            'record', 'record_body', 'gradient_parent_count', 'graph_parent_at', 'graph_edge_enabled', 'graph_audit')),
        'input_sha256': {str(path): SHA(body) for path, body in inputs.items()},
        'upstream_verified_source_files': upstream_pins,
        'host_production_numerical_runtime': False, 'all_rng_complete': False,
        'all_optimizer_complete': False, 'full_pytorch_rewrite_complete': False,
    }

    def save():
        (out / 'build.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')

    save()
    for name, source in units:
        (out / (name + '.hepta')).write_bytes(source)
        result = subprocess.run([str(compiler)], input=source, capture_output=True, env={'PATH': ''}, timeout=60)
        (out / (name + '.o')).write_bytes(result.stdout)
        (out / (name + '.compile.stderr')).write_bytes(result.stderr)
        row = {'name': name, 'source_sha256': SHA(source), 'source_bytes': len(source),
               'functions': len(re.findall(rb'^fn ', source, re.M)),
               'blocks': len(re.findall(rb'^block ', source, re.M)), 'instructions': source.count(b' = '),
               'compile_exit': result.returncode, 'object_sha256': SHA(result.stdout), 'object_bytes': len(result.stdout)}
        rows.append(row)
        save()
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), row
        objects.append(result.stdout)
    request = b'HIASTKS1' + struct.pack('<3Q', len(objects), 4, 0) + b'step'
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (out / 'link.request').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=60)
    (out / 'link.stderr').write_bytes(result.stderr)
    executable = out / 'tensor-adam-distributions-session'
    executable.write_bytes(result.stdout)
    report['link_exit'] = result.returncode
    save()
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    executable.chmod(0o500)
    for path, raw_input in inputs.items():
        assert path.read_bytes() == raw_input, path
    loaded = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(up):
                relative = str(path.relative_to(up))
                assert relative in upstream_pins and SHA(path.read_bytes()) == upstream_pins[relative], relative
                loaded[relative] = upstream_pins[relative]
    assert loaded == upstream_pins, (sorted(set(loaded) - set(upstream_pins)), sorted(set(upstream_pins) - set(loaded)))
    report.update(status='PASS', executable_sha256=SHA(result.stdout), executable_bytes=len(result.stdout),
                  actually_loaded_upstream_sources=loaded, input_custody_after=True)
    save()
    print(json.dumps({key: report[key] for key in ('status', 'runtime_source_sha256', 'executable_sha256',
                                                 'executable_bytes', 'core_functions', 'core_blocks', 'core_instructions')}, indent=2))


if __name__ == '__main__':
    main()
