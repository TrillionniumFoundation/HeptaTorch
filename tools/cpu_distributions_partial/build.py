#!/usr/bin/env python3
"""Author native HIA distributions over the frozen seven-kind owned runtime.

Python emits source and orchestrates actual native compilation/linking only.
Tensor and random-number computations execute in the resulting native ELF.
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
BASE = 'a72a98232c7b7f10efa67ce29b21e417c58201383bef9aacc266fe247ff356ce'
RNG = 'a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3'
OWNERS = '9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884'
COMMON_AUTHORING = '5f6578c483d97fdd75655aaff82176b44d5ab92f53ad47f1fcb6ac6a6d6e3438'
GRADIENT_AUTHORING = '5f6578c483d97fdd75655aaff82176b44d5ab92f53ad47f1fcb6ac6a6d6e3438'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
CONTRACT = '8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c'
FUNCTION = re.compile(rb'^(fn (\w+)\([^\n]* entry\(\w+\)\n.*?^end\n)', re.M | re.S)
CALL = re.compile(rb'\{"function":"(\w+)"\}')
EXTERNAL_RNG = ['rng_seed', 'rng_restore', 'rng_next', 'rng_next64', 'rng_unit32', 'rng_unit64']


def functions(raw):
    matches = FUNCTION.findall(raw)
    result = {name.decode(): body for body, name in matches}
    assert len(result) == len(matches)
    return result


def literal(tag):
    return b'{"value":' + str(int.from_bytes(tag, 'little')).encode() + b'}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'common-authoring', 'gradient-authoring', 'upstream', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base, up = a.baseline.resolve(), a.upstream.resolve()
    compiler = a.stack_build.resolve() / 'native/stack-module-compiler'
    linker = a.stack_build.resolve() / 'native/stack-linker-session'
    pins = {base / 'runtime.hepta': BASE, base / 'rng03.hepta': RNG, base / 'owners.hepta': OWNERS,
            a.common_authoring.resolve(): COMMON_AUTHORING, a.gradient_authoring.resolve(): GRADIENT_AUTHORING,
            compiler: COMPILER, linker: LINKER, up / 'docs/HIA_SELFHOST_CORE_SUPPORT.json': CONTRACT}
    inputs = {p: p.read_bytes() for p in pins}
    for path, pin in pins.items():
        assert SHA(inputs[path]) == pin, (path, SHA(inputs[path]), pin)
    inputs[Path(__file__).resolve()] = Path(__file__).read_bytes()
    inputs[HERE / 'partial.authoring'] = (HERE / 'partial.authoring').read_bytes()
    inputs[HERE / 'compact_functions.py'] = (HERE / 'compact_functions.py').read_bytes()
    baseline = functions(inputs[base / 'runtime.hepta'])
    rng = functions(inputs[base / 'rng03.hepta'])
    assert len(baseline) == 294 and len(rng) == 17
    combined = dict(baseline)
    source_contract = json.loads(inputs[up / 'docs/HIA_SELFHOST_CORE_SUPPORT.json'])['source_identity']['files']
    verified = {}
    for name, pin in source_contract.items():
        if name.startswith(('hepta/', 'tools/')) and (up / name).is_file():
            verified[name] = SHA((up / name).read_bytes())
            assert verified[name] == pin, name
    sys.path.insert(0, str(up))
    from tools.generate_hia_selfhost import Builder, compact_authoring_cfg, total_scalar_functions
    from hepta.hia_agent_assembly import _parse_assembly_document, _render_assembly_document
    spec = importlib.util.spec_from_file_location('distribution_body_compactor', HERE / 'compact_functions.py')
    compactor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compactor)

    class NativeBuilder(Builder):
        def expr(self, node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                x, y = self.expr(node.left), self.expr(node.right)
                op = 'int.div.checked' if isinstance(node.op, ast.FloorDiv) else 'int.rem.checked'
                return self.op(op, [x, y], x[1])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name in ('copy_range', 'copy_from_mut'):
                    target, source, at, old, size = [self.expr(x) for x in node.args]
                    loan = None
                    if name == 'copy_from_mut':
                        loan = self.op('borrow.read', [source], 'readbuf')
                        source = loan
                    self.op('buffer.copy_range', [target, source, self.cast(at, 'usize'), self.cast(old, 'usize'), self.cast(size, 'usize')])
                    if loan is not None:
                        self.op('loan.end', [loan])
                    return self.c(0)
                if name == 'move_within':
                    w, dest, src, size = [self.expr(x) for x in node.args]
                    self.op('buffer.move_within', [w, self.cast(dest, 'usize'), self.cast(src, 'usize'), self.cast(size, 'usize')])
                    return self.c(0)
            return super().expr(node)

    common_nodes = {n.name: n for n in ast.parse(inputs[a.common_authoring.resolve()]).body}
    gradient_nodes = {n.name: n for n in ast.parse(inputs[a.gradient_authoring.resolve()]).body}
    original_authoring = '\n\n'.join(ast.unparse(n) for n in (
        common_nodes['entry'], gradient_nodes['gradient_direct'], gradient_nodes['partial_apply_node']))
    entry = ast.unparse(common_nodes['entry'])
    needle = '            next = dist_command(s, o, w, scratch, 32 + i * 64, at)'
    assert entry.count(needle) == 1
    entry = entry.replace(needle, '''            if code == 86 or code == 87:
                next = dist_partial_command(s,o,w,scratch,32+i*64,at)
            else:
                next = dist_command(s,o,w,scratch,32+i*64,at)''')
    needle = 'if getr(s, 32 + i * 64) == 58 and getw(o, at + 16) != 0:'
    assert entry.count(needle) == 1
    entry = entry.replace(needle, 'if (code == 58 or code == 86 or code == 87) and getw(o, at + 16) != 0:')
    authoring = entry + '\n\n' + inputs[HERE / 'partial.authoring'].decode()
    (out / 'overlay-authoring.py').write_text(authoring)
    signature = {}
    for name, body in {**baseline, **rng}.items():
        match = re.match(rb'fn \w+\(([^)]*)\) -> \(([^)]*)\)', body)
        signature[name] = ([x.split(b':')[1].decode() for x in match[1].split(b',') if x], match[2].decode())
    original_context = []
    for body in list(baseline.values()) + [body for name, body in rng.items() if name not in baseline]:
        original_context += _parse_assembly_document(b'hia-asm.v1 hepta.hia.program.v8 dist_context\n' + body)['functions']

    compaction_receipts = []

    def emit(source, bounded=False):
        parsed = ast.parse(source).body
        local_signature = dict(signature)
        local_signature.update({n.name: ([x.annotation.id for x in n.args.args], n.returns.id) for n in parsed})
        names = {name: name for name in local_signature}
        total = total_scalar_functions(parsed)
        emitted = []
        for node in parsed:
            builder = NativeBuilder(node, names, local_signature, total)
            builder.statements(node.body)
            emitted.append({'name': node.name, 'params': [builder.param(v) for v in builder.params],
                            'returns': [builder.returns], 'effects': ['read', 'write', 'trap'],
                            'entry': 'b0', 'blocks': builder.blocks})
        changed = {n.name for n in parsed}
        context = {'schema': 'hepta.hia.program.v8', 'module': 'dist_context',
                   'functions': [f for f in original_context if f['name'] not in changed] + emitted}
        if bounded:
            document = dict(context)
            document['functions'], records = compactor.compact(context['functions'], changed)
            compaction_receipts.append(records)
        else:
            document = compact_authoring_cfg(context)
            document['functions'] = [f for f in document['functions'] if f['name'] in changed]
        return _render_assembly_document(document).encode()

    original_emitted = functions(emit(original_authoring, bounded=True))
    for name, body in original_emitted.items():
        assert body == baseline[name], ('authoring is not frozen baseline body', name)
    fragment = emit(authoring, bounded=True)
    (out / 'body-compaction.json').write_text(json.dumps({
        'boundary': 'individual actual bodies checked against interfaces from actual source; no aggregate Core verification claim',
        'baseline_matches_frozen_source': True, 'aggregate_core_check': False, 'groups': compaction_receipts}, indent=2) + '\n')
    additions = functions(fragment)
    combined.update(additions)
    tag_functions = ('init_state', 'state_header', 'so_state_init', 'so_state_header')
    for name in tag_functions:
        old = b'HTDIST01' if name in tag_functions[:2] else b'HTODIST1'
        new = b'HTDIST02' if name in tag_functions[:2] else b'HTODIST2'
        assert combined[name].count(literal(old)) == 1
        combined[name] = combined[name].replace(literal(old), literal(new))
    assert {name for name in baseline if combined[name] != baseline[name]} == set(tag_functions) | {'entry'}
    assert all(combined[name] == baseline[name] for name in baseline if name.startswith('dist_'))
    (out / 'overlay.hepta').write_bytes(fragment)
    raw = b'hia-asm.v1 hepta.hia.program.v8 tensor_distributions_partial_runtime\n' + b''.join(combined.values())
    (out / 'runtime.hepta').write_bytes(raw)
    active, done, ordered = set(), set(), []

    def visit(name):
        if name in EXTERNAL_RNG:
            return
        assert name in combined, name
        if name in done:
            return
        assert name not in active, ('call cycle', name)
        active.add(name)
        for target in sorted({s.decode() for s in CALL.findall(combined[name])}):
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
    units = [('rng03', inputs[base / 'rng03.hepta'])]
    for index, group in enumerate(groups):
        imports = sorted({s.decode() for name in group for s in CALL.findall(combined[name])} - set(group))
        assert len(group) + len(imports) <= 256
        name = 'distributions_' + str(index)
        source = ('hia-asm.v1 hepta.hia.program.v8 ' + name + '\nexports(' + ','.join(group) + ')\n').encode()
        for target in imports:
            body = rng[target] if target in EXTERNAL_RNG else combined[target]
            source += re.sub(rb' entry\(\w+\)$', b' external()', body.split(b'\n', 1)[0]) + b'\n'
        source += b''.join(combined[target] for target in group)
        units.append((name, source))
    units.append(('owners', inputs[base / 'owners.hepta']))
    rows, objects = [], []
    report = {
        'schema': 'heptatorch.cpu-distributions-partial-native-build.v1', 'status': 'BUILDING',
        'profile': 'heptatorch.cpu-distributions.partial-errors.f32.seven-kind.abi3.v1',
        'baseline_source_sha256': BASE, 'runtime_source_sha256': SHA(raw),
        'compiler_sha256': COMPILER, 'linker_sha256': LINKER,
        'core_functions': len(combined), 'core_blocks': len(re.findall(rb'^block ', raw, re.M)),
        'core_instructions': raw.count(b' = '), 'units': rows,
        'replaced_functions': sorted(set(additions) & set(baseline)),
        'added_functions': sorted(set(additions) - set(baseline)),
        'tag_functions': list(tag_functions), 'old_authoring_reproduced_exactly': True,
        'host_check_boundary': 'each changed actual function before and after compaction; final modules admitted by native compiler and ABI3 linker',
        'aggregate_core_host_check': False,
        'base_function_body_map': {name: {'before': SHA(body), 'after': SHA(combined[name]), 'unchanged': body == combined[name]}
                                   for name, body in baseline.items()},
        'rng_module_byte_identical': True, 'owner_module_byte_identical': True,
        'all_predecessor_dist_functions_byte_identical': True, 'old_atomic_commands_80_85_unchanged': True,
        'input_sha256': {str(p): SHA(b) for p, b in inputs.items()},
        'upstream_verified_source_files': verified,
        'host_production_numerical_runtime': False, 'all_rng_complete': False,
        'full_pytorch_rewrite_complete': False,
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
    executable = out / 'tensor-distributions-session'
    executable.write_bytes(result.stdout)
    report['link_exit'] = result.returncode
    save()
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    executable.chmod(0o500)
    for path, data in inputs.items():
        assert path.read_bytes() == data, path
    loaded = {}
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            path = Path(file).resolve()
            if path.is_relative_to(up):
                relative = str(path.relative_to(up))
                assert relative in verified and SHA(path.read_bytes()) == verified[relative], relative
                loaded[relative] = verified[relative]
    report.update(status='PASS', executable_sha256=SHA(result.stdout), executable_bytes=len(result.stdout),
                  actually_loaded_upstream_sources=loaded, input_custody_after=True)
    save()
    print(json.dumps({k: report[k] for k in ('status', 'runtime_source_sha256', 'executable_sha256',
                                           'executable_bytes', 'core_functions', 'core_blocks', 'core_instructions')}, indent=2))


if __name__ == '__main__':
    main()
