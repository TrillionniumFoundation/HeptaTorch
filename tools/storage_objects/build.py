#!/usr/bin/env python3
"""Emit HIA from authoring inputs; compile/link only through pinned native tools.

The host builds source/requests and records bytes. It implements no storage
operation, allocation provider, HIA interpreter, or native linking algorithm.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
COMPILER_SHA = '0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73'
LINKER_SHA = '23eed229e329fb5b7e2116012f74f210d5de4acd2b4f8475e364890748da86ee'
CONTRACT_SHA = '8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c'
AUTHORING_FILES = '''hepta/__init__.py
hepta/artifact_publication.py
hepta/hia_agent_assembly.py
hepta/hia_capability_flow.py
hepta/hia_core_admission_native.py
hepta/hia_function_shape_native.py
hepta/hia_instruction_policy_native.py
hepta/hia_json_canonical_native.py
hepta/hia_json_decode.py
hepta/hia_json_decode_native.py
hepta/hia_json_preflight.py
hepta/hia_json_preflight_native.py
hepta/hia_machine_x64.py
hepta/hia_opcodes.py
hepta/hia_program.py
hepta/hia_program_shape_native.py
hepta/hia_program_tools.py
hepta/hia_schema.py
hepta/hia_sha256_native.py
tools/generate_hia_selfhost.py
tools/hia_selfhost_chain_elf.py
tools/hia_selfhost_cli.py
tools/hia_selfhost_effects.py
tools/hia_selfhost_exports.py
tools/hia_selfhost_keywords.py
tools/hia_selfhost_limits.py
tools/hia_selfhost_multi_results.py
tools/hia_selfhost_names.py
tools/hia_selfhost_owned_elf.py
tools/hia_selfhost_returned_loans.py
tools/hia_selfhost_signed.py
tools/hia_selfhost_traps.py'''.splitlines()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--linker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    upstream, compiler, linker = (p.resolve() for p in (args.upstream, args.compiler, args.linker))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert sha(compiler.read_bytes()) == COMPILER_SHA
    assert sha(linker.read_bytes()) == LINKER_SHA
    contract_raw = (upstream/'docs/HIA_SELFHOST_CORE_SUPPORT.json').read_bytes()
    assert sha(contract_raw) == CONTRACT_SHA
    pinned = json.loads(contract_raw)['source_identity']['files']
    custody = {}
    for path in AUTHORING_FILES:
        custody[path] = sha((upstream/path).read_bytes())
        assert custody[path] == pinned[path], path
    sys.path.insert(0, str(upstream))
    from tools.generate_hia_selfhost import Builder, compact_authoring_cfg, total_scalar_functions
    from hepta.hia_agent_assembly import _render_assembly_document

    class NativeBuilder(Builder):
        def expr(self, value):
            if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.FloorDiv, ast.Mod)):
                left, right = self.expr(value.left), self.expr(value.right)
                return self.op('int.div.checked' if isinstance(value.op, ast.FloorDiv)
                               else 'int.rem.checked', [left, right], left[1])
            if isinstance(value, ast.Call) and value.func.id in ('copy_range', 'copy_from_mut'):
                dest, source, dest_at, source_at, size = [self.expr(x) for x in value.args]
                loan = None
                if value.func.id == 'copy_from_mut':
                    loan = self.op('borrow.read', [source], 'readbuf')
                    source = loan
                self.op('buffer.copy_range', [dest, source, self.cast(dest_at, 'usize'),
                                             self.cast(source_at, 'usize'), self.cast(size, 'usize')])
                if loan is not None:
                    self.op('loan.end', [loan])
                return self.c(0)
            return super().expr(value)

    constants = {name: int.from_bytes(magic.encode(), 'little') for name, magic in {
        'STATE_MAGIC': 'HTOSST01', 'REQUEST_MAGIC': 'HTOSOBJ1',
        'RESPONSE_MAGIC': 'HTOSRES1', 'TRACE_REQUEST_MAGIC': 'HTOSTRC1',
        'TRACE_RESPONSE_MAGIC': 'HTOSTRR1'}.items()}
    modules, all_nodes, input_hashes = {}, [], {}
    for module in ('bytes', 'records', 'audit', 'ops'):
        path = ROOT/(module+'.authoring')
        raw = path.read_bytes()
        input_hashes[path.name] = sha(raw)
        source = raw.decode()
        for key in sorted(constants, key=len, reverse=True):
            source = source.replace(key, str(constants[key]))
        nodes = ast.parse(source).body
        assert all(isinstance(node, ast.FunctionDef) for node in nodes)
        modules[module] = [node.name for node in nodes]
        all_nodes.extend(nodes)
    signatures = {node.name: ([p.annotation.id for p in node.args.args], node.returns.id)
                  for node in all_nodes}
    names = {name: name for name in signatures}
    total = total_scalar_functions(all_nodes)
    functions = []
    for node in all_nodes:
        b = NativeBuilder(node, names, signatures, total)
        b.statements(node.body)
        functions.append({'name': node.name, 'params': [b.param(p) for p in b.params],
                          'returns': [b.returns], 'effects': ['read', 'write', 'trap'],
                          'entry': 'b0', 'blocks': b.blocks})
    doc = compact_authoring_cfg({'schema': 'hepta.hia.program.v8',
                                 'module': 'storage_authoring', 'functions': functions})
    by_name = {f['name']: f for f in doc['functions']}
    def external(name):
        params, result = signatures[name]
        return ('fn '+name+'('+','.join('p'+str(i)+':'+t for i,t in enumerate(params))
                +') -> ('+result+') effects(read,write,trap) external()\n')
    sources, structures = {}, {}
    for module, fnames in modules.items():
        fns = [by_name[name] for name in fnames]
        calls = {op['attrs']['function'] for f in fns for block in f['blocks'] for op in block['ops']
                 if op['op'] == 'call'}
        imports = sorted(calls-set(fnames))
        body = _render_assembly_document({'schema': 'hepta.hia.program.v8', 'module': 'storage_'+module,
                                         'exports': fnames, 'functions': fns})
        first, exports, tail = body.split('\n', 2)
        sources[module] = first+'\n'+exports+'\n'+''.join(external(n) for n in imports)+tail
        structures[module] = {
            'definitions': len(fns), 'imports': len(imports),
            'blocks': sum(len(f['blocks']) for f in fns),
            'instructions': sum(len(b['ops']) for f in fns for b in f['blocks']),
            'max_blocks_per_function': max(len(f['blocks']) for f in fns),
            'max_block_parameters': max(len(b['params']) for f in fns for b in f['blocks'])}
    for module in ('owners', 'trace'):
        path = ROOT/(module+'.hia.in')
        if not path.exists():
            if args.source_only:
                continue
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        input_hashes[path.name] = sha(raw)
        source = raw.decode()
        for key in sorted(constants, key=len, reverse=True):
            source = source.replace(key, str(constants[key]))
        source = '\n'.join(line for line in source.splitlines() if line.strip())+'\n'
        sources[module] = source
    for name, source in sources.items():
        (output/(name+'.hepta')).write_text(source)
    actual_loaded = {}
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path is not None:
            path = Path(path).resolve()
            if path.is_relative_to(upstream):
                relative = str(path.relative_to(upstream))
                actual_loaded[relative] = sha(path.read_bytes())
    assert set(actual_loaded) == set(custody), (set(actual_loaded)-set(custody), set(custody)-set(actual_loaded))
    assert actual_loaded == custody
    report = {'schema': 'heptatorch.native-storage-object-build.v1',
              'status': 'SOURCE_ONLY' if args.source_only else 'BUILDING',
              'upstream_commit': 'f33d8da316de8fdf27b504371f1a5310997ff548',
              'upstream_contract_sha256': CONTRACT_SHA, 'upstream_authoring_inputs': custody,
              'authoring_sources': input_hashes, 'authoring_host_supplies_runtime': False,
              'compiler_sha256': COMPILER_SHA, 'linker_sha256': LINKER_SHA,
              'modules': {name: {'source_sha256': sha(source.encode()), 'source_bytes': len(source.encode()),
                                **structures.get(name, {})} for name,source in sources.items()}}
    def save():
        (output/'build.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    save()
    if args.source_only:
        print(json.dumps({'status':report['status'], 'modules':report['modules']}))
        return
    objects = []
    for name, source in sources.items():
        result = subprocess.run([str(compiler)], input=source.encode(), capture_output=True,
                                env={'PATH': ''}, timeout=60)
        (output/(name+'.compile.stdout')).write_bytes(result.stdout)
        (output/(name+'.compile.stderr')).write_bytes(result.stderr)
        report['modules'][name]['compile_exit'] = result.returncode
        save()
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), (name,result.returncode,result.stderr)
        (output/(name+'.o')).write_bytes(result.stdout)
        report['modules'][name]['object_sha256'] = sha(result.stdout)
        objects.append(result.stdout)
    entry = b'entry'
    request = b'HIAOWN02'+struct.pack('<3Q',len(objects),len(entry),0)+entry
    request += b''.join(struct.pack('<Q',len(obj))+obj for obj in objects)
    (output/'link-request.bin').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=60)
    (output/'link.stdout').write_bytes(result.stdout)
    (output/'link.stderr').write_bytes(result.stderr)
    report['link_exit'] = result.returncode
    save()
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), (result.returncode,result.stderr)
    application = output/'storage-trace'
    application.write_bytes(result.stdout)
    application.chmod(0o700)
    report.update(status='PASS', executable_sha256=sha(result.stdout),
                  executable_bytes=len(result.stdout), link_request_sha256=sha(request),
                  complete_tensor_storage_claimed=False, interactive_claimed=False)
    save()
    print(json.dumps({'status':'PASS','executable_sha256':report['executable_sha256'], 'modules':report['modules']}))


if __name__ == '__main__':
    main()
