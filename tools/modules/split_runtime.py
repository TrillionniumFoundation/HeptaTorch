#!/usr/bin/env python3
"""Author a complete two-unit HIA partition without changing function bodies.

This host-side source organizer is not a compiler, signature verifier, linker,
or numerical runtime. The native compiler and linker must admit its outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


ROOTS = ('binary', 'unary', 'reduce', 'matmul', 'view', 'transpose', 'clone',
         'detach', 'reduce_axis', 'softmax', 'index_select', 'gather', 'gadd',
         'tnew', 'init', 'graph_reset')
FUNCTION = re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]*\n.*?^end\n)',
                      re.MULTILINE | re.DOTALL)
CALL = re.compile(rb'\bcall\([^\n)]*\)[ \t]*(\{[^\n]*\})')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def split(source_path, output):
    source_path = source_path.resolve()
    source = source_path.read_bytes()
    header, separator, rest = source.partition(b'\n')
    require(separator and header == b'hia-asm.v1 hepta.hia.program.v8 selfhost_exec',
            'expected the complete ordinary HeptaTorch program')
    functions = {}
    end = 0
    for match in FUNCTION.finditer(rest):
        require(not rest[end:match.start()].strip(), 'unparsed source between functions')
        body, raw_name = match.groups()
        name = raw_name.decode('ascii')
        require(name not in functions, 'duplicate function: ' + name)
        functions[name] = body
        end = match.end()
    require(functions and not rest[end:].strip(), 'unparsed source after functions')
    require('entry' in functions and all(name in functions for name in ROOTS),
            'runtime entry or partition root missing')
    callees = {}
    for name, body in functions.items():
        matches = list(CALL.finditer(body))
        require(len(matches) == len(re.findall(rb'= call\(', body)),
                'unparsed call in ' + name)
        targets = set()
        for match in matches:
            attributes = json.loads(match.group(1))
            require(set(attributes) == {'function'}, 'unexpected call attributes')
            target = attributes['function']
            require(target in functions, 'undefined source function: ' + str(target))
            targets.add(target)
        callees[name] = targets
    kernel = set(ROOTS)
    while True:
        expanded = kernel | set().union(*(callees[name] for name in kernel))
        if expanded == kernel:
            break
        kernel = expanded
    runtime = set(functions) - kernel
    require('entry' in runtime and not any(callees[name] - kernel for name in kernel),
            'partition does not form the required dependency DAG')
    imports = set().union(*(callees[name] for name in runtime)) & kernel
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []

    def ordered(selection):
        return [name for name in functions if name in selection]

    for name, local, external, exports in (
        ('ht_kernel', kernel, set(), imports),
        ('ht_runtime', runtime, imports, {'entry'}),
    ):
        raw = ('hia-asm.v1 hepta.hia.program.v8 ' + name + '\nexports(' +
               ','.join(ordered(exports)) + ')\n').encode()
        for target in ordered(external):
            signature = functions[target].split(b'\n', 1)[0]
            declaration, count = re.subn(rb'entry\([^)]*\)$', b'external()', signature)
            require(count == 1 and b'return_loans' not in signature,
                    'unsupported external boundary for ' + target)
            raw += declaration + b'\n'
        for target in ordered(local):
            raw += functions[target]
        (output / (name + '.hepta')).write_bytes(raw)
        rows.append({
            'module': name,
            'functions': ordered(local),
            'imports': ordered(external),
            'exports': ordered(exports),
            'function_count': len(local),
            'import_count': len(external),
            'export_count': len(exports),
            'function_bodies_byte_identical': True,
            'function_body_sha256': {target: digest(functions[target]) for target in ordered(local)},
            'source_sha256': digest(raw),
            'source_bytes': len(raw),
        })
    require(source_path.read_bytes() == source, 'source changed during partitioning')
    report = {
        'schema': 'heptatorch.hia.source-partition.v1',
        'source_sha256': digest(source),
        'source_bytes': len(source),
        'total_function_count': len(functions),
        'total_blocks': len(re.findall(rb'^block ', source, re.MULTILINE)),
        'total_instructions': len(re.findall(rb' = ', source)),
        'partition_complete': kernel | runtime == set(functions) and not (kernel & runtime),
        'module_dag': [['ht_runtime', 'ht_kernel']],
        'native_compilation_performed': False,
        'native_admission_and_linking_performed': False,
        'rows': rows,
    }
    (output / 'split-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
    print(json.dumps([{key: value for key, value in row.items()
                       if key not in ('functions', 'imports', 'exports', 'function_body_sha256')}
                      for row in rows], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    split(options.source, options.output)
