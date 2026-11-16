"""Emit a cmd95-only HIA helper and execute an isolated real native kernel.

The filter embeds the actual unchanged numerical callees from candidate02.
It does not modify the candidate or execute numerical authoring as Python.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_SHA = '84a088b4e36771251f240a8d293e2041ca9fbfcc25877815774a371dcedd01a8'
COMPILER_SHA = 'c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6'
ACTUAL_SHA = 'e664c9ff717a104b8c7317dfc55677e0248e58f981cb8315eacb3975049048ff'
FUNCTION = re.compile(rb'^(fn (\w+)\([^\n]* entry\([^\n]+\)\n.*?^end\n)', re.M | re.S)
CALL = re.compile(rb'\{"function":"(\w+)"\}')
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'compiler', 'observations', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {args.source: args.source.read_bytes(), args.compiler: args.compiler.read_bytes(),
              args.observations: args.observations.read_bytes(), HERE/'authoring.py': (HERE/'authoring.py').read_bytes(),
              Path(__file__): Path(__file__).read_bytes()}
    assert sha(inputs[args.source]) == SOURCE_SHA and sha(inputs[args.compiler]) == COMPILER_SHA
    assert sha(inputs[args.observations]) == ACTUAL_SHA
    old = {name.decode(): body for body, name in FUNCTION.findall(inputs[args.source])}
    sys.path.insert(0, str(ROOT/'dev'))
    import emit_hia
    fragment, mapping = emit_hia.emit(inputs[HERE/'authoring.py'].decode())
    fragment = fragment.encode()
    added = {name.decode(): body for body, name in FUNCTION.findall(fragment)}
    assert set(added) == {'scalar_add_preserving_bits'}
    (output/'scalar_add_preserving_bits.hepta').write_bytes(fragment)
    (output/'functions.json').write_text(json.dumps(mapping, indent=2) + '\n')
    before = old['scalar_add_nograd']
    needle = b'{"function":"addf"}'
    replacement = b'{"function":"scalar_add_preserving_bits"}'
    assert before.count(needle) == 1
    after = before.replace(needle, replacement, 1)
    assert after.replace(replacement, needle, 1) == before
    (output/'scalar_add_nograd.hepta').write_bytes(after)
    probe = '''def entry(s:readbuf, o:mutbuf, w:mutbuf) -> u32:
    n = length(s)
    if n % 8 != 0:
        return 4294967295
    count = n // 8
    if count > length(o) // 4:
        return 4294967295
    i = 0
    while i < count:
        a = read4(s,i*8)
        b = read4(s,i*8+4)
        put4(o,i*4,scalar_add_preserving_bits(a,b))
        i = i+1
    return count*4
'''
    entry, _ = emit_hia.emit(inputs[HERE/'authoring.py'].decode() + '\n\n' + probe)
    emitted_probe = {name.decode(): body for body, name in FUNCTION.findall(entry.encode())}
    assert emitted_probe['scalar_add_preserving_bits'] == added['scalar_add_preserving_bits']
    functions = dict(old, **added)
    functions['entry'] = emitted_probe['entry']
    selected = []
    visited = set()
    def visit(name):
        if name in visited:
            return
        assert name in functions
        visited.add(name)
        selected.append(name)
        for callee in sorted({name.decode() for name in CALL.findall(functions[name])}):
            visit(callee)
    visit('entry')
    source = b'hia-asm.v1 hepta.hia.program.v8 selfhost_exec\n' + b''.join(functions[name] for name in selected)
    (output/'helper-probe.hepta').write_bytes(source)
    compiled = subprocess.run([str(args.compiler)], input=source, capture_output=True,
                              env={'PATH': ''}, timeout=120)
    (output/'compiler.stderr').write_bytes(compiled.stderr)
    assert compiled.returncode == 0 and not compiled.stderr and compiled.stdout.startswith(b'\x7fELF')
    executable = output/'helper-probe'
    executable.write_bytes(compiled.stdout)
    executable.chmod(0o500)
    actual = json.loads(inputs[args.observations])
    assert actual['status'] == 'OBSERVED' and actual['cases'] == 12 and actual['variants'] == 48
    request = bytearray()
    expected = bytearray()
    offsets = []
    for row in actual['rows']:
        for variant in row['variants']:
            offsets.append({'case': row['case'], 'variant': variant['variant'],
                            'first_word': len(expected)//4,
                            'words': len(variant['before']['binary32_words'])})
            for before_word, after_word in zip(variant['before']['binary32_words'], variant['after']['binary32_words']):
                request.extend(struct.pack('<II', before_word, variant['scalar_admitted_binary32_word']))
                expected.extend(struct.pack('<I', after_word))
    result = subprocess.run([str(executable)], input=bytes(request), capture_output=True, env={'PATH': ''}, timeout=30)
    (output/'request.bin').write_bytes(request)
    (output/'expected.bin').write_bytes(expected)
    (output/'response.bin').write_bytes(result.stdout)
    (output/'execution.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr and result.stdout == expected
    assert all(path.read_bytes() == value for path, value in inputs.items())
    report = {'schema': 'heptatorch.command95-nan-helper-native-qualification.v1', 'status': 'PASS',
              'actual_reference_sha256': ACTUAL_SHA, 'cases': 12, 'variants': 48,
              'exact_output_words': len(expected)//4, 'source_input_sha256': SOURCE_SHA,
              'compiler_sha256': COMPILER_SHA, 'helper_authoring_sha256': sha(inputs[HERE/'authoring.py']),
              'helper_hia_sha256': sha(fragment), 'caller_before_sha256': sha(before), 'caller_after_sha256': sha(after),
              'caller_only_one_callee_replacement_exact_inverse': True,
              'harness_source_sha256': sha(source), 'harness_native_sha256': sha(compiled.stdout),
              'unchanged_real_callees': {name: sha(functions[name]) for name in selected if name not in ('entry', 'scalar_add_preserving_bits')},
              'request_sha256': sha(request), 'expected_sha256': sha(expected), 'response_sha256': sha(result.stdout),
              'all_input_bytes_unchanged_after': True,
              'scope': 'Scalar arithmetic helper against the fixed actual48. The production cmd95 owner/strided/version path still requires final candidate replay.',
              'Torch_executed_here': False, 'native_executed': True, 'offsets': offsets}
    (output/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: report[name] for name in ('status','cases','variants','exact_output_words',
                     'helper_hia_sha256','caller_after_sha256','harness_native_sha256')}))


if __name__ == '__main__':
    main()
