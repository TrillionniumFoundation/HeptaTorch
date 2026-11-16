"""Qualification-only native owner transfer between real predecessor/new steps.

Every function, import and export is reversibly namespaced. All complete units
are checked by the unchanged native ABI3 compiler and linker; no callee stubs,
foreign pointers, or reconstructed owners are introduced.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import subprocess

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
HERE = Path(__file__).resolve().parent
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
NEW = '7c68a96b08aa9a21a52e2fd12be912c5f47dbde889e462f9a0dbb38cb3c09edf'
PREDECESSORS = {
    'nullable': ('490fc144160f6e4440aed001363f86ccff56c335f9781e38b058fc6106b1be09', 'HTRNGN01', False),
    'optimizer001': ('1b143984d6fce05ba9a92f5f528c7a1b59b4e8aaa9ceae2c6e2299c43a651297', 'HTOPT001', True),
}


def rename(raw, prefix, reverse=False):
    def symbol(value):
        if reverse:
            assert value.startswith(prefix)
            return value[len(prefix):]
        return prefix + value
    raw = re.sub(rb'^(hia-asm\.v1 hepta\.hia\.program\.v8 )(\w+)$',
                 lambda m: m[1] + symbol(m[2]), raw, flags=re.M)
    raw = re.sub(rb'^(fn )(\w+)(\()', lambda m: m[1] + symbol(m[2]) + m[3], raw, flags=re.M)
    raw = re.sub(rb'^exports\(([^)]*)\)$',
                 lambda m: b'exports(' + b','.join(symbol(n) for n in m[1].split(b',')) + b')', raw, flags=re.M)
    return re.sub(rb'(\{"function":")(\w+)("\})', lambda m: m[1] + symbol(m[2]) + m[3], raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predecessor', choices=sorted(PREDECESSORS), required=True)
    for name in ('old', 'new', 'compiler', 'linker', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    compiler, linker = a.compiler.resolve(), a.linker.resolve()
    assert SHA(compiler.read_bytes()) == COMPILER and SHA(linker.read_bytes()) == LINKER
    paths = (Path(__file__).resolve(), HERE / 'state_handoff.hepta')
    snapshots = {p: p.read_bytes() for p in paths}
    assert SHA(snapshots[HERE / 'state_handoff.hepta']) == '4093b8a1a4ce1b19d6943f457649355c950cfba50743ac7afe8a8396437cc7ee'
    old_sha, old_tag, old_optimizer = PREDECESSORS[a.predecessor]
    report = {'schema': 'heptatorch.optimizer-owner-state-handoff-build.v1', 'status': 'BUILDING',
              'qualification_only': True, 'old_core_sha256': old_sha, 'new_core_sha256': NEW,
              'old_tag': old_tag, 'new_tag': 'HTOPT002', 'old_has_optimizer': old_optimizer,
              'compiler_sha256': COMPILER, 'linker_sha256': LINKER, 'source_bindings': [], 'units': []}
    def save():
        (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    units = []
    for label, directory, core_sha in [('old', a.old.resolve(), old_sha), ('new', a.new.resolve(), NEW)]:
        core, receipt = directory / 'runtime.hepta', directory / 'build.json'
        snapshots[core], snapshots[receipt] = core.read_bytes(), receipt.read_bytes()
        assert SHA(snapshots[core]) == core_sha
        original_report = json.loads(snapshots[receipt])
        assert original_report['status'] == 'PASS'
        for unit in original_report['units']:
            path = directory / (unit['name'] + '.hepta')
            raw = path.read_bytes()
            snapshots[path] = raw
            assert SHA(raw) == unit['source_sha256']
            transformed = rename(raw, (label + '_').encode())
            assert rename(transformed, (label + '_').encode(), True) == raw
            name = label + '_' + unit['name']
            units.append((name, transformed))
            report['source_bindings'].append({'side': label, 'name': name,
                'original_sha256': SHA(raw), 'renamed_sha256': SHA(transformed),
                'inverse_symbol_rename_byte_identical': True})
    units.append(('state_handoff', snapshots[HERE / 'state_handoff.hepta']))
    report['inputs_sha256'] = {str(p): SHA(raw) for p, raw in snapshots.items()}
    save()
    objects = []
    for name, source in units:
        (a.output / (name + '.hepta')).write_bytes(source)
        result = subprocess.run([str(compiler)], input=source, capture_output=True, timeout=120, env={'PATH': ''})
        (a.output / (name + '.o')).write_bytes(result.stdout)
        (a.output / (name + '.stderr')).write_bytes(result.stderr)
        report['units'].append({'name': name, 'source_sha256': SHA(source), 'source_bytes': len(source),
            'compiler_exit': result.returncode, 'object_sha256': SHA(result.stdout), 'object_bytes': len(result.stdout)})
        save()
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        objects.append(result.stdout)
    request = b'HIASTKS1' + struct.pack('<3Q', len(objects), 4, 0) + b'step'
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (a.output / 'link.request.bin').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, timeout=120, env={'PATH': ''})
    (a.output / 'link.stderr').write_bytes(result.stderr)
    report['link_exit'] = result.returncode
    save()
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    expected = {'nullable': '1de3f152119def317df9661ea40441328f2683fec83e1878b900cad90683dc4f',
                'optimizer001': 'a8f7819c6d9e7b1a0fd4303554f5a6051ba67813a300237de144adee65a50db6'}
    assert SHA(result.stdout) == expected[a.predecessor]
    binary = a.output / 'state-handoff-session'
    binary.write_bytes(result.stdout)
    binary.chmod(0o500)
    for path, raw in snapshots.items():
        assert path.read_bytes() == raw, path
    for (name, source), obj in zip(units, objects):
        assert (a.output / (name + '.hepta')).read_bytes() == source
        assert (a.output / (name + '.o')).read_bytes() == obj
    assert SHA(compiler.read_bytes()) == COMPILER and SHA(linker.read_bytes()) == LINKER
    report.update(status='PASS', native_sha256=SHA(result.stdout), native_bytes=len(result.stdout),
                  module_count=len(objects), link_request_bytes=len(request))
    save()
    print(json.dumps({k: report[k] for k in ('status', 'native_sha256', 'native_bytes', 'module_count')}))


if __name__ == '__main__':
    main()
