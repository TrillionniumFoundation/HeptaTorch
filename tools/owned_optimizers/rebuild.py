"""Reproduce the frozen native owned SGD profile with source custody checks."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SHA = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    names = ('baseline', 'objects', 'common-authoring', 'nullable-authoring', 'partial-authoring',
             'materialization-authoring', 'repository', 'source-root', 'compiler', 'linker', 'output')
    for name in names:
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    manifest_path = HERE / 'source-pins.sha256'
    manifest = manifest_path.read_bytes()
    snapshots = {manifest_path: manifest}
    for line in manifest.decode().splitlines():
        pin, name = line.split('  ', 1)
        path = (HERE / name).resolve()
        assert path.is_relative_to(HERE)
        raw = path.read_bytes()
        assert SHA(raw) == pin, path
        snapshots[path] = raw
    profile = json.loads(snapshots[HERE / 'profile.json'])
    argv = [sys.executable, '-I', '-S', '-B', str(HERE / 'build.py')]
    for name in names:
        argv += ['--' + name, str(getattr(a, name.replace('-', '_')).resolve())]
    subprocess.run(argv, check=True)
    report = json.loads((a.output / 'build.json').read_text())
    assert report['status'] == 'PASS'
    for name in ('source_sha256', 'native_sha256', 'compiler_sha256', 'linker_sha256'):
        assert report[name] == profile[name], name
    actual = {u['name']: (u['source_sha256'], u['object_sha256']) for u in report['units']}
    expected = {u['name']: (u['source_sha256'], u['object_sha256']) for u in profile['units']}
    assert actual == expected
    for path, raw in snapshots.items():
        assert path.read_bytes() == raw, path
    result = {'schema': 'heptatorch.owned-optimizer-reproduction.v1', 'status': 'PASS',
              'source_pins_sha256': SHA(manifest), 'source_sha256': profile['source_sha256'],
              'native_sha256': profile['native_sha256'], 'native_modules': len(actual),
              'all_source_object_image_bytes_identical': True,
              'host_authoring_aggregate_document_admission_claimed': False,
              'all_complete_native_units_and_actual_linked_call_graph_checked': True}
    (a.output / 'reproduction.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
