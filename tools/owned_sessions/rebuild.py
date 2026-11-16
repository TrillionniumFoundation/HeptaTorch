#!/usr/bin/env python3
"""Rebuild the distinct production and qualification owned-session linkers."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--owned-build', type=Path, required=True)
    parser.add_argument('--owned-profile', type=Path, required=True)
    parser.add_argument('--borrowed-tools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = {}
    for line in (spec / 'source-pins.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        actual = digest((spec / relative).read_bytes())
        assert actual == expected, relative
        identity[relative] = actual
    profile = json.loads((spec / 'profile.json').read_text())
    rows = []

    def execute(label, command):
        result = subprocess.run(command, capture_output=True, timeout=180)
        (output / (label + '.stdout')).write_bytes(result.stdout)
        (output / (label + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr, (label, result.returncode, result.stderr)

    for label, wrapper in [('production', 'wrapper.py'), ('qualification', 'close_probe_wrapper.py'),
                           ('signal_qualification', 'signal_wrapper.py')]:
        expected = profile[label]
        assert digest((spec / wrapper).read_bytes()) == expected['startup_author_sha256']
        source = output / (label + '-source')
        native = output / (label + '-native')
        command = [sys.executable, '-I', '-S', '-B', str(spec / 'build_linker.py'),
                   '--source-root', str(args.source_root.resolve()), '--owned-build', str(args.owned_build.resolve()),
                   '--owned-profile', str(args.owned_profile.resolve()), '--startup-template', str(spec / wrapper),
                   '--output', str(source)]
        if label != 'production':
            command += ['--qualification-template-sha256', expected['startup_author_sha256'],
                        '--qualification-prefix-sha256', expected['startup_prefix_sha256']]
        execute(label + '-authoring', command)
        raw = (source / 'owned-session-linker.hepta').read_bytes()
        assert digest(raw) == expected['source_sha256']
        execute(label + '-native-build', [sys.executable, '-I', '-S', '-B', str(spec / 'native_build.py'),
                '--generated', str(source), '--compiler', str(args.owned_build.resolve() / 'native/owned-module-compiler'),
                '--borrowed-tools', str(args.borrowed_tools.resolve()), '--output', str(native)])
        assert digest((native / 'owned-session-linker').read_bytes()) == expected['linker_sha256']
        assert digest((native / 'linker.image').read_bytes()) == expected['image_sha256']
        rows.append({'profile': label, 'authoring': json.loads((source / 'authoring.json').read_text()),
                     'native_build': json.loads((native / 'build.json').read_text())})
    for relative, expected in identity.items():
        assert digest((spec / relative).read_bytes()) == expected, relative
    report = {'schema': 'heptatorch.owned-sessions-native-rebuild.v1', 'status': 'PASS',
              'package_source_sha256': identity, 'rows': rows,
              'native_compilation_and_linking': True, 'host_owner_runtime': False,
              'production_linker_sha256': profile['production']['linker_sha256'],
              'qualification_linker_sha256': profile['qualification']['linker_sha256'],
              'signal_qualification_linker_sha256': profile['signal_qualification']['linker_sha256']}
    (output / 'rebuild.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('rows', 'package_source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
