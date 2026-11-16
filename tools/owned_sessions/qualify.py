#!/usr/bin/env python3
"""Qualify the real owned service with independent native admission and I/O tests."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import subprocess
import sys


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('service_sources', type=Path, nargs='+')
    args = parser.parse_args()
    spec = Path(__file__).resolve().parent
    build, compiler = args.build.resolve(), args.compiler.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = {}
    for line in (spec / 'source-pins.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        actual = digest((spec / relative).read_bytes())
        assert actual == expected, relative
        identity[relative] = actual
    profile = json.loads((spec / 'profile.json').read_text())
    assert digest(compiler.read_bytes()) == profile['compiler_sha256']
    tools = {}
    for label in ('production', 'qualification', 'signal_qualification'):
        tools[label] = build / (label + '-native/owned-session-linker')
        assert digest(tools[label].read_bytes()) == profile[label]['linker_sha256']

    def execute(label, script, options):
        command = [sys.executable, '-I', '-S', '-B', str(spec / script), *options]
        result = subprocess.run(command, capture_output=True, timeout=180)
        (output / (label + '.stdout')).write_bytes(result.stdout)
        (output / (label + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr, (label, result.returncode, result.stderr)

    execute('entry-admission', 'tests/entries.py', [
        '--compiler', str(compiler), '--linker', str(tools['production']),
        '--output', str(output / 'entry-admission')])
    execute('service-build', 'link_service.py', [
        '--compiler', str(compiler), '--linker', str(tools['production']),
        '--qualification-linker', str(tools['qualification']), '--output', str(output / 'service'),
        *[str(path.resolve()) for path in args.service_sources]])
    service = output / 'service/service'
    close_probe = output / 'service/service-close-probe'
    objects = [(output / f'service/unit-{index}.o').read_bytes() for index in range(len(args.service_sources))]
    request = b'HIAOWNT1' + struct.pack('<3Q', len(objects), 4, 0) + b'step'
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (output / 'signal-service.request').write_bytes(request)
    result = subprocess.run([str(tools['signal_qualification'])], input=request,
                            capture_output=True, env={'PATH': ''}, timeout=30)
    (output / 'signal-service.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    signal_probe = output / 'signal-service'
    signal_probe.write_bytes(result.stdout)
    signal_probe.chmod(0o700)
    assert signal_probe.read_bytes()[4096:] == service.read_bytes()[4096:]
    execute('transport', 'test_transport.py', ['--service', str(service), '--close-probe', str(close_probe),
        '--build-report', str(output / 'service/build.json'), '--output', str(output / 'transport')])
    execute('returned-length', 'test_result_bounds.py', ['--compiler', str(compiler),
        '--linker', str(tools['production']), '--qualification-linker', str(tools['qualification']),
        '--output', str(output / 'returned-length')])
    execute('real-signals', 'test_real_signals.py', ['--service', str(service), '--signal-probe', str(signal_probe),
        '--signal-metadata', str(build / 'signal_qualification-source/startup.json'),
        '--output', str(output / 'real-signals')])
    rows = []
    for label, cases in [('entry-admission', 103), ('transport', 20), ('returned-length', 6), ('real-signals', 2)]:
        path = output / label / 'report.json'
        report = json.loads(path.read_text())
        assert report.get('status') == 'PASS' or report.get('pass') is True, label
        assert report['cases'] == cases, (label, report['cases'])
        rows.append({'suite': label, 'cases': cases, 'report_sha256': digest(path.read_bytes()),
                     'report': label + '/report.json', 'status': 'PASS'})
    for relative, expected in identity.items():
        assert digest((spec / relative).read_bytes()) == expected, relative
    for label, path in tools.items():
        assert digest(path.read_bytes()) == profile[label]['linker_sha256']
    assert digest(compiler.read_bytes()) == profile['compiler_sha256']
    report = {'schema': 'heptatorch.owned-sessions-portable-qualification.v1', 'status': 'PASS',
              'cases': 131, 'rows': rows, 'package_source_sha256': identity,
              'compiler_sha256': profile['compiler_sha256'],
              'production_service_sha256': digest(service.read_bytes()),
              'close_probe_service_sha256': digest(close_probe.read_bytes()),
              'signal_probe_service_sha256': digest(signal_probe.read_bytes()),
              'service_build': json.loads((output / 'service/build.json').read_text()),
              'signal_service_request_sha256': digest(request),
              'production_and_signal_object_text_identical': True,
              'scope': 'Native ABI2 entry admission, real owned state across frames, transport, cleanup and actual kernel I/O interruptions.'}
    (output / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('rows', 'package_source_sha256', 'service_build')}, indent=2))


if __name__ == '__main__':
    main()
