#!/usr/bin/env python3
"""Execute the old and new command admission contracts on all three native ELFs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

SHA = lambda data: hashlib.sha256(data).hexdigest()
PINS = {
    'atomic03': ('a72a98232c7b7f10efa67ce29b21e417c58201383bef9aacc266fe247ff356ce',
                 '515790a191b3db1ecbfcb3f2afab644ef980b51ad85c5d6daf1b8c5a9a5a6e28', 'tensor-distributions-session'),
    'partial01': ('55ed0c6e16317add03974c01ffd85e38684fc535a3e5cb6423ed043f4217f05b',
                  'bb4c25e35a170f08276d464db6b630cf03765ff816e6a9ca9aa1651360a1ccf4', 'tensor-distributions-session'),
    'combined02': ('5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873',
                   '3fc59ab7ed9c32dd75ff440ef9bfbec31756f50edd77715d665bdac098cc3f22', 'tensor-adam-distributions-session'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('atomic-build', 'partial-build', 'build', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    inputs = {Path(__file__).resolve(): Path(__file__).read_bytes(), a.protocol.resolve(): a.protocol.read_bytes()}
    assert SHA(a.protocol.read_bytes()) == '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
    spec = importlib.util.spec_from_file_location('admission_contract_wire', a.protocol)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    roots = {'atomic03': a.atomic_build, 'partial01': a.partial_build, 'combined02': a.build}
    report = {'schema': 'heptatorch.adam-distributions.new-command-admission.v1', 'status': 'RUNNING',
              'rows': [], 'frames': 0, 'complete_owner_rollback_bytes': 0,
              'old_atomic_controls_remain_expected_fail_on_new_profile': True,
              'old_atomic_controls_sha256': '908dff3b128a15d6dc4559bdde9483517f13696c981122e5cd46fbe1bd9990f3',
              'old_opcode80_85_expectations_changed': False,
              'new_contract_source': 'frozen partial01 dist_partial_command checks typed Tensor/Generator handles before execution; reserved88/89 still route to unknown command',
              'pytorch_executed_here': False}
    for label, root in roots.items():
        source_pin, image_pin, image_name = PINS[label]
        source, image = root / 'runtime.hepta', root / image_name
        inputs[source.resolve()] = source.read_bytes()
        inputs[image.resolve()] = image.read_bytes()
        assert SHA(inputs[source.resolve()]) == source_pin and SHA(inputs[image.resolve()]) == image_pin
        session = wire.Session(image, directory=a.output / label)
        session.run([wire.full((1,), .5)])
        for opcode in (86, 87, 88, 89):
            expected = 1 if label != 'atomic03' and opcode in (86, 87) else 4
            before = session.owner()
            actual, records = wire.parse(session.exchange(wire.packet([(opcode,)])))
            assert actual == expected and not records and session.owner() == before
            report['complete_owner_rollback_bytes'] += before['length']
            report['rows'].append({'profile': label, 'source_sha256': source_pin, 'executable_sha256': image_pin,
                                   'opcode': opcode, 'arguments': [], 'expected_error': expected,
                                   'actual_error': actual, 'owner_identity_and_bytes_unchanged': True})
        session.run([(53,)])
        session.finish()
        report['frames'] += session.frames
    for path, raw in inputs.items():
        assert path.read_bytes() == raw, path
    report.update(status='PASS', cases=len(report['rows']), input_custody_after=True,
                  input_sha256={str(path): SHA(raw) for path, raw in inputs.items()})
    (a.output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'frames', 'complete_owner_rollback_bytes')}))


if __name__ == '__main__':
    main()
