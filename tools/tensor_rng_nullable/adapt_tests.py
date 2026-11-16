#!/usr/bin/env python3
"""Adapt only private observer layout/tag constants for the combined profile."""
import argparse
import hashlib
import json
from pathlib import Path

SHA = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('nullable-package', 'common-package', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    plans = [
        (a.nullable_package / 'test_nullable.py',
         '7dea3d709405a07df2d0cb5df14670ca6a878130059e75f343abad88b5f989ab',
         [("b'HTSTATE6'", "b'HTRNGN01'"), ('144+424*', '144+488*')]),
        (a.nullable_package / 'replay_grad_clear.py',
         'fdaedbb56e4ce4b526e084f97842892bb1862ce0950d010b7182016b5364a476',
         [("b'HTSTATE6'", "b'HTRNGN01'"), ('144+424*', '144+488*')]),
        (a.common_package / 'test_joint.py',
         '381a65215dc7d238f46ffbf0856414d98cc1f34d0f4eebe9129c123894437ba7',
         [("b'HTRNGP01'", "b'HTRNGN01'")]),
    ]
    rows = []
    for path, pin, edits in plans:
        original = path.read_bytes()
        assert SHA(original) == pin, path
        source = original.decode()
        for before, after in edits:
            assert source.count(before) == 1, (path, before)
            source = source.replace(before, after)
        (a.output / path.name).write_text(source)
        assert path.read_bytes() == original
        rows.append({'name': path.name, 'original_sha256': pin, 'adapted_sha256': SHA(source.encode()),
                     'exact_edits': [{'before': b, 'after': c, 'count': 1} for b, c in edits]})
    report = {'schema': 'heptatorch.tensor-rng-nullable-test-layout-adaptation.v1', 'status': 'PASS',
              'scope': 'read-only owner observer private state magic and seven-kind data offset',
              'changed_input_packets': False, 'changed_expected_values': False,
              'changed_numerical_tolerances': False, 'changed_error_or_commit_expectations': False,
              'scripts': rows}
    (a.output / 'adaptation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
