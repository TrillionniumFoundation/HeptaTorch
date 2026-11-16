"""Rebind frozen tests only to the eight-kind read-only observer layout.

Numerical recipes, expected values, tolerances, error statuses and commit
assertions are byte-preserved. Every textual edit has an exact inverse.
"""
from pathlib import Path
import argparse
import hashlib
import json

SHA = lambda data: hashlib.sha256(data).hexdigest()
IMAGE = 'b6634158e079512e797d8237cca12d65052db18df23b12b8414adeb8d0b4b0e1'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('grad-clear', 'common-joint', 'nullable-joint', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    plans = [
        (a.grad_clear, 'replay_grad_clear.py',
         'fdaedbb56e4ce4b526e084f97842892bb1862ce0950d010b7182016b5364a476',
         [("b'HTSTATE6'", "b'HTOPT002'"), ('144+424*', '144+552*')]),
        (a.common_joint, 'test_joint.py',
         '381a65215dc7d238f46ffbf0856414d98cc1f34d0f4eebe9129c123894437ba7',
         [("b'HTRNGP01'", "b'HTOPT002'"), ('144+488*', '144+552*')]),
        (a.nullable_joint, 'test_combination.py',
         'fb8648b0b2331349c85c448d16cc448faa5514b26a13ae7303868916a9995bd1',
         [("'5baa9cd5e5625f0e2324941486982b714dcdedfe897f5c16686a3ac8882ce242'", repr(IMAGE)),
          ("b'HTRNGN01'", "b'HTOPT002'"), ("b'HTORNGN1'", "b'HTOOPT02'"),
          ('144 + 488 *', '144 + 552 *')]),
    ]
    rows = []
    for path, name, pin, edits in plans:
        original = path.read_bytes()
        assert SHA(original) == pin, path
        text = original.decode()
        for before, after in edits:
            assert text.count(before) == 1 and after not in text, (path, before)
            text = text.replace(before, after)
        reverse = text
        for before, after in reversed(edits):
            assert reverse.count(after) == 1
            reverse = reverse.replace(after, before)
        assert reverse.encode() == original
        (a.output / name).write_text(text)
        (a.output / ('original-' + name)).write_bytes(original)
        assert path.read_bytes() == original
        rows.append({'name': name, 'original_sha256': pin, 'adapted_sha256': SHA(text.encode()),
                     'inverse_byte_identical': True,
                     'exact_edits': [{'before': b, 'after': c, 'count': 1} for b, c in edits]})
    report = {'schema': 'heptatorch.optimizer-regression-observer-adaptation.v1', 'status': 'PASS',
              'native_executable_sha256': IMAGE,
              'scope': 'executable identity, internal state tags, and read-only eight-kind data offsets',
              'changed_input_packets': False, 'changed_expected_numeric_values': False,
              'changed_numerical_tolerances': False, 'changed_error_or_commit_expectations': False,
              'scripts': rows}
    (a.output / 'adaptation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
