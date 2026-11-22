"""Exercise the exact legacy reader qualification-loader transition in isolation."""
import importlib.util
from pathlib import Path
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
PATH = REPOSITORY / 'tests/current_runtime_f64_fixtures.py'
SPEC = importlib.util.spec_from_file_location('current_f64_shared_admission_test', PATH)
FIXTURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURES)


class Inputs:
    def __init__(self, payload=None):
        self.payload = payload
        self.calls = []

    def bind(self, path, expected):
        self.calls.append((path, expected))
        return self.payload if self.payload is not None else path.read_bytes()


class ExactAdmissionTransition(unittest.TestCase):
    def test_current_source_restores_full_frozen_sha(self):
        inputs = Inputs()
        path, current, proof = FIXTURES.bind_shared_admission(
            REPOSITORY, inputs, FIXTURES.FROZEN_SHARED_ADMISSION)
        self.assertEqual(inputs.calls, [(path, FIXTURES.CURRENT_SHARED_ADMISSION)])
        self.assertEqual(path, REPOSITORY / 'tools/current_runtime/qualification.py')
        self.assertEqual(current, path.read_bytes())
        self.assertEqual(proof, {
            'frozen_sha256': FIXTURES.FROZEN_SHARED_ADMISSION,
            'executed_sha256': FIXTURES.CURRENT_SHARED_ADMISSION,
            'exact_round_trip': True,
            'altered_only_builder_loading': True,
        })

    def test_wrong_predecessor_refuses_before_any_read(self):
        inputs = Inputs()
        with self.assertRaisesRegex(ValueError, 'exact frozen qualification helper'):
            FIXTURES.bind_shared_admission(REPOSITORY, inputs, '0' * 64)
        self.assertEqual(inputs.calls, [])

    def test_refuses_frozen_or_changed_current_bytes(self):
        current = (REPOSITORY / 'tools/current_runtime/qualification.py').read_bytes()
        for payload in (current + b'\n', current.replace(
                FIXTURES._ADMITTED_CURRENT_LOAD, FIXTURES._ADMITTED_OLD_LOAD, 1)):
            with self.subTest(size=len(payload)), self.assertRaisesRegex(
                    ValueError, 'exact current shared admission'):
                FIXTURES.bind_shared_admission(
                    REPOSITORY, Inputs(payload), FIXTURES.FROZEN_SHARED_ADMISSION)


if __name__ == '__main__':
    unittest.main()
