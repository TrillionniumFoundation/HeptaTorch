"""Tests for the diagnostic-only current35 failure summary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from current_runtime35 import stage_failure_details


class Current35FailureDetailsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)

    def write_stage(self, name, data):
        (self.output / (name + '-stage.json')).write_text(json.dumps(data))

    def test_reports_original_stage_causes(self):
        error = {'type': 'AssertionError', 'message': 'constructor or layout contract changed: linear'}
        for phase in ('native', 'strict', 'controls'):
            self.write_stage(phase, dict(phase=phase, status='FAIL',
                                         error={**error, 'traceback': 'not emitted'}))
        self.assertEqual(stage_failure_details(self.output),
                         dict.fromkeys(('native', 'strict', 'controls'), error))

    def test_ignores_missing_malformed_and_wrong_phase(self):
        self.write_stage('native', dict(phase='strict', status='FAIL',
                                         error=dict(type='x', message='x')))
        (self.output / 'strict-stage.json').write_bytes(b'\xff')
        self.assertEqual(stage_failure_details(self.output), {})

    def test_ignores_success_and_non_object_error(self):
        self.write_stage('native', dict(phase='native', status='PASS',
                                         error=dict(type='x', message='x')))
        self.write_stage('strict', dict(phase='strict', status='FAIL', error=['not a dictionary']))
        self.assertEqual(stage_failure_details(self.output), {})

    def test_bounds_fields_and_ignores_oversize(self):
        self.write_stage('native', dict(phase='native', status='FAIL',
                                         error=dict(type='x' * 100, message='y' * 1000)))
        (self.output / 'strict-stage.json').write_bytes(b'x' * (1024 * 1024 + 1))
        self.assertEqual(stage_failure_details(self.output),
                         {'native': dict(type='x' * 80, message='y' * 240)})

    def test_refuses_symlink(self):
        stage = self.output / 'not-a-receipt.json'
        stage.write_text(json.dumps(dict(phase='native', status='FAIL',
                                         error=dict(type='x', message='x'))))
        (self.output / 'native-stage.json').symlink_to(stage)
        self.assertEqual(stage_failure_details(self.output), {})


if __name__ == '__main__':
    unittest.main()
