#!/usr/bin/env python3
"""Independent refusal controls for actual object migration and raw process custody."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest

parser = argparse.ArgumentParser()
parser.add_argument('--repository', type=Path, required=True)
parser.add_argument('--implementation', type=Path)
args, rest = parser.parse_known_args()
ROOT = args.repository.resolve()
HERE = (args.implementation or ROOT/'tools/current_f64_consumer').resolve()
sys.path.insert(0, str(HERE))
from build import adapter, dump, identity, sha
from support import read_process, original


class Refusals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = (ROOT/'tools/dtype-f64/f64.hepta').read_bytes()
        cls.current = (ROOT/'tools/current_runtime/source/runtime.hepta').read_bytes()
        cls.profile = (ROOT/'tools/dtype-f64/profile.json').read_bytes()

    def test_complete_source_inverse(self):
        _, proof = adapter(self.old, self.current, self.profile)
        self.assertEqual(11, len(proof['reused_current_functions']))
        self.assertEqual(3, proof['compiled_adapter_function_count'])
        self.assertTrue(proof['complete_inverse'])

    def test_original_numeric_mutation_refused(self):
        changed = self.old.replace(b'"value":60000', b'"value":60001', 1)
        self.assertNotEqual(changed, self.old)
        with self.assertRaises(ValueError):
            adapter(changed, self.current, self.profile)

    def test_current_kernel_mutation_refused(self):
        changed = self.current.replace(b'fn add64(', b'fn changed_add64(', 1)
        self.assertNotEqual(changed, self.current)
        with self.assertRaises(ValueError):
            adapter(self.old, changed, self.profile)

    def test_original_comparison_inverse(self):
        _, _, proof = original(ROOT)
        self.assertEqual(55, proof['original_expected_cases'])
        self.assertEqual(16, proof['original_bad_packet_cases'])
        self.assertTrue(proof['original_comparison_unmodified'])

    def test_physical_process_refusals(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            executable = path/'native'; executable.write_bytes(b'fixed-fixture-image'); executable.chmod(0o500)
            request, stdout = b'original-request', b'original-response'
            intent = {'name': 'native-0000', 'argv': [str(executable)], 'cwd': str(path),
                      'environment': {'PATH': ''}, 'timeout_seconds': 30,
                      'executable': identity(executable), 'supervisor_pid': 101, 'started': False}
            receipt = dict(intent, started=True, pid=102, exit_code=0, timed_out=False,
                           stdout_EOF=True, stderr_EOF=True, seconds=0.1,
                           request_bytes=len(request), request_sha256=sha(request),
                           stdout_bytes=len(stdout), stdout_sha256=sha(stdout),
                           stderr_bytes=0, stderr_sha256=sha(b''))
            dump(path/'native-0000.intent.json', intent)
            (path/'native-0000.request.bin').write_bytes(request)
            (path/'native-0000.stdout.bin').write_bytes(stdout)
            (path/'native-0000.stderr.bin').write_bytes(b'')
            dump(path/'native-0000.process.json', receipt)
            self.assertEqual(receipt, read_process(path, 'native-0000', executable, request, stdout, timeout=30))
            mutations = [('exit_code', 1), ('timed_out', True), ('stdout_EOF', False),
                         ('stderr_EOF', False), ('pid', 0), ('started', False),
                         ('request_bytes', len(request)+1), ('request_sha256', '0'*64),
                         ('stdout_bytes', len(stdout)+1), ('stdout_sha256', '0'*64)]
            for key, value in mutations:
                with self.subTest(field=key):
                    changed = dict(receipt, **{key: value}); dump(path/'native-0000.process.json', changed)
                    with self.assertRaises(ValueError):
                        read_process(path, 'native-0000', executable, request, stdout, timeout=30)
            dump(path/'native-0000.process.json', receipt)
            for suffix, bad in [('request.bin', request+b'x'), ('stdout.bin', stdout+b'x'), ('stderr.bin', b'x')]:
                item = path/('native-0000.'+suffix); saved = item.read_bytes(); item.write_bytes(bad)
                with self.subTest(stream=suffix), self.assertRaises(ValueError):
                    read_process(path, 'native-0000', executable, request, stdout, timeout=30)
                item.write_bytes(saved)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *rest])
