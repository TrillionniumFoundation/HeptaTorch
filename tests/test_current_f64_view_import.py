"""Verify the F64 view adapter is imported from the exact isolated checkout."""
import importlib.util
import os
import py_compile
import subprocess
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/current_runtime_f64_fixtures.py'
spec = importlib.util.spec_from_file_location('isolated_current_f64_fixtures', FIXTURES)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class CurrentF64ViewImportTests(unittest.TestCase):
    def test_resolves_local_package_with_tests_path_absent(self):
        tests = str(ROOT / 'tests')
        original = sys.path[:]
        try:
            sys.path[:] = [path for path in sys.path if path != tests]
            snapshot = sys.path[:]
            with fixtures.load_view_adapter(ROOT) as adapter:
                self.assertEqual(adapter.__module__, 'f64_view.legacy_binding')
                self.assertEqual(Path(sys.modules[adapter.__module__].__file__).resolve(),
                                 ROOT / 'tests/f64_view/legacy_binding.py')
                from f64_view import evidence
                self.assertEqual(Path(evidence.__file__).resolve(),
                                 ROOT / 'tests/f64_view/evidence.py')
            self.assertEqual(snapshot, sys.path)
        finally:
            sys.path[:] = original

    def test_scope_restores_path_after_reader_failure(self):
        snapshot = sys.path[:]
        with self.assertRaisesRegex(RuntimeError, 'reader failure'):
            with fixtures.load_view_adapter(ROOT):
                raise RuntimeError('reader failure')
        self.assertEqual(snapshot, sys.path)

    def test_stale_cached_bytecode_does_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            package = repository / 'tests/f64_view'
            package.mkdir(parents=True)
            (package / '__init__.py').write_text('')
            binding = package / 'legacy_binding.py'
            source = 'def adapter_type(): return "source"\n'
            decoy = 'def adapter_type(): return "pyc!!!"\n'
            self.assertEqual(len(source), len(decoy))
            binding.write_text(decoy)
            metadata = binding.stat()
            py_compile.compile(str(binding), doraise=True,
                               invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
            binding.write_text(source)
            os.utime(binding, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
            program = """
import importlib.util
import sys
from pathlib import Path

repository = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('fixture', sys.argv[2])
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
if sys.argv[3] == 'ordinary':
    sys.path.insert(0, str(repository / 'tests'))
    from f64_view.legacy_binding import adapter_type
    print(adapter_type())
else:
    with fixtures.load_view_adapter(repository) as adapter:
        print(adapter())
"""
            def run(mode):
                return subprocess.run([sys.executable, '-I', '-S', '-B', '-c', program,
                                       str(repository), str(FIXTURES), mode],
                                      capture_output=True, check=True, text=True).stdout.strip()
            self.assertEqual(run('ordinary'), 'pyc!!!')
            self.assertEqual(run('bound'), 'source')

    def test_refuses_cached_foreign_package(self):
        previous = sys.modules.get('f64_view')
        wrong = types.ModuleType('f64_view')
        wrong.__file__ = str(ROOT / 'tests/not_f64_view.py')
        sys.modules['f64_view'] = wrong
        snapshot = sys.path[:]
        try:
            with self.assertRaisesRegex(ValueError, 'cached F64 view module was not source-loaded from the bound repository'):
                with fixtures.load_view_adapter(ROOT):
                    self.fail("foreign package was admitted")
            self.assertEqual(snapshot, sys.path)
        finally:
            if previous is None:
                sys.modules.pop('f64_view', None)
            else:
                sys.modules['f64_view'] = previous


if __name__ == '__main__':
    unittest.main()
