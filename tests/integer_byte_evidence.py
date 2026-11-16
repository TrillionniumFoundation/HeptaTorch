"""Record existing integer-oracle observations without computing numerical values."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Evidence:
    def __init__(self, directory, profile, reference, source, compiler, oracle, repo):
        self.directory = Path(directory).resolve()
        self.profile = profile
        self.reference = reference
        self.compilers = []
        self.cases = []
        self.current = None
        self.inputs = {}
        self.finalized = False
        self._capture_input('kernel.hepta', source)
        self._capture_input('compiler.elf', compiler)
        self._capture_input('oracle.py', oracle)
        self._capture_input('observer.py', Path(__file__))
        self._capture_input('reader.py', Path(__file__).with_name('check_integer_byte_evidence.py'))
        self._capture_input('pytorch_oracle.py', Path(repo) / 'tests/pytorch_oracle.py')
        self._capture_input('pytorch-oracle-lock.json', Path(repo) / 'compatibility/pytorch-oracle-lock.json')
        if reference == 'pytorch':
            lock = json.loads((Path(repo) / 'compatibility/pytorch-oracle-lock.json').read_bytes())
            self._capture_input('pytorch-install.json', Path(repo) / lock['oracle']['install_report'])

    def _write(self, relative, raw):
        assert isinstance(raw, bytes)
        path = PurePosixPath(relative)
        assert not path.is_absolute() and '..' not in path.parts
        target = self.directory / path
        assert not target.exists(), str(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return relative

    def _json(self, relative, value):
        return self._write(relative, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())

    def _capture_input(self, role, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        copied = self._write('evidence/sources/' + role, raw)
        self.inputs[role] = {'original_path': str(path), 'snapshot': copied,
                             'bytes': len(raw), 'sha256': _sha(raw)}

    def compile_result(self, generation, result):
        assert generation == len(self.compilers)
        prefix = 'evidence/compiler-' + str(generation)
        stdout = self._write(prefix + '.stdout.bin', result.stdout)
        stderr = self._write(prefix + '.stderr.bin', result.stderr)
        record = {'generation': generation, 'returncode': result.returncode,
                  'stdout': stdout, 'stderr': stderr}
        self._json(prefix + '.json', record)
        self.compilers.append(record)

    def begin_case(self, name):
        assert re.fullmatch(r'[a-zA-Z0-9_-]+', name)
        assert name not in {case['case'] for case in self.cases}
        self.current = {'case': name, 'references': {}}
        self.cases.append(self.current)

    def _case_path(self, filename):
        assert self.current is not None
        return 'evidence/cases/' + self.current['case'] + '/' + filename

    def request(self, raw):
        self.current['request'] = self._write(self._case_path('request.bin'), raw)

    def native_result(self, result):
        self.current['native'] = {
            'returncode': result.returncode,
            'stdout': self._write(self._case_path('response.bin'), result.stdout),
            'stderr': self._write(self._case_path('stderr.bin'), result.stderr),
        }
        self._json(self._case_path('native.json'), self.current['native'])

    def reference_bytes(self, name, raw):
        assert name in ('cpython', 'pytorch')
        assert name not in self.current['references']
        self.current['references'][name] = self._write(self._case_path('reference-' + name + '.bin'), raw)

    def expected_bytes(self, raw, kind, origin):
        assert kind in ('payload', 'full-response')
        assert origin in ('cpython', 'pytorch', 'empty-no-framework-operation', 'protocol-literal')
        self.current['expected'] = {
            'kind': kind, 'origin': origin,
            'path': self._write(self._case_path('expected.bin'), raw),
        }

    def finalize(self, report_path):
        assert not self.finalized
        self.finalized = True
        report_path = Path(report_path).resolve()
        assert report_path.parent == self.directory
        report_raw = report_path.read_bytes()
        report = json.loads(report_raw)
        unchanged = {}
        for name, record in self.inputs.items():
            current = Path(record['original_path']).read_bytes()
            unchanged[name] = len(current) == record['bytes'] and _sha(current) == record['sha256']
        files = {}
        for path in sorted(self.directory.rglob('*')):
            if path.is_file():
                mode = path.lstat().st_mode
                assert stat.S_ISREG(mode) and not path.is_symlink()
                raw = path.read_bytes()
                files[path.relative_to(self.directory).as_posix()] = {
                    'bytes': len(raw), 'sha256': _sha(raw), 'mode': stat.S_IMODE(mode),
                }
        document = {
            'schema': 'heptatorch.integer-original-byte-evidence.v1',
            'status': 'COMPLETE_RECORDED_EXECUTION' if report.get('status') == 'PASS' and all(unchanged.values()) else 'RECORDED_FAILED_EXECUTION',
            'profile': self.profile, 'reference': self.reference,
            'report': {'path': report_path.name, 'bytes': len(report_raw), 'sha256': _sha(report_raw)},
            'inputs': self.inputs, 'inputs_unchanged': unchanged,
            'compiler_invocations': self.compilers, 'cases': self.cases,
            'files': files, 'manifest_self_exclusion': 'byte-evidence.json',
            'runtime_behavior_modified': False,
            'empty_framework_operation_claimed': False,
        }
        self._json('byte-evidence.json', document)
