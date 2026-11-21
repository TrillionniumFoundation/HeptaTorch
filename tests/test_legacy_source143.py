"""Exercise historical source custody using actual pinned143 bytes and bad inputs."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'tools/legacy_source143/restore.py'
SPEC = importlib.util.spec_from_file_location('historical143_custody', HELPER)
RESTORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESTORE)
SOURCE_FILE = ROOT / 'src/heptatorch.hepta'
SOURCE = None


class Response(io.BytesIO):
    def __init__(self, raw, status=200, headers=None, url=RESTORE.RAW_URL):
        super().__init__(raw)
        self.status, self.url = status, url
        self.headers = headers if headers is not None else {'Content-Length': str(len(raw))}

    def geturl(self):
        return self.url


class SourceCustodyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='heptatorch-source143-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.store = self.root / 'custody'
        self.target = self.repo / 'src/heptatorch.hepta'
        self.supplied = self.root / 'original.hepta'
        self.supplied.write_bytes(SOURCE)
        self.supplied.chmod(0o644)

    def local_restore(self, **kwargs):
        return RESTORE.restore(self.repo, self.store, self.supplied, offline=True, **kwargs)

    def complete_receipt(self, record, receipt):
        self.assertEqual(json.loads(Path(receipt['path']).read_text()), record)
        self.assertEqual(hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest(), receipt['sha256'])
        for key in ('manifest_snapshot', 'restorer_snapshot'):
            path = Path(record[key]['path'])
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record[key]['sha256'])

    def test_fresh_cli_restores_exact_original_bytes_mode_and_custody(self):
        process = subprocess.run([sys.executable, '-I', '-S', '-B', str(HELPER),
            '--repository', str(self.repo), '--store', str(self.store),
            '--source-file', str(self.supplied), '--offline'], capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, b'')
        summary = json.loads(process.stdout)
        self.assertEqual(self.target.read_bytes(), SOURCE)
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o644)
        self.assertEqual((self.store / 'canonical.hepta').read_bytes(), SOURCE)
        record = json.loads(Path(summary['receipt']['path']).read_text())
        self.complete_receipt(record, summary['receipt'])
        self.assertTrue(record['source_installed'] and record['cache_installed'])
        self.assertEqual(record['verified_identity']['git_blob_sha1'], '2e994b4c15f5c6760d65aa555c0bca6c0bd88a56')
        self.assertEqual(record['verified_identity']['functions'], 143)
        self.assertFalse(record['native_executed'] or record['current_runtime_dependency_added'])

    def test_existing_source_and_cache_are_verified_without_replacement(self):
        first, first_receipt = self.local_restore()
        before = self.target.stat()
        cache_before = (self.store / 'canonical.hepta').stat()
        with mock.patch.object(RESTORE, 'download_source', side_effect=AssertionError('network must not run')):
            record, receipt = RESTORE.restore(self.repo, self.store, offline=True)
        self.assertEqual(record['status'], 'PASS_HISTORICAL143_SOURCE_CUSTODY')
        self.assertFalse(record['source_installed'] or record['cache_installed'])
        self.assertEqual(record['source_before'], record['source_after'])
        self.assertEqual(record['cache_before'], record['cache_after'])
        unchanged = lambda value: (value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        self.assertEqual(unchanged(self.target.stat()), unchanged(before))
        self.assertEqual(unchanged((self.store / 'canonical.hepta').stat()), unchanged(cache_before))
        self.assertNotEqual(receipt['path'], first_receipt['path'])
        self.complete_receipt(first, first_receipt)
        self.complete_receipt(record, receipt)

    def test_missing_target_restores_from_verified_cache_offline(self):
        self.local_restore()
        self.target.unlink()
        with mock.patch.object(RESTORE, 'download_source', side_effect=AssertionError('network must not run')):
            record, receipt = RESTORE.restore(self.repo, self.store, offline=True)
        self.assertEqual(record['status'], 'PASS_HISTORICAL143_SOURCE_CUSTODY')
        self.assertEqual(record['acquisition']['kind'], 'verified-cache')
        self.assertEqual(self.target.read_bytes(), SOURCE)
        self.assertTrue(record['source_installed'])
        self.assertFalse(record['cache_installed'])

    def test_same_size_bad_target_is_rejected_without_overwrite(self):
        self.target.parent.mkdir()
        bad = SOURCE[:-2] + bytes([SOURCE[-2] ^ 1]) + SOURCE[-1:]
        self.target.write_bytes(bad)
        before = self.target.stat()
        record, receipt = self.local_restore()
        self.assertEqual(record['status'], 'FAIL_HISTORICAL143_SOURCE_CUSTODY')
        self.assertEqual(self.target.read_bytes(), bad)
        self.assertEqual(self.target.stat().st_ino, before.st_ino)
        self.assertFalse((self.store / 'canonical.hepta').exists())
        self.assertIn('SHA-256', record['errors'][0]['message'])

    def test_bad_cache_is_rejected_even_when_existing_target_is_valid(self):
        self.local_restore()
        cache = self.store / 'canonical.hepta'
        bad = SOURCE[:-1] + b'x'
        cache.write_bytes(bad)
        record, receipt = RESTORE.restore(self.repo, self.store, offline=True)
        self.assertEqual(record['status'], 'FAIL_HISTORICAL143_SOURCE_CUSTODY')
        self.assertEqual(cache.read_bytes(), bad)
        self.assertEqual(self.target.read_bytes(), SOURCE)

    def test_source_symlink_and_hardlink_are_rejected(self):
        self.target.parent.mkdir()
        for kind in ('symbolic', 'hard'):
            with self.subTest(kind=kind):
                if kind == 'symbolic':
                    self.target.symlink_to(self.supplied)
                else:
                    os.link(self.supplied, self.target)
                record, receipt = self.local_restore()
                self.assertEqual(record['status'], 'FAIL_HISTORICAL143_SOURCE_CUSTODY')
                self.assertEqual(self.supplied.read_bytes(), SOURCE)
                self.target.unlink()

    def test_parent_directory_symlink_is_not_followed(self):
        unrelated = self.root / 'unrelated'
        unrelated.mkdir()
        (self.repo / 'src').symlink_to(unrelated, target_is_directory=True)
        record, receipt = self.local_restore()
        self.assertEqual(record['status'], 'FAIL_HISTORICAL143_SOURCE_CUSTODY')
        self.assertEqual(list(unrelated.iterdir()), [])

    def test_supplied_symlink_is_not_followed(self):
        alias = self.root / 'alias.hepta'
        alias.symlink_to(self.supplied)
        record, receipt = RESTORE.restore(self.repo, self.store, alias, offline=True)
        self.assertEqual(record['status'], 'FAIL_HISTORICAL143_SOURCE_CUSTODY')
        self.assertFalse(self.target.exists())

    def test_fifo_source_is_rejected_without_blocking(self):
        self.target.parent.mkdir()
        os.mkfifo(self.target, 0o644)
        process = subprocess.run([sys.executable, '-I', '-S', '-B', str(HELPER),
            '--repository', str(self.repo), '--store', str(self.store), '--offline'],
            capture_output=True, timeout=5)
        self.assertEqual(process.returncode, 1)
        self.assertTrue(stat.S_ISFIFO(self.target.lstat().st_mode))

    def test_wrong_mode_is_rejected_without_chmod(self):
        self.supplied.chmod(0o600)
        record, receipt = self.local_restore()
        self.assertEqual(record['status'], 'FAIL_HISTORICAL143_SOURCE_CUSTODY')
        self.assertEqual(stat.S_IMODE(self.supplied.stat().st_mode), 0o600)
        self.assertFalse(self.target.exists())

    def test_short_or_long_input_is_not_published(self):
        for bad in (SOURCE[:-1], SOURCE + b'x'):
            with self.subTest(bytes=len(bad)):
                self.supplied.write_bytes(bad)
                record, receipt = self.local_restore()
                self.assertEqual(record['status'], 'FAIL_HISTORICAL143_SOURCE_CUSTODY')
                self.assertFalse(self.target.exists())
                self.assertFalse((self.store / 'canonical.hepta').exists())

    def test_altered_manifest_is_rejected_before_source_installation(self):
        copied = self.root / 'copied-helper'
        copied.mkdir()
        shutil.copyfile(HELPER, copied / 'restore.py')
        manifest = dict(RESTORE.EXPECTED_MANIFEST, functions=144)
        (copied / 'manifest.json').write_text(json.dumps(manifest))
        process = subprocess.run([sys.executable, '-I', '-S', '-B', str(copied / 'restore.py'),
            '--repository', str(self.repo), '--store', str(self.store),
            '--source-file', str(self.supplied), '--offline'], capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 1)
        self.assertFalse(self.target.exists())
        self.assertIn('manifest differs', process.stdout.decode())

    def test_download_accepts_only_complete_exact_fixed_response(self):
        opener = mock.Mock()
        opener.open.return_value = Response(SOURCE)
        with mock.patch.object(RESTORE.urllib.request, 'build_opener', return_value=opener):
            record, receipt = RESTORE.restore(self.repo, self.store)
        self.assertEqual(record['status'], 'PASS_HISTORICAL143_SOURCE_CUSTODY')
        self.assertEqual(self.target.read_bytes(), SOURCE)
        self.assertTrue(record['acquisition']['complete_response_EOF'])
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, RESTORE.RAW_URL)
        self.assertIsNone(request.get_header('Authorization'))

    def test_download_rejects_bad_bytes_encoding_url_or_trailing_response(self):
        variants = [dict(raw=SOURCE[:-1]), dict(raw=SOURCE + b'x'),
            dict(raw=SOURCE[:-1] + b'x'), dict(raw=SOURCE, url='https://example.invalid/source'),
            dict(raw=SOURCE, headers={'Content-Encoding': 'gzip'}), dict(raw=SOURCE, status=206)]
        for variant in variants:
            with self.subTest(variant={k: len(v) if k == 'raw' else v for k, v in variant.items()}):
                opener = mock.Mock()
                opener.open.return_value = Response(**variant)
                with mock.patch.object(RESTORE.urllib.request, 'build_opener', return_value=opener):
                    record, receipt = RESTORE.restore(self.repo, self.store)
                self.assertEqual(record['status'], 'FAIL_HISTORICAL143_SOURCE_CUSTODY')
                self.assertFalse(self.target.exists())
                self.assertFalse((self.store / 'canonical.hepta').exists())

    def test_redirect_handler_refuses_even_https_redirects(self):
        with self.assertRaises(RESTORE.CustodyError):
            RESTORE.NoRedirect().redirect_request(None, None, 302, 'Found', {}, RESTORE.RAW_URL)

    def test_absent_offline_inputs_do_not_create_source_or_cache(self):
        with mock.patch.object(RESTORE, 'download_source', side_effect=AssertionError('network must not run')):
            record, receipt = RESTORE.restore(self.repo, self.store, offline=True)
        self.assertEqual(record['status'], 'FAIL_HISTORICAL143_SOURCE_CUSTODY')
        self.assertFalse(self.target.exists())
        self.assertFalse((self.store / 'canonical.hepta').exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source', type=Path, default=SOURCE_FILE)
    args, remaining = parser.parse_known_args()
    SOURCE, identity = RESTORE.read_source(args.source)
    unittest.main(argv=[sys.argv[0]] + remaining)
