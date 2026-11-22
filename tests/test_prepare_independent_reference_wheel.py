"""Physical custody regression tests; all archives here are small test bodies."""

import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import current_runtime_f64_readback as reader
import prepare_independent_reference_wheel as wheel

BODY = b'test fixed wheel bytes'
MODE = 0o640


class DiskRead:
    def bind(self, path, expected):
        return path.read_bytes()


class WheelCustody(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.repo = Path(directory) / 'repo'
        self.cache = Path(directory) / 'cache'
        self.body = self.cache / 'http-v2/a/b/official.body'
        self.body.parent.mkdir(parents=True)
        self.body.write_bytes(BODY)
        self.body.chmod(MODE)
        self.target = self.repo / wheel.custody.WHEEL_ARCHIVE_PATH
        self.target.parent.mkdir(parents=True)
        self.enterContext(patch.multiple(wheel.custody, WHEEL_BYTES=len(BODY),
                                        WHEEL_SHA256=wheel.custody.sha(BODY)))

    def require_independent(self):
        self.assertEqual(self.target.stat().st_nlink, 1)
        self.assertEqual(self.body.stat().st_dev, self.target.stat().st_dev)
        self.assertNotEqual(self.body.stat().st_ino, self.target.stat().st_ino)
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), MODE)
        self.assertEqual(reader.Bound(DiskRead()).read(
            self.target, wheel.custody.WHEEL_SHA256, len(BODY), MODE), BODY)

    def test_same_filesystem_cache_with_wrong_same_size_candidate(self):
        self.body.with_name('first.body').write_bytes(bytes(len(BODY)))
        self.assertEqual(wheel.materialize(self.repo, cache=self.cache), self.target)
        self.require_independent()

    def test_preparation_reuses_exact_independent_entity(self):
        wheel.materialize(self.repo, wheel=self.body)
        identity = self.target.stat()
        self.assertEqual(wheel.materialize(self.repo, cache=self.cache / 'missing'), self.target)
        self.assertEqual(self.target.stat().st_ino, identity.st_ino)
        self.assertEqual(self.target.stat().st_mtime_ns, identity.st_mtime_ns)
        self.require_independent()

    def test_old_hardlink_fails_reader_before_migration(self):
        os.link(self.body, self.target)
        with self.assertRaisesRegex(ValueError, 'unique ordinary physical evidence'):
            reader.Bound(DiskRead()).read(self.target, wheel.custody.WHEEL_SHA256)
        inode = self.body.stat().st_ino
        self.assertEqual(wheel.materialize(self.repo, cache=self.cache / 'missing'), self.target)
        self.require_independent()
        self.assertEqual(self.body.stat().st_ino, inode)
        self.assertEqual(self.body.read_bytes(), BODY)

    def test_wrong_concurrent_target_is_never_clobbered(self):
        wrong = bytes(len(BODY))
        link = os.link

        def collide(source, destination):
            destination.write_bytes(wrong)
            link(source, destination)

        with patch.object(wheel.os, 'link', side_effect=collide):
            with self.assertRaisesRegex(ValueError, 'concurrent target is not the exact'):
                wheel.materialize(self.repo, self.body)
        self.assertEqual(self.target.read_bytes(), wrong)
        self.assertFalse(list(self.target.parent.glob('*.materializing-*')))

    def test_wrong_existing_regular_file_is_not_replaced(self):
        wrong = bytes(len(BODY))
        self.target.write_bytes(wrong)
        with self.assertRaisesRegex(ValueError, 'existing original wheel archive differs'):
            wheel.materialize(self.repo, self.body)
        self.assertEqual(self.target.read_bytes(), wrong)

    def test_symlink_is_not_followed_or_replaced(self):
        self.target.symlink_to(self.body)
        with self.assertRaisesRegex(ValueError, 'regular original installed source entity'):
            wheel.materialize(self.repo)
        self.assertTrue(self.target.is_symlink())

    def test_no_exact_cache_bytes_leaves_target_absent(self):
        self.body.write_bytes(bytes(len(BODY)))
        with self.assertRaisesRegex(ValueError, 'exact original wheel absent'):
            wheel.materialize(self.repo, cache=self.cache)
        self.assertFalse(self.target.exists())


if __name__ == '__main__':
    unittest.main()
