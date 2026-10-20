from pathlib import Path
import errno
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gw._files import install_verified_file


class FileInstallation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.target = self.root / 'source', self.root / 'target'
        self.body = bytes(range(256)) * 8193
        self.source.write_bytes(self.body)

    def test_local_installation(self):
        install_verified_file(self.source, self.target)
        self.assertEqual(self.target.read_bytes(), self.body)
        self.assertEqual(self.source.read_bytes(), self.body)

    def test_unsupported_link_uses_streamed_copy(self):
        for code in (errno.ENOTSUP, errno.EPERM, errno.EXDEV, errno.ENOSYS):
            with self.subTest(errno=code), patch('gw._files.os.link', side_effect=OSError(code, 'link')):
                install_verified_file(self.source, self.target)
                self.assertEqual(self.target.read_bytes(), self.body)
                self.target.unlink()

    def test_existing_destination_is_preserved(self):
        self.target.write_bytes(b'user data')
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                if fallback:
                    with patch('gw._files.os.link', side_effect=OSError(errno.ENOTSUP, 'link')):
                        with self.assertRaises(FileExistsError):
                            install_verified_file(self.source, self.target)
                else:
                    with self.assertRaises(FileExistsError):
                        install_verified_file(self.source, self.target)
                self.assertEqual(self.target.read_bytes(), b'user data')

    def test_existing_symlink_is_preserved(self):
        outside = self.root / 'outside'
        outside.write_bytes(b'user data')
        try:
            self.target.symlink_to(outside)
        except OSError as error:
            if error.errno in (errno.ENOTSUP, errno.EPERM):
                self.skipTest('Filesystem has no symbolic links')
            raise
        with patch('gw._files.os.link', side_effect=OSError(errno.ENOTSUP, 'link')):
            with self.assertRaises(FileExistsError):
                install_verified_file(self.source, self.target)
        self.assertEqual(outside.read_bytes(), b'user data')
        self.assertTrue(self.target.is_symlink())

    def test_copy_failure_cleans_created_destination(self):
        def fail(source, output, **kwargs):
            output.write(b'partial')
            raise OSError(errno.ENOSPC, 'disk full')
        with patch('gw._files.os.link', side_effect=OSError(errno.ENOTSUP, 'link')):
            with patch('gw._files.shutil.copyfileobj', side_effect=fail):
                with self.assertRaises(OSError):
                    install_verified_file(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.source.read_bytes(), self.body)

    def test_disk_reserve_is_preserved(self):
        with patch('gw._files.os.link', side_effect=OSError(errno.ENOTSUP, 'link')):
            with patch('gw._files.shutil.disk_usage', return_value=SimpleNamespace(free=len(self.body))):
                with self.assertRaises(RuntimeError):
                    install_verified_file(self.source, self.target, reserve_bytes=1)
        self.assertFalse(self.target.exists())

    def test_concurrent_replacement_survives_copy_failure(self):
        replacement = self.root / 'replacement'
        replacement.write_bytes(b'user replacement')
        def fail(source, output, **kwargs):
            output.write(b'partial')
            replacement.replace(self.target)
            raise OSError(errno.EIO, 'io')
        with patch('gw._files.os.link', side_effect=OSError(errno.ENOTSUP, 'link')):
            with patch('gw._files.shutil.copyfileobj', side_effect=fail):
                with self.assertRaises(OSError):
                    install_verified_file(self.source, self.target)
        self.assertEqual(self.target.read_bytes(), b'user replacement')

    def test_other_link_errors_are_reported(self):
        with patch('gw._files.os.link', side_effect=OSError(errno.EIO, 'io')):
            with self.assertRaises(OSError):
                install_verified_file(self.source, self.target)
        self.assertFalse(self.target.exists())


if __name__ == '__main__':
    unittest.main()
