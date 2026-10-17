"""Test source verification across documentation-only edits."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.source_provenance import ARCHIVE, verify_source


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = b'"""Original description."""\n# Original comment\ndef value():\n    """Return a number."""\n    return 7\n'
        self.expected = hashlib.sha256(self.original).hexdigest()
        self.path = self.root / 'sample.py'
        self.path.write_bytes(self.original)
        archive = self.root / ARCHIVE
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, 'w') as handle:
            handle.writestr('sample.py', self.original)

    def test_comment_and_docstring_edits_are_allowed(self):
        self.path.write_text('"""Short description."""\ndef value():\n    return 7\n', encoding='utf-8')
        self.assertEqual(verify_source(self.root, 'sample.py', self.expected), self.original)

    def test_changed_calculation_is_rejected(self):
        self.path.write_bytes(self.original.replace(b'return 7', b'return 8'))
        with self.assertRaisesRegex(ValueError, 'Executable source differs'):
            verify_source(self.root, 'sample.py', self.expected)

    def test_changed_runtime_string_is_rejected(self):
        self.path.write_bytes(self.original.replace(b'return 7', b'return "seven"'))
        with self.assertRaisesRegex(ValueError, 'Executable source differs'):
            verify_source(self.root, 'sample.py', self.expected)

    def test_missing_working_source_is_rejected(self):
        self.path.unlink()
        with self.assertRaises(OSError):
            verify_source(self.root, 'sample.py', self.expected)

    def test_changed_snapshot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Original frozen source changed'):
            verify_source(self.root, 'sample.py', '0' * 64)

    def test_path_outside_root_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_source(self.root, '../sample.py', self.expected)

    def test_without_snapshot_exact_bytes_are_required(self):
        (self.root / ARCHIVE).unlink()
        self.assertEqual(verify_source(self.root, 'sample.py', self.expected), self.original)
        self.path.write_bytes(self.original + b'# Another comment\n')
        with self.assertRaisesRegex(ValueError, 'Original frozen source changed'):
            verify_source(self.root, 'sample.py', self.expected)


if __name__ == '__main__':
    unittest.main()
