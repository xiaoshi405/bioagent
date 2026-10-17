"""Exercise release maintenance boundaries in disposable Git repositories."""
import hashlib
from contextlib import redirect_stdout
import io
from unittest.mock import patch
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.update_release_manifest import plan_update


class ManifestMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_git("init", "-q")
        self.run_git("config", "user.name", "Offline Test")
        self.run_git("config", "user.email", "offline@example.invalid")
        content = {"README.md": "Initial documentation.\n", "LICENSE": "Test license fixture.\n",
                   "results/archived.csv": "status\nrecorded\n"}
        files = {}
        for name, text in content.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "source": "test fixture", "mode": "byte_exact_copy"}
        (self.root / "RELEASE_MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")
        self.run_git("add", ".")
        self.run_git("commit", "-qm", "Initial test fixture")

    def run_git(self, *args):
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def test_preview_updates_document_checksum_without_writing(self):
        original = (self.root / "RELEASE_MANIFEST.json").read_bytes()
        (self.root / "README.md").write_text("Updated documentation.\n", encoding="utf-8")
        manifest, report = plan_update(self.root)
        self.assertEqual(report["changed"], ["README.md"])
        self.assertEqual(manifest["files"]["README.md"]["mode"], "public_maintenance")
        self.assertEqual((self.root / "RELEASE_MANIFEST.json").read_bytes(), original)

    def test_only_tracked_new_files_are_added(self):
        (self.root / "new.md").write_text("A staged document", encoding="utf-8")
        (self.root / "untracked.md").write_text("Local notes", encoding="utf-8")
        self.run_git("add", "new.md")
        manifest, report = plan_update(self.root)
        self.assertEqual(report["added"], ["new.md"])
        self.assertNotIn("untracked.md", manifest["files"])

    def test_changing_archived_results_is_rejected(self):
        (self.root / "results/archived.csv").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Protected historical/source asset"):
            plan_update(self.root)

    def test_deleting_archived_results_is_rejected(self):
        self.run_git("rm", "results/archived.csv")
        with self.assertRaisesRegex(ValueError, "Protected historical/source asset"):
            plan_update(self.root)

    def test_staged_local_environment_is_rejected(self):
        (self.root / ".env").write_text("API_KEY=placeholder\n", encoding="utf-8")
        self.run_git("add", ".env")
        with self.assertRaisesRegex(ValueError, "Local environment file"):
            plan_update(self.root)

    def test_manuscript_administration_is_rejected(self):
        (self.root / "Title_Page.docx").write_bytes(b"test fixture")
        self.run_git("add", "Title_Page.docx")
        with self.assertRaisesRegex(ValueError, "Manuscript administration"):
            plan_update(self.root)

    def test_bad_startup_configuration_does_not_echo_its_value(self):
        from scripts import start_demo
        output = io.StringIO()
        with patch.object(start_demo, "check_environment", side_effect=ValueError("private-config-test-value")), patch("sys.argv", ["start_demo.py", "--check-only"]), redirect_stdout(output):
            status = start_demo.main()
        self.assertEqual(status, 1)
        self.assertNotIn("private-config-test-value", output.getvalue())
        self.assertIn("check_environment.py", output.getvalue())

    def test_source_archive_without_git_is_not_rewritten(self):
        folder = self.root / "source_archive"
        folder.mkdir()
        with self.assertRaisesRegex(ValueError, "Git checkout"):
            plan_update(folder)


if __name__ == "__main__":
    unittest.main()
