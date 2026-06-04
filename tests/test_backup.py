"""Unit tests for hive.backup — the scratch undo store behind apply --write.

Coverage:
  ① create_bundle snapshots originals verbatim and writes a manifest
  ② restore_bundle copies the saved originals back over the codebase
  ③ purge_expired removes only bundles past their manifest expiry
  ④ latest_bundle returns the most recent bundle
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import backup


class TestBackupStore(unittest.TestCase):
    def setUp(self):
        self.code = tempfile.mkdtemp()
        self.store = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.code, "pkg"), exist_ok=True)
        self.rel = os.path.join("pkg", "a.py")
        # Write the fixture in BINARY with an explicit LF so the on-disk bytes are
        # deterministic across platforms (text mode would emit CRLF on Windows).
        self._original = "x = 1\n"
        with open(os.path.join(self.code, self.rel), "wb") as f:
            f.write(self._original.encode("utf-8"))

    def test_create_bundle_snapshots_and_manifests(self):
        rec = backup.create_bundle(self.store, "spec.json", self.code,
                                   [self.rel], ttl_hours=24)
        self.assertTrue(os.path.isdir(rec["dir"]))
        # originals are now snapshotted as verbatim BYTES (EOL-faithful undo).
        self.assertEqual(rec["originals"][self.rel], self._original.encode("utf-8"))
        saved = os.path.join(rec["dir"], backup.FILES_SUBDIR, self.rel)
        with open(saved, encoding="utf-8") as f:
            self.assertEqual(f.read(), self._original)
        manifest = backup.load_manifest(rec["dir"])
        self.assertEqual(manifest["files"], [self.rel])
        self.assertEqual(manifest["ttl_hours"], 24)

    def test_restore_puts_originals_back(self):
        rec = backup.create_bundle(self.store, "spec.json", self.code,
                                   [self.rel], ttl_hours=24)
        # Mutate live file, then restore from the bundle.
        with open(os.path.join(self.code, self.rel), "w", encoding="utf-8") as f:
            f.write("x = 999\n")
        restored = backup.restore_bundle(rec["dir"])
        self.assertEqual(len(restored), 1)
        with open(os.path.join(self.code, self.rel), encoding="utf-8") as f:
            self.assertEqual(f.read(), self._original)

    def test_snapshot_and_restore_are_byte_exact_regardless_of_eol(self):
        # Regression (FlowGate EOL drift): a backup must preserve the file's exact
        # bytes — including CRLF / LF / BOM — so a restore reproduces pre-edit state
        # byte-for-byte. Text-mode IO would re-normalize EOL to the host's os.linesep.
        cases = {
            os.path.join("crlf.py"): b"a = 1\r\nb = 2\r\n",          # Windows EOL
            os.path.join("lf.ts"): b"const a = 1\nconst b = 2\n",    # Unix EOL
            os.path.join("bom.py"): b"\xef\xbb\xbfx = 1\r\n",        # UTF-8 BOM + CRLF
        }
        for rel, data in cases.items():
            with open(os.path.join(self.code, rel), "wb") as f:
                f.write(data)
        rec = backup.create_bundle(self.store, "spec.json", self.code,
                                   list(cases), ttl_hours=24)
        for rel, data in cases.items():
            self.assertEqual(rec["originals"][rel], data)  # in-memory rollback bytes
            snap = os.path.join(rec["dir"], backup.FILES_SUBDIR, rel)
            with open(snap, "rb") as f:
                self.assertEqual(f.read(), data)           # on-disk snapshot bytes
        # Clobber every file, then restore and confirm byte-for-byte recovery.
        for rel in cases:
            with open(os.path.join(self.code, rel), "wb") as f:
                f.write(b"// trashed\n")
        backup.restore_bundle(rec["dir"])
        for rel, data in cases.items():
            with open(os.path.join(self.code, rel), "rb") as f:
                self.assertEqual(f.read(), data)

    def test_purge_removes_only_expired(self):
        past = datetime.now(timezone.utc) - timedelta(hours=48)
        fresh = backup.create_bundle(self.store, "fresh.json", self.code,
                                     [self.rel], ttl_hours=24)
        old = backup.create_bundle(self.store, "old.json", self.code,
                                   [self.rel], ttl_hours=1, now=past)
        removed = backup.purge_expired(self.store)
        self.assertIn(os.path.basename(old["dir"]), removed)
        self.assertNotIn(os.path.basename(fresh["dir"]), removed)
        self.assertTrue(os.path.isdir(fresh["dir"]))
        self.assertFalse(os.path.isdir(old["dir"]))

    def test_latest_bundle(self):
        self.assertIsNone(backup.latest_bundle(self.store))
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        backup.create_bundle(self.store, "old.json", self.code, [self.rel],
                             ttl_hours=24, now=old)
        newer = backup.create_bundle(self.store, "new.json", self.code,
                                     [self.rel], ttl_hours=24)
        self.assertEqual(backup.latest_bundle(self.store), newer["dir"])


if __name__ == "__main__":
    unittest.main()
