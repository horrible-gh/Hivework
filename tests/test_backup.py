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
        self._original = "x = 1\n"
        with open(os.path.join(self.code, self.rel), "w", encoding="utf-8") as f:
            f.write(self._original)

    def test_create_bundle_snapshots_and_manifests(self):
        rec = backup.create_bundle(self.store, "spec.json", self.code,
                                   [self.rel], ttl_hours=24)
        self.assertTrue(os.path.isdir(rec["dir"]))
        self.assertEqual(rec["originals"][self.rel], self._original)
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
