"""Fan-out tests — focused on the comb-write path and the G8 race guard.

Regression for the G8 race: combs_dir is created once before the worker pool
launches, but each worker's call_worker can run for minutes. If the dir is
removed externally in that window (a concurrent run sharing the default
workdir, tmp cleanup), the comb write used to die with FileNotFoundError and
abort the whole pipeline. The guard re-creates the parent right before writing.
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

from hive import fanout
from hive.providers import WorkerResult


class TestFanoutG8Race(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="hive_fanout_test_")
        self.combs_dir = os.path.join(self.workdir, "combs")

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_comb_write_survives_combs_dir_removed_mid_worker(self):
        """Worker deletes combs/ during its run; the write must still succeed."""
        axes = [{"id": "G8", "title": "last axis", "brief": "b"}]

        def fake_call_worker(provider, model, prompt, cwd=None, timeout=600, **kw):
            # Simulate the external race: combs/ vanishes while the worker runs,
            # i.e. after run_fanout created it but before we write the comb.
            shutil.rmtree(self.combs_dir, ignore_errors=True)
            self.assertFalse(os.path.isdir(self.combs_dir))
            return WorkerResult(stdout='{"axis_id":"G8"}', stderr="", exit_code=0,
                                latency_s=0.1)

        with mock.patch.object(fanout, "call_worker", fake_call_worker):
            comb_files = fanout.run_fanout(
                axes=axes, seed_text="seed", codebase_root=self.workdir,
                workdir=self.workdir, max_workers=1,
            )

        comb_path = comb_files["G8"]
        self.assertTrue(os.path.exists(comb_path), "comb file should be written despite the race")
        with open(comb_path, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"axis_id":"G8"}')

    def test_all_axes_written_when_dir_stable(self):
        """Baseline: every axis produces a comb file when nothing interferes."""
        axes = [{"id": f"G{i}", "title": f"axis {i}", "brief": "b"} for i in range(1, 4)]

        def fake_call_worker(provider, model, prompt, cwd=None, timeout=600, **kw):
            return WorkerResult(stdout="ok", stderr="", exit_code=0, latency_s=0.1)

        with mock.patch.object(fanout, "call_worker", fake_call_worker):
            comb_files = fanout.run_fanout(
                axes=axes, seed_text="seed", codebase_root=self.workdir,
                workdir=self.workdir, max_workers=2,
            )

        self.assertEqual(set(comb_files), {"G1", "G2", "G3"})
        for path in comb_files.values():
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
