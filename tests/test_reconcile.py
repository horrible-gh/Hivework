"""Unit tests for hive.reconcile — convergence bookkeeping.

Covers the fix that stops the conflict count from GROWING every round: a reconcile
round's own comb is a resolution, not a new conflict source, so it is excluded from
conflict detection (but kept in the returned combs for assemble).
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import reconcile
from hive.providers import WorkerResult


class TestReconcileHelpers(unittest.TestCase):
    def test_is_reconcile_comb(self):
        self.assertTrue(reconcile._is_reconcile_comb({"axis_id": "RECONCILE_R1"}))
        self.assertTrue(reconcile._is_reconcile_comb({"axis_id": "RECONCILE2"}))
        self.assertFalse(reconcile._is_reconcile_comb({"axis_id": "A"}))
        self.assertFalse(reconcile._is_reconcile_comb({"axis_id": ""}))

    def test_detectable_excludes_reconcile(self):
        combs = [{"axis_id": "A"}, {"axis_id": "RECONCILE_R1"}, {"axis_id": "B"}]
        self.assertEqual([c["axis_id"] for c in reconcile._detectable(combs)], ["A", "B"])


class TestReconcileLoopNoGrowth(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_reconcile_comb_is_not_a_conflict_source(self):
        # Two original axes that conflict at the SAME locus with different conclusions.
        combs = [
            {"axis_id": "A", "termination": "resolved",
             "root_cause_signal": "foo.py:10 — cause X", "findings": []},
            {"axis_id": "B", "termination": "resolved",
             "root_cause_signal": "foo.py:10 — cause Y", "findings": []},
        ]
        # The reconcile worker returns a comb that ALSO cites foo.py:10 — under the old
        # code this third voice grew the conflict count every round.
        recon = {"axis_id": "RECONCILE_R1", "termination": "resolved",
                 "root_cause_signal": "foo.py:10 — cause X is the real one",
                 "findings": []}
        wr = WorkerResult(stdout=json.dumps(recon), stderr="", exit_code=0, latency_s=0.0)
        with mock.patch.object(reconcile, "call_worker", return_value=wr):
            final, remaining, rounds = reconcile.run_reconcile_loop(
                combs=combs, comb_files={"A": "a", "B": "b"}, seed_text="seed",
                codebase_root=".", workdir=self.tmp, comb_contract="contract",
                round_cap=2)
        # The reconcile comb is kept so assemble still sees its findings...
        self.assertTrue(any(c["axis_id"] == "RECONCILE_R1" for c in final))
        # ...but it is never itself reported as a conflicting axis (no growth).
        for c in remaining:
            joined = f"{c.get('axis_a', '')}{c.get('axis_b', '')}".upper()
            self.assertNotIn("RECONCILE", joined)
        # The genuine A/B conflict still surfaces (not suppressed).
        self.assertTrue(any(c["type"] == "root_cause_mismatch" for c in remaining))


if __name__ == "__main__":
    unittest.main()
