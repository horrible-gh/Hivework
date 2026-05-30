"""Unit tests for hive.conflict_scan — inter-comb conflict detection.

Uses real parsed comb data from smoke/loop/combs/ to verify:
  - D2's termination=needs_pm with reachable=conditional triggers conflict
  - G's termination=needs_runtime is detected
  - RECONCILE2 resolves the conditional reachability (termination=resolved)
  - The actual conflict that led to RECONCILE2 being fired is reproduced
"""

import os
import sys
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive.parse import parse_comb_file
from hive.conflict_scan import scan_conflicts

# Paths to real comb files
COMBS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "smoke", "loop", "combs"
)


class TestConflictScanWithRealCombs(unittest.TestCase):
    """Test conflict detection using real comb files from the smoke loop."""

    @classmethod
    def setUpClass(cls):
        """Parse real comb files once for all tests."""
        cls.comb_d2 = parse_comb_file(os.path.join(COMBS_DIR, "comb_D2.txt"))
        cls.comb_g = parse_comb_file(os.path.join(COMBS_DIR, "comb_G.txt"))
        cls.comb_reconcile2 = parse_comb_file(
            os.path.join(COMBS_DIR, "comb_RECONCILE2.txt")
        )

    def test_d2_has_needs_pm(self):
        """Verify D2 parsed with termination=needs_pm."""
        self.assertEqual(self.comb_d2["termination"], "needs_pm")

    def test_d2_has_conditional_reachable(self):
        """Verify D2 has a finding with reachable=conditional."""
        has_conditional = False
        for f in self.comb_d2.get("findings", []):
            if "conditional" in str(f.get("reachable", "")).lower():
                has_conditional = True
                break
        self.assertTrue(has_conditional,
                        "D2 should have a finding with reachable=conditional")

    def test_d2_triggers_unresolved_conditional(self):
        """D2 alone triggers an 'unresolved_conditional' conflict.

        D2 has termination=needs_pm (not resolved) AND a finding with
        reachable=conditional → conflict_scan should flag it.
        """
        conflicts = scan_conflicts([self.comb_d2])
        conflict_types = [c["type"] for c in conflicts]
        self.assertIn("unresolved_conditional", conflict_types,
                      f"Expected unresolved_conditional in {conflict_types}")

        # The specific conflict should reference axis D2
        uc = [c for c in conflicts if c["type"] == "unresolved_conditional"]
        self.assertTrue(any(c["axis_a"] == "D2" for c in uc),
                        f"Expected D2 in unresolved_conditional: {uc}")

    def test_d2_g_together_detect_conflicts(self):
        """D2 + G together produce conflicts — the real scenario that fired RECONCILE.

        D2: termination=needs_pm, root_cause_signal mentions queries.json:128
        G:  termination=needs_runtime, root_cause_signal mentions queries.json

        These should produce:
        - unresolved_conditional (D2 has conditional reachable + non-resolved)
        - Potentially termination_divergence if they're related
        """
        conflicts = scan_conflicts([self.comb_d2, self.comb_g])
        self.assertGreater(len(conflicts), 0,
                           "D2 + G should produce at least one conflict")

        # At minimum, D2's unresolved_conditional should be present
        conflict_types = [c["type"] for c in conflicts]
        self.assertIn("unresolved_conditional", conflict_types,
                      f"Expected unresolved_conditional: {conflict_types}")

    def test_reconcile2_resolves_conditional(self):
        """RECONCILE2 has termination=resolved — should not trigger its own conflicts."""
        conflicts = scan_conflicts([self.comb_reconcile2])
        # RECONCILE2 is resolved, so no unresolved_conditional from it
        uc_conflicts = [c for c in conflicts
                        if c["type"] == "unresolved_conditional"
                        and c["axis_a"] == "RECONCILE2"]
        self.assertEqual(len(uc_conflicts), 0,
                         "RECONCILE2 (resolved) should not trigger unresolved_conditional")

    def test_full_set_d2_g_reconcile2_fewer_conflicts(self):
        """Adding RECONCILE2 to the set should not increase unresolved_conditional.

        RECONCILE2's resolved status should at least not add new conflicts.
        """
        without = scan_conflicts([self.comb_d2, self.comb_g])
        with_r = scan_conflicts([self.comb_d2, self.comb_g, self.comb_reconcile2])

        uc_without = [c for c in without if c["type"] == "unresolved_conditional"]
        uc_with = [c for c in with_r if c["type"] == "unresolved_conditional"]

        # RECONCILE2 is resolved, so it shouldn't add new unresolved_conditional
        self.assertLessEqual(len(uc_with), len(uc_without) + 0,
                             "RECONCILE2 should not add unresolved_conditional conflicts")


class TestConflictScanSynthetic(unittest.TestCase):
    """Synthetic tests for conflict detection logic."""

    def test_no_conflicts_when_all_resolved(self):
        """No conflicts when all combs are resolved."""
        combs = [
            {"axis_id": "A", "termination": "resolved",
             "root_cause_signal": None, "findings": []},
            {"axis_id": "B", "termination": "resolved",
             "root_cause_signal": None, "findings": []},
        ]
        conflicts = scan_conflicts(combs)
        self.assertEqual(len(conflicts), 0)

    def test_root_cause_mismatch(self):
        """Two axes with different root_cause_signal should conflict."""
        combs = [
            {"axis_id": "A", "termination": "resolved",
             "root_cause_signal": "foo.py:10 — bug here",
             "findings": []},
            {"axis_id": "B", "termination": "resolved",
             "root_cause_signal": "bar.py:99 — different bug",
             "findings": []},
        ]
        conflicts = scan_conflicts(combs)
        mismatch = [c for c in conflicts if c["type"] == "root_cause_mismatch"]
        self.assertGreater(len(mismatch), 0,
                           "Different root_cause_signals should trigger mismatch")

    def test_termination_divergence_with_cross_refs(self):
        """Resolved vs needs_pm with cross-reference should trigger divergence."""
        combs = [
            {"axis_id": "X", "termination": "resolved",
             "root_cause_signal": "queries.json:128 gate",
             "cross_refs": ["Y"], "findings": []},
            {"axis_id": "Y", "termination": "needs_pm",
             "root_cause_signal": "queries.json:128 different angle",
             "cross_refs": ["X"], "findings": []},
        ]
        conflicts = scan_conflicts(combs)
        divergence = [c for c in conflicts if c["type"] == "termination_divergence"]
        self.assertGreater(len(divergence), 0,
                           "Cross-referenced resolved vs needs_pm should trigger divergence")

    def test_empty_combs_no_error(self):
        """Empty comb list produces no conflicts and no error."""
        conflicts = scan_conflicts([])
        self.assertEqual(len(conflicts), 0)

    def test_single_comb_no_mismatch(self):
        """Single comb can't have root_cause_mismatch or termination_divergence."""
        combs = [
            {"axis_id": "A", "termination": "needs_pm",
             "root_cause_signal": "some.py:1", "findings": []},
        ]
        conflicts = scan_conflicts(combs)
        mismatch = [c for c in conflicts if c["type"] == "root_cause_mismatch"]
        divergence = [c for c in conflicts if c["type"] == "termination_divergence"]
        self.assertEqual(len(mismatch), 0)
        self.assertEqual(len(divergence), 0)


if __name__ == "__main__":
    unittest.main()
