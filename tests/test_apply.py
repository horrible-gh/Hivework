"""Unit tests for hive.apply — edit-spec → proposal (Stage-1: propose only).

apply calls no worker; these tests use a real temp codebase on disk. Coverage:
  ① render_unified_diff produces a/b-labelled +/- hunks
  ② evaluate_edit: applicable (unique anchor) | ambiguous | missing | drift |
     already_applied | file_missing | no_change
  ③ build_proposal is ready ONLY when termination=ready_to_apply and every edit
     is applicable; deferred-only / drifted / ambiguous specs are not ready
  ④ Stage-1 safety: run_apply never writes to the target codebase, even when the
     spec sets gate.apply = True
  ⑤ run_apply writes the proposal markdown and resolves codebase_root from the
     spec when not passed explicitly
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import apply


def _spec(edits, termination="ready_to_apply", deferred=None, gate=None,
          codebase_root="/code"):
    return {
        "source_honey": "honey.md",
        "codebase_root": codebase_root,
        "edits": edits,
        "deferred": deferred or [],
        "gate": gate if gate is not None else {"commands": ["pytest x"], "apply": False},
        "termination": termination,
        "notes": "",
    }


def _edit(anchor_old, replacement_new, file="a.py", eid="E1",
          anchor_status="verified"):
    return {
        "id": eid, "file": file,
        "anchor_old": anchor_old, "replacement_new": replacement_new,
        "rationale": "fix", "evidence": [f"{file}:1"],
        "confidence": "high", "anchor_status": anchor_status,
    }


class TestRenderUnifiedDiff(unittest.TestCase):
    def test_diff_has_labels_and_changes(self):
        diff = apply.render_unified_diff(
            "a.py", "x = 1\ny = 2\n", "x = 2\ny = 2\n")
        self.assertIn("a/a.py", diff)
        self.assertIn("b/a.py", diff)
        self.assertIn("-x = 1", diff)
        self.assertIn("+x = 2", diff)


class TestEvaluateEdit(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def _write(self, name, text):
        with open(os.path.join(self.root, name), "w", encoding="utf-8") as f:
            f.write(text)

    def test_applicable_when_anchor_unique(self):
        self._write("a.py", "before\nx = 1\nafter\n")
        r = apply.evaluate_edit(_edit("x = 1", "x = 2"), self.root)
        self.assertEqual(r["status"], apply.APPLICABLE)
        self.assertTrue(r["applicable"])
        self.assertIn("+x = 2", r["diff"])

    def test_ambiguous_when_anchor_repeats(self):
        self._write("a.py", "x = 1\nx = 1\n")
        r = apply.evaluate_edit(_edit("x = 1", "x = 2"), self.root)
        self.assertEqual(r["status"], apply.ANCHOR_AMBIGUOUS)
        self.assertFalse(r["applicable"])

    def test_missing_anchor(self):
        self._write("a.py", "totally different\n")
        r = apply.evaluate_edit(_edit("x = 1", "x = 2"), self.root)
        self.assertEqual(r["status"], apply.ANCHOR_MISSING)
        self.assertFalse(r["applicable"])

    def test_drift_flagged_when_verified_but_absent(self):
        self._write("a.py", "totally different\n")
        r = apply.evaluate_edit(
            _edit("x = 1", "x = 2", anchor_status="verified"), self.root)
        self.assertEqual(r["status"], apply.ANCHOR_MISSING)
        self.assertTrue(any("DRIFT" in m for m in r["messages"]))

    def test_already_applied(self):
        self._write("a.py", "x = 2\n")
        r = apply.evaluate_edit(_edit("x = 1", "x = 2"), self.root)
        self.assertEqual(r["status"], apply.ALREADY_APPLIED)
        self.assertFalse(r["applicable"])

    def test_file_missing(self):
        r = apply.evaluate_edit(_edit("x = 1", "x = 2", file="nope.py"), self.root)
        self.assertEqual(r["status"], apply.FILE_MISSING)

    def test_no_change_when_anchor_equals_replacement(self):
        self._write("a.py", "x = 1\n")
        r = apply.evaluate_edit(_edit("x = 1", "x = 1"), self.root)
        self.assertEqual(r["status"], apply.NO_CHANGE)
        self.assertFalse(r["applicable"])


class TestBuildProposal(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        with open(os.path.join(self.root, "a.py"), "w", encoding="utf-8") as f:
            f.write("before\nx = 1\nafter\n")

    def test_ready_when_all_applicable_and_ready_termination(self):
        spec = _spec([_edit("x = 1", "x = 2")])
        p = apply.build_proposal(spec, self.root)
        self.assertTrue(p["ready"])
        self.assertEqual(p["n_applicable"], 1)
        self.assertEqual(p["not_ready_reasons"], [])

    def test_not_ready_when_termination_not_ready(self):
        spec = _spec([_edit("x = 1", "x = 2")], termination="needs_reinvestigation")
        p = apply.build_proposal(spec, self.root)
        self.assertFalse(p["ready"])

    def test_not_ready_when_no_edits(self):
        spec = _spec([], deferred=[{"issue": "X", "reason": "policy_direction"}])
        p = apply.build_proposal(spec, self.root)
        self.assertFalse(p["ready"])
        self.assertTrue(any("no edits" in r for r in p["not_ready_reasons"]))

    def test_not_ready_when_one_edit_drifted(self):
        spec = _spec([
            _edit("x = 1", "x = 2", eid="E1"),
            _edit("gone = 0", "gone = 1", eid="E2"),  # not in file
        ])
        p = apply.build_proposal(spec, self.root)
        self.assertFalse(p["ready"])
        self.assertEqual(p["n_applicable"], 1)
        self.assertTrue(any("E2" in r for r in p["not_ready_reasons"]))

    def test_gate_apply_true_is_flagged(self):
        spec = _spec([_edit("x = 1", "x = 2")],
                     gate={"commands": [], "apply": True})
        p = apply.build_proposal(spec, self.root)
        self.assertTrue(any("gate.apply" in r for r in p["not_ready_reasons"]))


class TestRunApply(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.target = os.path.join(self.root, "a.py")
        self._original = "before\nx = 1\nafter\n"
        with open(self.target, "w", encoding="utf-8") as f:
            f.write(self._original)
        self.tmp = tempfile.mkdtemp()
        self.spec_path = os.path.join(self.tmp, "spec.json")
        self.out = os.path.join(self.tmp, "proposal.md")

    def _write_spec(self, spec):
        with open(self.spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f)

    def test_writes_proposal_and_resolves_root_from_spec(self):
        self._write_spec(_spec([_edit("x = 1", "x = 2")], codebase_root=self.root))
        proposal = apply.run_apply(self.spec_path, output_path=self.out)
        self.assertTrue(proposal["ready"])
        self.assertTrue(os.path.exists(self.out))
        with open(self.out, encoding="utf-8") as f:
            md = f.read()
        self.assertIn("READY TO APPLY", md)
        self.assertIn("```diff", md)

    def test_never_writes_to_target_codebase(self):
        # Even a gate.apply=True spec must leave the live file untouched.
        self._write_spec(_spec([_edit("x = 1", "x = 2")],
                               gate={"commands": [], "apply": True},
                               codebase_root=self.root))
        apply.run_apply(self.spec_path, output_path=self.out)
        with open(self.target, encoding="utf-8") as f:
            self.assertEqual(f.read(), self._original)  # unchanged

    def test_missing_root_raises(self):
        spec = _spec([_edit("x = 1", "x = 2")])
        spec.pop("codebase_root")
        self._write_spec(spec)
        with self.assertRaises(ValueError):
            apply.run_apply(self.spec_path)

    def test_no_output_path_still_returns_proposal(self):
        self._write_spec(_spec([_edit("x = 1", "x = 2")], codebase_root=self.root))
        proposal = apply.run_apply(self.spec_path)
        self.assertTrue(proposal["ready"])


class TestRunApplyWrite(unittest.TestCase):
    """--write path: applies READY edits, backs up, rolls back, refuses non-ready."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.target = os.path.join(self.root, "a.py")
        self._original = "before\nx = 1\nafter\n"
        with open(self.target, "w", encoding="utf-8") as f:
            f.write(self._original)
        self.tmp = tempfile.mkdtemp()
        self.spec_path = os.path.join(self.tmp, "spec.json")
        self.out = os.path.join(self.tmp, "proposal.md")
        self.backup_root = tempfile.mkdtemp()

    def _write_spec(self, spec):
        with open(self.spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f)

    def _read_target(self):
        with open(self.target, encoding="utf-8") as f:
            return f.read()

    def test_write_applies_ready_edit_and_backs_up(self):
        self._write_spec(_spec([_edit("x = 1", "x = 2")], codebase_root=self.root))
        proposal = apply.run_apply(
            self.spec_path, output_path=self.out, write=True,
            backup_root=self.backup_root, ttl_hours=24)
        self.assertEqual(self._read_target(), "before\nx = 2\nafter\n")
        w = proposal["write"]
        self.assertTrue(w["ok"])
        self.assertIn("a.py", w["written"])
        # Backup bundle holds the pre-write original.
        self.assertTrue(os.path.isdir(w["bundle"]))
        saved = os.path.join(w["bundle"], "files", "a.py")
        with open(saved, encoding="utf-8") as f:
            self.assertEqual(f.read(), self._original)
        with open(self.out, encoding="utf-8") as f:
            self.assertIn("WRITTEN to live codebase", f.read())

    def test_not_ready_proposal_is_never_written(self):
        # Drifted second edit makes the proposal not ready → no write at all.
        self._write_spec(_spec([
            _edit("x = 1", "x = 2", eid="E1"),
            _edit("gone = 0", "gone = 1", eid="E2"),
        ], codebase_root=self.root))
        proposal = apply.run_apply(
            self.spec_path, write=True, backup_root=self.backup_root)
        self.assertFalse(proposal["ready"])
        self.assertEqual(self._read_target(), self._original)  # untouched
        self.assertFalse(proposal["write"]["attempted"])

    def test_default_is_propose_only(self):
        self._write_spec(_spec([_edit("x = 1", "x = 2")], codebase_root=self.root))
        proposal = apply.run_apply(self.spec_path)  # write defaults to False
        self.assertEqual(self._read_target(), self._original)
        self.assertIsNone(proposal["write"])

    def test_write_without_backup_root_raises(self):
        self._write_spec(_spec([_edit("x = 1", "x = 2")], codebase_root=self.root))
        with self.assertRaises(ValueError):
            apply.run_apply(self.spec_path, write=True)

    def test_multiple_edits_same_file_compose(self):
        with open(self.target, "w", encoding="utf-8") as f:
            f.write("a = 1\nb = 2\n")
        self._write_spec(_spec([
            _edit("a = 1", "a = 10", eid="E1"),
            _edit("b = 2", "b = 20", eid="E2"),
        ], codebase_root=self.root))
        proposal = apply.run_apply(
            self.spec_path, write=True, backup_root=self.backup_root)
        self.assertTrue(proposal["write"]["ok"])
        self.assertEqual(self._read_target(), "a = 10\nb = 20\n")


if __name__ == "__main__":
    unittest.main()
