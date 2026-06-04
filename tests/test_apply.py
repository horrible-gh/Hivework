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

    # ── post-apply verification: unique anchor but broken result (T889) ──────────

    def test_overlap_anchor_duplicates_following_lines(self):
        # anchor ends mid-block; replacement re-states the lines that FOLLOW it →
        # applying duplicates them (the exact shape that crashed in T889's E1).
        self._write("a.py",
                    "if head is not None:\n"
                    "    out['title'] = head.get('title')\n"
                    "    out['status'] = head.get('status')\n"
                    "    out['kind'] = head.get('kind')\n"
                    "elif other_condition_holds:\n"
                    "    fallback_value = compute_default()\n")
        anchor = ("if head is not None:\n"
                  "    out['title'] = head.get('title')\n")
        repl = ("if head is not None:\n"
                "    out['title'] = head.get('title')\n"
                "    out['status'] = head.get('status')\n"
                "    out['kind'] = head.get('kind')\n"
                "elif other_condition_holds:\n"
                "    fallback_value = compute_default()\n"
                "    extra_added = True\n")
        r = apply.evaluate_edit(_edit(anchor, repl), self.root)
        self.assertEqual(r["status"], apply.POST_APPLY_BROKEN)
        self.assertFalse(r["applicable"])
        self.assertTrue(any("duplicate" in m for m in r["messages"]))

    def test_python_syntax_break_is_caught(self):
        self._write("a.py", "def f():\n    return 1\n")
        r = apply.evaluate_edit(
            _edit("    return 1", "    return (1"), self.root)  # unbalanced paren
        self.assertEqual(r["status"], apply.POST_APPLY_BROKEN)
        self.assertFalse(r["applicable"])
        self.assertTrue(any("compile" in m for m in r["messages"]))

    def test_vue_template_dotvalue_is_caught(self):
        self._write("c.vue",
                    "<template>\n"
                    "  <div :class=\"flag ? 'on' : ''\"></div>\n"
                    "</template>\n"
                    "<script setup>\nconst flag = computed(() => true)\n</script>\n")
        r = apply.evaluate_edit(
            _edit("flag ? 'on' : ''", "flag.value ? 'on' : ''", file="c.vue"),
            self.root)
        self.assertEqual(r["status"], apply.POST_APPLY_BROKEN)
        self.assertFalse(r["applicable"])
        self.assertTrue(any(".value" in m for m in r["messages"]))

    def test_clean_edit_still_applicable(self):
        # a normal, sound edit must still pass the new checks.
        self._write("a.py", "x = 1\ny = 2\n")
        r = apply.evaluate_edit(_edit("x = 1", "x = 42"), self.root)
        self.assertEqual(r["status"], apply.APPLICABLE)
        self.assertTrue(r["applicable"])

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

    def test_needs_runtime_is_valid_and_not_ready(self):
        # N174: needs_runtime is first-class (investigate.py emits it). apply must accept
        # it as a known termination — no "invalid" warning — and treat it as non-ready.
        self.assertIn("needs_runtime", apply._VALID_TERMINATION)
        spec = _spec([_edit("x = 1", "x = 2")], termination="needs_runtime")
        p = apply.build_proposal(spec, self.root)
        self.assertFalse(p["ready"])
        self.assertTrue(any("needs_runtime" in r for r in p["not_ready_reasons"]))

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


class TestResolveEffectiveRoot(unittest.TestCase):
    """Code and design docs in separate trees: apply must resolve an edit's file
    against the tree it actually lives under, not blindly against --codebase
    (T890: a doc edit-spec applied with --codebase <source> file_missing'd)."""

    def setUp(self):
        self.code = tempfile.mkdtemp()   # source tree (no doc file)
        self.docs = tempfile.mkdtemp()   # design-doc tree (the doc file lives here)
        os.makedirs(os.path.join(self.docs, "210_design"), exist_ok=True)
        self.docfile = os.path.join("210_design", "D031.md")
        with open(os.path.join(self.docs, self.docfile), "w", encoding="utf-8") as f:
            f.write("before\nstate = old\nafter\n")
        self.tmp = tempfile.mkdtemp()
        self.spec_path = os.path.join(self.tmp, "spec.json")

    def _write_spec(self, spec):
        with open(self.spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f)

    def test_picks_root_holding_the_file_over_explicit_codebase(self):
        # The runner passes --codebase <source>, but the file lives under docs.
        spec = _spec([_edit("state = old", "state = new", file=self.docfile)],
                     codebase_root=self.docs)
        self._write_spec(spec)
        proposal = apply.run_apply(self.spec_path, codebase_root=self.code)
        self.assertTrue(proposal["ready"])
        self.assertEqual(proposal["codebase_root"], os.path.abspath(self.docs))

    def test_explicit_docs_root_resolves_doc_edit(self):
        # Spec's own codebase_root is wrong/absent; --docs supplies the base.
        spec = _spec([_edit("state = old", "state = new", file=self.docfile)],
                     codebase_root=self.code)
        self._write_spec(spec)
        proposal = apply.run_apply(
            self.spec_path, codebase_root=self.code, docs_root=self.docs)
        self.assertTrue(proposal["ready"])
        self.assertEqual(proposal["codebase_root"], os.path.abspath(self.docs))

    def test_unresolvable_file_keeps_first_candidate_and_reports_missing(self):
        spec = _spec([_edit("state = old", "state = new", file="nope/ghost.md")],
                     codebase_root=self.code)
        self._write_spec(spec)
        proposal = apply.run_apply(self.spec_path, codebase_root=self.code)
        self.assertFalse(proposal["ready"])
        self.assertEqual(proposal["codebase_root"], os.path.abspath(self.code))
        self.assertEqual(proposal["edits"][0]["status"], apply.FILE_MISSING)


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

    def test_write_preserves_crlf_and_restore_round_trips(self):
        # Regression (FlowGate EOL drift): a CRLF file must stay CRLF after a write,
        # and the backup must restore it byte-for-byte. The LF-based anchor still
        # matches the CRLF file because matching runs on '\n'-normalized text.
        crlf = b"before\r\nx = 1\r\nafter\r\n"
        with open(self.target, "wb") as f:
            f.write(crlf)
        self._write_spec(_spec([_edit("x = 1", "x = 2")], codebase_root=self.root))
        proposal = apply.run_apply(
            self.spec_path, write=True, backup_root=self.backup_root, ttl_hours=24)
        self.assertTrue(proposal["write"]["ok"])
        with open(self.target, "rb") as f:
            self.assertEqual(f.read(), b"before\r\nx = 2\r\nafter\r\n")  # CRLF kept
        # Snapshot is the verbatim pre-write bytes → restore is byte-exact.
        from hive import backup as _bk
        _bk.restore_bundle(proposal["write"]["bundle"])
        with open(self.target, "rb") as f:
            self.assertEqual(f.read(), crlf)

    def test_write_preserves_lf_on_any_platform(self):
        # A pure-LF file must NOT be rewritten to the host os.linesep (LF→CRLF on
        # Windows was the exact corruption that broke restore byte-fidelity).
        lf = b"const a = 1\nconst b = 2\n"
        target = os.path.join(self.root, "x.ts")
        with open(target, "wb") as f:
            f.write(lf)
        self._write_spec(_spec(
            [_edit("const a = 1", "const a = 99", file="x.ts")],
            codebase_root=self.root))
        proposal = apply.run_apply(
            self.spec_path, write=True, backup_root=self.backup_root, ttl_hours=24)
        self.assertTrue(proposal["write"]["ok"])
        with open(target, "rb") as f:
            self.assertEqual(f.read(), b"const a = 99\nconst b = 2\n")  # still LF

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


class TestPartialApply(unittest.TestCase):
    """Defect 3 (T892): a verified, effective edit must be writable even when an
    unrelated sibling item defers and flips termination to needs_reinvestigation."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.target = os.path.join(self.root, "a.py")
        self._original = "before\nx = 1\nafter\n"
        with open(self.target, "w", encoding="utf-8") as f:
            f.write(self._original)
        self.tmp = tempfile.mkdtemp()
        self.spec_path = os.path.join(self.tmp, "spec.json")
        self.backup_root = tempfile.mkdtemp()

    def _write_spec(self, spec):
        with open(self.spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f)

    def _read_target(self):
        with open(self.target, encoding="utf-8") as f:
            return f.read()

    def _deferred_spec(self):
        # E1 applicable + verified, but a sibling deferred → needs_reinvestigation.
        return _spec(
            [_edit("x = 1", "x = 2", eid="E1")],
            termination="needs_reinvestigation",
            deferred=[{"issue": "spec.ts guard", "reason": "anchor_not_grounded"}],
            codebase_root=self.root)

    def test_writable_subset_surfaced_when_sibling_defers(self):
        p = apply.build_proposal(self._deferred_spec(), self.root)
        self.assertFalse(p["ready"])
        self.assertTrue(p["partial_ready"])
        self.assertEqual(p["writable_ids"], ["E1"])

    def test_partial_write_applies_only_writable_edit(self):
        self._write_spec(self._deferred_spec())
        proposal = apply.run_apply(
            self.spec_path, write=True, partial=True,
            backup_root=self.backup_root, ttl_hours=24)
        # E1 shipped despite the non-ready termination
        self.assertEqual(self._read_target(), "before\nx = 2\nafter\n")
        self.assertTrue(proposal["applied_partial"])
        self.assertTrue(proposal["write"]["ok"])
        self.assertTrue(proposal["write"]["partial"])
        self.assertEqual(proposal["write"]["applied_ids"], ["E1"])

    def test_without_partial_flag_nothing_is_written(self):
        self._write_spec(self._deferred_spec())
        proposal = apply.run_apply(
            self.spec_path, write=True, backup_root=self.backup_root)
        self.assertEqual(self._read_target(), self._original)  # untouched
        self.assertFalse(proposal["write"]["attempted"])
        # but the operator is told a ready subset exists
        self.assertTrue(proposal["partial_ready"])

    def test_ineffective_edit_is_not_writable(self):
        # An edit the effectiveness gate flagged ineffective must NOT be partial-applied.
        spec = self._deferred_spec()
        spec["effectiveness"] = {"inconclusive": False, "ineffective_ids": ["E1"]}
        self._write_spec(spec)
        proposal = apply.run_apply(
            self.spec_path, write=True, partial=True, backup_root=self.backup_root)
        self.assertEqual(self._read_target(), self._original)  # not applied
        self.assertEqual(proposal["writable_ids"], [])
        self.assertFalse(proposal["partial_ready"])

    def test_partial_only_writes_writable_when_a_sibling_edit_drifts(self):
        # Two edits: E1 applicable, E2 drifted. Partial writes only E1.
        self._write_spec(_spec([
            _edit("x = 1", "x = 2", eid="E1"),
            _edit("gone = 0", "gone = 1", eid="E2"),  # not in file → not applicable
        ], termination="needs_reinvestigation", codebase_root=self.root))
        proposal = apply.run_apply(
            self.spec_path, write=True, partial=True, backup_root=self.backup_root)
        self.assertEqual(self._read_target(), "before\nx = 2\nafter\n")
        self.assertEqual(proposal["write"]["applied_ids"], ["E1"])
        self.assertTrue(proposal["applied_partial"])


class TestIsTestFile(unittest.TestCase):
    """Path-convention recognition that drives partial atomicity."""

    def test_recognizes_test_conventions(self):
        for p in ("tests/test_documents.py", "server/tests/foo.py",
                  "test_documents.py", "documents_test.py",
                  "client/Comp.spec.ts", "client/Comp.test.tsx",
                  "src/__tests__/x.js", "a\\tests\\b.py"):
            self.assertTrue(apply._is_test_file(p), p)

    def test_rejects_source_paths(self):
        for p in ("server/documents.py", "client/MainPanel.vue",
                  "hive/specify.py", "", "latest.py"):
            self.assertFalse(apply._is_test_file(p), p)


class TestPartialAtomicity(unittest.TestCase):
    """Defect 3 (M-head): --partial must not write a test-expectation edit ahead of the
    source edit it asserts. If a source edit is unwritable, its test siblings are held."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        # source file present; test file present
        with open(os.path.join(self.root, "documents.py"), "w", encoding="utf-8") as f:
            f.write("HEAD = {'R'}\nother = 1\n")
        os.makedirs(os.path.join(self.root, "tests"), exist_ok=True)
        with open(os.path.join(self.root, "tests", "test_documents.py"), "w",
                  encoding="utf-8") as f:
            f.write("assert head == 'DS'\n")

    def _spec_code_fails_test_ok(self):
        # E1 (source) anchor is NOT in the file → unwritable; E2 (test) IS applicable.
        return _spec([
            _edit("MISSING = {'R','M','Q'}", "MISSING = {'R','Q'}",
                  file="documents.py", eid="E1"),
            _edit("assert head == 'DS'", "assert head == 'M'",
                  file="tests/test_documents.py", eid="E2"),
        ], termination="needs_reinvestigation", codebase_root=self.root)

    def test_test_edit_held_when_source_edit_unwritable(self):
        p = apply.build_proposal(self._spec_code_fails_test_ok(), self.root)
        # the applicable test edit must NOT be writable while its source sibling failed
        self.assertEqual(p["writable_ids"], [])
        self.assertFalse(p["partial_ready"])
        e2 = next(r for r in p["edits"] if r["id"] == "E2")
        self.assertIn("held_reason", e2)

    def test_test_edit_writable_when_all_source_edits_writable(self):
        # source edit now applicable → its test sibling is free to ship in the same round
        with open(os.path.join(self.root, "documents.py"), "w", encoding="utf-8") as f:
            f.write("HEAD = {'R','M','Q'}\nother = 1\n")
        spec = _spec([
            _edit("HEAD = {'R','M','Q'}", "HEAD = {'R','Q'}",
                  file="documents.py", eid="E1"),
            _edit("assert head == 'DS'", "assert head == 'M'",
                  file="tests/test_documents.py", eid="E2"),
        ], termination="needs_reinvestigation", codebase_root=self.root)
        p = apply.build_proposal(spec, self.root)
        self.assertEqual(sorted(p["writable_ids"]), ["E1", "E2"])

    def test_pure_test_only_spec_is_unaffected(self):
        # no source edits at all → vacuously all source writable → test edit stays writable
        spec = _spec([
            _edit("assert head == 'DS'", "assert head == 'M'",
                  file="tests/test_documents.py", eid="E2"),
        ], termination="needs_reinvestigation", codebase_root=self.root)
        p = apply.build_proposal(spec, self.root)
        self.assertEqual(p["writable_ids"], ["E2"])


if __name__ == "__main__":
    unittest.main()
