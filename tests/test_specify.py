"""Unit tests for hive.specify — honey → edit-spec lowering stage.

The provider is mocked so these run without the copilot CLI. Coverage:
  ① build_specify_prompt embeds the contract, codebase root, and honey
  ② run_specify parses the author's JSON, writes it as the SSOT, returns the dict
  ③ Stage-1 safety: gate.apply is forced false even if the author set it true
  ④ A stale/not_found edit downgrades ready_to_apply → needs_reinvestigation
  ⑤ run_specify raises ValueError when the author emits no JSON
  ⑥ config exposes a 'specify' role and --model override reaches it
  ⑦ effectiveness gate: deterministic no-op + model review downgrade a ready spec
    whose edits do not change the reported behavior; inconclusive review → needs_pm
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import specify
from hive.config import load_config
from hive.providers import WorkerResult


def _wr(stdout: str) -> WorkerResult:
    return WorkerResult(stdout=stdout, stderr="", exit_code=0, latency_s=0.01)


def _review(reviews: list) -> str:
    """A well-formed effectiveness-review worker response."""
    return json.dumps({"reviews": reviews})


def _fresh_ready() -> dict:
    """A fresh deep copy of the ready spec (tests mutate it)."""
    return json.loads(json.dumps(_READY_SPEC))


_READY_SPEC = {
    "source_honey": "honey.md",
    "codebase_root": "/code",
    "edits": [{
        "id": "E1", "file": "a.py",
        "anchor_old": "x = 1", "replacement_new": "x = 2",
        "rationale": "fix", "evidence": ["a.py:1"],
        "confidence": "high", "anchor_status": "verified",
    }],
    "deferred": [],
    "gate": {"commands": ["pytest a"], "apply": False},
    "termination": "ready_to_apply",
    "notes": "",
}


class TestBuildPrompt(unittest.TestCase):
    def test_embeds_contract_codebase_and_honey(self):
        prompt = specify.build_specify_prompt(
            honey_text="HONEY_BODY",
            contract_text="CONTRACT_RULES",
            codebase_root=r"C:\code\proj",
        )
        self.assertIn("CONTRACT_RULES", prompt)
        self.assertIn("HONEY_BODY", prompt)
        self.assertIn(r"C:\code\proj", prompt)
        # The "lift anchors from live code" instruction must be present.
        self.assertIn("byte-for-byte", prompt)

    def test_grounded_only_prompt_is_tool_off(self):
        # tool-OFF author: lift from the ground-truth block, never re-open files.
        prompt = specify.build_specify_prompt(
            honey_text="H", contract_text="C", codebase_root=r"C:\code",
            grounded_only=True)
        self.assertIn("NO file-system tools", prompt)
        self.assertIn("Anchor ground truth", prompt)
        self.assertIn("anchor_not_grounded", prompt)
        self.assertNotIn("Re-open every file", prompt)

    def test_tool_on_prompt_reopens_files(self):
        prompt = specify.build_specify_prompt(
            honey_text="H", contract_text="C", codebase_root=r"C:\code")
        self.assertIn("Re-open every file", prompt)
        self.assertNotIn("NO file-system tools", prompt)

    def test_no_docs_block_when_docs_root_absent(self):
        prompt = specify.build_specify_prompt(
            honey_text="H", contract_text="C", codebase_root=r"C:\code")
        self.assertNotIn("Design-docs root", prompt)

    def test_docs_block_present_when_docs_root_given(self):
        prompt = specify.build_specify_prompt(
            honey_text="H", contract_text="C", codebase_root=r"C:\code",
            docs_root=r"C:\docs\tree")
        self.assertIn("Design-docs root", prompt)
        self.assertIn(r"C:\docs\tree", prompt)
        # The author must be told to prefer the doc over the nearest source file.
        self.assertIn("Prefer the design document", prompt)


class TestStampRoot(unittest.TestCase):
    """codebase_root is stamped with the tree that actually holds the edited
    files, so apply resolves a doc edit even without --docs on the apply call."""

    def setUp(self):
        self.code = tempfile.mkdtemp()
        self.docs = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.docs, "210_design"), exist_ok=True)
        with open(os.path.join(self.docs, "210_design", "D031.md"), "w",
                  encoding="utf-8") as f:
            f.write("doc body\n")

    def _spec(self, file):
        return {"edits": [{"id": "E1", "file": file, "kind": "edit",
                           "anchor_old": "a", "replacement_new": "b"}]}

    def test_doc_edit_stamps_docs_root(self):
        spec = self._spec(os.path.join("210_design", "D031.md"))
        self.assertEqual(specify._stamp_root(spec, self.code, self.docs), self.docs)

    def test_code_edit_stamps_codebase_root(self):
        with open(os.path.join(self.code, "a.py"), "w", encoding="utf-8") as f:
            f.write("x = 1\n")
        spec = self._spec("a.py")
        self.assertEqual(specify._stamp_root(spec, self.code, self.docs), self.code)

    def test_no_docs_root_returns_codebase(self):
        spec = self._spec("whatever.md")
        self.assertEqual(specify._stamp_root(spec, self.code, None), self.code)

    def test_create_file_only_spec_returns_codebase(self):
        spec = {"edits": [{"id": "E1", "file": "new.md", "kind": "create_file",
                           "content": "x"}]}
        self.assertEqual(specify._stamp_root(spec, self.code, self.docs), self.code)


class TestNormalizeSpec(unittest.TestCase):
    def test_forces_apply_false(self):
        spec = {"gate": {"apply": True}, "edits": [], "deferred": [],
                "termination": "needs_pm"}
        out = specify._normalize_spec(spec)
        self.assertIs(out["gate"]["apply"], False)

    def test_missing_gate_gets_apply_false(self):
        out = specify._normalize_spec({"edits": [], "deferred": []})
        self.assertIs(out["gate"]["apply"], False)

    def test_stale_edit_downgrades_ready(self):
        spec = {
            "gate": {"apply": False},
            "edits": [{"id": "E1", "anchor_status": "stale"}],
            "deferred": [],
            "termination": "ready_to_apply",
        }
        out = specify._normalize_spec(spec)
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_verified_edit_keeps_ready(self):
        spec = {
            "gate": {"apply": False},
            "edits": [{"id": "E1", "anchor_status": "verified"}],
            "deferred": [],
            "termination": "ready_to_apply",
        }
        out = specify._normalize_spec(spec)
        self.assertEqual(out["termination"], "ready_to_apply")


class TestValidateSpec(unittest.TestCase):
    def test_clean_spec_has_no_problems(self):
        self.assertEqual(specify._validate_spec(_READY_SPEC), [])

    def test_missing_keys_flagged(self):
        problems = specify._validate_spec({"edits": []})
        self.assertTrue(any("deferred" in p for p in problems))
        self.assertTrue(any("gate" in p for p in problems))

    def test_bad_termination_flagged(self):
        problems = specify._validate_spec(
            {"edits": [], "deferred": [], "gate": {}, "termination": "bogus"})
        self.assertTrue(any("termination" in p for p in problems))


class TestRunSpecify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.honey = os.path.join(self.tmp, "honey.md")
        with open(self.honey, "w", encoding="utf-8") as f:
            f.write("# honey\nFix direction: change x.\n")
        self.contract = os.path.join(self.tmp, "contract.md")
        with open(self.contract, "w", encoding="utf-8") as f:
            f.write("[Role] specify author contract")
        self.out = os.path.join(self.tmp, "spec.json")

    def _run(self, stdout: str):
        # review=False: these tests cover authoring/normalize, not the gate, so a
        # single mocked worker call is enough.
        with mock.patch.object(specify, "call_worker", return_value=_wr(stdout)):
            return specify.run_specify(
                honey_path=self.honey,
                codebase_root=self.tmp,
                output_path=self.out,
                contract_path=self.contract,
                review=False,
            )

    def test_parses_writes_and_returns(self):
        # Author wraps JSON in tool-trace lines — extract_first_json must strip them.
        stdout = "● Read a.py\n  └ done\n" + json.dumps(_READY_SPEC)
        spec = self._run(stdout)
        self.assertEqual(len(spec["edits"]), 1)
        self.assertTrue(os.path.exists(self.out))
        with open(self.out, encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["termination"], "ready_to_apply")
        self.assertIs(on_disk["gate"]["apply"], False)

    def test_forces_apply_false_end_to_end(self):
        leaky = dict(_READY_SPEC)
        leaky["gate"] = {"commands": [], "apply": True}
        spec = self._run(json.dumps(leaky))
        self.assertIs(spec["gate"]["apply"], False)

    def test_fills_provenance_when_absent(self):
        bare = {"edits": [], "deferred": [], "gate": {"apply": False},
                "termination": "needs_reinvestigation"}
        spec = self._run(json.dumps(bare))
        self.assertEqual(spec["source_honey"], self.honey)
        self.assertEqual(spec["codebase_root"], os.path.abspath(self.tmp))

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            self._run("● Read a.py\n  └ nothing parseable here\n")

    def test_author_echoed_codebase_root_is_overridden_to_docs(self):
        # The author worker echoes the prompt's CODE root into the spec, but the
        # edit targets a doc that lives under docs_root. The deterministic stamp
        # must overwrite the untrusted author value so apply can resolve the path.
        docs = tempfile.mkdtemp()
        os.makedirs(os.path.join(docs, "210_design"), exist_ok=True)
        rel = os.path.join("210_design", "D031.md")
        with open(os.path.join(docs, rel), "w", encoding="utf-8") as f:
            f.write("doc body\n")
        leaky = {
            "edits": [{"id": "E1", "file": rel, "kind": "edit",
                       "anchor_old": "doc body", "replacement_new": "doc body!",
                       "anchor_status": "verified"}],
            "deferred": [], "gate": {"apply": False},
            "termination": "ready_to_apply",
            "codebase_root": self.tmp,  # author echoed the CODE root (wrong)
        }
        with mock.patch.object(specify, "call_worker",
                               return_value=_wr(json.dumps(leaky))):
            spec = specify.run_specify(
                honey_path=self.honey, codebase_root=self.tmp,
                output_path=self.out, contract_path=self.contract,
                review=False, docs_root=docs)
        self.assertEqual(spec["codebase_root"], os.path.abspath(docs))


class TestDeterministicNoop(unittest.TestCase):
    def test_whitespace_only_change_is_noop(self):
        spec = {"edits": [
            {"id": "E1", "anchor_old": "  return x", "replacement_new": "  return x  "},
        ]}
        self.assertEqual(specify._deterministic_noop_ids(spec), ["E1"])

    def test_real_change_is_not_noop(self):
        spec = {"edits": [
            {"id": "E1", "anchor_old": "x = 1", "replacement_new": "x = 2"},
        ]}
        self.assertEqual(specify._deterministic_noop_ids(spec), [])

    def test_indentation_change_is_not_noop(self):
        # Indentation is significant (Python) — it must NOT be treated as a no-op.
        spec = {"edits": [
            {"id": "E1", "anchor_old": "if a:\n    b()", "replacement_new": "if a:\n  b()"},
        ]}
        self.assertEqual(specify._deterministic_noop_ids(spec), [])


class TestIncompleteWiring(unittest.TestCase):
    """N175: an edit that adds an import nobody uses (the binding/call site was never
    wired) is incomplete and must be flagged, not marked applicable."""

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def _write(self, name, text):
        p = os.path.join(self.root, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return p

    def test_unused_added_import_is_flagged(self):
        # the reported shape: useToast imported, but the call site uses showToast and the
        # `const { showToast } = useToast()` binding is missing → useToast is dead.
        self._write("Modal.vue",
                    "<script setup>\nconst x = 1\nshowToast('hi')\n</script>\n")
        spec = {"edits": [{
            "id": "E1", "file": "Modal.vue",
            "anchor_old": "<script setup>",
            "replacement_new": "<script setup>\nimport { useToast } from '@/composables/useToast'",
        }]}
        flagged = specify._incomplete_wiring_ids(spec, self.root)
        self.assertIn("E1", flagged)
        self.assertIn("useToast", flagged["E1"])

    def test_used_added_import_is_not_flagged(self):
        # same import, but this time the binding IS added in the same edit → used → ok
        self._write("Modal.vue",
                    "<script setup>\nconst x = 1\nshowToast('hi')\n</script>\n")
        spec = {"edits": [{
            "id": "E1", "file": "Modal.vue",
            "anchor_old": "<script setup>\nconst x = 1",
            "replacement_new": ("<script setup>\nimport { useToast } from '@/x'\n"
                                "const { showToast } = useToast()\nconst x = 1"),
        }]}
        self.assertEqual(specify._incomplete_wiring_ids(spec, self.root), {})

    def test_import_used_by_sibling_edit_not_flagged(self):
        # import added in E1, used by code added in E2 to the SAME file → not flagged
        self._write("m.js", "// head\nlineA\nlineB\n")
        spec = {"edits": [
            {"id": "E1", "file": "m.js", "anchor_old": "// head",
             "replacement_new": "// head\nimport { fmt } from './fmt'"},
            {"id": "E2", "file": "m.js", "anchor_old": "lineB",
             "replacement_new": "fmt(lineB)"},
        ]}
        self.assertEqual(specify._incomplete_wiring_ids(spec, self.root), {})

    def test_unreadable_file_is_skipped(self):
        spec = {"edits": [{
            "id": "E1", "file": "nope.js", "anchor_old": "a",
            "replacement_new": "import X from 'x'\na"}]}
        self.assertEqual(specify._incomplete_wiring_ids(spec, self.root), {})

    def test_gate_downgrades_on_incomplete_wiring(self):
        spec = _fresh_ready()
        out = specify._apply_effectiveness_gate(
            spec, [], {}, False, incomplete_wiring={"E1": "incomplete wiring — imported 'useToast' is never used"})
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertIn("E1", out["effectiveness"]["ineffective_ids"])


class TestEffectivenessGateUnit(unittest.TestCase):
    def test_ineffective_review_downgrades(self):
        out = specify._apply_effectiveness_gate(
            _fresh_ready(), [],
            {"E1": {"effective": False, "coherent": True, "reason": "never reached"}}, False)
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertIn("E1", out["effectiveness"]["ineffective_ids"])
        self.assertFalse(out["edits"][0]["effectiveness"]["ok"])

    def test_incoherent_review_downgrades(self):
        out = specify._apply_effectiveness_gate(
            _fresh_ready(), [],
            {"E1": {"effective": True, "coherent": False, "reason": "contradicts honey"}}, False)
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_over_scope_review_downgrades(self):
        # T891 v2: the edit is effective and coherent but OVER-APPLIES (recoloured a
        # shared rule the seed said to leave neutral), so in_scope=false must
        # downgrade the ready spec exactly like an ineffective edit.
        out = specify._apply_effectiveness_gate(
            _fresh_ready(), [],
            {"E1": {"effective": True, "coherent": True, "in_scope": False,
                    "reason": "recolours shared .wf-undecided, regresses other steps"}},
            False)
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertIn("E1", out["effectiveness"]["ineffective_ids"])
        self.assertIn("over-scope", out["edits"][0]["effectiveness"]["reason"])

    def test_in_scope_true_keeps_ready(self):
        out = specify._apply_effectiveness_gate(
            _fresh_ready(), [],
            {"E1": {"effective": True, "coherent": True, "in_scope": True}}, False)
        self.assertEqual(out["termination"], "ready_to_apply")

    def test_deterministic_noop_downgrades_even_if_review_says_ok(self):
        out = specify._apply_effectiveness_gate(
            _fresh_ready(), ["E1"], {"E1": {"effective": True, "coherent": True}}, False)
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_inconclusive_downgrades_to_needs_pm(self):
        out = specify._apply_effectiveness_gate(_fresh_ready(), [], {}, True)
        self.assertEqual(out["termination"], "needs_pm")

    def test_all_good_keeps_ready(self):
        out = specify._apply_effectiveness_gate(
            _fresh_ready(), [], {"E1": {"effective": True, "coherent": True}}, False)
        self.assertEqual(out["termination"], "ready_to_apply")

    def test_never_upgrades_a_non_ready_spec(self):
        spec = _fresh_ready()
        spec["termination"] = "needs_pm"
        out = specify._apply_effectiveness_gate(
            spec, ["E1"], {"E1": {"effective": False}}, True)
        self.assertEqual(out["termination"], "needs_pm")

    # N174 #3: a review-only "ineffective" verdict on VERIFIED edits in a CROSS-FILE
    # wiring must not declare the fix wrong and loop back — defer to a human (needs_pm).
    def _cross_file_ready(self):
        return {
            "source_honey": "h.md", "codebase_root": "/code", "gate": {"apply": False},
            "deferred": [], "termination": "ready_to_apply",
            "edits": [
                {"id": "E1", "file": "server/guard.py", "anchor_old": "a",
                 "replacement_new": "b", "anchor_status": "verified"},
                {"id": "E2", "file": "ui/Toast.vue", "anchor_old": "c",
                 "replacement_new": "d", "anchor_status": "verified"},
            ]}

    def test_cross_file_verified_review_ineffective_defers_to_needs_pm(self):
        out = specify._apply_effectiveness_gate(
            self._cross_file_ready(), [],
            {"E1": {"effective": False, "reason": "alone doesn't change behavior"}}, False)
        self.assertEqual(out["termination"], "needs_pm")
        self.assertIn("E1", out["effectiveness"]["ineffective_ids"])

    def test_cross_file_but_unverified_still_reinvestigates(self):
        spec = self._cross_file_ready()
        spec["edits"][0]["anchor_status"] = "stale"  # the flagged edit is not verified
        out = specify._apply_effectiveness_gate(
            spec, [], {"E1": {"effective": False, "reason": "x"}}, False)
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_cross_file_with_deterministic_noop_still_reinvestigates(self):
        # a CERTAIN finding (no-op) is present → loop back regardless of cross-file
        out = specify._apply_effectiveness_gate(
            self._cross_file_ready(), ["E1"], {}, False)
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_single_file_review_ineffective_still_reinvestigates(self):
        # _fresh_ready has one edit/one file → softening does not apply
        out = specify._apply_effectiveness_gate(
            _fresh_ready(), [], {"E1": {"effective": False, "reason": "x"}}, False)
        self.assertEqual(out["termination"], "needs_reinvestigation")


class TestRunSpecifyWithReview(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.honey = os.path.join(self.tmp, "honey.md")
        with open(self.honey, "w", encoding="utf-8") as f:
            f.write("# honey\nSymptom: X is wrong. Fix: change x.\n")
        self.contract = os.path.join(self.tmp, "contract.md")
        with open(self.contract, "w", encoding="utf-8") as f:
            f.write("[Role] specify author contract")
        self.out = os.path.join(self.tmp, "spec.json")

    def _run(self, author_stdout: str, review_stdout: str):
        # First worker call = author; second = effectiveness review.
        with mock.patch.object(specify, "call_worker",
                               side_effect=[_wr(author_stdout), _wr(review_stdout)]):
            return specify.run_specify(
                honey_path=self.honey,
                codebase_root=self.tmp,
                output_path=self.out,
                contract_path=self.contract,
            )

    def test_review_keeps_ready_when_effective(self):
        spec = self._run(
            json.dumps(_READY_SPEC),
            _review([{"id": "E1", "effective": True, "coherent": True, "reason": "ok"}]))
        self.assertEqual(spec["termination"], "ready_to_apply")

    def test_review_downgrades_when_ineffective(self):
        spec = self._run(
            json.dumps(_READY_SPEC),
            _review([{"id": "E1", "effective": False, "coherent": True,
                      "reason": "guard can never be true"}]))
        self.assertEqual(spec["termination"], "needs_reinvestigation")
        # The downgrade reason is persisted to the SSOT on disk.
        with open(self.out, encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["termination"], "needs_reinvestigation")
        self.assertIn("E1", on_disk["effectiveness"]["ineffective_ids"])

    def test_unparseable_review_downgrades_to_needs_pm(self):
        # Unusable on BOTH the attempt and the retry → inconclusive → needs_pm.
        with mock.patch.object(specify, "call_worker",
                               side_effect=[_wr(json.dumps(_READY_SPEC)),
                                            _wr("● no json\n"), _wr("still no json\n")]):
            spec = specify.run_specify(
                honey_path=self.honey, codebase_root=self.tmp,
                output_path=self.out, contract_path=self.contract)
        self.assertEqual(spec["termination"], "needs_pm")


class TestReviewJsonRetry(unittest.TestCase):
    """The effectiveness review retries once on unusable JSON (mirrors judge)."""

    SPEC = _READY_SPEC

    def test_retry_recovers_and_keeps_ready(self):
        # 1st review = prose (unusable) → retry with reminder → valid reviews.
        seq = [_wr("Here is my assessment, the edit looks fine."),
               _wr(_review([{"id": "E1", "effective": True, "coherent": True}]))]
        with mock.patch.object(specify, "call_worker", side_effect=seq) as cw:
            judgments, inconclusive = specify.review_effectiveness(
                "honey", self.SPEC, "/code", "openai/gpt-oss-120b", "deepinfra")
        self.assertEqual(cw.call_count, 2)              # one retry happened
        self.assertFalse(inconclusive)
        self.assertIn("E1", judgments)

    def test_retry_prompt_carries_reminder(self):
        seq = [_wr("no json"),
               _wr(_review([{"id": "E1", "effective": True, "coherent": True}]))]
        with mock.patch.object(specify, "call_worker", side_effect=seq) as cw:
            specify.review_effectiveness("honey", self.SPEC, "/code",
                                         "m", "deepinfra")
        self.assertNotIn("[Retry]", cw.call_args_list[0].args[2])
        self.assertIn("[Retry]", cw.call_args_list[1].args[2])

    def test_missing_reviews_list_triggers_retry(self):
        # Valid JSON but no 'reviews' array is also unusable → retry.
        seq = [_wr(json.dumps({"verdict": "ok"})),
               _wr(_review([{"id": "E1", "effective": True, "coherent": True}]))]
        with mock.patch.object(specify, "call_worker", side_effect=seq) as cw:
            judgments, inconclusive = specify.review_effectiveness(
                "honey", self.SPEC, "/code", "m", "deepinfra")
        self.assertEqual(cw.call_count, 2)
        self.assertFalse(inconclusive)

    def test_worker_exception_not_retried(self):
        with mock.patch.object(specify, "call_worker",
                               side_effect=RuntimeError("boom")) as cw:
            judgments, inconclusive = specify.review_effectiveness(
                "honey", self.SPEC, "/code", "m", "deepinfra")
        self.assertEqual(cw.call_count, 1)             # transport failure: no retry
        self.assertTrue(inconclusive)

    def test_both_attempts_recorded_to_ledger(self):
        led = mock.Mock()
        seq = [_wr("garbage"),
               _wr(_review([{"id": "E1", "effective": True, "coherent": True}]))]
        with mock.patch.object(specify, "call_worker", side_effect=seq):
            specify.review_effectiveness("honey", self.SPEC, "/code", "m",
                                         "deepinfra", ledger=led)
        self.assertEqual(led.record_call.call_count, 2)  # paid retry recorded


class TestDecisivenessGate(unittest.TestCase):
    """specify._apply_decisiveness_gate — guarded needs_pm -> ready_to_apply promotion."""

    def _needs_pm(self, **over):
        spec = {
            "edits": [{
                "id": "E1", "file": "a.py",
                "anchor_old": "x = 1", "replacement_new": "x = 2",
                "confidence": "medium", "anchor_status": "verified",
            }],
            "deferred": [{"issue": "optional UX option", "reason": "policy_direction"}],
            "termination": "needs_pm",
            "effectiveness": {"inconclusive": False, "ineffective_ids": []},
            "notes": "PM please confirm authoritative UX",
        }
        spec.update(over)
        return spec

    def test_promotes_verified_effective_with_policy_deferred(self):
        out = specify._apply_decisiveness_gate(self._needs_pm())
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertIn("decisiveness gate", out["notes"])

    def test_create_file_edit_promotes(self):
        spec = self._needs_pm(edits=[{
            "id": "E1", "kind": "create_file", "file": "new.py",
            "content": "x = 1\n", "confidence": "medium",
        }])
        self.assertEqual(
            specify._apply_decisiveness_gate(spec)["termination"], "ready_to_apply")

    def test_low_confidence_blocks(self):
        spec = self._needs_pm()
        spec["edits"][0]["confidence"] = "low"
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"], "needs_pm")

    def test_ineffective_id_blocks(self):
        spec = self._needs_pm()
        spec["effectiveness"]["ineffective_ids"] = ["E1"]
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"], "needs_pm")

    def test_inconclusive_blocks(self):
        spec = self._needs_pm()
        spec["effectiveness"]["inconclusive"] = True
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"], "needs_pm")

    def test_non_optional_deferred_blocks(self):
        spec = self._needs_pm()
        spec["deferred"] = [{"issue": "x", "reason": "needs_runtime"}]
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"], "needs_pm")

    def test_missing_effectiveness_key_blocks(self):
        spec = self._needs_pm()
        del spec["effectiveness"]
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"], "needs_pm")

    def test_unverified_anchor_blocks(self):
        spec = self._needs_pm()
        spec["edits"][0]["anchor_status"] = "stale"
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"], "needs_pm")

    def test_never_upgrades_needs_reinvestigation(self):
        spec = self._needs_pm(termination="needs_reinvestigation")
        self.assertEqual(
            specify._apply_decisiveness_gate(spec)["termination"], "needs_reinvestigation")

    def test_leaves_ready_untouched(self):
        spec = self._needs_pm(termination="ready_to_apply")
        self.assertEqual(
            specify._apply_decisiveness_gate(spec)["termination"], "ready_to_apply")


class TestGroundAnchors(unittest.TestCase):
    """Anchor-grounding pre-flight: lift CURRENT live values at cited file:line
    into the honey (NR164/NR165/TR891 — location present, value absent)."""

    def setUp(self):
        self.code = tempfile.mkdtemp()
        comp = os.path.join(self.code, "client", "src")
        os.makedirs(comp, exist_ok=True)
        # The CSS rule whose VALUE the honey never quoted (grey, not blue).
        with open(os.path.join(comp, "DocWorkflow.vue"), "w", encoding="utf-8") as f:
            f.write("\n".join([
                "line1", "line2", "line3",
                ".wf-step.wf-undecided {", "  color: #999999;", "}",
            ]) + "\n")

    def test_lifts_value_at_cited_location(self):
        honey = "## Fix directions\n- target: client/src/DocWorkflow.vue:4-6\n"
        out, diag = specify.ground_anchors(honey, self.code)
        self.assertIn("client/src/DocWorkflow.vue:4-6", diag["lifted"])
        self.assertIn("color: #999999;", out)          # the VALUE is now in the honey
        self.assertIn("Anchor ground truth", out)

    def test_no_citation_leaves_honey_unchanged(self):
        honey = "## Fix directions\n- change the undecided step color to blue.\n"
        out, diag = specify.ground_anchors(honey, self.code)
        self.assertEqual(out, honey)
        self.assertEqual(diag["lifted"], [])

    def test_unresolvable_citation_skipped_not_fabricated(self):
        honey = "- target: client/src/Ghost.vue:10-12\n"
        out, diag = specify.ground_anchors(honey, self.code)
        self.assertEqual(out, honey)
        self.assertIn("client/src/Ghost.vue:10-12", diag["unresolved"])

    def test_already_quoted_value_is_not_relifted(self):
        # honey already carries the exact line → dedup guard skips it.
        honey = ("- target: client/src/DocWorkflow.vue:5-5\n"
                 "current code:\n  color: #999999;\n")
        out, diag = specify.ground_anchors(honey, self.code)
        self.assertEqual(diag["lifted"], [])
        self.assertIn("client/src/DocWorkflow.vue:5-5", diag["skipped_present"])

    def test_resolves_against_docs_root(self):
        docs = tempfile.mkdtemp()
        d = os.path.join(docs, "210_design")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "D031.md"), "w", encoding="utf-8") as f:
            f.write("# D031\n## policy\nthe action-bar must stay visible\n")
        honey = "- target: 210_design/D031.md:3-3\n"
        out, diag = specify.ground_anchors(honey, self.code, docs)
        self.assertIn("the action-bar must stay visible", out)
        self.assertIn("under docs root", out)

    def test_range_clamped_to_max_lines(self):
        big = os.path.join(self.code, "big.txt")
        with open(big, "w", encoding="utf-8") as f:
            f.write("\n".join(f"row{i}" for i in range(1, 200)) + "\n")
        honey = "- target: big.txt:1-150\n"
        out, diag = specify.ground_anchors(honey, self.code, max_lines=10)
        self.assertIn("(truncated)", out)
        self.assertIn("row1", out)
        self.assertNotIn("row120", out)

    def test_timestamp_is_not_a_citation(self):
        # "21:33:01" has no file extension → must not be mistaken for file:line.
        honey = "log at 21:33:01 says the step is grey\n"
        out, diag = specify.ground_anchors(honey, self.code)
        self.assertEqual(out, honey)
        self.assertEqual(diag["lifted"], [])

    def test_seed_target_gets_generous_forward_window(self):
        # Defect 2 (T892): a seed cites an APPROXIMATE range (e.g. 4-6) but the real
        # block to rewrite sits past it. A wide_files target lifts a forward window,
        # so a downstream line the tight cap would truncate is still pulled in.
        spec = os.path.join(self.code, "client", "src", "view.spec.ts")
        with open(spec, "w", encoding="utf-8") as f:
            f.write("\n".join(f"assert_{i} = {i}" for i in range(1, 80)) + "\n")
        honey = "- target: client/src/view.spec.ts:4-6\n"
        # Without wide: tight cap stops well before line 60.
        narrow, _ = specify.ground_anchors(honey, self.code, max_lines=10)
        self.assertNotIn("assert_60", narrow)
        # With the file marked as a seed target: the forward window reaches it.
        wide, diag = specify.ground_anchors(
            honey, self.code, wide_files={"client/src/view.spec.ts"}, wide_lines=120)
        self.assertIn("assert_60", wide)
        self.assertIn("client/src/view.spec.ts", diag["lifted"][0])

    def test_overlapping_windows_merged_to_one(self):
        # T892 balloon: three near-duplicate windows of ONE region (queries.json
        # 124-130 / 127-127 / 129-135, each widened to ~120 lines of a short file)
        # were each lifted in full, tripling the author prompt. Merging overlapping
        # ranges collapses them to a single lift.
        f = os.path.join(self.code, "q.json")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("\n".join(f"row{i}" for i in range(1, 40)) + "\n")
        honey = ("- a: q.json:5-8\n- b: q.json:7-7\n- c: q.json:9-12\n")
        out, diag = specify.ground_anchors(
            honey, self.code, wide_files={"q.json"}, wide_lines=120)
        # One merged window, not three separate lifts of the same region.
        self.assertEqual(len(diag["lifted"]), 1)
        # The merged window starts at the earliest cited line and covers the region.
        self.assertTrue(diag["lifted"][0].startswith("q.json:5-"))
        self.assertIn("row9", out)

    def test_disjoint_regions_stay_separate(self):
        # Non-overlapping regions of the same file are NOT merged (they are genuinely
        # different blocks the author needs — workflowViewState.ts 51-170 vs 200-319).
        f = os.path.join(self.code, "big.txt")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("\n".join(f"row{i}" for i in range(1, 400)) + "\n")
        honey = "- a: big.txt:10-12\n- b: big.txt:300-302\n"
        out, diag = specify.ground_anchors(honey, self.code, max_lines=20)
        self.assertEqual(len(diag["lifted"]), 2)

    def test_total_line_budget_caps_runaway(self):
        # A citation-storm of DISTINCT regions can't balloon the prompt past the
        # total-line ceiling even when each window is individually within the cap.
        f = os.path.join(self.code, "huge.txt")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("\n".join(f"row{i}" for i in range(1, 2000)) + "\n")
        # 20 disjoint 100-line windows = 2000 lines requested, well over the budget.
        cites = "".join(f"- x: huge.txt:{1 + i*120}-{100 + i*120}\n" for i in range(20))
        out, diag = specify.ground_anchors(
            cites, self.code, max_lines=100, max_anchors=99)
        total = sum(block.count("\n") for block in out.split("```")[1::2])
        self.assertLessEqual(total, specify._GROUND_MAX_TOTAL_LINES + 100)
        self.assertLess(len(diag["lifted"]), 20)


class TestRunSpecifyGrounding(unittest.TestCase):
    """run_specify feeds the grounded honey to the author prompt."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "client"), exist_ok=True)
        with open(os.path.join(self.tmp, "client", "View.vue"), "w",
                  encoding="utf-8") as f:
            f.write("a\nb\n.badge { color: #999999; }\nd\n")
        self.honey = os.path.join(self.tmp, "honey.md")
        with open(self.honey, "w", encoding="utf-8") as f:
            f.write("## Fix directions\n- target: client/View.vue:3-3\n")
        self.contract = os.path.join(self.tmp, "contract.md")
        with open(self.contract, "w", encoding="utf-8") as f:
            f.write("[Role] contract")
        self.out = os.path.join(self.tmp, "spec.json")

    def test_grounded_value_reaches_author_prompt(self):
        bare = {"edits": [], "deferred": [], "gate": {"apply": False},
                "termination": "needs_reinvestigation"}
        with mock.patch.object(specify, "call_worker",
                               return_value=_wr(json.dumps(bare))) as cw:
            specify.run_specify(
                honey_path=self.honey, codebase_root=self.tmp,
                output_path=self.out, contract_path=self.contract, review=False)
        author_prompt = cw.call_args_list[0].args[2]
        self.assertIn("color: #999999;", author_prompt)
        self.assertIn("Anchor ground truth", author_prompt)

    def test_ground_false_skips_lift(self):
        bare = {"edits": [], "deferred": [], "gate": {"apply": False},
                "termination": "needs_reinvestigation"}
        with mock.patch.object(specify, "call_worker",
                               return_value=_wr(json.dumps(bare))) as cw:
            specify.run_specify(
                honey_path=self.honey, codebase_root=self.tmp,
                output_path=self.out, contract_path=self.contract,
                review=False, ground=False)
        author_prompt = cw.call_args_list[0].args[2]
        self.assertNotIn("Anchor ground truth", author_prompt)

    def test_deepinfra_author_gets_grounded_only_prompt(self):
        # A tool-OFF (deepinfra) author must get the grounded-only contract and
        # grounding forced on, so it lifts from the ground-truth block not files.
        bare = {"edits": [], "deferred": [], "gate": {"apply": False},
                "termination": "needs_reinvestigation"}
        with mock.patch.object(specify, "call_worker",
                               return_value=_wr(json.dumps(bare))) as cw:
            specify.run_specify(
                honey_path=self.honey, codebase_root=self.tmp,
                output_path=self.out, contract_path=self.contract,
                review=False, provider="deepinfra", model="openai/gpt-oss-120b")
        author_prompt = cw.call_args_list[0].args[2]
        self.assertIn("NO file-system tools", author_prompt)
        self.assertIn("color: #999999;", author_prompt)   # grounding forced on
        self.assertNotIn("Re-open every file", author_prompt)

    def test_deepinfra_author_grounding_forced_even_if_ground_false(self):
        bare = {"edits": [], "deferred": [], "gate": {"apply": False},
                "termination": "needs_reinvestigation"}
        with mock.patch.object(specify, "call_worker",
                               return_value=_wr(json.dumps(bare))) as cw:
            specify.run_specify(
                honey_path=self.honey, codebase_root=self.tmp,
                output_path=self.out, contract_path=self.contract,
                review=False, ground=False,
                provider="deepinfra", model="openai/gpt-oss-120b")
        author_prompt = cw.call_args_list[0].args[2]
        self.assertIn("color: #999999;", author_prompt)   # forced on despite ground=False

    def test_author_timeout_passes_through(self):
        # The per-call author timeout is forwarded to call_worker so the operator
        # can right-size the slow agentic CLI (codex) cap via config (T892).
        bare = {"edits": [], "deferred": [], "gate": {"apply": False},
                "termination": "needs_reinvestigation"}
        with mock.patch.object(specify, "call_worker",
                               return_value=_wr(json.dumps(bare))) as cw:
            specify.run_specify(
                honey_path=self.honey, codebase_root=self.tmp,
                output_path=self.out, contract_path=self.contract,
                review=False, author_timeout=900)
        self.assertEqual(cw.call_args_list[0].kwargs["timeout"], 900)

    def test_author_retries_on_timeout(self):
        # A transient timeout is retried up to author_retries before giving up, so
        # one slow call does not discard the completed investigate stage (T892).
        bare = {"edits": [], "deferred": [], "gate": {"apply": False},
                "termination": "needs_reinvestigation"}
        calls = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            calls.append(1)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(cmd="codex", timeout=timeout)
            return _wr(json.dumps(bare))

        with mock.patch.object(specify, "call_worker", side_effect=fake):
            spec = specify.run_specify(
                honey_path=self.honey, codebase_root=self.tmp,
                output_path=self.out, contract_path=self.contract,
                review=False, author_retries=1)
        self.assertEqual(len(calls), 2)            # first timed out, second succeeded
        self.assertEqual(spec["termination"], "needs_reinvestigation")

    def test_author_reraises_after_retries_exhausted(self):
        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            raise subprocess.TimeoutExpired(cmd="codex", timeout=timeout)

        with mock.patch.object(specify, "call_worker", side_effect=fake):
            with self.assertRaises(subprocess.TimeoutExpired):
                specify.run_specify(
                    honey_path=self.honey, codebase_root=self.tmp,
                    output_path=self.out, contract_path=self.contract,
                    review=False, author_retries=1)


class TestReviewerProvider(unittest.TestCase):
    """The effectiveness reviewer can run on a different provider than the author
    (cost lever: move the tool-OFF single-shot review off copilot)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.honey = os.path.join(self.tmp, "honey.md")
        with open(self.honey, "w", encoding="utf-8") as f:
            f.write("# honey\nSymptom: X. Fix: change x.\n")
        self.contract = os.path.join(self.tmp, "contract.md")
        with open(self.contract, "w", encoding="utf-8") as f:
            f.write("[Role] contract")
        self.out = os.path.join(self.tmp, "spec.json")

    def test_reviewer_uses_its_own_provider(self):
        seen = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            seen.append((provider, model))
            # 1st call = author (spec), 2nd = review.
            if len(seen) == 1:
                return _wr(json.dumps(_READY_SPEC))
            return _wr(_review([{"id": "E1", "effective": True, "coherent": True}]))

        with mock.patch.object(specify, "call_worker", side_effect=fake):
            spec = specify.run_specify(
                honey_path=self.honey, codebase_root=self.tmp,
                output_path=self.out, contract_path=self.contract,
                model="gpt-5-mini", provider="copilot",
                review_model="openai/gpt-oss-120b", review_provider="deepinfra")
        self.assertEqual(seen[0], ("copilot", "gpt-5-mini"))      # author
        self.assertEqual(seen[1], ("deepinfra", "openai/gpt-oss-120b"))  # reviewer
        self.assertEqual(spec["termination"], "ready_to_apply")

    def test_reviewer_defaults_to_author_provider(self):
        seen = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            seen.append((provider, model))
            if len(seen) == 1:
                return _wr(json.dumps(_READY_SPEC))
            return _wr(_review([{"id": "E1", "effective": True, "coherent": True}]))

        with mock.patch.object(specify, "call_worker", side_effect=fake):
            specify.run_specify(
                honey_path=self.honey, codebase_root=self.tmp,
                output_path=self.out, contract_path=self.contract,
                model="gpt-5-mini", provider="copilot")  # no review_* → fall back
        self.assertEqual(seen[1], ("copilot", "gpt-5-mini"))


class TestConfigSpecifyRole(unittest.TestCase):
    def test_specify_role_exists(self):
        # hive.config.json routes the specify author onto codex (gpt-5.4-mini).
        cfg = load_config()
        role = cfg.role("specify")
        self.assertEqual(role.provider, "codex")
        self.assertEqual(role.model, "gpt-5.4-mini")

    def test_review_role_routes_to_deepinfra(self):
        # hive.config.json opts the reviewer onto deepinfra (cost lever).
        cfg = load_config()
        role = cfg.role("review")
        self.assertEqual(role.provider, "deepinfra")
        self.assertEqual(role.model, "openai/gpt-oss-120b")

    def test_cli_model_override_reaches_specify(self):
        cfg = load_config()
        cfg.apply_cli_model("claude-sonnet-4.6")
        self.assertEqual(cfg.role("specify").model, "claude-sonnet-4.6")


_HONEY_WITH_TARGETS = (
    "## Seed-specified edit targets (the user named these files explicitly — AUTHOR them)\n\n"
    "Author the seed's specified change at each.\n\n"
    "- server/sql/queries/queries.json:129-129\n"
    "- client/tests/main/workflowViewState.spec.ts:300-320\n\n"
    "## Axes without a confident localisation (do NOT fabricate edits here)\n\n"
    "- SEED_ANCHOR: not located\n"
)


class TestAnchorNotGroundedGate(unittest.TestCase):
    """_apply_anchor_not_grounded_gate: an edit for the same file as an
    anchor_not_grounded deferred item is a contradiction and must be removed."""

    def test_removes_contradictory_edit_and_downgrades_ready(self):
        # The author put queries.json in deferred(anchor_not_grounded) AND in edits[].
        spec = {
            "edits": [{"id": "E1", "file": "server/sql/queries/queries.json",
                       "anchor_old": "x", "replacement_new": "y",
                       "anchor_status": "verified"}],
            "deferred": [{"issue": "get_pending_head_by_group queries.json not in evidence",
                          "reason": "anchor_not_grounded", "stays_as": "investigation"}],
            "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_anchor_not_grounded_gate(spec)
        self.assertEqual(out["edits"], [])
        self.assertEqual(out["termination"], "needs_pm")
        self.assertIn("anchor-not-grounded gate", out["notes"])

    def test_no_anchor_not_grounded_deferred_leaves_spec_unchanged(self):
        spec = {
            "edits": [{"id": "E1", "file": "a.py", "anchor_status": "verified"}],
            "deferred": [{"issue": "optional UX", "reason": "policy_direction"}],
            "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_anchor_not_grounded_gate(spec)
        self.assertEqual(len(out["edits"]), 1)
        self.assertEqual(out["termination"], "ready_to_apply")

    def test_different_file_edit_is_kept(self):
        # deferred names queries.json, edit targets routers/main.py → no contradiction
        spec = {
            "edits": [{"id": "E1", "file": "server/routers/main.py",
                       "anchor_status": "verified"}],
            "deferred": [{"issue": "queries.json path not grounded",
                          "reason": "anchor_not_grounded"}],
            "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_anchor_not_grounded_gate(spec)
        self.assertEqual(len(out["edits"]), 1)
        self.assertEqual(out["termination"], "ready_to_apply")

    def test_does_not_change_non_ready_termination(self):
        # The edit is still removed; only ready_to_apply is downgraded
        spec = {
            "edits": [{"id": "E1", "file": "queries.json", "anchor_status": "verified"}],
            "deferred": [{"issue": "queries.json not grounded",
                          "reason": "anchor_not_grounded"}],
            "termination": "needs_reinvestigation", "notes": "",
        }
        out = specify._apply_anchor_not_grounded_gate(spec)
        self.assertEqual(out["edits"], [])
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_match_via_evidence_field(self):
        # The file reference is in the evidence list, not the issue text
        spec = {
            "edits": [{"id": "E1", "file": "server/sql/queries.json",
                       "anchor_status": "verified"}],
            "deferred": [{"issue": "path unknown",
                          "reason": "anchor_not_grounded",
                          "evidence": ["server/sql/queries.json:45"]}],
            "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_anchor_not_grounded_gate(spec)
        self.assertEqual(out["edits"], [])
        self.assertEqual(out["termination"], "needs_pm")

    def test_needs_runtime_termination_is_valid(self):
        problems = specify._validate_spec({
            "edits": [],
            "deferred": [{"issue": "needs row state", "reason": "needs_runtime"}],
            "gate": {},
            "termination": "needs_runtime",
        })
        self.assertEqual(problems, [])


class TestSeedCoverageGate(unittest.TestCase):
    """Defect 2 (T892): a seed-named edit target must become an edit, or a ready
    spec is downgraded to needs_pm with the dropped target reported."""

    def test_seed_target_files_parsed_from_section(self):
        self.assertEqual(
            specify._seed_target_files(_HONEY_WITH_TARGETS),
            ["server/sql/queries/queries.json",
             "client/tests/main/workflowViewState.spec.ts"])

    def test_downgrades_ready_when_seed_target_missing(self):
        spec = {
            "edits": [{"id": "E1",
                       "file": "client/src/main/workflow/workflowViewState.ts"}],
            "deferred": [{"issue": "Update queries.json get_pending_head_by_group …",
                          "reason": "not_expressible_as_edit"}],
            "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_seed_coverage_gate(spec, _HONEY_WITH_TARGETS)
        self.assertEqual(out["termination"], "needs_pm")
        # both seed targets are missing (only the FE view-state file was edited)
        self.assertIn("server/sql/queries/queries.json", out["seed_coverage"]["missing"])
        self.assertIn("client/tests/main/workflowViewState.spec.ts",
                      out["seed_coverage"]["missing"])
        # the author's stated defer reason is reported, not hidden
        self.assertIn("not_expressible_as_edit",
                      out["seed_coverage"]["reasons"]["server/sql/queries/queries.json"])
        self.assertIn("seed-coverage gate", out["notes"])

    def test_passes_when_all_seed_targets_edited(self):
        spec = {
            "edits": [
                {"id": "E1", "file": "server/sql/queries/queries.json"},
                {"id": "E2", "file": "client/tests/main/workflowViewState.spec.ts"}],
            "deferred": [], "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_seed_coverage_gate(spec, _HONEY_WITH_TARGETS)
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertEqual(out["seed_coverage"]["missing"], [])

    def test_noop_when_no_seed_section(self):
        spec = {"edits": [], "termination": "ready_to_apply"}
        out = specify._apply_seed_coverage_gate(spec, "no seed section here")
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertNotIn("seed_coverage", out)

    def test_records_diagnostics_but_does_not_upgrade_non_ready(self):
        # A spec already at needs_reinvestigation is not vouching for completeness;
        # the gate records coverage but never promotes it.
        spec = {"edits": [], "deferred": [], "termination": "needs_reinvestigation"}
        out = specify._apply_seed_coverage_gate(spec, _HONEY_WITH_TARGETS)
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertEqual(len(out["seed_coverage"]["missing"]), 2)


class TestVerifyAnchorsLive(unittest.TestCase):
    """N175 E7: a 'verified' anchor must be re-confirmed against LIVE disk, not trusted
    from the author's claim. _verify_anchors_live downgrades a verified-but-absent or
    non-unique anchor so the spec is not presented as ready."""

    def _spec(self, anchor_old, status="verified", file="f.py"):
        return {"edits": [{"id": "E1", "file": file, "anchor_old": anchor_old,
                           "replacement_new": "y = 2", "anchor_status": status}],
                "deferred": [], "gate": {"apply": False},
                "termination": "ready_to_apply"}

    def test_verified_but_absent_downgraded_to_not_found(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "f.py"), "w", encoding="utf-8") as fh:
                fh.write("a = 1\nb = 2\n")  # does NOT contain the claimed anchor
            spec = self._spec("y = 1")      # author claimed verified, but it's not there
            out = specify._verify_anchors_live(spec, root)
        self.assertEqual(out["edits"][0]["anchor_status"], "not_found")
        self.assertIn("anchor_drift", out["edits"][0])
        # and the normalize step then refuses to present it as ready
        out = specify._normalize_spec(out)
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_verified_and_present_once_survives(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "f.py"), "w", encoding="utf-8") as fh:
                fh.write("x = 1\nb = 2\n")
            out = specify._verify_anchors_live(self._spec("x = 1"), root)
        self.assertEqual(out["edits"][0]["anchor_status"], "verified")
        self.assertNotIn("anchor_drift", out["edits"][0])

    def test_verified_but_ambiguous_downgraded_to_stale(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "f.py"), "w", encoding="utf-8") as fh:
                fh.write("x = 1\nx = 1\n")  # anchor occurs twice → not uniquely targetable
            out = specify._verify_anchors_live(self._spec("x = 1"), root)
        self.assertEqual(out["edits"][0]["anchor_status"], "stale")

    def test_non_verified_status_is_left_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "f.py"), "w", encoding="utf-8") as fh:
                fh.write("a = 1\n")
            out = specify._verify_anchors_live(self._spec("y = 1", status="stale"), root)
        self.assertEqual(out["edits"][0]["anchor_status"], "stale")  # not upgraded/changed

    def test_unreadable_file_left_for_apply_to_recheck(self):
        # File missing under root → cannot read → status untouched (apply re-verifies).
        with tempfile.TemporaryDirectory() as root:
            out = specify._verify_anchors_live(self._spec("y = 1"), root)
        self.assertEqual(out["edits"][0]["anchor_status"], "verified")


if __name__ == "__main__":
    unittest.main()
