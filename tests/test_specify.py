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
        spec = self._run(json.dumps(_READY_SPEC), "● no parseable json here\n")
        self.assertEqual(spec["termination"], "needs_pm")


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


class TestConfigSpecifyRole(unittest.TestCase):
    def test_specify_role_exists(self):
        cfg = load_config()
        role = cfg.role("specify")
        self.assertEqual(role.provider, "copilot")
        self.assertTrue(role.model)

    def test_cli_model_override_reaches_specify(self):
        cfg = load_config()
        cfg.apply_cli_model("claude-sonnet-4.6")
        self.assertEqual(cfg.role("specify").model, "claude-sonnet-4.6")


if __name__ == "__main__":
    unittest.main()
