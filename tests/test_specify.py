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
        cfg = load_config()
        role = cfg.role("specify")
        self.assertEqual(role.provider, "copilot")
        self.assertTrue(role.model)

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


if __name__ == "__main__":
    unittest.main()
