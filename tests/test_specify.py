"""Unit tests for hive.specify — honey → edit-spec lowering stage.

The provider is mocked so these run without the copilot CLI. Coverage:
  ① build_specify_prompt embeds the contract, codebase root, and honey
  ② run_specify parses the author's JSON, writes it as the SSOT, returns the dict
  ③ Stage-1 safety: gate.apply is forced false even if the author set it true
  ④ A stale/not_found edit downgrades ready_to_apply → needs_reinvestigation
  ⑤ run_specify raises ValueError when the author emits no JSON
  ⑥ config exposes a 'specify' role and --model override reaches it
  ⑦ effectiveness gate: deterministic no-op + model review downgrade a ready spec
    whose edits do not change the reported behavior; inconclusive review →
    needs_reinvestigation (there is no human-handoff terminal)
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import specify
from hive.config import DbConnection, load_config
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

    def test_contract_disambiguates_existing_fixture_and_test_only_requests(self):
        contract = specify.load_contract()
        self.assertIn(
            "Requesting an existing fixture as a test-function argument",
            contract)
        self.assertIn(
            "either (a) anchor_old → replacement_new", contract)
        self.assertIn("TEST-ONLY REQUESTS are different from runtime proof", contract)
        self.assertIn(
            "Every id in `verify.test_edit_ids` MUST name an actual entry",
            contract)


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
                "termination": "needs_reinvestigation"}
        out = specify._normalize_spec(spec)
        self.assertIs(out["gate"]["apply"], False)

    def test_retired_needs_pm_coerced_to_reinvestigation(self):
        # needs_pm is retired (no human-handoff terminal); a stray emission is coerced.
        spec = {"gate": {"apply": False}, "edits": [], "deferred": [],
                "termination": "needs_pm"}
        out = specify._normalize_spec(spec)
        self.assertEqual(out["termination"], "needs_reinvestigation")

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


class TestRewrapFlattenedEdit(unittest.TestCase):
    """specify._rewrap_flattened_edit — salvage a single edit the author flattened
    onto the spec root (no edits[] envelope) so it is not dropped as 0 edits (N175)."""

    def _flattened(self):
        # The exact N175 shape: one verified, high-confidence edit's fields hoisted to
        # the root with envelope-level gate/effectiveness alongside, but no edits[].
        return {
            "id": "E5",
            "file": "client/src/main/components/NewRequirementModal.vue",
            "anchor_old": "} finally {",
            "replacement_new": "  showToast(msg, 'danger')\n} finally {",
            "rationale": "surface readable toast",
            "confidence": "high",
            "anchor_status": "verified",
            "gate": {"apply": False},
            "effectiveness": {"inconclusive": False, "ineffective_ids": []},
        }

    def test_flattened_edit_is_wrapped(self):
        out = specify._rewrap_flattened_edit(self._flattened())
        self.assertIsInstance(out.get("edits"), list)
        self.assertEqual(len(out["edits"]), 1)
        self.assertEqual(out["edits"][0]["id"], "E5")
        self.assertEqual(out["edits"][0]["anchor_status"], "verified")
        # Edit-level fields are moved off the root.
        self.assertNotIn("anchor_old", out)
        self.assertNotIn("replacement_new", out)

    def test_envelope_level_keys_stay_at_root(self):
        out = specify._rewrap_flattened_edit(self._flattened())
        self.assertEqual(out["gate"], {"apply": False})
        self.assertIn("effectiveness", out)
        self.assertEqual(out["deferred"], [])
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_wrapped_flattened_edit_reaches_ready_via_decisiveness(self):
        # The salvage itself only loops back; the existing decisiveness gate is what
        # promotes a verified/effective/confident edit to ready_to_apply.
        out = specify._rewrap_flattened_edit(self._flattened())
        out = specify._apply_decisiveness_gate(out)
        self.assertEqual(out["termination"], "ready_to_apply")

    def test_create_file_flattened_is_wrapped(self):
        spec = {"kind": "create_file", "file": "a/b.py", "content": "x = 1\n",
                "confidence": "high"}
        out = specify._rewrap_flattened_edit(spec)
        self.assertEqual(len(out["edits"]), 1)
        self.assertEqual(out["edits"][0]["kind"], "create_file")

    def test_proper_envelope_untouched(self):
        spec = {"edits": [{"id": "E1", "anchor_status": "verified"}], "deferred": [],
                "gate": {"apply": False}, "termination": "ready_to_apply"}
        out = specify._rewrap_flattened_edit(spec)
        self.assertEqual(out["edits"], [{"id": "E1", "anchor_status": "verified"}])
        self.assertEqual(out["termination"], "ready_to_apply")

    def test_non_edit_root_untouched(self):
        # A spec with no edits[] and no edit-shaped root (e.g. a pure NR) is left alone.
        spec = {"deferred": [], "termination": "needs_reinvestigation",
                "notes": "no concrete edit"}
        out = specify._rewrap_flattened_edit(spec)
        self.assertNotIn("edits", out)


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

    def test_verify_reference_to_missing_edit_is_flagged(self):
        problems = specify._validate_spec({
            "edits": [], "deferred": [], "gate": {},
            "termination": "needs_reinvestigation",
            "verify": {"red_test_node": "tests/test_x.py::test_x",
                       "test_edit_ids": ["E1"]},
        })
        self.assertTrue(any("missing edits: E1" in p for p in problems))


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

    def test_qwen_shaped_ghost_verify_is_routed_as_reauthoring_error(self):
        contradictory = {
            "edits": [],
            "deferred": [{
                "issue": "Create tests/test_x.py using the existing test_db fixture",
                "reason": "multi_file_design",
            }],
            "gate": {"commands": [], "apply": False},
            "verify": {
                "red_test_node": "tests/test_x.py::test_x",
                "test_edit_ids": ["E1"],
            },
            "termination": "needs_reinvestigation",
            "notes": "new file requires multi-file fixture wiring",
        }
        spec = self._run(json.dumps(contradictory))
        self.assertNotIn("verify", spec)
        self.assertEqual(
            spec["reinvestigation"]["reason_code"],
            specify.RI_VERIFY_INCONSISTENT)
        self.assertEqual(
            spec["verify_consistency"]["missing_test_edit_ids"], ["E1"])

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

    def test_test_file_edit_exempt_from_wiring(self):
        # a red test (create_file under tests/) with an unused import must NOT be flagged:
        # its correctness is the red→green run, not import usage.
        spec = {"edits": [{
            "id": "E2", "kind": "create_file", "file": "server/tests/test_x.py",
            "content": "import os\n\ndef test_x():\n    assert 1 + 1 == 2\n"}],
            "verify": {"test_edit_ids": ["E2"]}}
        self.assertEqual(specify._incomplete_wiring_ids(spec, self.root), {})

    def test_test_edit_id_exempt_even_off_tests_path(self):
        # exemption also keys off verify.test_edit_ids, not only the path
        spec = {"edits": [{
            "id": "E2", "kind": "create_file", "file": "checks/probe_x.py",
            "content": "import os\n\ndef test_x():\n    assert True\n"}],
            "verify": {"test_edit_ids": ["E2"]}}
        self.assertEqual(specify._incomplete_wiring_ids(spec, self.root), {})

    def test_future_import_in_create_file_not_flagged(self):
        # Real regression: a create_file red test opens with the idiomatic
        # `from __future__ import annotations`. That binding is a compiler directive
        # never referenced by name, so the unused-import check must NOT flag it (it
        # previously did, downgrading a valid red test to needs_reinvestigation).
        spec = {"edits": [{
            "id": "E3", "kind": "create_file", "file": "tests/test_new.py",
            "content": ("from __future__ import annotations\n\n"
                        "def test_x():\n    assert 1 + 1 == 2\n"),
        }]}
        self.assertEqual(specify._incomplete_wiring_ids(spec, self.root), {})

    def test_future_import_line_yields_no_names(self):
        # the unit underneath: a future-statement binds no usable name
        self.assertEqual(specify._imported_names("from __future__ import annotations"), [])

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


class TestDatasourceRegressionGate(unittest.TestCase):
    """N176: an edit that switches a SQL read's primary FROM table to an empty / sparser /
    missing table (checked against the live DB) is a data-source regression — flagged
    deterministically so a ready spec loops back to re-retrieve the real source."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "t.db")
        c = sqlite3.connect(self.db)
        c.execute("CREATE TABLE groups (project_id TEXT, module TEXT)")
        c.executemany("INSERT INTO groups VALUES (?,?)",
                      [("test", "none"), ("test", "alpha")])  # 2 rows
        c.execute("CREATE TABLE project_modules (project_id TEXT, name TEXT)")
        c.execute("INSERT INTO project_modules VALUES ('test','alpha')")  # 1 row
        c.execute("CREATE TABLE empty_modules (project_id TEXT, name TEXT)")  # 0 rows
        c.commit()
        c.close()
        self.conn = DbConnection(kind="sqlite", path=self.db)

    def _spec(self, old_sql, new_sql):
        return {"edits": [{"id": "E1", "file": "list_routes.py",
                           "anchor_old": old_sql, "replacement_new": new_sql}]}

    def test_swap_to_sparser_ssot_table_not_flagged(self):
        # M036/회귀2: groups (2 denormalized rows) -> project_modules (1 clean SSOT row) is
        # the CORRECT fix. Row count is not coverage, so the fewer-rows swap must NOT be
        # flagged — the earlier "strictly fewer rows" branch mis-fired on exactly this swap.
        spec = self._spec(
            'rows = q("SELECT DISTINCT module FROM groups WHERE project_id = ?", [p])',
            'rows = q("SELECT name AS module FROM project_modules WHERE project_id = ?", [p])')
        self.assertEqual(specify._datasource_regression_ids(spec, self.conn), {})

    def test_swap_to_empty_table_is_flagged(self):
        spec = self._spec('SELECT module FROM groups WHERE project_id = ?',
                          'SELECT name FROM empty_modules WHERE project_id = ?')
        flagged = specify._datasource_regression_ids(spec, self.conn)
        self.assertIn("E1", flagged)
        self.assertIn("EMPTY", flagged["E1"])

    def test_swap_to_missing_table_is_flagged(self):
        spec = self._spec('SELECT module FROM groups',
                          'SELECT name FROM nonexistent_table')
        flagged = specify._datasource_regression_ids(spec, self.conn)
        self.assertIn("E1", flagged)
        self.assertIn("does not exist", flagged["E1"])

    def test_swap_to_richer_table_not_flagged(self):
        # project_modules (1) -> groups (2): coverage grows, not a regression
        spec = self._spec('SELECT name FROM project_modules',
                          'SELECT DISTINCT module FROM groups')
        self.assertEqual(specify._datasource_regression_ids(spec, self.conn), {})

    def test_same_table_not_flagged(self):
        spec = self._spec('SELECT module FROM groups WHERE project_id = ?',
                          'SELECT DISTINCT module FROM groups WHERE project_id = ? ORDER BY module')
        self.assertEqual(specify._datasource_regression_ids(spec, self.conn), {})

    def test_no_db_conn_is_no_op(self):
        spec = self._spec('SELECT module FROM groups',
                          'SELECT name FROM project_modules')
        self.assertEqual(specify._datasource_regression_ids(spec, None), {})

    def test_non_sql_edit_ignored(self):
        spec = self._spec("x = 1", "x = 2")
        self.assertEqual(specify._datasource_regression_ids(spec, self.conn), {})

    def test_gate_downgrades_and_routes_to_retrieve(self):
        spec = _fresh_ready()
        out = specify._apply_effectiveness_gate(
            spec, [], {}, False,
            datasource_ids={"E1": "data-source regression — groups -> project_modules"})
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertIn("E1", out["effectiveness"]["ineffective_ids"])
        self.assertEqual(out["reinvestigation"]["reason_code"],
                         specify.RI_DATASOURCE_REGRESSION)


class TestUndefinedColumnGate(unittest.TestCase):
    """T906/T907: an edit whose SQL names a column ABSENT from the live schema (or a test
    fixture that INSERTs such a column) is unrunnable — flagged deterministically so a
    ready spec loops back instead of shipping SQL that cannot execute. Schema mirrors
    FlowGate migration 028: project_modules has name/title, NOT module/is_active."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "t.db")
        c = sqlite3.connect(self.db)
        c.execute("CREATE TABLE projects (project_id TEXT PRIMARY KEY, project_name TEXT, "
                  "is_active INTEGER, created_at TEXT, updated_at TEXT)")
        c.execute("CREATE TABLE project_modules (module_id TEXT PRIMARY KEY, "
                  "project_id TEXT, name TEXT, title TEXT, created_at TEXT, updated_at TEXT)")
        c.commit()
        c.close()
        self.conn = DbConnection(kind="sqlite", path=self.db)

    def _spec(self, **edit):
        edit.setdefault("id", "E1")
        edit.setdefault("file", "store.py")
        return {"edits": [edit]}

    def test_qualified_missing_column_flagged(self):
        # T906: COALESCE(pm.module, '') — project_modules has no 'module' column.
        spec = self._spec(anchor_old="SELECT 1", replacement_new=(
            "SELECT p.project_id, COALESCE(pm.module, '') AS module FROM projects p "
            "LEFT JOIN project_modules pm ON pm.project_id = p.project_id"))
        flagged = specify._undefined_column_ids(spec, self.conn)
        self.assertIn("E1", flagged)
        self.assertIn("project_modules.module", flagged["E1"])

    def test_join_condition_missing_column_flagged(self):
        # T907: ... AND pm.is_active = 1 — project_modules has no 'is_active' column.
        spec = self._spec(anchor_old="SELECT 1", replacement_new=(
            "SELECT projects.project_id, pm.name AS module FROM projects "
            "LEFT JOIN project_modules pm ON pm.project_id = projects.project_id "
            "AND pm.is_active = 1 WHERE projects.is_active = 1"))
        flagged = specify._undefined_column_ids(spec, self.conn)
        self.assertIn("E1", flagged)
        self.assertIn("project_modules.is_active", flagged["E1"])
        # projects.is_active is real and must NOT be flagged
        self.assertNotIn("projects.is_active", flagged["E1"])

    def test_correct_columns_not_flagged(self):
        # The right fix (pm.name, no is_active) must pass clean.
        spec = self._spec(anchor_old="SELECT 1", replacement_new=(
            "SELECT projects.project_id AS project, projects.project_name, "
            "pm.name AS module FROM projects LEFT JOIN project_modules pm "
            "ON pm.project_id = projects.project_id WHERE projects.is_active = 1"))
        self.assertEqual(specify._undefined_column_ids(spec, self.conn), {})

    def test_fixture_insert_missing_column_flagged(self):
        # T907 fixture: INSERT INTO project_modules (..., is_active, ...) — column absent.
        spec = self._spec(kind="create_file", file="server/tests/test_x.py", content=(
            "def test_x(test_db):\n"
            "    test_db.execute(\"INSERT INTO project_modules "
            "(project_id, name, title, is_active, created_at, updated_at) "
            "VALUES ('p','m','M',1,'now','now')\")\n"))
        flagged = specify._undefined_column_ids(spec, self.conn)
        self.assertIn("E1", flagged)
        self.assertIn("project_modules.is_active", flagged["E1"])

    def test_fixture_insert_real_columns_not_flagged(self):
        spec = self._spec(kind="create_file", file="server/tests/test_x.py", content=(
            "test_db.execute(\"INSERT INTO project_modules "
            "(project_id, name, title) VALUES ('p','m','M')\")\n"))
        self.assertEqual(specify._undefined_column_ids(spec, self.conn), {})

    def test_unresolved_alias_is_skipped(self):
        # A qualifier that is not a known table alias (CTE / object) must not be flagged.
        spec = self._spec(anchor_old="x", replacement_new=(
            "rows = res.data.items.map(m => m.module)"))
        self.assertEqual(specify._undefined_column_ids(spec, self.conn), {})

    def test_unknown_table_is_skipped(self):
        spec = self._spec(anchor_old="x", replacement_new=(
            "SELECT t.bogus FROM some_unknown_table t"))
        self.assertEqual(specify._undefined_column_ids(spec, self.conn), {})

    def test_no_db_conn_is_no_op(self):
        spec = self._spec(anchor_old="x", replacement_new=(
            "SELECT pm.module FROM project_modules pm"))
        self.assertEqual(specify._undefined_column_ids(spec, None), {})

    def test_non_sql_edit_ignored(self):
        spec = self._spec(anchor_old="x = 1", replacement_new="x = 2")
        self.assertEqual(specify._undefined_column_ids(spec, self.conn), {})

    def test_gate_downgrades_via_effectiveness(self):
        spec = _fresh_ready()
        out = specify._apply_effectiveness_gate(
            spec, [], {}, False,
            datasource_ids={"E1": "undefined SQL column(s) — project_modules.is_active "
                            "not present in the live DB schema"})
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertIn("E1", out["effectiveness"]["ineffective_ids"])

    def test_alias_map_resolves_join_aliases(self):
        amap = specify._sql_alias_map(
            "FROM projects p LEFT JOIN project_modules pm ON pm.x = p.y WHERE p.z = 1")
        self.assertEqual(amap.get("p"), "projects")
        self.assertEqual(amap.get("pm"), "project_modules")
        # the bare table name maps to itself; a trailing keyword is never read as an alias
        self.assertEqual(amap.get("projects"), "projects")
        self.assertNotIn("where", amap)


class TestCalleeContractGrounding(unittest.TestCase):
    """N175 round-2: an edit adds a USED call (showToast) but with the wrong argument
    order, because the callee's real signature was never on the table. We lift each
    called symbol's real definition out of the module the edit imports it from so the
    author writes the call right AND the tool-OFF reviewer can flag a mis-invoked call."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        comp = os.path.join(self.root, "client", "src", "composables")
        os.makedirs(comp, exist_ok=True)
        # The toast composable whose REAL signature takes (options) — NOT (msg, severity).
        with open(os.path.join(comp, "useToast.ts"), "w", encoding="utf-8") as f:
            f.write(
                "export function useToast() {\n"
                "  function showToast(options: { message: string; severity?: string }) {\n"
                "    toasts.push(options)\n"
                "  }\n"
                "  return { showToast }\n"
                "}\n")

    def _modal_edit(self):
        return {"edits": [{
            "id": "E1", "file": "client/src/NewRequirementModal.vue",
            "anchor_old": "// marker",
            "replacement_new": (
                "import { useToast } from '@/composables/useToast'\n"
                "const { showToast } = useToast()\n"
                "showToast(t('main.x.error_group_r_exists'), 'danger')\n"),
        }]}

    def test_called_symbols_filters_keywords(self):
        syms = specify._called_symbols("if (a) { showToast(x); JSON(y); foo() }")
        self.assertIn("showToast", syms)
        self.assertIn("foo", syms)
        self.assertNotIn("if", syms)
        self.assertNotIn("JSON", syms)

    def test_resolve_alias_module(self):
        f = specify._resolve_module_file(
            "@/composables/useToast", None, [self.root])
        self.assertIsNotNone(f)
        self.assertTrue(f.endswith("useToast.ts"))

    def test_resolve_relative_module(self):
        f = specify._resolve_module_file(
            "./useToast", "client/src/composables/Other.ts", [self.root])
        self.assertIsNotNone(f)
        self.assertTrue(f.endswith("useToast.ts"))

    def test_external_module_resolves_to_none(self):
        self.assertIsNone(specify._resolve_module_file("vue", None, [self.root]))

    def test_collect_lifts_real_signature(self):
        contracts = specify._collect_callee_contracts(
            specify._edit_call_items(self._modal_edit()), [self.root])
        syms = {s for s, _ in contracts}
        self.assertIn("showToast", syms)
        sig = dict(contracts)["showToast"]["text"]
        # The real signature (options object) is now on the table — the author's guessed
        # positional ('danger') call can be seen to not match it.
        self.assertIn("options", sig)
        self.assertIn("severity", sig)

    def test_unresolved_callee_is_not_fabricated(self):
        # A call to a symbol with no resolvable def yields no contract (no phantom).
        edit = {"edits": [{
            "id": "E1", "file": "client/src/NewRequirementModal.vue",
            "anchor_old": "// m",
            "replacement_new": "mysteryHelper(a, b)\n"}]}
        contracts = specify._collect_callee_contracts(
            specify._edit_call_items(edit), [self.root])
        self.assertEqual(contracts, [])

    def test_review_prompt_carries_callee_signature(self):
        prompt = specify.build_review_prompt("honey", self._modal_edit(), self.root)
        self.assertIn("## Callee contracts", prompt)   # the rendered block (not just mandate)
        self.assertIn("showToast", prompt)
        self.assertIn("severity", prompt)        # the real signature reached the reviewer
        self.assertIn("MIS-INVOKED", prompt)     # the mandate to flag a mismatched call

    def test_review_prompt_carries_full_create_file_content(self):
        # TR909: a long create_file (a red test) must reach the reviewer in FULL, with no
        # "(truncated)" marker injected into the content — that marker once read as the
        # file's real final line and false-failed a valid test, downgrading a ready spec.
        body = "\n".join(f"line_{i} = {i}" for i in range(1, 61)) + "\nassert line_60 == 60\n"
        spec = {"edits": [{"id": "E1", "kind": "create_file",
                           "file": "server/tests/test_big.py", "content": body,
                           "rationale": "red test"}]}
        prompt = specify.build_review_prompt("honey", spec, self.root)
        self.assertIn("assert line_60 == 60", prompt)   # the final line survived
        self.assertNotIn("(truncated)", prompt)

    def test_review_prompt_no_block_when_nothing_resolves(self):
        # The MANDATE always names "Callee contracts"; the rendered BLOCK ('## …') only
        # appears when a callee actually resolved. Nothing resolves here → no block.
        edit = {"edits": [{"id": "E1", "file": "x.py",
                           "anchor_old": "a", "replacement_new": "b = 1"}]}
        prompt = specify.build_review_prompt("honey", edit, self.root)
        self.assertNotIn("## Callee contracts", prompt)

    def test_real_n175_shape_import_and_call_in_separate_edits(self):
        # The ACTUAL N175 spec: E2 adds the import, E3 the binding, E4 the wrong-order
        # call — three edits on ONE file. Per-edit grounding would see E4's call with no
        # import and miss it; per-file aggregation resolves the module from E2 so the real
        # signature still reaches the reviewer. Mirrors FlowGate's relative './common/useToast'.
        comp = os.path.join(self.root, "client", "src", "main", "components", "common")
        os.makedirs(comp, exist_ok=True)
        with open(os.path.join(comp, "useToast.ts"), "w", encoding="utf-8") as f:
            f.write("export function useToast() {\n"
                    "  function showToast(message: string, type: ToastType = 'info') {\n"
                    "    toasts.value.push({ message, type })\n"
                    "  }\n"
                    "  return { toasts, showToast }\n"
                    "}\n")
        modal = "client/src/main/components/NewRequirementModal.vue"
        os.makedirs(os.path.dirname(os.path.join(self.root, modal)), exist_ok=True)
        with open(os.path.join(self.root, modal), "w", encoding="utf-8") as f:
            f.write("// imports\n// binding\n// handler\n")
        spec = {"edits": [
            {"id": "E2", "file": modal, "anchor_old": "// imports",
             "replacement_new": "// imports\nimport { useToast } from './common/useToast'"},
            {"id": "E3", "file": modal, "anchor_old": "// binding",
             "replacement_new": "// binding\nconst { showToast } = useToast()"},
            {"id": "E4", "file": modal, "anchor_old": "// handler",
             "replacement_new": "// handler\nshowToast('danger', t('main.x.error_group_r_exists'))"},
        ]}
        contracts = dict(specify._collect_callee_contracts(
            specify._edit_call_items(spec, self.root), [self.root]))
        self.assertIn("showToast", contracts)
        # The real signature (message FIRST, type SECOND) is grounded — the swapped call
        # ('danger' first) can now be seen to be mis-invoked.
        self.assertIn("message: string, type", contracts["showToast"]["text"])
        prompt = specify.build_review_prompt("honey", spec, self.root)
        self.assertIn("## Callee contracts", prompt)
        self.assertIn("showToast(message: string", prompt)

    def test_local_def_resolves_without_import(self):
        # A callee defined in the EDITED file itself (no import) still grounds.
        with open(os.path.join(self.root, "helpers.py"), "w", encoding="utf-8") as f:
            f.write("def compute(scale, offset):\n    return scale + offset\n")
        edit = {"edits": [{"id": "E1", "file": "helpers.py",
                           "anchor_old": "# call", "replacement_new": "compute(1, 2)\n"}]}
        contracts = dict(specify._collect_callee_contracts(
            specify._edit_call_items(edit), [self.root]))
        self.assertIn("compute", contracts)
        self.assertIn("def compute(scale, offset)", contracts["compute"]["text"])


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

    def test_inconclusive_downgrades_to_reinvestigation(self):
        out = specify._apply_effectiveness_gate(_fresh_ready(), [], {}, True)
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_all_good_keeps_ready(self):
        out = specify._apply_effectiveness_gate(
            _fresh_ready(), [], {"E1": {"effective": True, "coherent": True}}, False)
        self.assertEqual(out["termination"], "ready_to_apply")

    def test_never_upgrades_a_non_ready_spec(self):
        spec = _fresh_ready()
        spec["termination"] = "needs_reinvestigation"
        out = specify._apply_effectiveness_gate(
            spec, ["E1"], {"E1": {"effective": False}}, True)
        self.assertEqual(out["termination"], "needs_reinvestigation")

    # N174 #3: a review-only "ineffective" verdict on VERIFIED edits in a CROSS-FILE
    # wiring must not assert the fix is wrong; it is held as inconclusive, which loops
    # back to re-investigate (there is no human-handoff terminal).
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

    def test_cross_file_verified_review_ineffective_holds_inconclusive(self):
        out = specify._apply_effectiveness_gate(
            self._cross_file_ready(), [],
            {"E1": {"effective": False, "reason": "alone doesn't change behavior"}}, False)
        # held as inconclusive → loops back to re-investigate, not asserted-wrong
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertIn("E1", out["effectiveness"]["ineffective_ids"])
        # the distinctive cross-file branch fired (note records why it is not asserted-wrong)
        self.assertIn("cross-file wiring", out["notes"])

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

    def test_unparseable_review_downgrades_to_reinvestigation(self):
        # Unusable on BOTH the attempt and the retry → inconclusive → needs_reinvestigation.
        with mock.patch.object(specify, "call_worker",
                               side_effect=[_wr(json.dumps(_READY_SPEC)),
                                            _wr("● no json\n"), _wr("still no json\n")]):
            spec = specify.run_specify(
                honey_path=self.honey, codebase_root=self.tmp,
                output_path=self.out, contract_path=self.contract)
        self.assertEqual(spec["termination"], "needs_reinvestigation")


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
        self.assertEqual(led.begin_call.call_count, 2)   # paid retry recorded
        self.assertEqual(led.finish_call.call_count, 2)


class TestDecisivenessGate(unittest.TestCase):
    """specify._apply_decisiveness_gate — guarded needs_reinvestigation -> ready_to_apply
    promotion of an over-conservative hedge whose edits are all verified+effective."""

    def _hedged(self, **over):
        spec = {
            "edits": [{
                "id": "E1", "file": "a.py",
                "anchor_old": "x = 1", "replacement_new": "x = 2",
                "confidence": "medium", "anchor_status": "verified",
            }],
            "deferred": [{"issue": "optional UX option", "reason": "policy_direction"}],
            "termination": "needs_reinvestigation",
            "effectiveness": {"inconclusive": False, "ineffective_ids": []},
            "notes": "hedged on authoritative UX",
        }
        spec.update(over)
        return spec

    def test_promotes_verified_effective_with_policy_deferred(self):
        out = specify._apply_decisiveness_gate(self._hedged())
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertIn("decisiveness gate", out["notes"])

    def test_create_file_edit_promotes(self):
        spec = self._hedged(edits=[{
            "id": "E1", "kind": "create_file", "file": "new.py",
            "content": "x = 1\n", "confidence": "medium",
        }])
        self.assertEqual(
            specify._apply_decisiveness_gate(spec)["termination"], "ready_to_apply")

    def test_low_confidence_blocks(self):
        spec = self._hedged()
        spec["edits"][0]["confidence"] = "low"
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"],
                         "needs_reinvestigation")

    def test_ineffective_id_blocks(self):
        spec = self._hedged()
        spec["effectiveness"]["ineffective_ids"] = ["E1"]
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"],
                         "needs_reinvestigation")

    def test_inconclusive_blocks(self):
        spec = self._hedged()
        spec["effectiveness"]["inconclusive"] = True
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"],
                         "needs_reinvestigation")

    def test_non_optional_deferred_blocks(self):
        spec = self._hedged()
        spec["deferred"] = [{"issue": "x", "reason": "needs_runtime"}]
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"],
                         "needs_reinvestigation")

    def test_missing_effectiveness_key_blocks(self):
        spec = self._hedged()
        del spec["effectiveness"]
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"],
                         "needs_reinvestigation")

    def test_unverified_anchor_blocks(self):
        spec = self._hedged()
        spec["edits"][0]["anchor_status"] = "stale"
        self.assertEqual(specify._apply_decisiveness_gate(spec)["termination"],
                         "needs_reinvestigation")

    def test_leaves_needs_runtime_untouched(self):
        # the gate only ever promotes a needs_reinvestigation hedge; needs_runtime is
        # a concrete blocked-on-a-datum verdict and must never be promoted.
        spec = self._hedged(termination="needs_runtime")
        self.assertEqual(
            specify._apply_decisiveness_gate(spec)["termination"], "needs_runtime")

    def test_leaves_ready_untouched(self):
        spec = self._hedged(termination="ready_to_apply")
        self.assertEqual(
            specify._apply_decisiveness_gate(spec)["termination"], "ready_to_apply")

    def test_substantive_deferred_no_longer_promotes(self):
        # multi_file_design / not_expressible_as_edit left the optional set (N176):
        # the decisiveness gate must NOT promote a hedge that punted a substantive fix.
        for reason in ("multi_file_design", "not_expressible_as_edit"):
            spec = self._hedged()
            spec["deferred"] = [{"issue": "real fix spans BE+FE", "reason": reason}]
            self.assertEqual(
                specify._apply_decisiveness_gate(spec)["termination"],
                "needs_reinvestigation", reason)


class TestDeferredSubstanceGate(unittest.TestCase):
    """specify._apply_deferred_substance_gate — a ready_to_apply spec that punted a
    substantive fix to deferred[] is downgraded (N176 cheap-path guard)."""

    def _ready(self, deferred):
        return {
            "edits": [{"id": "E1", "file": "Modal.vue",
                       "anchor_old": "x", "replacement_new": "y",
                       "confidence": "high", "anchor_status": "verified"}],
            "deferred": deferred,
            "termination": "ready_to_apply",
            "notes": "fixed the front-end",
        }

    def test_multi_file_design_downgrades(self):
        spec = specify._apply_deferred_substance_gate(
            self._ready([{"issue": "list_routes ignores project_modules (BE+FE)",
                          "reason": "multi_file_design"}]))
        self.assertEqual(spec["termination"], "needs_reinvestigation")
        self.assertEqual(spec["reinvestigation"]["reason_code"],
                         specify.RI_DEFERRED_ROOT_CAUSE)
        self.assertIn("deferred-substance gate", spec["notes"])

    def test_not_expressible_as_edit_downgrades(self):
        spec = specify._apply_deferred_substance_gate(
            self._ready([{"issue": "backend query rewrite",
                          "reason": "not_expressible_as_edit"}]))
        self.assertEqual(spec["termination"], "needs_reinvestigation")
        self.assertEqual(spec["reinvestigation"]["reason_code"],
                         specify.RI_DEFERRED_ROOT_CAUSE)

    def test_policy_direction_is_exempt(self):
        spec = specify._apply_deferred_substance_gate(
            self._ready([{"issue": "could add a tooltip", "reason": "policy_direction"}]))
        self.assertEqual(spec["termination"], "ready_to_apply")
        self.assertNotIn("reinvestigation", spec)

    def test_live_refuted_deferral_does_not_downgrade(self):
        # T907: a substantive-reason deferral whose OWN evidence shows the claim is refuted
        # by live code is a disproven hypothesis, not a punted root cause — it must NOT
        # downgrade an otherwise-complete ready spec.
        spec = specify._apply_deferred_substance_gate(self._ready([{
            "issue": "RequirementCreateView.vue:147-171 claimed shape-mismatch",
            "reason": "not_expressible_as_edit",
            "evidence": ["RequirementCreateView.vue:129-136 maps API modules into "
                         "{id,label} objects, so the claim is not observed in live code."],
        }]))
        self.assertEqual(spec["termination"], "ready_to_apply")
        self.assertNotIn("reinvestigation", spec)

    def test_live_refuted_korean_evidence_does_not_downgrade(self):
        spec = specify._apply_deferred_substance_gate(self._ready([{
            "issue": "shape mismatch 주장", "reason": "multi_file_design",
            "evidence": ["라이브 코드에서 {id,label} 매핑이 확인되어 반박됨"],
        }]))
        self.assertEqual(spec["termination"], "ready_to_apply")

    def test_substantive_deferral_without_refutation_still_downgrades(self):
        # Guard the N176 protection survives: a genuine punt (no refutation marker) still
        # downgrades.
        spec = specify._apply_deferred_substance_gate(self._ready([{
            "issue": "backend query must aggregate project_modules across modules",
            "reason": "multi_file_design"}]))
        self.assertEqual(spec["termination"], "needs_reinvestigation")

    def test_only_acts_on_ready(self):
        spec = self._ready([{"issue": "x", "reason": "multi_file_design"}])
        spec["termination"] = "needs_runtime"
        self.assertEqual(
            specify._apply_deferred_substance_gate(spec)["termination"], "needs_runtime")

    def test_no_deferred_is_noop(self):
        spec = specify._apply_deferred_substance_gate(self._ready([]))
        self.assertEqual(spec["termination"], "ready_to_apply")

    # ── Converge-certified escape (N182) ───────────────────────────────────────
    def _honey(self, locus="Modal.vue:31-53", ungrounded=False, multi=False):
        flag = " (⚠ attributed file not in evidence — re-confirm it exists)" \
            if ungrounded else ""
        lines = [
            "## Converged call path (the single executed path — START HERE)",
            "",
            "### Primary edit target — attributed defect",
            f"- location: {locus}{flag}",
            "- node: render",
            "",
        ]
        if multi:  # a multi-locus declaration the coverage gate owns instead
            lines += [specify.CONVERGE_TARGET_SECTION + " (author or defer EACH)", "",
                      "- Modal.vue:31-53", "- other_service.py:80-90", ""]
        return "\n".join(lines)

    def test_converge_certified_locus_escape_keeps_ready(self):
        # converge certified the single locus the edit lands on → substantive deferred
        # peer is secondary, ship the certified fix ready.
        spec = specify._apply_deferred_substance_gate(
            self._ready([{"issue": "backend query rewrite",
                          "reason": "multi_file_design"}]),
            self._honey())
        self.assertEqual(spec["termination"], "ready_to_apply")
        self.assertNotIn("reinvestigation", spec)
        self.assertEqual(spec["deferred_substance_escape"]["certified_locus"], "Modal.vue")

    def test_escape_requires_edit_on_certified_locus(self):
        # converge certified a DIFFERENT file than the edit touches → no escape, downgrade.
        spec = specify._apply_deferred_substance_gate(
            self._ready([{"issue": "x", "reason": "multi_file_design"}]),
            self._honey(locus="other_service.py:80-90"))
        self.assertEqual(spec["termination"], "needs_reinvestigation")
        self.assertNotIn("deferred_substance_escape", spec)

    def test_escape_blocked_when_attribution_ungrounded(self):
        # converge warned the attributed file may not exist → not a solid certification.
        spec = specify._apply_deferred_substance_gate(
            self._ready([{"issue": "x", "reason": "multi_file_design"}]),
            self._honey(ungrounded=True))
        self.assertEqual(spec["termination"], "needs_reinvestigation")

    def test_escape_not_applied_in_multilocus(self):
        # multi-locus convergence keeps its own stricter coverage gate — no escape here.
        spec = specify._apply_deferred_substance_gate(
            self._ready([{"issue": "x", "reason": "multi_file_design"}]),
            self._honey(multi=True))
        self.assertEqual(spec["termination"], "needs_reinvestigation")


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
    """Specify/review role routing.

    Loads an EXPLICIT fixture config (not the live repo-root hive.config.json — that
    file is now gitignored and absent in a fresh clone / CI, so reading it here was
    fragile). The fixture encodes the canonical deployment choice: specify is a
    tool-ON author (re-opens live files to lift anchors) so it stays on the file-tool
    provider copilot; the reviewer is a tool-OFF single-shot, opted onto the
    OpenAI-compatible HTTP provider as a cost lever (see hivework-model-placement)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "hive.config.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump({"roles": {
                "specify": {"provider": "copilot", "model": "gpt-5-mini"},
                "review":  {"provider": "deepinfra", "model": "openai/gpt-oss-120b"},
            }}, f)

    def test_specify_role_exists(self):
        role = load_config(path=self.config_path).role("specify")
        self.assertEqual(role.provider, "copilot")
        self.assertEqual(role.model, "gpt-5-mini")

    def test_review_role_routes_to_deepinfra(self):
        role = load_config(path=self.config_path).role("review")
        self.assertEqual(role.provider, "deepinfra")
        self.assertEqual(role.model, "openai/gpt-oss-120b")

    def test_cli_model_override_reaches_specify(self):
        cfg = load_config(path=self.config_path)
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
        self.assertEqual(out["termination"], "needs_reinvestigation")
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
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_needs_runtime_termination_is_valid(self):
        problems = specify._validate_spec({
            "edits": [],
            "deferred": [{"issue": "needs row state", "reason": "needs_runtime"}],
            "gate": {},
            "termination": "needs_runtime",
        })
        self.assertEqual(problems, [])

    def test_needs_pm_termination_is_retired_and_invalid(self):
        # needs_pm is no longer part of the vocabulary (no human-handoff terminal).
        self.assertNotIn("needs_pm", specify.VALID_TERMINATION)
        problems = specify._validate_spec({
            "edits": [], "deferred": [], "gate": {}, "termination": "needs_pm",
        })
        self.assertTrue(any("invalid termination" in p for p in problems))


class TestSeedCoverageGate(unittest.TestCase):
    """Defect 2 (T892): a seed-named edit target must become an edit, or a ready
    spec is downgraded to needs_reinvestigation with the dropped target reported."""

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
        self.assertEqual(out["termination"], "needs_reinvestigation")
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


_HONEY_MULTI_LOCUS = (
    "## Converged — MULTIPLE INDEPENDENT defects (START HERE)\n\n"
    "### Additional independent defects — each needs its OWN edit\n\n"
    "- client/src/main/DocInfoPanel.vue:40-52 [fe] — badge never flips\n"
    "- server/workflow_decision_service.py:70-95 [handler] — step index off\n\n"
    "## Converge-attributed edit targets (author or explicitly defer EACH)\n\n"
    "- client/src/app.css:12-14\n"
    "- client/src/main/DocInfoPanel.vue:40-52\n"
    "- server/workflow_decision_service.py:70-95\n\n"
    "## Grounded localisations (investigation evidence — NOT a list of edit sites)\n\n"
    "- SEED_ANCHOR: located\n"
)


class TestConvergeCoverageGate(unittest.TestCase):
    """N179: a genuine MULTI-locus convergence must not ship a ready spec that authored
    an edit for only some of the independent loci. Fires only when converge declared
    ≥2 independent loci (single-defect converges emit no section → no-op)."""

    def test_loci_parsed_from_section(self):
        self.assertEqual(
            specify._converge_target_loci(_HONEY_MULTI_LOCUS),
            ["client/src/app.css",
             "client/src/main/DocInfoPanel.vue",
             "server/workflow_decision_service.py"])

    def test_downgrades_ready_when_only_one_locus_covered(self):
        # The N179 shape: 3 independent loci declared, only the app.css selector authored.
        spec = {
            "edits": [{"id": "E1", "file": "client/src/app.css"}],
            "deferred": [], "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_converge_coverage_gate(spec, _HONEY_MULTI_LOCUS)
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertEqual(out["reinvestigation"]["reason_code"],
                         specify.RI_CONVERGE_LOCUS_UNCOVERED)
        uncovered = out["converge_coverage"]["uncovered"]
        self.assertEqual(len(uncovered), 2)  # both non-css loci are uncovered
        self.assertIn("client/src/main/DocInfoPanel.vue", uncovered)
        self.assertIn("server/workflow_decision_service.py", uncovered)
        self.assertIn("converge-coverage gate", out["notes"])

    def test_passes_when_every_locus_edited(self):
        spec = {
            "edits": [
                {"id": "E1", "file": "client/src/app.css"},
                {"id": "E2", "file": "client/src/main/DocInfoPanel.vue"},
                {"id": "E3", "file": "server/workflow_decision_service.py"}],
            "deferred": [], "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_converge_coverage_gate(spec, _HONEY_MULTI_LOCUS)
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertEqual(out["converge_coverage"]["uncovered"], [])

    def test_count_escape_tolerates_reground(self):
        # Three distinct edits (enough to plausibly cover every locus) even though one
        # lands on a re-grounded file converge did not literally name → not downgraded.
        spec = {
            "edits": [
                {"id": "E1", "file": "client/src/app.css"},
                {"id": "E2", "file": "client/src/main/DocInfoPanel.vue"},
                {"id": "E3", "file": "server/workflow_decision_other.py"}],
            "deferred": [], "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_converge_coverage_gate(spec, _HONEY_MULTI_LOCUS)
        self.assertEqual(out["termination"], "ready_to_apply")

    def test_explicit_defer_counts_as_covered(self):
        spec = {
            "edits": [
                {"id": "E1", "file": "client/src/app.css"},
                {"id": "E2", "file": "client/src/main/DocInfoPanel.vue"}],
            "deferred": [{"issue": "workflow_decision_service.py needs runtime row state",
                          "reason": "needs_runtime"}],
            "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_converge_coverage_gate(spec, _HONEY_MULTI_LOCUS)
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertEqual(out["converge_coverage"]["uncovered"], [])

    def test_noop_when_single_locus_or_no_section(self):
        spec = {"edits": [{"id": "E1", "file": "x.css"}],
                "termination": "ready_to_apply"}
        out = specify._apply_converge_coverage_gate(spec, "no converge section here")
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertNotIn("converge_coverage", out)

    def test_records_diagnostics_but_does_not_upgrade_non_ready(self):
        spec = {"edits": [{"id": "E1", "file": "client/src/app.css"}],
                "deferred": [], "termination": "needs_reinvestigation"}
        out = specify._apply_converge_coverage_gate(spec, _HONEY_MULTI_LOCUS)
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertEqual(len(out["converge_coverage"]["uncovered"]), 2)


class TestReinvestigationReason(unittest.TestCase):
    """Step A: every gate that lands a spec in needs_reinvestigation stamps a
    machine-readable ``spec['reinvestigation']`` reason so the reactive bridge can
    route by cause. The structured field is the single source of truth (last gate
    wins) and is cleared when a spec is promoted back to ready_to_apply."""

    def test_normalize_stale_anchor_stamps_reason(self):
        spec = _fresh_ready()
        spec["edits"][0]["anchor_status"] = "stale"
        out = specify._normalize_spec(spec)
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertEqual(out["reinvestigation"]["reason_code"], specify.RI_STALE_ANCHOR)
        self.assertEqual(out["reinvestigation"]["gate"], "normalize")

    def test_normalize_legacy_needs_pm_stamps_reason(self):
        out = specify._normalize_spec({"edits": [], "termination": "needs_pm"})
        self.assertEqual(out["reinvestigation"]["reason_code"], specify.RI_LEGACY_COERCE)

    def test_effectiveness_ineffective_stamps_reason(self):
        out = specify._apply_effectiveness_gate(
            _fresh_ready(), [],
            {"E1": {"effective": False, "coherent": True, "reason": "never reached"}}, False)
        self.assertEqual(out["reinvestigation"]["reason_code"], specify.RI_INEFFECTIVE)
        self.assertEqual(out["reinvestigation"]["gate"], "effectiveness")

    def test_effectiveness_inconclusive_stamps_reason(self):
        out = specify._apply_effectiveness_gate(_fresh_ready(), [], {}, True)
        self.assertEqual(out["reinvestigation"]["reason_code"], specify.RI_INCONCLUSIVE)

    def test_anchor_not_grounded_stamps_reason(self):
        spec = {
            "edits": [{"id": "E1", "file": "server/sql/queries/queries.json",
                       "anchor_old": "x", "replacement_new": "y",
                       "anchor_status": "verified"}],
            "deferred": [{"issue": "queries.json not in evidence",
                          "reason": "anchor_not_grounded"}],
            "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_anchor_not_grounded_gate(spec)
        self.assertEqual(out["reinvestigation"]["reason_code"],
                         specify.RI_ANCHOR_NOT_GROUNDED)

    def test_seed_coverage_stamps_reason(self):
        spec = {
            "edits": [{"id": "E1", "file": "server/sql/queries/queries.json"}],
            "deferred": [], "termination": "ready_to_apply", "notes": "",
        }
        out = specify._apply_seed_coverage_gate(spec, _HONEY_WITH_TARGETS)
        self.assertEqual(out["reinvestigation"]["reason_code"],
                         specify.RI_SEED_TARGET_UNCOVERED)
        self.assertIn("workflowViewState.spec.ts", out["reinvestigation"]["detail"])

    def test_verify_consistency_removes_ghost_reference_and_stamps_reason(self):
        spec = {
            "edits": [],
            "deferred": [{"issue": "create test", "reason": "multi_file_design"}],
            "gate": {"apply": False},
            "verify": {
                "red_test_node": "server/tests/test_x.py::test_x",
                "test_edit_ids": ["E1"],
            },
            "termination": "needs_reinvestigation",
            "notes": "",
        }
        out = specify._apply_verify_consistency_gate(spec)
        self.assertNotIn("verify", out)
        self.assertEqual(
            out["verify_consistency"]["missing_test_edit_ids"], ["E1"])
        self.assertEqual(
            out["reinvestigation"]["reason_code"],
            specify.RI_VERIFY_INCONSISTENT)
        self.assertEqual(out["reinvestigation"]["gate"], "verify_consistency")

    def test_verify_consistency_keeps_valid_ids_and_drops_only_missing(self):
        spec = _fresh_ready()
        spec["verify"] = {
            "red_test_node": "tests/test_x.py::test_x",
            "test_edit_ids": ["E1", "E9"],
        }
        out = specify._apply_verify_consistency_gate(spec)
        self.assertEqual(out["verify"]["test_edit_ids"], ["E1"])
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_decisiveness_cannot_repromote_verify_inconsistency(self):
        spec = _fresh_ready()
        spec["termination"] = "needs_reinvestigation"
        spec["effectiveness"] = {"inconclusive": False, "ineffective_ids": []}
        spec["verify_consistency"] = {"missing_test_edit_ids": ["E9"]}
        out = specify._apply_decisiveness_gate(spec)
        self.assertEqual(out["termination"], "needs_reinvestigation")

    def test_decisiveness_promotion_clears_reason(self):
        # A downgraded spec carrying a structured reason, when promoted back to ready,
        # must not keep a stale reinvestigation field.
        spec = {
            "edits": [{"id": "E1", "file": "a.py", "anchor_old": "x = 1",
                       "replacement_new": "x = 2", "confidence": "high",
                       "anchor_status": "verified"}],
            "deferred": [{"issue": "optional", "reason": "policy_direction"}],
            "termination": "needs_reinvestigation",
            "effectiveness": {"inconclusive": False, "ineffective_ids": []},
            "reinvestigation": {"reason_code": specify.RI_INEFFECTIVE, "gate": "x",
                                "detail": "stale"},
            "notes": "",
        }
        out = specify._apply_decisiveness_gate(spec)
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertNotIn("reinvestigation", out)

    def test_author_declared_reason_filled_by_finalizer(self):
        spec = {"termination": "needs_reinvestigation",
                "notes": "author could not ground the head hop"}
        out = specify._ensure_reinvestigation_reason(spec)
        self.assertEqual(out["reinvestigation"]["reason_code"],
                         specify.RI_AUTHOR_DECLARED)
        self.assertIn("head hop", out["reinvestigation"]["detail"])

    def test_finalizer_strips_reason_from_non_nr_spec(self):
        spec = {"termination": "ready_to_apply",
                "reinvestigation": {"reason_code": "stale"}}
        out = specify._ensure_reinvestigation_reason(spec)
        self.assertNotIn("reinvestigation", out)


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


class TestDisambiguateAnchors(unittest.TestCase):
    """M-head case: the same literal repeats in two branches. Sibling edits sharing that
    non-unique anchor are widened with adjacent live lines so each targets one occurrence
    uniquely — instead of dead-ending in apply's anchor_ambiguous."""

    # a file where NON_HEAD_TYPES = {"R","M","Q"} appears in TWO distinct branches
    _SRC = (
        "def group_head(items):\n"
        "    # group branch\n"
        "    NON_HEAD_TYPES = {\"R\",\"M\",\"Q\"}\n"
        "    return pick(items, NON_HEAD_TYPES)\n"
        "\n"
        "def seq_items(items):\n"
        "    # sequence branch\n"
        "    NON_HEAD_TYPES = {\"R\",\"M\",\"Q\"}\n"
        "    return order(items, NON_HEAD_TYPES)\n"
    )

    def _edit(self, eid):
        return {"id": eid, "file": "documents.py",
                "anchor_old": "NON_HEAD_TYPES = {\"R\",\"M\",\"Q\"}",
                "replacement_new": "NON_HEAD_TYPES = {\"R\",\"Q\"}",
                "anchor_status": "verified"}

    def _spec(self, edits):
        return {"edits": edits, "deferred": [], "gate": {"apply": False},
                "termination": "ready_to_apply"}

    def test_identical_sibling_anchors_widened_to_unique(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "documents.py"), "w", encoding="utf-8") as fh:
                fh.write(self._SRC)
            spec = self._spec([self._edit("E1"), self._edit("E2")])
            out = specify._disambiguate_anchors(spec, root)
            with open(os.path.join(root, "documents.py"), encoding="utf-8") as fh:
                text = fh.read()
        anchors = [e["anchor_old"] for e in out["edits"]]
        # each widened anchor is now uniquely present in live source
        for a in anchors:
            self.assertEqual(text.count(a), 1, a)
        self.assertNotEqual(anchors[0], anchors[1])
        for e in out["edits"]:
            self.assertIn("anchor_disambiguated", e)
            # the inner change survives byte-for-byte inside the widened replacement
            self.assertIn("NON_HEAD_TYPES = {\"R\",\"Q\"}", e["replacement_new"])

    def test_widened_pair_then_survives_live_verify(self):
        # the whole point: after widening, _verify_anchors_live keeps them 'verified'
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "documents.py"), "w", encoding="utf-8") as fh:
                fh.write(self._SRC)
            spec = self._spec([self._edit("E1"), self._edit("E2")])
            spec = specify._disambiguate_anchors(spec, root)
            out = specify._verify_anchors_live(spec, root)
        self.assertEqual([e["anchor_status"] for e in out["edits"]],
                         ["verified", "verified"])

    def test_single_edit_non_unique_is_left_alone(self):
        # a lone edit on a non-unique anchor is genuinely ambiguous → not widened
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "documents.py"), "w", encoding="utf-8") as fh:
                fh.write(self._SRC)
            spec = self._spec([self._edit("E1")])
            out = specify._disambiguate_anchors(spec, root)
        self.assertNotIn("anchor_disambiguated", out["edits"][0])
        self.assertEqual(out["edits"][0]["anchor_old"],
                         "NON_HEAD_TYPES = {\"R\",\"M\",\"Q\"}")

    def test_differing_replacements_not_disambiguated(self):
        # if the two edits want DIFFERENT replacements, the occurrence↔edit mapping is
        # not safe to guess → leave them for the downgrade path
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "documents.py"), "w", encoding="utf-8") as fh:
                fh.write(self._SRC)
            e1, e2 = self._edit("E1"), self._edit("E2")
            e2["replacement_new"] = "NON_HEAD_TYPES = {\"R\"}"
            out = specify._disambiguate_anchors(self._spec([e1, e2]), root)
        for e in out["edits"]:
            self.assertNotIn("anchor_disambiguated", e)


class TestFixtureGrounding(unittest.TestCase):
    """Red-test isolation grounding: lift the target's pytest fixtures from conftest so
    the author builds a data-dependent red test on the isolated harness, not get_store()."""

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def _conftest(self, rel, text):
        p = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)

    def test_lifts_fixture_name_and_summary(self):
        self._conftest("server/tests/conftest.py",
                       'import pytest\n\n'
                       '@pytest.fixture(scope="session")\n'
                       'def all_migrations_db():\n'
                       '    """Create a test database with all migrations applied."""\n'
                       '    yield None\n\n'
                       '@pytest.fixture\n'
                       'def test_db(all_migrations_db):\n'
                       '    """Provide a test database connection."""\n'
                       '    yield all_migrations_db\n')
        fx = specify._collect_test_fixtures(self.root)
        names = dict(fx)
        self.assertIn("all_migrations_db", names)
        self.assertIn("test_db", names)
        self.assertEqual(names["all_migrations_db"],
                         "Create a test database with all migrations applied.")
        block = specify._render_fixture_block(fx)
        self.assertIn("test_db", block)
        self.assertIn("NEVER let a red test read/write the production", block)

    def test_no_conftest_yields_empty(self):
        self.assertEqual(specify._collect_test_fixtures(self.root), [])
        self.assertEqual(specify._render_fixture_block([]), "")

    def test_missing_root_is_safe(self):
        self.assertEqual(specify._collect_test_fixtures(""), [])
        self.assertEqual(specify._collect_test_fixtures("/no/such/dir/xyz"), [])

    def test_lifts_db_test_wiring_example(self):
        # a test that patches get_store to a TestStore on a test DB — the wiring pattern
        self._conftest("server/tests/test_settings.py",
                       'from unittest.mock import patch\n'
                       'from modules.flow_gate.db.connection import get_store\n\n'
                       '@pytest.fixture(autouse=True)\n'
                       'def mock_db(test_db_path):\n'
                       '    store = TestStore(test_db_path)\n'
                       '    with patch("modules.flow_gate.db.connection.get_store", return_value=store):\n'
                       '        yield store\n\n'
                       'def test_thing():\n'
                       '    from modules.flow_gate.settings import get_all\n'
                       '    assert get_all() == []\n')
        ex = specify._lift_db_test_example(self.root)
        self.assertIn("from modules.flow_gate", ex)          # correct import root
        self.assertIn("patch(", ex)                           # the get_store patch
        self.assertIn("get_store", ex)
        self.assertIn("test_settings.py", ex)

    def test_db_example_none_when_no_store_pattern(self):
        self._conftest("server/tests/test_plain.py",
                       "def test_x():\n    assert 1 == 1\n")
        self.assertEqual(specify._lift_db_test_example(self.root), "")

    def test_db_example_prefers_string_target_patch_over_monkeypatch(self):
        # A big integration file that wires get_store via the error-prone
        # monkeypatch.setattr(alias, ...) style must NOT win just because it has many
        # `def test`s — even one string-target `patch("...get_store", ...)` file beats it.
        monkey = "from unittest.mock import patch\n"
        monkey += "import modules.flow_gate.db.groups as db_g\n\n"
        for i in range(30):
            monkey += (f"def test_case_{i}(monkeypatch):\n"
                       "    monkeypatch.setattr(db_g, 'get_store', lambda: store)\n"
                       "    assert True\n\n")
        self._conftest("server/tests/test_big_monkey.py", monkey)
        self._conftest("server/tests/test_clean.py",
                       'from unittest.mock import patch\n'
                       'from modules.flow_gate.db.connection import get_store\n\n'
                       'def test_one():\n'
                       '    with patch("modules.flow_gate.db.connection.get_store", '
                       'return_value=store):\n'
                       '        assert True\n')
        ex = specify._lift_db_test_example(self.root)
        self.assertIn("test_clean.py", ex)
        self.assertNotIn("test_big_monkey.py", ex)


class TestHttpShapeSynthesisPass(unittest.TestCase):
    """Lever ⑦: specify attaches an HTTP-shape red test so apply observes red→green."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self._tmp.name, "code")
        os.makedirs(os.path.join(self.root, "app"))
        os.makedirs(os.path.join(self.root, "tests"))
        with open(os.path.join(self.root, "app", "routes.py"), "w",
                  encoding="utf-8") as f:
            f.write('from fastapi import APIRouter\n'
                    'router = APIRouter(prefix="/api/v1")\n\n'
                    '@router.get("/projects")\n'
                    'def list_projects():\n'
                    '    return {"projects": _rows()}\n')
        with open(os.path.join(self.root, "tests", "conftest.py"), "w",
                  encoding="utf-8") as f:
            f.write('import pytest\n'
                    'from fastapi.testclient import TestClient\n\n'
                    '@pytest.fixture\n'
                    'def client():\n'
                    '    from app.routes import router\n'
                    '    from fastapi import FastAPI\n'
                    '    app = FastAPI(); app.include_router(router)\n'
                    '    return TestClient(app)\n')
        self.honey = ('const res = await getRequest("/api/v1/projects")\n'
                      'modules: Array.isArray(it.modules) ? it.modules : []\n')

    def tearDown(self):
        self._tmp.cleanup()

    def _spec(self):
        return {"edits": [{"id": "E1", "file": "app/routes.py",
                           "anchor_old": "a", "replacement_new": "b"}],
                "verify": {}}

    def test_attaches_red_test_and_wires_verify(self):
        spec = specify._synthesize_http_shape_red_test(
            self._spec(), self.honey, self.root)
        self.assertEqual(spec["verify"]["test_edit_ids"], ["HTTP_SHAPE_RED"])
        # node = "<test file>::<generated test fn>" (the URL is sanitised into the name)
        self.assertIn("test_http_shape_modules.py::", spec["verify"]["red_test_node"])
        self.assertIn("api_v1_projects", spec["verify"]["red_test_node"])
        red = [e for e in spec["edits"] if e["id"] == "HTTP_SHAPE_RED"]
        self.assertEqual(len(red), 1)
        self.assertEqual(red[0]["kind"], "create_file")

    def test_noop_when_author_red_test_already_present(self):
        spec = self._spec()
        spec["verify"] = {"red_test_node": "tests/test_x.py::t",
                          "test_edit_ids": ["E9"]}
        out = specify._synthesize_http_shape_red_test(spec, self.honey, self.root)
        self.assertEqual(out["verify"]["red_test_node"], "tests/test_x.py::t")
        self.assertFalse(any(e["id"] == "HTTP_SHAPE_RED" for e in out["edits"]))

    def test_noop_when_no_source_edit(self):
        spec = {"edits": [], "verify": {}}
        out = specify._synthesize_http_shape_red_test(spec, self.honey, self.root)
        self.assertNotIn("red_test_node", out.get("verify", {}))
        self.assertEqual(out["edits"], [])

    def test_kill_switch_disables_pass(self):
        with mock.patch.dict(os.environ, {specify._HTTP_SHAPE_ENV_OFF: "1"}):
            out = specify._synthesize_http_shape_red_test(
                self._spec(), self.honey, self.root)
        self.assertNotIn("red_test_node", out.get("verify", {}))
        self.assertFalse(any(e["id"] == "HTTP_SHAPE_RED" for e in out["edits"]))

    def test_fail_open_when_symptom_absent(self):
        out = specify._synthesize_http_shape_red_test(
            self._spec(), "no http symptom here", self.root)
        self.assertNotIn("red_test_node", out.get("verify", {}))

    def test_configured_test_dir_and_setup_block_are_forwarded(self):
        # The #1 wiring: config supplies a target-specific test_dir + setup_block
        # (its own seeded TestClient), so synthesis binds to THAT harness and places
        # the red test under the configured dir — not the "tests/" default.
        setup = ("import pytest\n"
                 "from fastapi.testclient import TestClient\n\n"
                 "@pytest.fixture\n"
                 "def seeded_client():\n"
                 "    from app.routes import router\n"
                 "    from fastapi import FastAPI\n"
                 "    app = FastAPI(); app.include_router(router)\n"
                 "    return TestClient(app)\n")
        spec = specify._synthesize_http_shape_red_test(
            self._spec(), self.honey, self.root,
            setup_block=setup, test_dir="server/tests")
        red = [e for e in spec["edits"] if e["id"] == "HTTP_SHAPE_RED"][0]
        self.assertTrue(red["file"].startswith("server/tests/"))
        self.assertTrue(spec["verify"]["red_test_node"].startswith("server/tests/"))
        # the supplied harness is prepended and its fixture name drives the test sig
        self.assertTrue(red["content"].startswith("import pytest"))
        self.assertIn("(seeded_client):", red["content"])


class TestWinningPathAndLayerGates(unittest.TestCase):
    """T909: off-path source/test edits cannot survive effectiveness."""

    def _honey(self, *, attributed="server/modules/flow_gate/db/projects.py"):
        nodes = [
            {"url": "/api/v1/projects", "verb": "GET", "role": "handler",
             "file": "server/modules/flow_gate/settings/routers/project_settings.py",
             "lines": "45-49", "symbol": "list_projects_endpoint", "depth": 0},
            {"url": "/api/v1/projects", "verb": "GET", "role": "producer",
             "file": "server/modules/flow_gate/db/projects.py",
             "lines": "19-27", "symbol": "list_projects", "depth": 1},
        ]
        lines = [
            "<!-- hive-winning-http-path: "
            + json.dumps(node, sort_keys=True) + " -->"
            for node in nodes
        ]
        lines.append(
            "<!-- hive-converge-attribution: "
            + json.dumps({
                "file": attributed, "lines": "19-27", "node": "db_fn",
                "converged": True, "causal_verdict": "consistent",
            }, sort_keys=True)
            + " -->"
        )
        return "\n".join(lines)

    @staticmethod
    def _ready(edits):
        return {
            "edits": edits, "deferred": [], "gate": {"apply": False},
            "termination": "ready_to_apply",
            "effectiveness": {"inconclusive": False, "ineffective_ids": []},
            "notes": "",
        }

    def test_t909_off_path_store_and_process_service_test_are_both_downgraded(self):
        spec = self._ready([
            {"id": "E1", "file": "server/modules/flow_gate/store.py",
             "anchor_old": "old", "replacement_new": "new",
             "anchor_status": "verified", "confidence": "high"},
            {"id": "E2", "kind": "create_file",
             "file": "server/tests/test_projects_endpoint_modules.py",
             "content": (
                 "from modules.flow_gate import process_service\n"
                 "def test_projects(test_db):\n"
                 "    assert process_service.get_projects_with_modules()\n"
             ),
             "confidence": "high"},
        ])
        out = specify._apply_winning_path_gate(spec, self._honey())
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertEqual(out["reinvestigation"]["gate"], "winning_path")
        self.assertEqual(out["effectiveness"]["ineffective_ids"], ["E1", "E2"])
        self.assertIn("off winning HTTP path",
                      out["edits"][0]["effectiveness"]["reason"])
        self.assertIn("off winning HTTP path",
                      out["edits"][1]["effectiveness"]["reason"])

    def test_legitimate_on_path_source_and_route_test_survive(self):
        spec = self._ready([
            {"id": "E1", "file": "server/modules/flow_gate/db/projects.py",
             "anchor_old": "old", "replacement_new": "new",
             "anchor_status": "verified", "confidence": "high"},
            {"id": "E2", "kind": "create_file",
             "file": "server/tests/test_projects_route_modules.py",
             "content": (
                 "def test_projects(client):\n"
                 "    response = client.get('/api/v1/projects')\n"
                 "    assert response.status_code == 200\n"
             ),
             "confidence": "high"},
        ])
        out = specify._apply_winning_path_gate(spec, self._honey())
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertEqual(out["effectiveness"]["ineffective_ids"], [])

    def test_converge_fe_author_be_contradiction_is_downgraded(self):
        spec = self._ready([
            {"id": "E1", "file": "server/modules/flow_gate/store.py",
             "anchor_old": "old", "replacement_new": "new",
             "anchor_status": "verified", "confidence": "high"},
        ])
        out = specify._apply_layer_consistency_gate(
            spec, self._honey(attributed="client/src/NewRequirementModal.vue"))
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertEqual(out["reinvestigation"]["gate"], "converge_author_layer")
        self.assertEqual(out["effectiveness"]["ineffective_ids"], ["E1"])

    def test_cross_layer_author_with_attributed_layer_present_is_allowed(self):
        spec = self._ready([
            {"id": "E1", "file": "client/src/NewRequirementModal.vue",
             "anchor_old": "old", "replacement_new": "new",
             "anchor_status": "verified", "confidence": "high"},
            {"id": "E2", "file": "server/modules/flow_gate/store.py",
             "anchor_old": "old2", "replacement_new": "new2",
             "anchor_status": "verified", "confidence": "high"},
        ])
        out = specify._apply_layer_consistency_gate(
            spec, self._honey(attributed="client/src/NewRequirementModal.vue"))
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertEqual(out["effectiveness"]["ineffective_ids"], [])

    def test_layer_gate_defers_to_winning_path_for_onpath_be_edit(self):
        # converge attributed FE, but the correct fix is the BE producer that the
        # deterministic winning-path proof reached. The layer gate must NOT fight that
        # producer grounding — the on-path BE edit is exonerated (winning-path is arbiter).
        spec = self._ready([
            {"id": "E1", "file": "server/modules/flow_gate/db/projects.py",
             "anchor_old": "old", "replacement_new": "new",
             "anchor_status": "verified", "confidence": "high"},
        ])
        out = specify._apply_layer_consistency_gate(
            spec, self._honey(attributed="client/src/NewRequirementModal.vue"))
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertEqual(out["effectiveness"]["ineffective_ids"], [])

    def test_winning_path_gate_skips_source_veto_when_proof_has_no_producer(self):
        # Handler-only proof (no producer node reached): the reachability proof is
        # incomplete for siting a producer fix, so a BE source edit is NOT failed-closed.
        # The off-path test rule still applies regardless of producer depth.
        handler_only = (
            "<!-- hive-winning-http-path: " + json.dumps(
                {"url": "/api/v1/projects", "verb": "GET", "role": "handler",
                 "file": "server/modules/flow_gate/settings/routers/project_settings.py",
                 "lines": "45-49", "symbol": "list_projects_endpoint", "depth": 0},
                sort_keys=True) + " -->"
        )
        spec = self._ready([
            {"id": "E1", "file": "server/modules/flow_gate/store.py",
             "anchor_old": "old", "replacement_new": "new",
             "anchor_status": "verified", "confidence": "high"},
            {"id": "E2", "kind": "create_file",
             "file": "server/tests/test_off_path.py",
             "content": (
                 "from modules.flow_gate import process_service\n"
                 "def test_x(test_db):\n"
                 "    assert process_service.get_projects_with_modules()\n"
             ),
             "confidence": "high"},
        ])
        out = specify._apply_winning_path_gate(spec, handler_only)
        # E1 (BE source) survives — proof reached no producer to site against.
        self.assertNotIn("E1", out["effectiveness"]["ineffective_ids"])
        # E2 (off-path test) is still flagged — the test-coverage contract is explicit.
        self.assertIn("E2", out["effectiveness"]["ineffective_ids"])


if __name__ == "__main__":
    unittest.main()
