"""Unit tests for hive.converge — the ④ stage that stitches per-axis verdicts
into one executed call path and attributes the defect to one node.

No real model calls: ``call_worker`` is patched with a scripted fake.
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import converge as C
from hive.investigate import classify_seed_kind, render_local_honey
from hive.providers import WorkerResult


def _wr(stdout: str, rc: int = 0) -> WorkerResult:
    return WorkerResult(stdout=stdout, stderr="", exit_code=rc, latency_s=0.01)


def _verdict(axis, located, file="", lines="", reason=""):
    return {"axis_id": axis, "title": axis,
            "verdict": {"located": located, "file": file, "lines": lines,
                        "reason": reason}}


# Two located fragments on different files (the N169 shape) + a bundle with a
# call-chain hop that links them.
LOCATED_VERDICTS = [
    _verdict("SEED_ANCHOR", True, "api/workflow_head_routes.py", "93-102",
             "handler returns full sequence"),
    _verdict("B", True, "db/workflow_sequences.py", "45-57",
             "get_effective_head mishandles completed R"),
]
BUNDLES = [
    {"axis_id": "SEED_ANCHOR",
     "code_snippets": [{"file": "api/workflow_head_routes.py", "lines": "93-102",
                        "text": "def get_workflow_head(...): return get_effective_head(...)"}],
     "call_chain": [{"file": "db/workflow_sequences.py", "lines": "45-57",
                     "text": "def get_effective_head(...): ORDER BY ...",
                     "via": "call-chain"}]},
    {"axis_id": "B",
     "code_snippets": [{"file": "db/workflow_sequences.py", "lines": "45-57",
                        "text": "def get_effective_head(...): ORDER BY in_progress"}],
     "call_chain": []},
]

CONVERGED_OUT = json.dumps({
    "converged": True,
    "path": [
        {"node": "endpoint", "file": "api/workflow_head_routes.py", "lines": "93-102",
         "symbol": "GET /workflow/{doc_id}/head"},
        {"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57",
         "symbol": "get_effective_head"},
    ],
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57", "why": "ORDER BY prefers in-progress over pending"},
    "missing_link": None,
})
MISSING_OUT = json.dumps({
    "converged": False,
    "path": [{"node": "endpoint", "file": "api/workflow_head_routes.py", "lines": "93-102"}],
    "attributed_defect": None,
    "missing_link": {"between": ["handler", "db_fn"],
                     "need": {"symbols": ["get_effective_head"], "greps": ["ORDER BY"],
                              "file_globs": []}},
})


class TestRunConverge(unittest.TestCase):
    def test_skips_when_fewer_than_two_located(self):
        """<2 located ⇒ nothing to stitch ⇒ NO model call (free skip)."""
        with mock.patch.object(C, "call_worker") as cw:
            res = C.run_converge(
                seed_text="trace the path",
                verdicts=[_verdict("A", True, "x.py", "1-2", "r"),
                          _verdict("B", False)],
                bundles=BUNDLES, provider="deepinfra", model="m")
        cw.assert_not_called()
        self.assertFalse(res.converged)
        self.assertIn("skipped", res.summary)

    def test_converges_and_attributes_one_node(self):
        with mock.patch.object(C, "call_worker", return_value=_wr(CONVERGED_OUT)) as cw:
            res = C.run_converge(seed_text="trace the path",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m")
        cw.assert_called_once()
        self.assertTrue(res.converged)
        self.assertEqual(res.attributed_defect["file"], "db/workflow_sequences.py")
        self.assertEqual(res.attributed_defect["node"], "db_fn")
        self.assertEqual(len(res.path), 2)
        self.assertNotIn("ungrounded", res.attributed_defect)

    def test_tool_off_single_shot(self):
        """Converge runs tool-OFF (available_tools=[]) like judge — no exploration."""
        seen = {}

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            seen.update(kw)
            return _wr(CONVERGED_OUT)

        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                           bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertEqual(seen.get("available_tools"), [])

    def test_missing_link_names_the_hop(self):
        with mock.patch.object(C, "call_worker", return_value=_wr(MISSING_OUT)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertFalse(res.converged)
        self.assertIsNotNone(res.missing_link)
        self.assertEqual(res.missing_link["between"], ["handler", "db_fn"])
        self.assertIn("get_effective_head", res.missing_link["need"]["symbols"])

    def test_unparseable_retries_then_degrades(self):
        """Garbage twice ⇒ one retry, then a graceful not-converged (never raises)."""
        with mock.patch.object(C, "call_worker", return_value=_wr("not json")) as cw:
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertEqual(cw.call_count, 2)   # initial + one JSON-only retry
        self.assertFalse(res.converged)

    def test_worker_failure_is_nonfatal(self):
        with mock.patch.object(C, "call_worker", side_effect=RuntimeError("boom")):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertFalse(res.converged)
        self.assertIn("failed", res.summary)

    def test_ungrounded_attribution_flagged_not_dropped(self):
        out = json.dumps({
            "converged": True, "path": [],
            "attributed_defect": {"node": "db_fn", "file": "totally/unseen.py",
                                  "lines": "1-2", "why": "x"},
            "missing_link": None})
        with mock.patch.object(C, "call_worker", return_value=_wr(out)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertTrue(res.converged)
        self.assertTrue(res.attributed_defect.get("ungrounded"))

    def test_converged_first_pass_makes_no_second_call(self):
        """A first pass that converges is trusted — no follow-up, single call."""
        with mock.patch.object(C, "call_worker", return_value=_wr(CONVERGED_OUT)) as cw, \
             mock.patch.object(C, "retrieve_followup") as rf:
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m",
                                 code_root="/repo")
        self.assertTrue(res.converged)
        self.assertEqual(cw.call_count, 1)
        rf.assert_not_called()

    def test_missing_link_triggers_followup_then_reconverges(self):
        """1st pass names a missing link → fetch it locally → 2nd pass converges."""
        outs = [_wr(MISSING_OUT), _wr(CONVERGED_OUT)]
        fu = {"seeds": [{"file": "db/workflow_sequences.py", "lines": "45-57",
                         "text": "def get_effective_head(): ORDER BY", "via": "need-symbol"}],
              "call_chain": [], "stats": {}}
        with mock.patch.object(C, "call_worker", side_effect=outs) as cw, \
             mock.patch.object(C, "retrieve_followup", return_value=fu) as rf:
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m",
                                 code_root="/repo")
        rf.assert_called_once()
        self.assertEqual(cw.call_count, 2)            # 1st + re-converge
        self.assertTrue(res.converged)               # adopted the re-pass
        self.assertEqual(res.attributed_defect["node"], "db_fn")

    def test_no_followup_without_code_root(self):
        """No code_root ⇒ cannot fetch the missing link ⇒ no second pass."""
        with mock.patch.object(C, "call_worker", return_value=_wr(MISSING_OUT)) as cw, \
             mock.patch.object(C, "retrieve_followup") as rf:
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        rf.assert_not_called()
        self.assertEqual(cw.call_count, 1)
        self.assertFalse(res.converged)
        self.assertIsNotNone(res.missing_link)        # link still named for the author

    def test_followup_that_still_fails_keeps_named_link(self):
        """Re-pass that still can't converge ⇒ keep the 1st result's missing link."""
        with mock.patch.object(C, "call_worker", side_effect=[_wr(MISSING_OUT), _wr(MISSING_OUT)]), \
             mock.patch.object(C, "retrieve_followup",
                               return_value={"seeds": [{"file": "x.py", "lines": "1-2",
                                                        "text": "y", "via": "need-symbol"}],
                                             "call_chain": []}):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m",
                                 code_root="/repo")
        self.assertFalse(res.converged)
        self.assertIsNotNone(res.missing_link)

    def test_converged_claim_without_node_is_demoted(self):
        out = json.dumps({"converged": True, "path": [],
                          "attributed_defect": None, "missing_link": None})
        with mock.patch.object(C, "call_worker", return_value=_wr(out)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertFalse(res.converged)


class TestClassifySeedKind(unittest.TestCase):
    def test_trace_seed_is_diagnostic(self):
        self.assertEqual(
            classify_seed_kind("trace the actual head query call path for r head bar"),
            "diagnostic")

    def test_fix_seed_is_fix(self):
        self.assertEqual(
            classify_seed_kind("fix the ORDER BY in get_effective_head"), "fix")

    def test_diagnostic_with_fix_verb_defaults_to_fix(self):
        # A seed that both traces AND asks to change should still be authored.
        self.assertEqual(
            classify_seed_kind("trace the path and fix the wrong head"), "fix")

    def test_unknown_defaults_to_fix(self):
        self.assertEqual(classify_seed_kind("the head bar is wrong"), "fix")


class TestHoneyConvergeSection(unittest.TestCase):
    def _result(self, converge, seed_kind):
        return {"axes_judged": 2, "axes_total": 2, "seed_kind": seed_kind,
                "converge": converge,
                "verdicts": LOCATED_VERDICTS}

    def test_converged_section_leads_with_single_target(self):
        converge = {
            "converged": True,
            "path": [{"node": "db_fn", "file": "db/workflow_sequences.py",
                      "lines": "45-57", "symbol": "get_effective_head"}],
            "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                                  "lines": "45-57", "why": "ORDER BY wrong"},
            "missing_link": None}
        honey = render_local_honey(self._result(converge, "fix"), "fix the head")
        self.assertIn("Converged call path", honey)
        self.assertIn("Primary edit target", honey)
        self.assertIn("db/workflow_sequences.py:45-57", honey)
        self.assertIn("do not return needs_reinvestigation", honey.lower())

    def test_diagnostic_seed_says_path_is_deliverable(self):
        converge = {
            "converged": True, "path": [],
            "attributed_defect": {"node": "db_fn", "file": "x.py", "lines": "1-2",
                                  "why": "w"},
            "missing_link": None}
        honey = render_local_honey(self._result(converge, "diagnostic"),
                                   "trace the path")
        self.assertIn("DIAGNOSTIC", honey)
        self.assertIn("deliverable", honey)

    def test_missing_link_section_named(self):
        converge = {"converged": False, "path": [], "attributed_defect": None,
                    "missing_link": {"between": ["handler", "db_fn"],
                                     "need": {"symbols": ["get_effective_head"],
                                              "greps": [], "file_globs": []}}}
        honey = render_local_honey(self._result(converge, "fix"), "fix it")
        self.assertIn("Convergence incomplete", honey)
        self.assertIn("get_effective_head", honey)

    def test_no_converge_section_when_skipped(self):
        honey = render_local_honey(self._result(None, "fix"), "fix it")
        self.assertNotIn("Converged call path", honey)
        self.assertNotIn("Convergence incomplete", honey)


if __name__ == "__main__":
    unittest.main()
