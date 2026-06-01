"""Unit tests for hive.judge — JUDGE call routing, parsing, and budget caps.

No real model calls: ``call_worker`` is patched with a scripted fake, and
``retrieve_followup`` is patched to a stub so the loop logic is tested in
isolation from the live codebase.
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import judge as J
from hive.config import JudgeConfig
from hive.providers import WorkerResult


def _wr(stdout: str, rc: int = 0) -> WorkerResult:
    return WorkerResult(stdout=stdout, stderr="", exit_code=rc, latency_s=0.01)


VERDICT_AND_NEED = json.dumps({
    "verdict": {"located": False, "file": "", "lines": "",
                "reason": "need to resolve callee"},
    "need": {"symbols": ["get_linked_result_documents"], "greps": ["d.type"],
             "file_globs": []},
})
FINAL_VERDICT = json.dumps({
    "verdict": {"located": True, "file": "server/store.py", "lines": "1444-1453",
                "reason": "selects d.type but column is type_code"},
})
VERDICT_NO_NEED = json.dumps({
    "verdict": {"located": True, "file": "server/store.py", "lines": "1444-1453",
                "reason": "bug visible in first pass"},
    "need": {"symbols": [], "greps": [], "file_globs": []},
})

PLAN_BUNDLE = {
    "axis_id": "regression-blame",
    "code_snippets": [{"file": "server/store.py", "lines": "1400-1412",
                       "text": "def get_linked_result_documents(...): ...",
                       "hits": ["type"]}],
    "call_chain": [],
    "call_sites": [{"file": "server/service.py", "line": 306, "text": "x"}],
    "git_history": [{"file": "server/store.py", "blame": "b425d0dc ...", "log": ""}],
    "design_excerpts": [],
}
FU_BUNDLE = {"axis_id": "regression-blame",
             "seeds": [{"file": "server/store.py", "lines": "1444-1453",
                        "text": "SELECT d.type ...", "via": "need-symbol"}],
             "call_chain": [], "stats": {}}


class TestRunJudgeCapTwo(unittest.TestCase):
    """max_calls_per_axis=2 with a need → two calls, final verdict from re-judge."""

    def setUp(self):
        self.calls = []

        def fake_call(provider, model, prompt, cwd=None, timeout=300, **kw):
            self.calls.append(prompt)
            return _wr(VERDICT_AND_NEED if len(self.calls) == 1 else FINAL_VERDICT)

        self.p1 = mock.patch.object(J, "call_worker", side_effect=fake_call)
        self.p2 = mock.patch.object(J, "retrieve_followup", return_value=FU_BUNDLE)
        self.p1.start(); self.p2.start()
        self.res = J.run_judge(
            plan_bundle=PLAN_BUNDLE, symptom="linked-result type lookup",
            axis_globs=["server/**/*.py"], code_root="/x",
            provider="copilot", model="gpt-5-mini", judge_cfg=JudgeConfig())

    def tearDown(self):
        self.p1.stop(); self.p2.stop()

    def test_two_calls_made(self):
        self.assertEqual(self.res["calls_made"], 2)

    def test_followup_ran(self):
        self.assertIsNotNone(self.res["followup_bundle"])

    def test_need_extracted(self):
        self.assertIn("get_linked_result_documents", self.res["need"].symbols)

    def test_final_verdict_from_rejudge(self):
        v = self.res["verdict"]
        self.assertTrue(v.located)
        self.assertEqual(v.lines, "1444-1453")

    def test_need_defaults_globs_to_axis(self):
        # judge left file_globs empty → falls back to axis globs
        self.assertEqual(self.res["need"].file_globs, ["server/**/*.py"])


class TestRunJudgeCapOne(unittest.TestCase):
    """max_calls_per_axis=1 → single call, verdict only, no follow-up even if asked."""

    def setUp(self):
        self.calls = []

        def fake_call(provider, model, prompt, cwd=None, timeout=300, **kw):
            self.calls.append(prompt)
            return _wr(VERDICT_NO_NEED)

        self.p1 = mock.patch.object(J, "call_worker", side_effect=fake_call)
        self.p2 = mock.patch.object(J, "retrieve_followup")
        self.p1.start(); self.fu = self.p2.start()
        self.res = J.run_judge(
            plan_bundle=PLAN_BUNDLE, symptom="s", axis_globs=["server/**/*.py"],
            code_root="/x", provider="copilot", model="gpt-5-mini",
            judge_cfg=JudgeConfig(max_calls_per_axis=1))

    def tearDown(self):
        self.p1.stop(); self.p2.stop()

    def test_one_call(self):
        self.assertEqual(self.res["calls_made"], 1)

    def test_no_followup_called(self):
        self.fu.assert_not_called()

    def test_no_need_requested(self):
        # want_need is False under cap=1, so need is never built
        self.assertIsNone(self.res["need"])

    def test_verdict_present(self):
        self.assertTrue(self.res["verdict"].located)


class TestRunJudgeNoNeed(unittest.TestCase):
    """cap=2 but judge returns an empty need → no second call."""

    def setUp(self):
        self.calls = []

        def fake_call(provider, model, prompt, cwd=None, timeout=300, **kw):
            self.calls.append(prompt)
            return _wr(VERDICT_NO_NEED)

        self.p1 = mock.patch.object(J, "call_worker", side_effect=fake_call)
        self.p2 = mock.patch.object(J, "retrieve_followup")
        self.p1.start(); self.fu = self.p2.start()
        self.res = J.run_judge(
            plan_bundle=PLAN_BUNDLE, symptom="s", axis_globs=["g"],
            code_root="/x", provider="copilot", model="gpt-5-mini",
            judge_cfg=JudgeConfig(max_calls_per_axis=2))

    def tearDown(self):
        self.p1.stop(); self.p2.stop()

    def test_single_call(self):
        self.assertEqual(self.res["calls_made"], 1)

    def test_followup_not_run(self):
        self.fu.assert_not_called()
        self.assertIsNone(self.res["followup_bundle"])


class TestRunJudgeDegradation(unittest.TestCase):
    """Unparseable / failed model output never raises; degrades to located=false."""

    def test_unparseable_output(self):
        with mock.patch.object(J, "call_worker", return_value=_wr("no json here")), \
             mock.patch.object(J, "retrieve_followup"):
            res = J.run_judge(plan_bundle=PLAN_BUNDLE, symptom="s",
                              axis_globs=["g"], code_root="/x", provider="copilot",
                              model="gpt-5-mini", judge_cfg=JudgeConfig())
        self.assertFalse(res["verdict"].located)
        self.assertEqual(res["calls_made"], 1)  # no need parsed → no re-judge

    def test_worker_exception(self):
        with mock.patch.object(J, "call_worker", side_effect=RuntimeError("boom")), \
             mock.patch.object(J, "retrieve_followup"):
            res = J.run_judge(plan_bundle=PLAN_BUNDLE, symptom="s",
                              axis_globs=["g"], code_root="/x", provider="copilot",
                              model="gpt-5-mini", judge_cfg=JudgeConfig())
        self.assertFalse(res["verdict"].located)

    def test_flaky_rejudge_keeps_first_verdict(self):
        # call1 locates AND asks for a need; call2 returns junk → keep call1 verdict.
        c1 = json.dumps({
            "verdict": {"located": True, "file": "a.py", "lines": "1-2", "reason": "r"},
            "need": {"symbols": ["s"], "greps": [], "file_globs": []}})
        seq = [_wr(c1), _wr("garbage no json")]
        with mock.patch.object(J, "call_worker", side_effect=seq), \
             mock.patch.object(J, "retrieve_followup", return_value=FU_BUNDLE):
            res = J.run_judge(plan_bundle=PLAN_BUNDLE, symptom="s",
                              axis_globs=["g"], code_root="/x", provider="copilot",
                              model="gpt-5-mini", judge_cfg=JudgeConfig())
        self.assertEqual(res["calls_made"], 2)
        self.assertTrue(res["verdict"].located)   # first verdict survived
        self.assertEqual(res["verdict"].file, "a.py")


class TestLedgerRecording(unittest.TestCase):
    """Each model call is recorded to the ledger under stage 'judge'."""

    def test_records_two_calls(self):
        led = mock.Mock()
        seq = [_wr(VERDICT_AND_NEED), _wr(FINAL_VERDICT)]
        with mock.patch.object(J, "call_worker", side_effect=seq), \
             mock.patch.object(J, "retrieve_followup", return_value=FU_BUNDLE):
            J.run_judge(plan_bundle=PLAN_BUNDLE, symptom="s", axis_globs=["g"],
                        code_root="/x", provider="copilot", model="gpt-5-mini",
                        judge_cfg=JudgeConfig(), ledger=led)
        self.assertEqual(led.record_call.call_count, 2)
        # stage positional arg is 'judge'
        self.assertEqual(led.record_call.call_args_list[0].args[0], "judge")


class TestSummarizeBundle(unittest.TestCase):
    """Bundle rendering is compact and truncates long windows."""

    def test_truncates_long_snippet(self):
        big = {"axis_id": "a", "code_snippets":
               [{"file": "f.py", "lines": "1-9", "text": "x" * 5000, "hits": []}],
               "call_chain": [], "call_sites": [], "git_history": [],
               "design_excerpts": []}
        out = J.summarize_bundle(big)
        self.assertIn("truncated", out)
        self.assertLess(len(out), 2000)

    def test_call_sites_surfaced_as_lines(self):
        out = J.summarize_bundle(PLAN_BUNDLE)
        # call_site lines are shown as file:line: text (seed-naming signal),
        # not collapsed to a bare file list.
        self.assertIn("server/service.py:306", out)

    def test_declarations_prioritised(self):
        b = {"axis_id": "a", "code_snippets": [], "call_chain": [],
             "git_history": [], "design_excerpts": [],
             "call_sites": [
                 {"file": "x.py", "line": 5, "text": "x = foo()"},
                 {"file": "x.py", "line": 9, "text": "def get_linked_result_documents(t):"},
             ]}
        out = J.summarize_bundle(b)
        self.assertIn("def get_linked_result_documents", out)
        self.assertIn("declarations", out)


if __name__ == "__main__":
    unittest.main()
