"""Fan-out tests — focused on the comb-write path and the G8 race guard.

Regression for the G8 race: combs_dir is created once before the worker pool
launches, but each worker's call_worker can run for minutes. If the dir is
removed externally in that window (a concurrent run sharing the default
workdir, tmp cleanup), the comb write used to die with FileNotFoundError and
abort the whole pipeline. The guard re-creates the parent right before writing.
"""
import os
import json
import shutil
import tempfile
import unittest
from unittest import mock

from hive import fanout
from hive.config import load_config
from hive.providers import WorkerResult
from hive.retriever import FollowupNeed, SearchPlan


class TestFanoutG8Race(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="hive_fanout_test_")
        self.combs_dir = os.path.join(self.workdir, "combs")

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_comb_write_survives_combs_dir_removed_mid_worker(self):
        """Worker deletes combs/ during its run; the write must still succeed."""
        axes = [{"id": "G8", "title": "last axis", "brief": "b"}]

        def fake_call_worker(provider, model, prompt, cwd=None, timeout=600, **kw):
            # Simulate the external race: combs/ vanishes while the worker runs,
            # i.e. after run_fanout created it but before we write the comb.
            shutil.rmtree(self.combs_dir, ignore_errors=True)
            self.assertFalse(os.path.isdir(self.combs_dir))
            return WorkerResult(stdout='{"axis_id":"G8"}', stderr="", exit_code=0,
                                latency_s=0.1)

        with mock.patch.object(fanout, "call_worker", fake_call_worker):
            comb_files = fanout.run_fanout(
                axes=axes, seed_text="seed", codebase_root=self.workdir,
                workdir=self.workdir, max_workers=1,
            )

        comb_path = comb_files["G8"]
        self.assertTrue(os.path.exists(comb_path), "comb file should be written despite the race")
        with open(comb_path, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"axis_id":"G8"}')

    def test_all_axes_written_when_dir_stable(self):
        """Baseline: every axis produces a comb file when nothing interferes."""
        axes = [{"id": f"G{i}", "title": f"axis {i}", "brief": "b"} for i in range(1, 4)]

        def fake_call_worker(provider, model, prompt, cwd=None, timeout=600, **kw):
            return WorkerResult(stdout="ok", stderr="", exit_code=0, latency_s=0.1)

        with mock.patch.object(fanout, "call_worker", fake_call_worker):
            comb_files = fanout.run_fanout(
                axes=axes, seed_text="seed", codebase_root=self.workdir,
                workdir=self.workdir, max_workers=2,
            )

        self.assertEqual(set(comb_files), {"G1", "G2", "G3"})
        for path in comb_files.values():
            self.assertTrue(os.path.exists(path))


class TestReinforceBudget(unittest.TestCase):
    """The per-run reinforcement ceiling drains correctly and never goes negative."""

    def test_take_respects_ceiling_and_drains(self):
        b = fanout.ReinforceBudget(3)
        self.assertEqual(b.take(2), 2)
        self.assertEqual(b.remaining, 1)
        self.assertEqual(b.take(2), 1)   # only 1 left → grants 1
        self.assertEqual(b.take(2), 0)   # exhausted
        self.assertEqual(b.remaining, 0)

    def test_zero_budget_grants_nothing(self):
        b = fanout.ReinforceBudget(0)
        self.assertEqual(b.take(5), 0)


class TestExtractCombEvidence(unittest.TestCase):
    def test_pulls_root_cause_and_evidence_files(self):
        comb = json.dumps({
            "root_cause_signal": "server/db/orders.py:42",
            "findings": [{"evidence": [{"file": "client/ui/Picker.vue", "lines": "9"}]}],
        })
        files, _ = fanout._extract_comb_evidence(comb)
        self.assertIn("server/db/orders.py", files)
        self.assertIn("client/ui/Picker.vue", files)

    def test_malformed_comb_is_empty(self):
        self.assertEqual(fanout._extract_comb_evidence("not json"), ([], []))
        self.assertEqual(fanout._extract_comb_evidence(""), ([], []))


class TestReinforceThinAxis(unittest.TestCase):
    """B3 reinforcement: gated, capped, returns a local FollowupNeed (no judge here)."""

    def setUp(self):
        self.cfg = load_config()  # real defaults; we toggle the reinforce gate per-test
        self.sp = SearchPlan(axis_id="A", keywords=["order_doc_id"],
                             file_globs=["server/**/*.py"], doc_topics=[])
        self.task = {"id": "A", "title": "t", "brief": "b"}

    def _stub(self, stdout):
        return mock.MagicMock(return_value=WorkerResult(
            stdout=stdout, stderr="", exit_code=0, latency_s=0.1))

    def test_disabled_is_noop(self):
        self.cfg.reinforce.enabled = False
        with mock.patch.object(fanout, "call_worker", self._stub("x")) as m:
            out = fanout.reinforce_thin_axis(
                self.task, self.sp, "seed", ".", cfg=self.cfg,
                budget=fanout.ReinforceBudget(4))
        self.assertIsNone(out)
        m.assert_not_called()                       # gated BEFORE any spend

    def test_enabled_returns_followupneed_scoped_to_cited_files(self):
        self.cfg.reinforce.enabled = True
        comb = json.dumps({"root_cause_signal": "server/db/orders.py:2",
                           "findings": []})
        with mock.patch.object(fanout, "call_worker", self._stub(comb)):
            need = fanout.reinforce_thin_axis(
                self.task, self.sp, "seed", ".", cfg=self.cfg,
                budget=fanout.ReinforceBudget(4))
        self.assertIsInstance(need, FollowupNeed)
        self.assertEqual(need.axis_id, "A")
        self.assertIn("server/db/orders.py", need.file_globs)
        self.assertEqual(need.greps, ["order_doc_id"])   # grepped by the axis's keywords

    def test_budget_exhausted_skips_without_spending(self):
        self.cfg.reinforce.enabled = True
        with mock.patch.object(fanout, "call_worker", self._stub("x")) as m:
            out = fanout.reinforce_thin_axis(
                self.task, self.sp, "seed", ".", cfg=self.cfg,
                budget=fanout.ReinforceBudget(0))
        self.assertIsNone(out)
        m.assert_not_called()

    def test_no_cited_files_returns_none(self):
        self.cfg.reinforce.enabled = True
        comb = json.dumps({"findings": [], "root_cause_signal": None})
        with mock.patch.object(fanout, "call_worker", self._stub(comb)):
            out = fanout.reinforce_thin_axis(
                self.task, self.sp, "seed", ".", cfg=self.cfg,
                budget=fanout.ReinforceBudget(4))
        self.assertIsNone(out)


class TestCombShape(unittest.TestCase):
    """is_comb_shaped separates a conclusion (findings array) from a search-memo."""

    def test_real_comb_is_shaped(self):
        comb = json.dumps({"axis_id": "A", "findings": [{"claim": "x"}]})
        self.assertTrue(fanout.is_comb_shaped(comb))

    def test_empty_findings_is_still_shaped(self):
        # A genuine "found nothing, concluded" comb is comb-SHAPED — shape != content.
        self.assertTrue(fanout.is_comb_shaped(json.dumps({"axis_id": "A", "findings": []})))

    def test_grep_arg_memo_is_not_shaped(self):
        # The dominant run-424 failure: the drone's NEXT search returned as the answer.
        memo = json.dumps({"path": "", "pattern": "create_button", "glob": "*.vue"})
        self.assertFalse(fanout.is_comb_shaped(memo))

    def test_empty_and_prose_are_not_shaped(self):
        self.assertFalse(fanout.is_comb_shaped(""))
        self.assertFalse(fanout.is_comb_shaped("ok"))
        self.assertFalse(fanout.is_comb_shaped("I will now search for the handler."))

    def test_findings_must_be_a_list(self):
        self.assertFalse(fanout.is_comb_shaped(json.dumps({"axis_id": "A", "findings": "x"})))


class _RecordingLedger:
    """Minimal ledger double capturing finish_call(ok, err) per call."""
    def __init__(self):
        self.finished = []  # list of (ok, err)
        self._n = 0
    def begin_call(self, *a, **kw):
        self._n += 1
        return self._n
    def mark_running(self, *a, **kw):
        pass
    def finish_call(self, call_id, output, latency_s, ok=True, err="", real_tokens=None):
        self.finished.append((bool(ok), err))


class TestShapeRejectRetry(unittest.TestCase):
    """Fix ① (CH 0004.0008): a non-comb gets ONE re-specification turn; the ledger
    records the search-memo as ok=0 even on exit 0."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="hive_fanout_respecify_")

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _comb(self):
        return json.dumps({"axis_id": "A1", "findings": [{"claim": "real"}]})

    def _memo(self):
        return json.dumps({"path": "", "pattern": "create_button", "glob": "*.vue"})

    def test_non_comb_triggers_respecify_and_adopts_real_comb(self):
        axes = [{"id": "A1", "title": "t", "brief": "b"}]
        calls = []
        outs = iter([self._memo(), self._comb()])  # 1st: search-memo, 2nd (retry): real comb

        def fake_call_worker(provider, model, prompt, cwd=None, timeout=600, **kw):
            calls.append(prompt)
            return WorkerResult(stdout=next(outs), stderr="", exit_code=0, latency_s=0.1)

        ledger = _RecordingLedger()
        with mock.patch.object(fanout, "call_worker", fake_call_worker):
            comb_files = fanout.run_fanout(
                axes=axes, seed_text="seed", codebase_root=self.workdir,
                workdir=self.workdir, max_workers=1, ledger=ledger,
            )

        self.assertEqual(len(calls), 2, "non-comb must be re-specified exactly once")
        self.assertIn("REJECTED", calls[1], "retry prompt must carry the rejection banner")
        with open(comb_files["A1"], encoding="utf-8") as f:
            self.assertEqual(f.read(), self._comb(), "the real comb (retry) must win")
        # Ledger honesty: 1st call recorded ok=0 (search-memo), 2nd ok=1 (real comb).
        self.assertEqual(ledger.finished[0][0], False)
        self.assertEqual(ledger.finished[1][0], True)

    def test_shaped_first_reply_skips_retry(self):
        axes = [{"id": "A1", "title": "t", "brief": "b"}]
        calls = []

        def fake_call_worker(provider, model, prompt, cwd=None, timeout=600, **kw):
            calls.append(prompt)
            return WorkerResult(stdout=self._comb(), stderr="", exit_code=0, latency_s=0.1)

        ledger = _RecordingLedger()
        with mock.patch.object(fanout, "call_worker", fake_call_worker):
            fanout.run_fanout(
                axes=axes, seed_text="seed", codebase_root=self.workdir,
                workdir=self.workdir, max_workers=1, ledger=ledger,
            )

        self.assertEqual(len(calls), 1, "a real comb on the first try is not retried")
        self.assertEqual(ledger.finished[0][0], True)

    def test_conclusion_mandate_in_default_contract(self):
        # Fix ②: the static instruction ships in the contract every drone receives.
        contract = fanout.load_comb_contract(None, self.workdir)
        self.assertIn("Conclusion mandate", contract)
        self.assertNotIn("{codebase_root}", contract)  # template fully substituted


if __name__ == "__main__":
    unittest.main()
