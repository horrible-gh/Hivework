"""Unit tests for hive.reinvestigate — reaction #3 routing brain (M013).

Pure, deterministic, no model calls. Pins the reason_code + coverage → action
mapping and the honesty gate (sufficient evidence → terminate, never busy-loop).
"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive.reinvestigate import (
    plan_reinvestigation, run_reinvestigation_loop, ReinvestPlan,
    ACTION_RE_RETRIEVE, ACTION_RE_CONVERGE, ACTION_TERMINATE,
)
from hive.specify import (
    RI_ANCHOR_NOT_GROUNDED, RI_AUTHOR_DECLARED, RI_INCONCLUSIVE, RI_INEFFECTIVE,
    RI_LEGACY_COERCE, RI_SEED_TARGET_UNCOVERED, RI_STALE_ANCHOR,
)


def _nr_spec(reason_code):
    return {"termination": "needs_reinvestigation",
            "reinvestigation": {"reason_code": reason_code, "gate": "g"}}


def _verdict(axis, *, located=False, thin=None):
    v = {"axis_id": axis,
         "verdict": {"located": located, "file": "f" if located else "",
                     "lines": "1" if located else "", "reason": "r"}}
    if thin is not None:
        v["coverage"] = {"thin": thin, "sufficient": not thin,
                         "needs_reinforcement": thin}
    return v


class TestRetrieveRouting(unittest.TestCase):
    """Missing/ungrounded evidence → re_retrieve, but ONLY where an axis is thin."""

    def test_uncovered_with_thin_axis_re_retrieves_it(self):
        verdicts = [_verdict("A", thin=True), _verdict("B", thin=False)]
        plan = plan_reinvestigation(_nr_spec(RI_SEED_TARGET_UNCOVERED), verdicts)
        self.assertEqual(plan.action, ACTION_RE_RETRIEVE)
        self.assertEqual(plan.axis_ids, ["A"])
        self.assertTrue(plan.will_rerun)

    def test_anchor_not_grounded_with_thin_axis_re_retrieves(self):
        plan = plan_reinvestigation(_nr_spec(RI_ANCHOR_NOT_GROUNDED),
                                    [_verdict("FE", thin=True)])
        self.assertEqual(plan.action, ACTION_RE_RETRIEVE)
        self.assertEqual(plan.axis_ids, ["FE"])

    def test_uncovered_but_all_sufficient_terminates_honestly(self):
        # Honesty gate: no thin axis → re-fetch buys nothing → terminate, not loop.
        verdicts = [_verdict("A", thin=False), _verdict("B", thin=False)]
        plan = plan_reinvestigation(_nr_spec(RI_SEED_TARGET_UNCOVERED), verdicts)
        self.assertEqual(plan.action, ACTION_TERMINATE)
        self.assertFalse(plan.will_rerun)
        self.assertIn("sufficient", plan.rationale)

    def test_uncovered_with_no_coverage_tags_terminates(self):
        # Older verdicts without a coverage tag are not treated as thin.
        plan = plan_reinvestigation(_nr_spec(RI_SEED_TARGET_UNCOVERED),
                                    [_verdict("A"), _verdict("B")])
        self.assertEqual(plan.action, ACTION_TERMINATE)


class TestConvergeRouting(unittest.TestCase):
    """Causal/effectiveness contradiction → re_converge when ≥2 located."""

    def test_ineffective_with_two_located_re_converges(self):
        verdicts = [_verdict("A", located=True), _verdict("B", located=True)]
        plan = plan_reinvestigation(_nr_spec(RI_INEFFECTIVE), verdicts)
        self.assertEqual(plan.action, ACTION_RE_CONVERGE)
        self.assertEqual(sorted(plan.axis_ids), ["A", "B"])

    def test_inconclusive_with_two_located_re_converges(self):
        verdicts = [_verdict("A", located=True), _verdict("B", located=True)]
        plan = plan_reinvestigation(_nr_spec(RI_INCONCLUSIVE), verdicts)
        self.assertEqual(plan.action, ACTION_RE_CONVERGE)

    def test_ineffective_with_one_located_terminates(self):
        plan = plan_reinvestigation(_nr_spec(RI_INEFFECTIVE),
                                    [_verdict("A", located=True), _verdict("B")])
        self.assertEqual(plan.action, ACTION_TERMINATE)
        self.assertIn("re-stitch", plan.rationale)


class TestTerminalReasons(unittest.TestCase):
    """Reasons with no cheap evidence-adding re-run terminate honestly."""

    def test_stale_anchor_terminates(self):
        plan = plan_reinvestigation(_nr_spec(RI_STALE_ANCHOR),
                                    [_verdict("A", thin=True)])
        self.assertEqual(plan.action, ACTION_TERMINATE)

    def test_legacy_coerce_terminates(self):
        plan = plan_reinvestigation(_nr_spec(RI_LEGACY_COERCE), [])
        self.assertEqual(plan.action, ACTION_TERMINATE)

    def test_author_declared_terminates(self):
        plan = plan_reinvestigation(_nr_spec(RI_AUTHOR_DECLARED), [])
        self.assertEqual(plan.action, ACTION_TERMINATE)

    def test_unknown_reason_terminates(self):
        plan = plan_reinvestigation(_nr_spec("something_new"), [])
        self.assertEqual(plan.action, ACTION_TERMINATE)
        self.assertIn("something_new", plan.rationale)


class TestGuards(unittest.TestCase):
    def test_non_nr_spec_terminates(self):
        spec = {"termination": "ready_to_apply"}
        plan = plan_reinvestigation(spec, [_verdict("A", thin=True)])
        self.assertEqual(plan.action, ACTION_TERMINATE)
        self.assertIn("not needs_reinvestigation", plan.rationale)

    def test_missing_reinvestigation_field_terminates(self):
        spec = {"termination": "needs_reinvestigation"}
        plan = plan_reinvestigation(spec, [])
        self.assertEqual(plan.action, ACTION_TERMINATE)

    def test_none_verdicts_is_safe(self):
        plan = plan_reinvestigation(_nr_spec(RI_SEED_TARGET_UNCOVERED), None)
        self.assertEqual(plan.action, ACTION_TERMINATE)


def _cfg(live, max_rounds):
    return SimpleNamespace(
        reinvestigation=SimpleNamespace(live=live, max_rounds=max_rounds))


def _route(spec, _verdicts):
    """Stand-in for log_plan: re_converge while NR, terminate once cleared."""
    if spec.get("termination") == "needs_reinvestigation":
        return ReinvestPlan(ACTION_RE_CONVERGE, "ineffective", axis_ids=["A"])
    return ReinvestPlan(ACTION_TERMINATE, "")


NR = {"termination": "needs_reinvestigation"}
READY = {"termination": "ready_to_apply"}


class TestReinvestigationLoop(unittest.TestCase):
    """The bounded live re-run loop: gating, the no-change early stop, and the cap."""

    def test_live_off_does_not_rerun(self):
        calls = []
        spec, _ = run_reinvestigation_loop(
            dict(NR), {"verdicts": []}, cfg=_cfg(False, 2),
            rerun=lambda p, r: calls.append("rerun") or r,
            respecify=lambda: calls.append("respec") or READY,
            read_honey=lambda: "h", log_plan=_route)
        self.assertEqual(spec["termination"], "needs_reinvestigation")
        self.assertEqual(calls, [])                 # plan logged, but no spend

    def test_terminate_plan_does_not_rerun(self):
        calls = []
        run_reinvestigation_loop(
            dict(NR), {"verdicts": []}, cfg=_cfg(True, 2),
            rerun=lambda p, r: calls.append("rerun") or r,
            respecify=lambda: calls.append("respec") or READY,
            read_honey=lambda: "h",
            log_plan=lambda s, v: ReinvestPlan(ACTION_TERMINATE, "stale_anchor"))
        self.assertEqual(calls, [])

    def test_one_round_clears_nr(self):
        honey = iter(["old", "new"])               # prev → "old", after rerun → "new"
        respec_calls = []
        spec, _ = run_reinvestigation_loop(
            dict(NR), {"verdicts": []}, cfg=_cfg(True, 2),
            rerun=lambda p, r: {"verdicts": []},
            respecify=lambda: respec_calls.append(1) or READY,
            read_honey=lambda: next(honey), log_plan=_route)
        self.assertEqual(spec["termination"], "ready_to_apply")
        self.assertEqual(len(respec_calls), 1)      # cleared NR in one round

    def test_no_change_stops_before_respecify(self):
        respec_calls = []
        spec, _ = run_reinvestigation_loop(
            dict(NR), {"verdicts": []}, cfg=_cfg(True, 2),
            rerun=lambda p, r: {"verdicts": []},
            respecify=lambda: respec_calls.append(1) or READY,
            read_honey=lambda: "same", log_plan=_route)   # honey never changes
        self.assertEqual(respec_calls, [])          # no wasted specify spend
        self.assertEqual(spec["termination"], "needs_reinvestigation")

    def test_persistent_nr_caps_at_max_rounds(self):
        n = {"i": 0}
        respec_calls = []
        def read():
            n["i"] += 1
            return str(n["i"])                      # always different → never no-change
        run_reinvestigation_loop(
            dict(NR), {"verdicts": []}, cfg=_cfg(True, 2),
            rerun=lambda p, r: {"verdicts": []},
            respecify=lambda: respec_calls.append(1) or dict(NR),
            read_honey=read, log_plan=_route)
        self.assertEqual(len(respec_calls), 2)      # bounded by max_rounds, no busy-loop

    def test_non_nr_spec_is_passthrough(self):
        spec, res = run_reinvestigation_loop(
            dict(READY), {"verdicts": []}, cfg=_cfg(True, 2),
            rerun=lambda p, r: r, respecify=lambda: READY,
            read_honey=lambda: "h", log_plan=_route)
        self.assertEqual(spec["termination"], "ready_to_apply")


if __name__ == "__main__":
    unittest.main()
