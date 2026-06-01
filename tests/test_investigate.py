"""Wiring test for hive.investigate — decompose → bridge → retrieve → judge.

No real model calls and no live codebase: ``call_worker`` (decompose + judge)
and ``retrieve`` are stubbed, so this verifies the *composition* (leaf gating,
bridge plan flowing into retrieve, judge verdict collected, report written)
without spending or touching ripgrep.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import investigate as INV
from hive.config import load_config
from hive.providers import WorkerResult
from hive.investigate import render_local_honey


def _wr(stdout: str) -> WorkerResult:
    return WorkerResult(stdout=stdout, stderr="", exit_code=0, latency_s=0.01)


DECOMPOSE_OUT = json.dumps({
    "fanout_decision": "fanout", "reason": "x",
    "steps": [["A", "B"], ["C"]],
    "tasks": [
        {"id": "A", "title": "sql", "depends_on": [],
         "brief": "inspect server/sql/queries/*.json for the group_head query",
         "search_plan": {"keywords": ["ORDER BY"],
                         "file_globs": ["server/sql/queries/*.json"],
                         "doc_topics": []}},
        {"id": "B", "title": "endpoints", "depends_on": [],
         "brief": "trace server/routers/main.py and db_docs.create()"},  # no plan
        {"id": "C", "title": "synthesis", "depends_on": ["A", "B"],
         "brief": "combine the findings"},  # dependent → must be skipped
    ],
})
VERDICT_OUT = json.dumps({
    "verdict": {"located": True, "file": "server/x.py", "lines": "10-12",
                "reason": "the bug"},
})


def _fake_retrieve(plan, code_root, docs_root=None, **kwargs):
    # Include the file the stub verdict cites so the judge verdict is GROUNDED
    # (a verdict citing a file absent from the bundle is downgraded by design).
    return {
        "axis_id": plan.axis_id,
        "code_snippets": [{"file": "server/x.py", "lines": "8-14",
                           "text": "def f(): ...", "hits": []}],
        "call_chain": [], "call_sites": [],
        "git_history": [], "design_excerpts": [],
        "stats": {"raw_hits": 3, "snippets": 1, "call_chain": 0},
    }


class TestInvestigateWiring(unittest.TestCase):
    def test_end_to_end_composition(self):
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1   # single judge call → no followup path
        cfg.judge.max_axes = 3

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "verdicts.json")
            with mock.patch("hive.decompose.call_worker", return_value=_wr(DECOMPOSE_OUT)), \
                 mock.patch("hive.judge.call_worker", return_value=_wr(VERDICT_OUT)) as judge_cw, \
                 mock.patch("hive.investigate.retrieve", side_effect=_fake_retrieve):
                result = INV.run_investigate(
                    seed_text="find the bug", recipe_path=None,
                    code_root=td, docs_root=None, output_path=out,
                    cfg=cfg, ledger=None,
                )

            # leaf gating: A and B judged, dependent C skipped
            self.assertEqual(result["axes_total"], 3)
            self.assertEqual(result["axes_judged"], 2)
            judged_ids = {v["axis_id"] for v in result["verdicts"]}
            self.assertEqual(judged_ids, {"A", "B"})
            self.assertNotIn("C", judged_ids)

            # judge was called once per judged axis (max_calls_per_axis=1)
            self.assertEqual(judge_cw.call_count, 2)

            # bridge plan flowed through: queen keyword survived for A,
            # local extraction filled B from its brief
            self.assertTrue(all(v["verdict"]["located"] for v in result["verdicts"]))
            a = next(v for v in result["verdicts"] if v["axis_id"] == "A")
            self.assertIn("ORDER BY", a["search_plan"]["keywords"])
            b = next(v for v in result["verdicts"] if v["axis_id"] == "B")
            self.assertIn("db_docs.create", b["search_plan"]["keywords"])

            # report artifacts written (JSON + sibling markdown)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(os.path.splitext(out)[0] + ".md"))

    def test_max_axes_caps_judged(self):
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.max_axes = 1   # only the first leaf judged

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "v.json")
            with mock.patch("hive.decompose.call_worker", return_value=_wr(DECOMPOSE_OUT)), \
                 mock.patch("hive.judge.call_worker", return_value=_wr(VERDICT_OUT)) as judge_cw, \
                 mock.patch("hive.investigate.retrieve", side_effect=_fake_retrieve):
                result = INV.run_investigate(
                    seed_text="x", recipe_path=None, code_root=td,
                    docs_root=None, output_path=out, cfg=cfg, ledger=None,
                )
            self.assertEqual(result["axes_judged"], 1)
            self.assertEqual(judge_cw.call_count, 1)

    # N164 regression: 5 leaf axes, a decisive one (css_rules) last in order.
    # The old cap (3) sliced leaves[:3] and silently dropped css_rules; the
    # runaway-ceiling default (12) must judge every leaf so the decisive axis
    # survives. Order matters — css_rules is positioned past the old cap.
    _FIVE_LEAVES = json.dumps({
        "fanout_decision": "fanout", "reason": "x",
        "steps": [["a1", "a2", "a3", "a4", "css_rules"]],
        "tasks": [
            {"id": "a1", "title": "t1", "depends_on": [], "brief": "b1"},
            {"id": "a2", "title": "t2", "depends_on": [], "brief": "b2"},
            {"id": "a3", "title": "t3", "depends_on": [], "brief": "b3"},
            {"id": "a4", "title": "t4", "depends_on": [], "brief": "b4"},
            {"id": "css_rules", "title": "color def", "depends_on": [],
             "brief": "grep .wf-step.wf-undecided color rule"},
        ],
    })

    def _run_five(self, cfg):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "v.json")
            with mock.patch("hive.decompose.call_worker", return_value=_wr(self._FIVE_LEAVES)), \
                 mock.patch("hive.judge.call_worker", return_value=_wr(VERDICT_OUT)), \
                 mock.patch("hive.investigate.retrieve", side_effect=_fake_retrieve):
                return INV.run_investigate(
                    seed_text="x", recipe_path=None, code_root=td,
                    docs_root=None, output_path=out, cfg=cfg, ledger=None,
                )

    def test_decisive_axis_not_dropped_under_default_ceiling(self):
        cfg = load_config()           # default max_axes=12 (runaway-ceiling)
        cfg.judge.max_calls_per_axis = 1
        result = self._run_five(cfg)
        self.assertEqual(result["axes_judged"], 5)
        self.assertIn("css_rules", {v["axis_id"] for v in result["verdicts"]})

    def test_truncation_warns_and_lists_dropped(self):
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.max_axes = 3        # force the old cap → css_rules dropped
        with self.assertLogs("hive.investigate", level="WARNING") as cm:
            result = self._run_five(cfg)
        self.assertEqual(result["axes_judged"], 3)
        joined = "\n".join(cm.output)
        self.assertIn("css_rules", joined)
        self.assertIn("DROPPING", joined)


class TestRenderLocalHoney(unittest.TestCase):
    """The free verdict→honey seam that lets the cheap path feed specify."""

    RESULT = {
        "axes_total": 3, "axes_judged": 2,
        "verdicts": [
            {"axis_id": "T7", "title": "Anchor in D031",
             "verdict": {"located": True, "file": "210_design/D031_x.md",
                         "lines": "69-81", "reason": "unique heading"}},
            {"axis_id": "T4", "title": "regression",
             "verdict": {"located": False, "file": "", "lines": "",
                         "reason": "ungrounded"}},
        ],
    }
    SEED = "Add a short Korean paragraph to D031 recording the action-bar policy."

    def setUp(self):
        self.honey = render_local_honey(self.RESULT, self.SEED)

    def test_seed_carried_as_requested_change(self):
        self.assertIn("Requested change", self.honey)
        self.assertIn("Korean paragraph", self.honey)

    def test_located_axis_becomes_fix_direction(self):
        self.assertIn("210_design/D031_x.md:69-81", self.honey)
        self.assertIn("unique heading", self.honey)

    def test_unlocated_axis_flagged_not_fabricated(self):
        # T4 (unlocated) must appear under the "do NOT fabricate" section, after
        # the fix-directions block — never promoted to a grounded fix direction.
        self.assertIn("do NOT fabricate", self.honey)
        tail = self.honey.split("do NOT fabricate", 1)[1]
        self.assertIn("T4", tail)
        # the only grounded target belongs to T7, above the fabricate section
        self.assertNotIn("210_design/D031_x.md", tail)

    def test_no_located_yields_defer_note(self):
        result = {"axes_total": 1, "axes_judged": 1, "verdicts": [
            {"axis_id": "X", "title": "t",
             "verdict": {"located": False, "file": "", "lines": "", "reason": "n/a"}}]}
        honey = render_local_honey(result, "do something")
        self.assertIn("defer", honey.lower())


if __name__ == "__main__":
    unittest.main()
