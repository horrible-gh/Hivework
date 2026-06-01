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
    return {
        "axis_id": plan.axis_id,
        "code_snippets": [], "call_chain": [], "call_sites": [],
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


if __name__ == "__main__":
    unittest.main()
