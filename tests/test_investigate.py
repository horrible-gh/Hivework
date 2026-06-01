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
from hive.investigate import render_local_honey, _prioritize_axes


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

    # N165 docs=(none) confound: an axis carries doc_topics but no docs tree was
    # supplied → the design channel is silently skipped. Surface it at WARNING.
    _DOC_AXIS = json.dumps({
        "fanout_decision": "fanout", "reason": "x",
        "steps": [["design_ssot"]],
        "tasks": [
            {"id": "design_ssot", "title": "design ssot", "depends_on": [],
             "brief": "confirm the D030 action-bar policy in the design doc",
             "search_plan": {"keywords": ["action-bar"],
                             "file_globs": ["Documents/projects/FlowGate/210_design/**"],
                             "doc_topics": ["action-bar policy"]}},
        ],
    })

    def test_warns_when_doc_topics_but_no_docs_root(self):
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "v.json")
            with mock.patch("hive.decompose.call_worker", return_value=_wr(self._DOC_AXIS)), \
                 mock.patch("hive.judge.call_worker", return_value=_wr(VERDICT_OUT)), \
                 mock.patch("hive.investigate.retrieve", side_effect=_fake_retrieve), \
                 self.assertLogs("hive.investigate", level="WARNING") as cm:
                INV.run_investigate(
                    seed_text="x", recipe_path=None, code_root=td,
                    docs_root=None, output_path=out, cfg=cfg, ledger=None)
        joined = "\n".join(cm.output)
        self.assertIn("--docs", joined)
        self.assertIn("design-doc channel is DISABLED", joined)


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

    def test_localisations_are_evidence_not_per_axis_edit_imperatives(self):
        # T891: the renderer must NOT print "apply the requested change ... at this
        # location" under every axis (that turned corroborating localisations into
        # N competing edit imperatives, and the author followed the wrong one).
        self.assertNotIn("apply the requested change above at this location",
                         self.honey)
        # The seed's scope must be elevated to BINDING, with precedence over any
        # single localisation that conflicts with it.
        self.assertIn("BINDING", self.honey)
        self.assertIn("OVERRIDES", self.honey)
        self.assertIn("not all are edit sites", self.honey)

    def test_same_file_loci_grouped_with_convergence_note(self):
        # Two axes locating different lines in the SAME file (the T891 shape:
        # placeholder div vs v-for :class in DocWorkflow.vue) must be grouped and
        # a convergence note emitted so the author sees the co-location explicitly.
        result = {
            "axes_total": 2, "axes_judged": 2,
            "verdicts": [
                {"axis_id": "logs_review", "title": "placeholder",
                 "verdict": {"located": True, "file": "client/DocWorkflow.vue",
                             "lines": "9-15", "reason": "placeholder renders gray"}},
                {"axis_id": "async_pipeline", "title": "binding",
                 "verdict": {"located": True, "file": "client/DocWorkflow.vue",
                             "lines": "35-47", "reason": "v-for omits current"}},
            ],
        }
        honey = render_local_honey(result, "make ONLY the placeholder div blue")
        self.assertIn("Convergence", honey)
        self.assertIn("logs_review", honey)
        self.assertIn("async_pipeline", honey)
        # both loci surfaced
        self.assertIn("9-15", honey)
        self.assertIn("35-47", honey)

    def test_no_located_yields_defer_note(self):
        result = {"axes_total": 1, "axes_judged": 1, "verdicts": [
            {"axis_id": "X", "title": "t",
             "verdict": {"located": False, "file": "", "lines": "", "reason": "n/a"}}]}
        honey = render_local_honey(result, "do something")
        self.assertIn("defer", honey.lower())


class TestPrioritizeAxes(unittest.TestCase):
    """Deterministic seed-relevance ranking + seed-anchor injection (T891)."""

    SEED = (
        "in client/src/main/components/DocWorkflow.vue the placeholder div must "
        "render blue like .wf-step.current (client/shared/app.css:341-342). "
        "DO NOT modify the .wf-step.wf-undecided rule. Add the current class to "
        "that placeholder, keep wf-current-clickable."
    )

    def _leaves(self):
        return [
            {"id": "sql_gate", "title": "drop wsi status",
             "brief": "033_drop_wsi_status.sql status column",
             "search_plan": {"keywords": ["status", "DROP"],
                             "file_globs": ["server/sql/**/*.sql"]}},
            {"id": "async_pipeline", "title": "class binding",
             "brief": "DocWorkflow class binding omits current",
             "search_plan": {"keywords": ["wf-current-clickable"],
                             "file_globs": ["client/src/main/components/DocWorkflow.vue"]}},
        ]

    def test_seed_anchor_injected_at_front(self):
        out = _prioritize_axes(self._leaves(), self.SEED)
        self.assertEqual(out[0]["id"], "SEED_ANCHOR")
        # scoped to the concrete file(s) the seed named, with seed keywords
        self.assertIn("client/src/main/components/DocWorkflow.vue",
                      out[0]["search_plan"]["file_globs"])
        self.assertEqual(out[0]["depends_on"], [])

    def test_rabbit_hole_axis_sinks_below_seed_relevant(self):
        out = _prioritize_axes(self._leaves(), self.SEED)
        order = [t["id"] for t in out]
        # the on-topic queen axis outranks the unrelated SQL-drop axis
        self.assertLess(order.index("async_pipeline"), order.index("sql_gate"))

    def test_no_concrete_file_skips_injection(self):
        # A seed naming no concrete file injects nothing (the SEED_ANCHOR guard is
        # only for an explicitly-named file) and preserves the leaf set.
        seed = "make the undecided step blue at rest"
        out = _prioritize_axes(self._leaves(), seed)
        self.assertNotIn("SEED_ANCHOR", [t["id"] for t in out])
        self.assertEqual(len(out), len(self._leaves()))

    def test_ranks_by_keyword_overlap_when_no_file(self):
        # With no concrete file but an extractable (code-ish) keyword the seed
        # shares with an axis, that axis still outranks the unrelated one.
        seed = "the workflow_decided getter returns a stale value"
        leaves = [
            {"id": "noise", "title": "x", "brief": "unrelated",
             "search_plan": {"keywords": ["zzz"], "file_globs": ["a/**/*.py"]}},
            {"id": "match", "title": "y", "brief": "z",
             "search_plan": {"keywords": ["workflow_decided"], "file_globs": ["b/**/*.py"]}},
        ]
        out = _prioritize_axes(leaves, seed)
        self.assertNotIn("SEED_ANCHOR", [t["id"] for t in out])
        order = [t["id"] for t in out]
        self.assertLess(order.index("match"), order.index("noise"))

    def test_anchor_survives_cap_when_rabbit_holes_would_truncate(self):
        # 13 rabbit-hole leaves + a seed naming a file: even at max_axes=12 the
        # SEED_ANCHOR rides at position 0 and is never truncated.
        leaves = [{"id": f"R{i}", "title": "noise", "brief": "unrelated",
                   "search_plan": {"keywords": ["zzz"], "file_globs": ["other/**/*.py"]}}
                  for i in range(13)]
        out = _prioritize_axes(leaves, self.SEED)
        self.assertEqual(out[0]["id"], "SEED_ANCHOR")
        self.assertEqual(out[:12][0]["id"], "SEED_ANCHOR")  # front of any cap window


if __name__ == "__main__":
    unittest.main()
