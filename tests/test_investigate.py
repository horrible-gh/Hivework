"""Wiring test for hive.investigate — decompose → bridge → retrieve → judge.

No real model calls and no live codebase: ``call_worker`` (decompose + judge)
and ``retrieve`` are stubbed, so this verifies the *composition* (leaf gating,
bridge plan flowing into retrieve, judge verdict collected, report written)
without spending or touching ripgrep.
"""
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import investigate as INV
from hive.config import load_config
from hive.providers import WorkerResult
from hive.investigate import (
    render_local_honey, _prioritize_axes, _axis_coverage, _coverage_phrase,
    rerun_reinvestigation, dissolve_scaffolding_leaves, _leaf_axes,
    _is_scaffolding_axis, ensure_mutation_path_axis, _fk_persistence_seed,
    _extract_table_hints, _is_mutation_path_axis, _RESERVED_MUTATION_SEATS,
    _mutation_symptom_seed, promote_mutation_path_axes,
    inject_mutation_path_anchor, _writer_layer_hits,
)
from hive.reinvestigate import (
    ReinvestPlan, ACTION_RE_CONVERGE, ACTION_RE_RETRIEVE,
)


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
# Converge stub: the ④ stage stitches the located fragments into one path. Patched
# so the wiring tests stay hermetic (no deepinfra network call when ≥2 axes locate).
CONVERGE_OUT = json.dumps({
    "converged": True,
    "path": [{"node": "db_fn", "file": "server/x.py", "lines": "10-12",
              "symbol": "f"}],
    "attributed_defect": {"node": "db_fn", "file": "server/x.py", "lines": "10-12",
                          "why": "the bug runs here"},
    "missing_link": None,
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
        cfg.judge.votes_per_axis = 1       # one judgment per axis (pin: config SSOT=5)
        cfg.judge.max_axes = 3

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "verdicts.json")
            with mock.patch("hive.decompose.call_worker", return_value=_wr(DECOMPOSE_OUT)), \
                 mock.patch("hive.judge.call_worker", return_value=_wr(VERDICT_OUT)) as judge_cw, \
                 mock.patch("hive.converge.call_worker", return_value=_wr(CONVERGE_OUT)), \
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
        cfg.judge.votes_per_axis = 1
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

    def test_m035_fe_candidate_reaches_converge_from_backend_only_decompose(self):
        """The conditional decompose axis must put the real FE node in ④ input."""
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.votes_per_axis = 1
        cfg.judge.max_axes = 12
        backend_only = json.dumps({
            "fanout_decision": "single",
            "reason": "queen only considered backend head ordering",
            "steps": [["SQL_HEAD"]],
            "tasks": [{
                "id": "SQL_HEAD", "title": "SQL head ordering",
                "brief": "Inspect get_effective_head ORDER BY.",
                "depends_on": [],
                "search_plan": {
                    "keywords": ["get_effective_head", "ORDER BY"],
                    "file_globs": ["server/sql/queries/*.json"],
                    "doc_topics": [],
                },
            }],
        })
        seed = (
            "The workflow head is shifted by one step: the wrong current stage "
            "is highlighted and done/current/future colors are off by one."
        )

        def judge_result(_provider, _model, prompt, **_kwargs):
            if 'axis "FE_DERIVED_STATE"' in prompt:
                return _wr(json.dumps({"verdict": {
                    "located": True,
                    "file": "client/src/main/workflow/workflowViewState.ts",
                    "lines": "2-5",
                    "reason": "headIndex drives done/current/future state",
                }}))
            return _wr(json.dumps({"verdict": {
                "located": True,
                "file": "server/sql/queries/queries.json",
                "lines": "1-1",
                "reason": "backend comparison candidate",
            }}))

        with tempfile.TemporaryDirectory() as td:
            fe = os.path.join(td, "client", "src", "main", "workflow",
                              "workflowViewState.ts")
            be = os.path.join(td, "server", "sql", "queries", "queries.json")
            os.makedirs(os.path.dirname(fe), exist_ok=True)
            os.makedirs(os.path.dirname(be), exist_ok=True)
            with open(fe, "w", encoding="utf-8") as f:
                f.write(
                    "export function buildStepStates(steps, headType) {\n"
                    "  const headIndex = steps.indexOf(headType)\n"
                    "  return steps.map((_, idx) => idx < headIndex ? 'done' : "
                    "idx === headIndex ? 'current' : 'future')\n"
                    "}\n")
            with open(be, "w", encoding="utf-8") as f:
                f.write('{"get_effective_head":"SELECT * ORDER BY sort_order ASC"}\n')

            cres = mock.Mock(
                converged=True,
                attributed_defect={
                    "file": "client/src/main/workflow/workflowViewState.ts",
                    "lines": "2-5",
                },
                causal_check={"verdict": "consistent"},
                missing_link=None,
            )
            cres.as_dict.return_value = {
                "converged": True,
                "attributed_defect": cres.attributed_defect,
            }
            out = os.path.join(td, "verdicts.json")
            with mock.patch("hive.decompose.call_worker",
                            return_value=_wr(backend_only)), \
                 mock.patch("hive.judge.call_worker",
                            side_effect=judge_result), \
                 mock.patch.object(INV, "run_converge",
                                   return_value=cres) as converge:
                result = INV.run_investigate(
                    seed_text=seed, recipe_path=None, code_root=td,
                    docs_root=None, output_path=out, cfg=cfg, ledger=None)

        fe_verdict = next(
            v for v in result["verdicts"]
            if v["axis_id"] == "FE_DERIVED_STATE")
        self.assertTrue(fe_verdict["verdict"]["located"])
        self.assertEqual(
            fe_verdict["verdict"]["file"],
            "client/src/main/workflow/workflowViewState.ts")
        converge_kwargs = converge.call_args.kwargs
        self.assertTrue(any(
            (v.get("verdict") or {}).get("file")
            == "client/src/main/workflow/workflowViewState.ts"
            for v in converge_kwargs["verdicts"]))
        pooled_files = {
            s["file"].removeprefix("./")
            for bundle in converge_kwargs["bundles"]
            for s in (bundle.get("code_snippets") or [])
        }
        self.assertIn(
            "client/src/main/workflow/workflowViewState.ts", pooled_files)

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
                 mock.patch("hive.converge.call_worker", return_value=_wr(CONVERGE_OUT)), \
                 mock.patch("hive.investigate.retrieve", side_effect=_fake_retrieve):
                return INV.run_investigate(
                    seed_text="x", recipe_path=None, code_root=td,
                    docs_root=None, output_path=out, cfg=cfg, ledger=None,
                )

    def test_decisive_axis_not_dropped_under_default_ceiling(self):
        cfg = load_config()           # default max_axes=12 (runaway-ceiling)
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.votes_per_axis = 1
        result = self._run_five(cfg)
        self.assertEqual(result["axes_judged"], 5)
        self.assertIn("css_rules", {v["axis_id"] for v in result["verdicts"]})

    def test_truncation_warns_and_lists_dropped(self):
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.votes_per_axis = 1
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
        cfg.judge.votes_per_axis = 1
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


class TestParallelJudging(unittest.TestCase):
    """The per-axis retrieve+judge fan-out runs concurrently (bounded by
    judge.max_parallel) yet returns verdicts in the original judged order — same
    result as the sequential path, only faster."""

    def test_verdicts_preserve_judged_order(self):
        # 5 leaves, generic seed (no concrete file → stable order a1..css_rules).
        # Even with parallel workers completing out of order, the reassembled
        # verdicts must follow the judged order, so converge's input is unchanged.
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.votes_per_axis = 1
        cfg.judge.max_parallel = 4
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "v.json")
            with mock.patch("hive.decompose.call_worker",
                            return_value=_wr(TestInvestigateWiring._FIVE_LEAVES)), \
                 mock.patch("hive.judge.call_worker", return_value=_wr(VERDICT_OUT)), \
                 mock.patch("hive.converge.call_worker", return_value=_wr(CONVERGE_OUT)), \
                 mock.patch("hive.investigate.retrieve", side_effect=_fake_retrieve):
                result = INV.run_investigate(
                    seed_text="x", recipe_path=None, code_root=td,
                    docs_root=None, output_path=out, cfg=cfg, ledger=None)
        order = [v["axis_id"] for v in result["verdicts"]]
        self.assertEqual(order, ["a1", "a2", "a3", "a4", "css_rules"])

    def test_axes_actually_run_concurrently(self):
        # A judge that blocks on a barrier proves overlap: with max_parallel>=3 and
        # 3 axes, all three enter judge at once. A sequential loop would deadlock the
        # barrier (only one thread ever inside), so a clean pass IS the concurrency proof.
        import threading
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.votes_per_axis = 1
        cfg.judge.max_parallel = 3
        barrier = threading.Barrier(3, timeout=10)
        peak = {"n": 0}
        live = {"n": 0}
        lock = threading.Lock()

        def _blocking_judge(*a, **kw):
            with lock:
                live["n"] += 1
                peak["n"] = max(peak["n"], live["n"])
            barrier.wait()            # all 3 must be here simultaneously
            with lock:
                live["n"] -= 1
            return _wr(VERDICT_OUT)

        three = json.dumps({
            "fanout_decision": "fanout", "reason": "x", "steps": [["a", "b", "c"]],
            "tasks": [{"id": i, "title": i, "depends_on": [], "brief": i}
                      for i in ("a", "b", "c")]})
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "v.json")
            with mock.patch("hive.decompose.call_worker", return_value=_wr(three)), \
                 mock.patch("hive.judge.call_worker", side_effect=_blocking_judge), \
                 mock.patch("hive.converge.call_worker", return_value=_wr(CONVERGE_OUT)), \
                 mock.patch("hive.investigate.retrieve", side_effect=_fake_retrieve):
                result = INV.run_investigate(
                    seed_text="x", recipe_path=None, code_root=td,
                    docs_root=None, output_path=out, cfg=cfg, ledger=None)
        self.assertEqual(result["axes_judged"], 3)
        self.assertEqual(peak["n"], 3)   # all three judged at the same instant


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


class TestFormatCallerContext(unittest.TestCase):
    """Opt-in requester comments fold into the seed (and thus the local honey)."""

    def test_none_or_empty_yields_empty_string(self):
        self.assertEqual(INV.format_caller_context(None), "")
        self.assertEqual(INV.format_caller_context([]), "")
        self.assertEqual(INV.format_caller_context(["", "   "]), "")

    def test_comments_rendered_verbatim_in_order(self):
        out = INV.format_caller_context(["color is #FFF", "place it top-right"])
        self.assertIn(INV.CALLER_CONTEXT_SECTION, out)
        self.assertIn("- color is #FFF", out)
        self.assertIn("- place it top-right", out)
        # order preserved
        self.assertLess(out.index("#FFF"), out.index("top-right"))

    def test_guardrail_sentence_present(self):
        # intent authoritative, locations only a hint to verify — the non-hallucination line
        out = INV.format_caller_context(["the logic lives in theme.ts"])
        self.assertIn("authoritative", out.lower())
        self.assertIn("verify", out.lower())

    def test_blank_entries_dropped_but_real_kept(self):
        out = INV.format_caller_context(["", "real hint", "  "])
        self.assertIn("- real hint", out)
        self.assertEqual(out.count("\n- "), 1)

    def test_folds_into_local_honey_so_specify_sees_it(self):
        # The seam: appending to seed_text means render_local_honey carries it through.
        seed = "Make the badge blue." + INV.format_caller_context(["exact color is #1E90FF"])
        result = {"axes_total": 1, "axes_judged": 1, "verdicts": [
            {"axis_id": "X", "title": "t",
             "verdict": {"located": True, "file": "ui/badge.ts",
                         "lines": "10-12", "reason": "render call"}}]}
        honey = render_local_honey(result, seed)
        self.assertIn("#1E90FF", honey)
        self.assertIn(INV.CALLER_CONTEXT_SECTION, honey)


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


class TestSeedEditTargets(unittest.TestCase):
    """Defect 2 (T892): the seed's own explicitly-named files must be groundable
    edit targets even when no judge axis located them."""

    def _write(self, td, rel, text):
        path = os.path.join(td, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_resolves_seed_named_file_to_its_keyword_line(self):
        with tempfile.TemporaryDirectory() as td:
            self._write(td, "server/sql/queries/queries.json",
                        '{\n  "get_in_progress_head_by_group": "SELECT 1",\n'
                        '  "get_pending_head_by_group": "SELECT x WHERE result_doc_id '
                        'IS NULL ORDER BY sort_order LIMIT 1"\n}\n')
            seed = ("[Edit 1] server/sql/queries/queries.json get_pending_head_by_group "
                    "WHERE result_doc_id IS NULL ORDER BY sort_order")
            targets = INV.seed_edit_targets(seed, td)
            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0]["file"], "server/sql/queries/queries.json")
            # the densest keyword line is the get_pending line (line 3)
            self.assertEqual(targets[0]["lines"].split("-")[0], "3")

    def test_honours_explicit_line_written_in_seed(self):
        with tempfile.TemporaryDirectory() as td:
            self._write(td, "client/src/view.ts", "a\nb\nc\nd\ne\n")
            seed = "fix client/src/view.ts:2-3 current-step derivation"
            targets = INV.seed_edit_targets(seed, td)
            self.assertEqual(targets[0]["lines"], "2-3")

    def test_honours_prose_lines_range_near_file_mention(self):
        # The seed writes the range as prose ("Around lines 5-7"), not path:line.
        with tempfile.TemporaryDirectory() as td:
            self._write(td, "client/tests/spec.ts", "1\n2\n3\n4\n5\n6\n7\n8\n")
            seed = ("[Edit] client/tests/spec.ts\nAround lines 5-7 there is the "
                    "R-head fixture pinned to the buggy state.")
            targets = INV.seed_edit_targets(seed, td)
            self.assertEqual(targets[0]["lines"], "5-7")

    def test_skips_file_not_on_disk(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(INV.seed_edit_targets("edit server/x/nope.py do_it", td), [])

    def test_no_code_root_returns_empty(self):
        self.assertEqual(INV.seed_edit_targets("edit a/b.py thing", None), [])

    def test_honey_emits_seed_target_section_with_citation(self):
        with tempfile.TemporaryDirectory() as td:
            self._write(td, "server/sql/queries/queries.json",
                        '{\n  "get_pending_head_by_group": "SELECT x WHERE result_doc_id '
                        'IS NULL ORDER BY sort_order"\n}\n')
            seed = ("[Edit 1] server/sql/queries/queries.json get_pending_head_by_group "
                    "result_doc_id IS NULL ORDER BY sort_order")
            result = {"axes_total": 1, "axes_judged": 1, "verdicts": [
                {"axis_id": "X", "title": "t",
                 "verdict": {"located": False, "file": "", "lines": "", "reason": "n/a"}}]}
            honey = render_local_honey(result, seed, td)
            self.assertIn("Seed-specified edit targets", honey)
            self.assertIn("AUTHOR them", honey)
            self.assertIn("server/sql/queries/queries.json:", honey)

    def test_orientation_context_map_is_not_an_edit_target(self):
        # N177: a [System context] file map names files purely to ORIENT ("verify
        # yourself … do NOT anchor on this prose"). None carry an edit-intent cue, so
        # none may become a binding edit target (else the seed-coverage gate forces
        # edits onto files converge ruled out).
        with tempfile.TemporaryDirectory() as td:
            self._write(td, "server/modules/api/workflow_head_routes.py", "x = 1\n")
            self._write(td, "client/src/workflowViewState.ts", "let a = 1\n")
            seed = (
                "[System context]\n"
                "Workflow head API: `server/modules/api/workflow_head_routes.py` (verify). "
                "FE view-state: `client/src/workflowViewState.ts` (verify). "
                "Verify every path/symbol yourself via grep/read — do NOT anchor on this prose.\n"
                "[Instruction]\n"
                "Trace the endpoint end-to-end and PIN the single off-by-one node with code.")
            self.assertEqual(INV.seed_edit_targets(seed, td), [])

    def test_ruled_out_file_is_not_an_edit_target(self):
        # The seed names a file only to FORBID editing it ("Do NOT author an edit
        # there"); even with a stray edit-ish word it must never be a target.
        with tempfile.TemporaryDirectory() as td:
            self._write(td, "server/sql/queries/queries.json",
                        '{\n  "get_effective_head": "SELECT 1 ORDER BY sort_order"\n}\n')
            seed = ("The ORDER BY change to server/sql/queries/queries.json is a CONFIRMED "
                    "no-op. Do NOT author an edit there.")
            self.assertEqual(INV.seed_edit_targets(seed, td), [])

    def test_designated_edit_target_still_resolved(self):
        # The complement: a genuinely DESIGNATED target (edit-intent cue on the line)
        # is still lifted — the tightening must not drop real targets.
        with tempfile.TemporaryDirectory() as td:
            self._write(td, "client/src/view.ts", "a\nb\nfunction buildStepStates() {}\n")
            seed = "Edit client/src/view.ts to fix the buildStepStates off-by-one."
            targets = INV.seed_edit_targets(seed, td)
            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0]["file"], "client/src/view.ts")

    def test_do_not_use_as_fix_site_is_not_target(self):
        # T906: "Do NOT use X as the primary fix site" is a prohibition — even though the
        # phrase "fix site" trips the edit-intent cue, the do-not-USE rule must win so the
        # off-path file is never handed to the seed-coverage gate as a mandatory target.
        with tempfile.TemporaryDirectory() as td:
            self._write(td, "server/modules/flow_gate/api/v1/list_routes.py",
                        "def list_modules():\n    return []\n")
            seed = ("Do not use `server/modules/flow_gate/api/v1/list_routes.py` as the "
                    "primary fix site unless live FE binding proves the modal calls it.")
            self.assertEqual(INV.seed_edit_targets(seed, td), [])

    def test_off_path_file_is_not_target(self):
        # An "off-path" mention marks a decoy, never an edit designation.
        with tempfile.TemporaryDirectory() as td:
            self._write(td, "server/api/list_routes.py", "x = 1\n")
            seed = ("T904 changed that off-path area `server/api/list_routes.py` and the "
                    "modal still did not show the selector.")
            self.assertEqual(INV.seed_edit_targets(seed, td), [])


class TestApplyCallBudget(unittest.TestCase):
    """max_total_calls: one-number cap → reduce votes first, then trim axes."""

    def _axes(self, n):
        return [{"id": f"A{i}"} for i in range(n)]

    def test_zero_budget_is_unlimited(self):
        j = self._axes(6)
        judged, votes = INV._apply_call_budget(j, votes_cfg=5, max_calls=2, budget=0)
        self.assertEqual(len(judged), 6)
        self.assertEqual(votes, 5)

    def test_budget_large_enough_no_reduction(self):
        # 6 axes × 5 votes × 2 = 60 worst-case; budget 60 fits exactly.
        judged, votes = INV._apply_call_budget(self._axes(6), 5, 2, budget=60)
        self.assertEqual(len(judged), 6)
        self.assertEqual(votes, 5)

    def test_budget_reduces_votes_keeps_axes(self):
        # 6 axes, max_calls=2, budget 40 → affordable votes = 40 // (6×2) = 3.
        judged, votes = INV._apply_call_budget(self._axes(6), 5, 2, budget=40)
        self.assertEqual(len(judged), 6)       # axis coverage preserved
        self.assertEqual(votes, 3)
        self.assertLessEqual(len(judged) * votes * 2, 40)   # worst-case ≤ budget

    def test_budget_too_small_trims_axes_to_single_vote(self):
        # 6 axes × 2 = 12 worst-case for even ONE vote each; budget 8 can't fit.
        judged, votes = INV._apply_call_budget(self._axes(6), 5, 2, budget=8)
        self.assertEqual(votes, 1)
        self.assertEqual(len(judged), 4)       # 8 // 2 = 4 axes
        self.assertLessEqual(len(judged) * votes * 2, 8)

    def test_single_shot_votes_full_depth_under_budget(self):
        # max_calls=1 (single-shot voting): 6 axes × 5 × 1 = 30 ≤ 40 → no cut.
        judged, votes = INV._apply_call_budget(self._axes(6), 5, 1, budget=40)
        self.assertEqual(len(judged), 6)
        self.assertEqual(votes, 5)

    def test_worst_case_never_exceeds_budget(self):
        for n, v, mc, b in [(6, 5, 2, 40), (12, 5, 1, 40), (8, 4, 2, 13),
                            (3, 7, 2, 5), (10, 5, 2, 100)]:
            judged, votes = INV._apply_call_budget(self._axes(n), v, mc, b)
            self.assertLessEqual(len(judged) * votes * mc, b,
                                 f"n={n} v={v} mc={mc} b={b}")


class TestConvergeFragments(unittest.TestCase):
    """Best-of-N union expands into one converge fragment per distinct locus."""

    def test_located_axis_expands_each_candidate(self):
        verdicts = [{"axis_id": "A", "title": "t",
                     "verdict": {"located": True, "file": "a.py", "lines": "1-2",
                                 "reason": "r"},
                     "candidates": [{"file": "a.py", "lines": "1-2", "reason": "r"},
                                    {"file": "b.py", "lines": "9-9", "reason": "r2"}]}]
        frags = INV._converge_fragments(verdicts)
        self.assertEqual(len(frags), 2)
        self.assertEqual({f["verdict"]["file"] for f in frags}, {"a.py", "b.py"})
        self.assertTrue(all(f["axis_id"] == "A" for f in frags))

    def test_unlocated_axis_passes_through_once(self):
        v = {"axis_id": "B", "title": "t",
             "verdict": {"located": False, "file": "", "lines": "", "reason": "refute"},
             "candidates": []}
        frags = INV._converge_fragments([v])
        self.assertEqual(frags, [v])

    def test_single_vote_is_one_fragment_per_axis(self):
        # votes_per_axis=1 → one candidate per located axis → unchanged shape.
        verdicts = [{"axis_id": "A", "title": "t",
                     "verdict": {"located": True, "file": "a.py", "lines": "1", "reason": "r"},
                     "candidates": [{"file": "a.py", "lines": "1", "reason": "r"}]}]
        frags = INV._converge_fragments(verdicts)
        self.assertEqual(len(frags), 1)
        self.assertEqual(frags[0]["verdict"]["file"], "a.py")


class TestAxisCoverage(unittest.TestCase):
    """(B2/#5) The (flag ∧ thin) sufficiency tag — free, deterministic."""

    def _stats(self, *, raw=5, snips=3, dropped=None, kept=None):
        gv = {}
        if dropped is not None:
            gv["dropped_empty"] = dropped
        if kept is not None:
            gv["kept"] = kept
        return {"raw_hits": raw, "snippets": snips, "glob_validation": gv}

    def test_flagged_and_empty_find_needs_reinforcement(self):
        cov = _axis_coverage(True, self._stats(raw=0, snips=0))
        self.assertTrue(cov["flagged"])
        self.assertTrue(cov["thin"])
        self.assertTrue(cov["needs_reinforcement"])
        self.assertFalse(cov["sufficient"])

    def test_flagged_but_find_not_empty_is_sufficient(self):
        # queen self-doubt that the FIND disproves → no reinforcement.
        cov = _axis_coverage(True, self._stats(raw=5, snips=3))
        self.assertTrue(cov["flagged"])
        self.assertFalse(cov["thin"])
        self.assertFalse(cov["needs_reinforcement"])
        self.assertTrue(cov["sufficient"])

    def test_thin_without_flag_does_not_reinforce(self):
        cov = _axis_coverage(False, self._stats(raw=0, snips=0))
        self.assertTrue(cov["thin"])
        self.assertFalse(cov["needs_reinforcement"])

    def test_all_globs_empty_is_thin(self):
        cov = _axis_coverage(True, self._stats(raw=2, snips=1,
                                               dropped=["x/*.py"], kept=[]))
        self.assertTrue(cov["thin"])
        self.assertTrue(cov["needs_reinforcement"])

    def test_globs_dropped_but_some_kept_not_thin_on_that_signal(self):
        # one empty glob but another kept + real hits → not starved.
        cov = _axis_coverage(False, self._stats(raw=4, snips=2,
                                                dropped=["x/*.py"], kept=["y/*.py"]))
        self.assertFalse(cov["thin"])


class TestCoveragePhrase(unittest.TestCase):
    """(C/#5) honey clause separating a retrieval gap from a reasoning gap."""

    def test_needs_reinforcement_phrase(self):
        p = _coverage_phrase({"needs_reinforcement": True, "thin": True})
        self.assertIn("THIN", p)
        self.assertIn("re-retrieve", p)

    def test_thin_unflagged_phrase(self):
        p = _coverage_phrase({"needs_reinforcement": False, "thin": True})
        self.assertIn("thin", p)
        self.assertNotIn("THIN", p)

    def test_sufficient_phrase_directs_honest_defer(self):
        p = _coverage_phrase({"needs_reinforcement": False, "thin": False,
                              "sufficient": True})
        self.assertIn("sufficient", p)
        self.assertIn("defer honestly", p)

    def test_missing_tag_is_empty(self):
        self.assertEqual(_coverage_phrase(None), "")
        self.assertEqual(_coverage_phrase({}), " — evidence sufficient (the FIND "
                         "retrieved code here): retrieval was NOT the gap, so defer "
                         "honestly — re-fetching the same scope will not help")


class TestHoneyCoverageRendering(unittest.TestCase):
    """The local honey surfaces the sufficiency tag on unlocated axes (C/#5)."""

    def _result(self, coverage):
        return {
            "axes_judged": 1, "axes_total": 1, "seed_kind": "fix", "converge": None,
            "verdicts": [{
                "axis_id": "A", "title": "thin axis",
                "verdict": {"located": False, "file": "", "lines": "", "reason": "no hit"},
                "candidates": [], "coverage": coverage,
            }],
        }

    def test_thin_axis_renders_reretrieve_hint(self):
        honey = render_local_honey(self._result(
            {"needs_reinforcement": True, "thin": True, "sufficient": False}),
            "fix the bug")
        self.assertIn("EVIDENCE note", honey)
        self.assertIn("re-retrieve", honey)

    def test_sufficient_axis_directs_honest_defer(self):
        honey = render_local_honey(self._result(
            {"needs_reinforcement": False, "thin": False, "sufficient": True}),
            "fix the bug")
        self.assertIn("defer honestly", honey)

    def test_verdict_without_coverage_omits_note(self):
        r = self._result(None)
        r["verdicts"][0].pop("coverage")
        honey = render_local_honey(r, "fix the bug")
        self.assertNotIn("EVIDENCE note", honey)


class TestRerunReinvestigation(unittest.TestCase):
    """Reaction #3 LIVE re-run: gated, bounded, reuses converge on re-grounded evidence."""

    def setUp(self):
        self.cfg = load_config()
        self.tmp = tempfile.mkdtemp(prefix="hive_reinv_")
        self.honey = os.path.join(self.tmp, "h.honey.md")

    def _located_verdicts(self):
        def v(axis):
            return {"axis_id": axis, "title": f"axis {axis}",
                    "search_plan": {"keywords": ["k"], "file_globs": ["s/*.py"],
                                    "doc_topics": []},
                    "verdict": {"located": True, "file": f"{axis}.py", "lines": "1-2",
                                "reason": "r"},
                    "candidates": [{"file": f"{axis}.py", "lines": "1-2", "reason": "r"}]}
        return [v("A"), v("B")]

    def _result(self, verdicts):
        return {"axes_judged": len(verdicts), "axes_total": len(verdicts),
                "seed_kind": "fix", "converge": None, "verdicts": verdicts}

    def _plan(self, action):
        return ReinvestPlan(action=action, reason_code="ineffective",
                            axis_ids=["A", "B"])

    def test_live_off_is_noop(self):
        self.cfg.reinvestigation.live = False
        with mock.patch.object(INV, "run_converge") as rc:
            out = rerun_reinvestigation(
                self._plan(ACTION_RE_CONVERGE), self._result(self._located_verdicts()),
                seed_text="s", code_root=".", docs_root=None, cfg=self.cfg,
                honey_out=self.honey)
        self.assertIsNone(out)
        rc.assert_not_called()                     # gated BEFORE any spend

    def test_re_converge_live_restitches_and_rerenders_honey(self):
        self.cfg.reinvestigation.live = True
        cres = mock.MagicMock()
        cres.as_dict.return_value = {"converged": True, "summary": "re-stitched"}
        with mock.patch.object(INV, "run_converge", return_value=cres), \
             mock.patch.object(INV, "retrieve", return_value={"axis_id": "X", "stats": {}}):
            out = rerun_reinvestigation(
                self._plan(ACTION_RE_CONVERGE), self._result(self._located_verdicts()),
                seed_text="s", code_root=".", docs_root=None, cfg=self.cfg,
                honey_out=self.honey)
        self.assertIsNotNone(out)
        self.assertEqual(out["converge"], {"converged": True, "summary": "re-stitched"})
        self.assertTrue(os.path.exists(self.honey))   # honey re-rendered for re-specify

    def test_re_converge_under_two_located_returns_none(self):
        self.cfg.reinvestigation.live = True
        one = self._located_verdicts()[:1]
        with mock.patch.object(INV, "run_converge") as rc, \
             mock.patch.object(INV, "retrieve", return_value={"stats": {}}):
            out = rerun_reinvestigation(
                self._plan(ACTION_RE_CONVERGE), self._result(one),
                seed_text="s", code_root=".", docs_root=None, cfg=self.cfg,
                honey_out=self.honey)
        self.assertIsNone(out)
        rc.assert_not_called()                     # nothing to re-stitch → no spend

    def test_re_retrieve_live_does_not_spend_yet(self):
        # re_retrieve re-judge is intentionally not wired — must not re-spend on specify.
        self.cfg.reinvestigation.live = True
        out = rerun_reinvestigation(
            self._plan(ACTION_RE_RETRIEVE), self._result(self._located_verdicts()),
            seed_text="s", code_root=".", docs_root=None, cfg=self.cfg,
            honey_out=self.honey)
        self.assertIsNone(out)

    def _verdict_at(self, axis, file):
        return {"axis_id": axis, "title": f"axis {axis}",
                "search_plan": {"keywords": ["k"], "file_globs": ["s/*.py"],
                                "doc_topics": []},
                "verdict": {"located": True, "file": file, "lines": "1-2", "reason": "r"},
                "candidates": [{"file": file, "lines": "1-2", "reason": "r"}]}

    def test_re_converge_excludes_refuted_locus_from_verdicts(self):
        """M035 ⑥→④: plan.exclude_loci drops the refuted locus from verdicts BEFORE the
        re-converge, so the re-stitch cannot re-crown it and must use the survivors."""
        self.cfg.reinvestigation.live = True
        verdicts = [self._verdict_at("BE", "db/workflow_sequences.py"),
                    self._verdict_at("FE", "client/src/workflow_view.ts"),
                    self._verdict_at("EP", "api/workflow_head_routes.py")]
        plan = ReinvestPlan(action=ACTION_RE_CONVERGE, reason_code="author_declared",
                            exclude_loci=[{"file": "db/workflow_sequences.py",
                                           "lines": "45-57"}])
        cres = mock.MagicMock()
        cres.as_dict.return_value = {"converged": True, "summary": "redirected"}
        with mock.patch.object(INV, "run_converge", return_value=cres) as rc, \
             mock.patch.object(INV, "retrieve", return_value={"axis_id": "X", "stats": {}}):
            out = rerun_reinvestigation(
                plan, self._result(verdicts), seed_text="s", code_root=".",
                docs_root=None, cfg=self.cfg, honey_out=self.honey)
        self.assertIsNotNone(out)
        passed = rc.call_args.kwargs["verdicts"]
        files = {f["verdict"]["file"] for f in passed}
        self.assertNotIn("db/workflow_sequences.py", files)   # refuted locus excluded
        self.assertEqual(files, {"client/src/workflow_view.ts",
                                 "api/workflow_head_routes.py"})

    def test_re_converge_exclusion_below_two_located_returns_none(self):
        """Excluding the refuted locus can drop below 2 survivors → honest NR, no spend."""
        self.cfg.reinvestigation.live = True
        verdicts = [self._verdict_at("BE", "db/workflow_sequences.py"),
                    self._verdict_at("FE", "client/src/workflow_view.ts")]
        plan = ReinvestPlan(action=ACTION_RE_CONVERGE, reason_code="author_declared",
                            exclude_loci=[{"file": "client/src/workflow_view.ts"},
                                          {"file": "db/workflow_sequences.py"}])
        with mock.patch.object(INV, "run_converge") as rc, \
             mock.patch.object(INV, "retrieve", return_value={"stats": {}}):
            out = rerun_reinvestigation(
                plan, self._result(verdicts), seed_text="s", code_root=".",
                docs_root=None, cfg=self.cfg, honey_out=self.honey)
        self.assertIsNone(out)
        rc.assert_not_called()


class TestScaffoldingLeafGuard(unittest.TestCase):
    """TSR hivework.0024.0003 (0082 Lv3, run475): a staircase DAG whose only leaf
    is a global-search/list funnel starves every diagnostic axis behind it. Fix A:
    dissolve the funnel leaf and promote its ungated dependents to leaves."""

    def _ids(self, tasks):
        return sorted(t.get("id") for t in tasks)

    def _leaf_ids(self, tasks):
        return sorted(t.get("id") for t in _leaf_axes(tasks))

    def test_promotes_dependents_when_sole_leaf_is_scaffolding(self):
        # The exact run475 shape: T0 = global search → list of file:line hits for
        # follow-ups; T1..T3 depend on it (incl. the FK axis); T4 synthesises.
        tasks = [
            {"id": "T0", "title": "Global code search for 'dispose'",
             "brief": "Search the codebase and produce a list of file:line hits "
                      "for follow-ups.", "depends_on": []},
            {"id": "T1", "title": "dispose handler", "brief": "inspect dispose",
             "depends_on": ["T0"]},
            {"id": "T2", "title": "close handler", "brief": "inspect close",
             "depends_on": ["T0"]},
            {"id": "T3", "title": "DB schema & constraints",
             "brief": "inspect group table, FOREIGN KEY and triggers",
             "depends_on": ["T0"]},
            {"id": "T4", "title": "synthesis", "brief": "combine",
             "depends_on": ["T1", "T2", "T3"]},
        ]
        # Before: the only leaf is the scaffolding funnel.
        self.assertEqual(self._leaf_ids(tasks), ["T0"])
        out = dissolve_scaffolding_leaves(tasks)
        # After: the funnel is gone and the 3 diagnostic axes became leaves; the
        # synthesis axis still depends on them and is NOT promoted.
        self.assertEqual(self._leaf_ids(out), ["T1", "T2", "T3"])
        self.assertNotIn("T0", self._ids(out))
        t4 = next(t for t in out if t["id"] == "T4")
        self.assertEqual(sorted(t4["depends_on"]), ["T1", "T2", "T3"])

    def test_noop_when_leaf_is_diagnostic(self):
        # A normal cut: independent diagnostic leaves + a synthesis dependent.
        tasks = [
            {"id": "A", "title": "sql gate", "brief": "inspect ORDER BY",
             "depends_on": []},
            {"id": "B", "title": "endpoint", "brief": "trace the handler",
             "depends_on": []},
            {"id": "C", "title": "synthesis", "brief": "combine A and B",
             "depends_on": ["A", "B"]},
        ]
        out = dissolve_scaffolding_leaves(tasks)
        self.assertIs(out, tasks)  # untouched
        self.assertEqual(self._leaf_ids(out), ["A", "B"])

    def test_noop_when_scaffolding_axis_has_no_dependents(self):
        # A lone scaffolding-worded leaf that gates nothing must NOT be dissolved
        # away — that would leave the run with nothing to judge.
        tasks = [
            {"id": "S", "title": "enumerate call sites",
             "brief": "find all occurrences of insert_event", "depends_on": []},
        ]
        out = dissolve_scaffolding_leaves(tasks)
        self.assertIs(out, tasks)
        self.assertEqual(self._leaf_ids(out), ["S"])

    def test_keeps_funnel_when_dissolving_strands_everything(self):
        # Scaffolding leaf gates a dependent that ALSO depends on a second
        # (non-existent-as-leaf) axis — guard never empties the leaf set.
        tasks = [
            {"id": "T0", "title": "global search",
             "brief": "produce a list of file:line hits for follow-ups",
             "depends_on": []},
            {"id": "T1", "title": "synth", "brief": "x",
             "depends_on": ["T0", "T1"]},  # self-cycle keeps T1 non-leaf
        ]
        out = dissolve_scaffolding_leaves(tasks)
        self.assertIs(out, tasks)  # kept the funnel rather than judge nothing

    def test_detector_matches_search_list_language_only(self):
        self.assertTrue(_is_scaffolding_axis(
            {"title": "Global code search", "brief": "produce a list of "
             "file:line hits for follow-ups"}))
        self.assertTrue(_is_scaffolding_axis(
            {"title": "enumerate", "brief": "find all occurrences of foo"}))
        self.assertFalse(_is_scaffolding_axis(
            {"title": "DB schema & constraints",
             "brief": "inspect the group table FOREIGN KEY and triggers"}))
        self.assertFalse(_is_scaffolding_axis(
            {"title": "dispose handler", "brief": "trace insert_event in dispose"}))

    def test_detector_does_not_flag_diagnostic_trace_with_fileline(self):
        # TSR hivework.0024.0005 live re-test (run2): an honest diagnostic TRACE
        # axis says "produce exact file:line locations" and "follow all internal
        # calls" because the seed asks for file:line. Matching that as scaffolding
        # would false-dissolve a real axis, so the detector must NOT flag it.
        self.assertFalse(_is_scaffolding_axis({
            "title": "Trace backend POST /groups/{id}/dispose endpoint",
            "brief": "follow all internal calls to the first thrown exception that "
                     "results in a 500. Produce exact file:line locations along the "
                     "call chain and a minimal call-path map (caller -> callee)."}))
        self.assertFalse(_is_scaffolding_axis({
            "title": "Locate DB/SQL gates that block disposal",
            "brief": "Report the exact SQL or ORM call, file:line, and highlight "
                     "predicates (NULL checks, FK constraints) that could throw."}))

    def test_diagnostic_trace_gating_synth_is_not_dissolved(self):
        # The same diagnostic trace, now gating a synthesis axis: the gates-dependents
        # rule alone is NOT enough — the detector precision is what prevents the real
        # trace axis from being dissolved and replaced by its synthesis dependent.
        tasks = [
            {"id": "TRACE", "title": "Trace dispose endpoint",
             "brief": "follow all internal calls; produce exact file:line locations "
                      "along the call chain to the 500.", "depends_on": []},
            {"id": "SYNTH", "title": "synthesis",
             "brief": "combine", "depends_on": ["TRACE"]},
        ]
        out = dissolve_scaffolding_leaves(tasks)
        self.assertIs(out, tasks)  # untouched — TRACE is diagnostic, not a funnel
        self.assertEqual(self._leaf_ids(out), ["TRACE"])


class TestMutationPathGuard(unittest.TestCase):
    """NR hivework.0034.0006 — pin write-path axes ahead of the max_axes cut for
    persistence-class (FK/constraint) bugs (run500 MISS root fix)."""

    # A run500-shaped FK symptom (bare SQLite message names no table).
    FK_SEED = ("POST /groups/{id}/dispose returns 500: sqlite3.IntegrityError: "
               "FOREIGN KEY constraint failed. The dispose event write fails.")
    # The same symptom but naming concrete write symbols (richer message).
    FK_SEED_NAMED = (
        "dispose 500: insert_event(group_id, ...) violates the events.doc_id "
        "FOREIGN KEY into documents; should use insert_group_event / group_events.")

    def _srv_axis(self):
        return {"id": "T3_srv_write", "title": "dispose service write path",
                "brief": "trace dispose_group event write to the events table",
                "search_plan": {"keywords": ["insert_event", "dispose_group"],
                                "file_globs": ["server/modules/**/*.py"]}}

    # ── seed classifier (a) ────────────────────────────────────────────────
    def test_fk_seed_is_persistence_class(self):
        self.assertTrue(_fk_persistence_seed(self.FK_SEED))
        self.assertTrue(_fk_persistence_seed(self.FK_SEED_NAMED))

    def test_plain_http_or_fe_seed_is_not_persistence_class(self):
        self.assertFalse(_fk_persistence_seed(
            "the workflow head badge shows the wrong colour in DocHeader.vue"))
        self.assertFalse(_fk_persistence_seed(
            "GET /api/list returns items in the wrong sort order"))

    def test_constraint_word_without_failure_cue_does_not_fire(self):
        # A constraint cue alone (no write/failure cue) must not arm the guard.
        self.assertFalse(_fk_persistence_seed(
            "document the foreign key relationships in the schema diagram"))

    # ── table hints (precision aid, optional) ──────────────────────────────
    def test_table_hints_extracted_and_noise_dropped(self):
        hints = _extract_table_hints(self.FK_SEED_NAMED)
        self.assertIn("insert_event", hints)
        self.assertIn("group_events", hints)
        self.assertIn("events", hints)          # from events.doc_id
        self.assertNotIn("group_id", hints)     # generic noise dropped
        self.assertNotIn("doc_id", hints)

    def test_bare_message_yields_no_table_hints(self):
        # SQLite bare message names no symbol → empty hints → grep fallback path.
        self.assertEqual(_extract_table_hints("FOREIGN KEY constraint failed"),
                         set())

    # ── axis classifier (b) — text route ───────────────────────────────────
    def test_axis_with_write_helper_in_text_is_mutation_path(self):
        self.assertTrue(_is_mutation_path_axis(
            {"id": "x", "title": "t", "brief": "calls insert_event(...) on dispose",
             "search_plan": {"keywords": [], "file_globs": []}},
            set(), None))

    def test_readonly_axis_is_not_mutation_path(self):
        self.assertFalse(_is_mutation_path_axis(
            {"id": "r", "title": "reader", "brief": "get_group / list rows via SELECT",
             "search_plan": {"keywords": ["get_group"], "file_globs": []}},
            set(), None))

    # ── axis classifier (b) — real-file grep route ─────────────────────────
    def test_grep_detects_write_in_globbed_file(self):
        with tempfile.TemporaryDirectory() as td:
            d = os.path.join(td, "server", "modules")
            os.makedirs(d)
            with open(os.path.join(d, "process_service.py"), "w",
                      encoding="utf-8") as f:
                f.write("def dispose_group(gid):\n"
                        "    db.insert_event(gid, 'group_disposed')\n")
            axis = {"id": "srv", "title": "service",
                    "brief": "dispose path",      # no write word in text
                    "search_plan": {"keywords": [],
                                    "file_globs": ["server/modules/**/*.py"]}}
            self.assertTrue(_is_mutation_path_axis(axis, set(), td))

    def test_grep_skips_test_files(self):
        with tempfile.TemporaryDirectory() as td:
            d = os.path.join(td, "tests")
            os.makedirs(d)
            with open(os.path.join(d, "test_x.py"), "w", encoding="utf-8") as f:
                f.write("db.insert_event(1, 'x')\n")
            axis = {"id": "t", "title": "t", "brief": "p",
                    "search_plan": {"keywords": [],
                                    "file_globs": ["tests/**/*.py"]}}
            self.assertFalse(_is_mutation_path_axis(axis, set(), td))

    # ── the guard: ordering / determinism ──────────────────────────────────
    def _run500_leaves(self):
        # 10 leaves with the SRV write-path axis buried at the back (low surface
        # relevance), mirroring run500's decompose where it lost the cap race.
        leaves = [{"id": f"N{i}", "title": f"noise {i}",
                   "brief": "unrelated read/list axis",
                   "search_plan": {"keywords": ["get_x"],
                                   "file_globs": ["client/**/*.vue"]}}
                  for i in range(9)]
        leaves.append(self._srv_axis())
        return leaves

    def test_srv_write_axis_pinned_inside_cap(self):
        # The core determinism guarantee: the write-path axis, last of 10, lands
        # INSIDE the max_axes=3 cut after the guard.
        leaves = self._run500_leaves()
        out = ensure_mutation_path_axis(leaves, self.FK_SEED, None)
        self.assertEqual(out[0]["id"], "T3_srv_write")
        self.assertIn("T3_srv_write", [a["id"] for a in out[:3]])

    def test_reserved_seats_bounded(self):
        # Even with many write axes, only _RESERVED_MUTATION_SEATS are pinned.
        leaves = [dict(self._srv_axis(), id=f"W{i}") for i in range(5)]
        out = ensure_mutation_path_axis(leaves, self.FK_SEED, None)
        # order preserved among the rest; pinned count is bounded.
        self.assertLessEqual(_RESERVED_MUTATION_SEATS, 2)
        self.assertEqual(len(out), len(leaves))   # reorder only, never adds/drops

    def test_noop_on_non_persistence_seed(self):
        leaves = self._run500_leaves()
        out = ensure_mutation_path_axis(leaves, "wrong sort order in the list", None)
        self.assertEqual([a["id"] for a in out], [a["id"] for a in leaves])

    def test_noop_when_no_mutation_axis(self):
        leaves = [{"id": f"N{i}", "title": "r", "brief": "read only get_x",
                   "search_plan": {"keywords": ["get_x"], "file_globs": []}}
                  for i in range(3)]
        out = ensure_mutation_path_axis(leaves, self.FK_SEED, None)
        self.assertEqual([a["id"] for a in out], [a["id"] for a in leaves])

    def test_bare_message_fallback_still_pins_via_text(self):
        # No table hints (bare SQLite) — the write signature alone qualifies.
        leaves = self._run500_leaves()
        out = ensure_mutation_path_axis(
            leaves, "FOREIGN KEY constraint failed during insert on dispose 500",
            None)
        self.assertEqual(out[0]["id"], "T3_srv_write")

    # ── end-to-end through the REAL pipeline (run500 reproduction) ──────────
    def _run500_decompose(self):
        # 10 leaves, SRV write-path axis LAST (lowest surface relevance — exactly
        # how run500 lost the cap race), plus a dependent synthesis axis.
        tasks = [{"id": f"N{i}", "title": f"reader {i}",
                  "brief": "list rows via get_x / SELECT for display",
                  "depends_on": [],
                  "search_plan": {"keywords": ["get_x"],
                                  "file_globs": ["client/**/*.vue"],
                                  "doc_topics": []}}
                 for i in range(9)]
        tasks.append({
            "id": "T3_srv_write", "title": "dispose service write",
            "brief": "trace dispose_group event write that hits the FK",
            "depends_on": [],
            "search_plan": {"keywords": ["insert_event", "dispose_group"],
                            "file_globs": ["server/**/*.py"], "doc_topics": []}})
        return json.dumps({"fanout_decision": "fanout", "reason": "x",
                           "steps": [[t["id"] for t in tasks]], "tasks": tasks})

    FK_SEED_E2E = ("POST /groups/{id}/dispose returns 500: "
                   "sqlite3.IntegrityError: FOREIGN KEY constraint failed.")

    def _judged_ids_for(self, seed):
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.votes_per_axis = 1
        cfg.judge.max_axes = 3                      # run500's cap
        seen = []

        def _record(_provider, _model, prompt, **_k):
            # axis id appears in the judge prompt's [Role] line ("for Hivework axis ...")
            m = re.search(r'axis "([^"]+)"', prompt)
            if m:
                seen.append(m.group(1))
            return _wr(VERDICT_OUT)

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "v.json")
            with mock.patch("hive.decompose.call_worker",
                            return_value=_wr(self._run500_decompose())), \
                 mock.patch("hive.judge.call_worker", side_effect=_record), \
                 mock.patch("hive.converge.call_worker",
                            return_value=_wr(CONVERGE_OUT)), \
                 mock.patch("hive.investigate.retrieve", side_effect=_fake_retrieve):
                INV.run_investigate(seed_text=seed, recipe_path=None, code_root=td,
                                    docs_root=None, output_path=out, cfg=cfg,
                                    ledger=None)
        return seen

    def test_e2e_srv_axis_judged_with_guard_on_fk_seed(self):
        # The fix: with an FK seed, the buried SRV write axis is judged despite cap=3.
        judged = self._judged_ids_for(self.FK_SEED_E2E)
        self.assertIn("T3_srv_write", judged)
        self.assertLessEqual(len(judged), 3)

    def test_e2e_srv_axis_cut_without_guard_on_plain_seed(self):
        # Negative control: a non-persistence seed → guard no-op → the SRV axis
        # stays last and is cut by max_axes=3 (run500's original MISS reproduced).
        judged = self._judged_ids_for("the list shows items in the wrong sort order")
        self.assertNotIn("T3_srv_write", judged)


class TestMutationPathPromotion(unittest.TestCase):
    """NR hivework.0035.0003 — promote buried NON-leaf write-path axes to leaves and
    broaden the mutation-symptom arm so a realistic seed (no FK word) arms the guard.
    Fixes leaf starvation (TSR hivework.0034.0012 / live run502 MISS)."""

    # The realistic run502 seed: "discard returns 500", NO FK/constraint word.
    REAL_SEED = ("Discarding a workflow group fails: POST /groups/{id}/dispose "
                 "returns 500 Internal Server Error. The close path fails too.")
    REAL_SEED_KO = ("그룹을 폐기하면 POST /groups/{id}/dispose 요청이 500 에러로 "
                    "떨어집니다. 마감 경로에서도 비슷하게 실패합니다.")
    FK_SEED = ("POST /groups/{id}/dispose returns 500: sqlite3.IntegrityError: "
               "FOREIGN KEY constraint failed.")

    # ── broadened arm (c) ──────────────────────────────────────────────────
    def test_realistic_dispose_500_seed_arms_without_fk_word(self):
        # The crux: the live seed carries NO FK word, so the OLD predicate slept.
        self.assertFalse(_fk_persistence_seed(self.REAL_SEED))
        self.assertTrue(_mutation_symptom_seed(self.REAL_SEED))

    def test_korean_dispose_seed_arms(self):
        self.assertTrue(_mutation_symptom_seed(self.REAL_SEED_KO))

    def test_fk_seed_still_arms(self):
        self.assertTrue(_mutation_symptom_seed(self.FK_SEED))

    def test_plain_read_seed_does_not_arm(self):
        # No mutation verb + no server error → no arm (over-fire gate holds).
        self.assertFalse(_mutation_symptom_seed(
            "GET /api/list returns items in the wrong sort order"))
        self.assertFalse(_mutation_symptom_seed(
            "the head badge shows the wrong colour in DocHeader.vue"))

    def test_error_without_mutation_verb_does_not_arm(self):
        # A 500 on a read path (no mutation verb) must NOT arm the broadened path.
        self.assertFalse(_mutation_symptom_seed(
            "GET /api/report returns 500 when rendering the chart"))

    # ── promotion guard ────────────────────────────────────────────────────
    def _run502_tasks(self):
        # run502's real shape: surface axes are leaves; the answer write-path axes
        # (service_logic, db_queries) are NON-leaf, buried behind the route axis.
        return [
            {"id": "api_route", "title": "route registration",
             "brief": "where POST /groups/{id}/dispose is registered",
             "depends_on": [],
             "search_plan": {"keywords": ["dispose"],
                             "file_globs": ["server/**/routes/*.py"]}},
            {"id": "frontend_ui", "title": "FE handler",
             "brief": "the dispose button handler", "depends_on": [],
             "search_plan": {"keywords": ["dispose"], "file_globs": ["client/**/*.vue"]}},
            {"id": "service_logic", "title": "dispose service write",
             "brief": "dispose_group calls insert_event(group_id, ...) on the events table",
             "depends_on": ["api_route"],
             "search_plan": {"keywords": ["insert_event", "dispose_group"],
                             "file_globs": ["server/**/*.py"]}},
            {"id": "db_queries", "title": "db write layer",
             "brief": "the insert_event helper that writes the dispose event row",
             "depends_on": ["service_logic"],
             "search_plan": {"keywords": ["insert_event"], "file_globs": ["server/**/*.py"]}},
            {"id": "synthesis_report", "title": "synthesise",
             "brief": "summarise findings for the report", "depends_on": ["db_queries"],
             "search_plan": {"keywords": [], "file_globs": []}},
        ]

    def test_buried_nonleaf_write_axis_promoted_to_leaf(self):
        tasks = self._run502_tasks()
        self.assertEqual([t["id"] for t in _leaf_axes(tasks)],
                         ["api_route", "frontend_ui"])   # writes buried
        out = promote_mutation_path_axes(tasks, self.REAL_SEED, None)
        leaf_ids = [t["id"] for t in _leaf_axes(out)]
        self.assertIn("service_logic", leaf_ids)         # rescued
        self.assertIn("db_queries", leaf_ids)
        self.assertNotIn("synthesis_report", leaf_ids)   # meta axis NOT promoted

    def test_promotion_bounded_by_reserved_seats(self):
        tasks = self._run502_tasks()
        out = promote_mutation_path_axes(tasks, self.REAL_SEED, None)
        # at most _RESERVED_MUTATION_SEATS newly-promoted (was 2 leaves).
        promoted = len(_leaf_axes(out)) - 2
        self.assertLessEqual(promoted, _RESERVED_MUTATION_SEATS)

    def test_promotion_is_noop_on_read_seed(self):
        tasks = self._run502_tasks()
        out = promote_mutation_path_axes(tasks, "wrong sort order in the list", None)
        self.assertEqual([t["id"] for t in _leaf_axes(out)],
                         ["api_route", "frontend_ui"])

    def test_promotion_is_noop_when_no_nonleaf_writes(self):
        # mutation-class seed but the only non-leaf axis is a read/synthesis axis.
        tasks = [
            {"id": "api_route", "title": "route", "brief": "dispose route",
             "depends_on": [], "search_plan": {"keywords": [], "file_globs": []}},
            {"id": "reader", "title": "reader", "brief": "list rows via get_x SELECT",
             "depends_on": ["api_route"],
             "search_plan": {"keywords": ["get_x"], "file_globs": []}},
        ]
        out = promote_mutation_path_axes(tasks, self.REAL_SEED, None)
        self.assertIs(out, tasks)                         # untouched (no-op)

    def test_promotion_does_not_mutate_input(self):
        tasks = self._run502_tasks()
        before = [dict(t) for t in tasks]
        promote_mutation_path_axes(tasks, self.REAL_SEED, None)
        self.assertEqual([t["depends_on"] for t in tasks],
                         [t["depends_on"] for t in before])

    # ── end-to-end: run502 reproduction through the real pipeline ───────────
    def _run502_decompose(self):
        return json.dumps({"fanout_decision": "fanout", "reason": "x",
                           "steps": [["api_route", "frontend_ui"],
                                     ["service_logic", "db_queries"],
                                     ["synthesis_report"]],
                           "tasks": self._run502_tasks()})

    def _judged_ids_for(self, seed):
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.votes_per_axis = 1
        cfg.judge.max_axes = 10                     # run502's cap (10 ≫ leaves)
        seen = []

        def _record(_provider, _model, prompt, **_k):
            m = re.search(r'axis "([^"]+)"', prompt)
            if m:
                seen.append(m.group(1))
            return _wr(VERDICT_OUT)

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "v.json")
            with mock.patch("hive.decompose.call_worker",
                            return_value=_wr(self._run502_decompose())), \
                 mock.patch("hive.judge.call_worker", side_effect=_record), \
                 mock.patch("hive.converge.call_worker",
                            return_value=_wr(CONVERGE_OUT)), \
                 mock.patch("hive.investigate.retrieve", side_effect=_fake_retrieve):
                INV.run_investigate(seed_text=seed, recipe_path=None, code_root=td,
                                    docs_root=None, output_path=out, cfg=cfg,
                                    ledger=None)
        return seen

    def test_e2e_buried_write_axis_judged_on_realistic_seed(self):
        # The fix end-to-end: the NON-leaf service_logic / db_queries axes are
        # promoted and judged despite carrying depends_on — run502 MISS resolved.
        judged = self._judged_ids_for(self.REAL_SEED)
        self.assertIn("service_logic", judged)
        self.assertIn("db_queries", judged)

    def test_e2e_buried_write_axis_cut_without_arm_on_read_seed(self):
        # Negative control: a read-only seed never arms promotion, so the buried
        # write axes stay non-leaf and unjudged (original starvation reproduced).
        judged = self._judged_ids_for("the list shows items in the wrong sort order")
        self.assertNotIn("service_logic", judged)
        self.assertNotIn("db_queries", judged)


class TestMutationPathInjection(unittest.TestCase):
    """NR hivework.0037.0007 — the GENERATION gap: when a stochastic decompose emits
    NO write-path axis at all (live run506/507/508/509), the reorder guards are no-ops
    (nothing to select) and recall stays 0/1. inject_mutation_path_anchor synthesises
    the missing candidate from a real-file grep — the recall lever the reorder guards
    cannot be."""

    FK_SEED = ("POST /groups/{id}/dispose returns 500: sqlite3.IntegrityError: "
               "FOREIGN KEY constraint failed.")
    REAL_SEED = ("Discarding a workflow group fails: POST /groups/{id}/dispose "
                 "returns 500 Internal Server Error.")

    def _mk_repo(self, td):
        # A realistic write site in a service layer + a noise reader.
        svc = os.path.join(td, "server", "modules")
        os.makedirs(svc)
        with open(os.path.join(svc, "process_service.py"), "w",
                  encoding="utf-8") as f:
            f.write("def dispose_group(group_id, reason):\n"
                    "    db.insert_event(group_id, 'group_disposed', note=reason)\n")
        with open(os.path.join(svc, "reader.py"), "w", encoding="utf-8") as f:
            f.write("def list_groups():\n    return db.get_all('SELECT * FROM groups')\n")

    def _read_only_leaves(self):
        # The exact failure mode: every leaf is a read/FE axis — NO write-path axis.
        return [{"id": f"N{i}", "title": f"reader {i}",
                 "brief": "list rows via get_x / SELECT for display",
                 "search_plan": {"keywords": ["get_x"],
                                 "file_globs": ["client/**/*.vue"], "doc_topics": []}}
                for i in range(4)]

    def test_injects_anchor_when_no_write_axis_exists(self):
        with tempfile.TemporaryDirectory() as td:
            self._mk_repo(td)
            out = inject_mutation_path_anchor(self._read_only_leaves(),
                                              self.FK_SEED, td)
            self.assertEqual(out[0]["id"], "MUTATION_ANCHOR")
            # scoped to the REAL write site, not the reader.
            globs = out[0]["search_plan"]["file_globs"]
            self.assertTrue(any("process_service.py" in g for g in globs))
            self.assertFalse(any("reader.py" in g for g in globs))

    def test_injects_on_realistic_seed_without_fk_word(self):
        with tempfile.TemporaryDirectory() as td:
            self._mk_repo(td)
            out = inject_mutation_path_anchor(self._read_only_leaves(),
                                              self.REAL_SEED, td)
            self.assertEqual(out[0]["id"], "MUTATION_ANCHOR")

    def test_noop_when_a_write_axis_already_exists(self):
        # The queen DID emit a write-path leaf → reorder guards own it, never duplicate.
        with tempfile.TemporaryDirectory() as td:
            self._mk_repo(td)
            leaves = self._read_only_leaves() + [
                {"id": "srv", "title": "service write",
                 "brief": "dispose_group insert_event write",
                 "search_plan": {"keywords": ["insert_event"],
                                 "file_globs": ["server/modules/**/*.py"],
                                 "doc_topics": []}}]
            out = inject_mutation_path_anchor(leaves, self.FK_SEED, td)
            self.assertIs(out, leaves)
            self.assertNotIn("MUTATION_ANCHOR", [a["id"] for a in out])

    def test_noop_on_non_mutation_seed(self):
        with tempfile.TemporaryDirectory() as td:
            self._mk_repo(td)
            out = inject_mutation_path_anchor(
                self._read_only_leaves(),
                "the list shows items in the wrong sort order", td)
            self.assertNotIn("MUTATION_ANCHOR", [a["id"] for a in out])

    def test_noop_when_repo_has_no_write_site(self):
        with tempfile.TemporaryDirectory() as td:
            d = os.path.join(td, "server")
            os.makedirs(d)
            with open(os.path.join(d, "reader.py"), "w", encoding="utf-8") as f:
                f.write("def list_groups():\n    return get_all('SELECT 1')\n")
            out = inject_mutation_path_anchor(self._read_only_leaves(),
                                              self.FK_SEED, td)
            self.assertNotIn("MUTATION_ANCHOR", [a["id"] for a in out])

    def test_noop_without_code_root(self):
        out = inject_mutation_path_anchor(self._read_only_leaves(),
                                          self.FK_SEED, None)
        self.assertNotIn("MUTATION_ANCHOR", [a["id"] for a in out])

    def test_writer_layer_hits_skip_test_files(self):
        with tempfile.TemporaryDirectory() as td:
            d = os.path.join(td, "tests")
            os.makedirs(d)
            with open(os.path.join(d, "test_dispose.py"), "w",
                      encoding="utf-8") as f:
                f.write("db.insert_event(1, 'x')\n")
            self.assertEqual(_writer_layer_hits(td, set()), [])

    # ── end-to-end: the run506/509 GENERATION gap, judged after injection ────
    def _decompose_no_write_axis(self):
        # Mirrors run509: decompose emits ONLY read/FE axes — the write site absent.
        tasks = [{"id": f"N{i}", "title": f"reader {i}",
                  "brief": "list rows via get_x / SELECT for display",
                  "depends_on": [],
                  "search_plan": {"keywords": ["get_x"],
                                  "file_globs": ["client/**/*.vue"],
                                  "doc_topics": []}}
                 for i in range(4)]
        return json.dumps({"fanout_decision": "fanout", "reason": "x",
                           "steps": [[t["id"] for t in tasks]], "tasks": tasks})

    def _judged_ids_for(self, seed, repo_builder):
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.votes_per_axis = 1
        cfg.judge.max_axes = 10
        seen = []

        def _record(_provider, _model, prompt, **_k):
            m = re.search(r'axis "([^"]+)"', prompt)
            if m:
                seen.append(m.group(1))
            return _wr(VERDICT_OUT)

        with tempfile.TemporaryDirectory() as td:
            repo_builder(td)
            out = os.path.join(td, "v.json")
            with mock.patch("hive.decompose.call_worker",
                            return_value=_wr(self._decompose_no_write_axis())), \
                 mock.patch("hive.judge.call_worker", side_effect=_record), \
                 mock.patch("hive.converge.call_worker",
                            return_value=_wr(CONVERGE_OUT)), \
                 mock.patch("hive.investigate.retrieve", side_effect=_fake_retrieve):
                INV.run_investigate(seed_text=seed, recipe_path=None, code_root=td,
                                    docs_root=None, output_path=out, cfg=cfg,
                                    ledger=None)
        return seen

    def test_e2e_injected_anchor_is_judged_on_fk_seed(self):
        # The whole point: with NO write axis in the decompose, the answer locus
        # STILL reaches the judge because the anchor was injected (recall 0→1 path).
        judged = self._judged_ids_for(self.FK_SEED, self._mk_repo)
        self.assertIn("MUTATION_ANCHOR", judged)

    def test_e2e_no_injection_on_read_seed(self):
        # Negative control: a read-only seed never arms injection — original MISS.
        judged = self._judged_ids_for(
            "the list shows items in the wrong sort order", self._mk_repo)
        self.assertNotIn("MUTATION_ANCHOR", judged)


class TestGateIndependentFKMisrouting(unittest.TestCase):
    """Lever A (hivework.default.0036.0005-NR): the deterministic FK-misrouting check
    must fire at investigate level even when the judge dismisses the real write-path
    axis (located=False) so converge is skipped (located_n < 2) — the run506 0082 MISS.
    """

    _MIG = (
        "CREATE TABLE documents (doc_id TEXT PRIMARY KEY);\n"
        "CREATE TABLE groups (group_id TEXT PRIMARY KEY);\n"
        "CREATE TABLE events (\n"
        "  event_id INTEGER PRIMARY KEY,\n"
        "  doc_id TEXT NOT NULL REFERENCES documents(doc_id),\n"
        "  note TEXT);\n"
        "CREATE TABLE group_events (\n"
        "  event_id INTEGER PRIMARY KEY,\n"
        "  group_id TEXT NOT NULL REFERENCES groups(group_id),\n"
        "  note TEXT);\n"
    )
    # The 0082 regression: group_id (FK→groups) routed into events.doc_id (FK→documents).
    _BUG_SRC = (
        "def dispose_group(group_id, reason):\n"
        "    db.insert_event(group_id, \"group_disposed\", note=reason)\n"
    )
    _LEGAL_SRC = (
        "def dispose_group(group_id, reason):\n"
        "    db.insert_group_event(group_id, \"group_disposed\", note=reason)\n"
    )
    # One leaf axis whose retrieve will surface the write-path file.
    _DECOMPOSE = json.dumps({
        "fanout_decision": "fanout", "reason": "x",
        "steps": [["B"]],
        "tasks": [
            {"id": "B", "title": "dispose service", "depends_on": [],
             "brief": "trace server/x.py dispose_group event write",
             "search_plan": {"keywords": ["dispose_group"],
                             "file_globs": ["server/x.py"], "doc_topics": []}},
        ],
    })
    # The judge dismisses it — exactly run506's "functions as designed" refute.
    _REFUTE = json.dumps({"verdict": {"located": False, "type": "refuted",
                                      "reason": "code functions as designed"}})

    def _repo(self, td, src):
        mig = os.path.join(td, "server", "sql", "migrations")
        os.makedirs(mig, exist_ok=True)
        with open(os.path.join(mig, "001.sql"), "w", encoding="utf-8") as fh:
            fh.write(self._MIG)
        with open(os.path.join(td, "server", "x.py"), "w", encoding="utf-8") as fh:
            fh.write(src)

    @staticmethod
    def _retrieve(plan, code_root, docs_root=None, **kwargs):
        # Surface server/x.py in the bundle so the FK check has it to scan, even
        # though the judge will refute the axis (bundles are kept regardless).
        return {
            "axis_id": plan.axis_id,
            "code_snippets": [{"file": "server/x.py", "lines": "1-3",
                               "text": "def dispose_group(): ...", "hits": []}],
            "call_chain": [], "call_sites": [],
            "git_history": [], "design_excerpts": [],
            "stats": {"raw_hits": 2, "snippets": 1, "call_chain": 0},
        }

    def _run(self, src):
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.votes_per_axis = 1
        cfg.judge.max_axes = 5
        with tempfile.TemporaryDirectory() as td:
            self._repo(td, src)
            out = os.path.join(td, "verdicts.json")
            with mock.patch("hive.decompose.call_worker", return_value=_wr(self._DECOMPOSE)), \
                 mock.patch("hive.judge.call_worker", return_value=_wr(self._REFUTE)), \
                 mock.patch("hive.investigate.retrieve", side_effect=self._retrieve):
                return INV.run_investigate(
                    seed_text="disposing a group returns 500", recipe_path=None,
                    code_root=td, docs_root=None, output_path=out, cfg=cfg, ledger=None,
                )

    def setUp(self):
        os.environ.pop("HIVE_NO_FK_MISROUTE", None)

    def _fk_verdicts(self, result):
        return [v for v in result["verdicts"]
                if (v.get("verdict") or {}).get("via") == "fk-misrouting"]

    def test_fk_facet_injected_despite_judge_refute_and_converge_skip(self):
        result = self._run(self._BUG_SRC)
        # No real axis located → converge skipped; the gate-independent check still fires.
        fk = self._fk_verdicts(result)
        self.assertEqual(len(fk), 1, [v["verdict"] for v in result["verdicts"]])
        vd = fk[0]["verdict"]
        self.assertTrue(vd["located"])
        self.assertEqual(_norm(vd["file"]), "server/x.py")
        self.assertEqual(vd["lines"], "2")            # the insert_event line
        self.assertIn("group_id", vd["reason"])
        self.assertIn("events.doc_id", vd["reason"])
        # converge gate: facet lifted located_n to ≥1 deterministically (found restored).
        self.assertTrue(any(v["verdict"]["located"] for v in result["verdicts"]))

    def test_no_facet_on_legal_group_event_writer(self):
        # The fixed code (insert_group_event) must NOT trip the check (no false positive).
        result = self._run(self._LEGAL_SRC)
        self.assertEqual(self._fk_verdicts(result), [])

    def test_kill_switch_disables_gate_independent_check(self):
        os.environ["HIVE_NO_FK_MISROUTE"] = "1"
        try:
            result = self._run(self._BUG_SRC)
        finally:
            os.environ.pop("HIVE_NO_FK_MISROUTE", None)
        self.assertEqual(self._fk_verdicts(result), [])

    # ── Lever B (NR0011): post-convergence re-scan ─────────────────────────────
    _DECOMPOSE2 = json.dumps({
        "fanout_decision": "fanout", "reason": "x",
        "steps": [["P", "Q"]],
        "tasks": [
            {"id": "P", "title": "close path", "depends_on": [],
             "brief": "server/decoy.py close transition",
             "search_plan": {"keywords": ["close"], "file_globs": ["server/decoy.py"],
                             "doc_topics": []}},
            {"id": "Q", "title": "500 site", "depends_on": [],
             "brief": "server/decoy.py fail500",
             "search_plan": {"keywords": ["_fail"], "file_globs": ["server/decoy.py"],
                             "doc_topics": []}},
        ],
    })
    # The judge locates only the DECOY (grounded by retrieve) — never the real write file,
    # exactly run507: process_service.py never reaches a pre-converge bundle.
    _DECOY_LOCATED = json.dumps({"verdict": {"located": True, "file": "server/decoy.py",
                                             "lines": "1-2", "reason": "decoy 500 site"}})

    @staticmethod
    def _retrieve_decoy(plan, code_root, docs_root=None, **kwargs):
        return {
            "axis_id": plan.axis_id,
            "code_snippets": [{"file": "server/decoy.py", "lines": "1-2",
                               "text": "def close_group(): ...", "hits": []}],
            "call_chain": [], "call_sites": [],
            "git_history": [], "design_excerpts": [],
            "stats": {"raw_hits": 1, "snippets": 1, "call_chain": 0},
        }

    def test_lever_b_post_converge_rescan_promotes_fk_locus(self):
        # run507 shape: judge locates 2 decoys (converge runs), and converge's path-tracing
        # REACHES the real write file (server/x.py) as its generic attributed_defect — but
        # only AFTER the pre-converge FK checks ran. Lever B re-scans converge's final path
        # and promotes the proven FK locus.
        cfg = load_config()
        cfg.judge.max_calls_per_axis = 1
        cfg.judge.votes_per_axis = 1
        cfg.judge.max_axes = 5
        with tempfile.TemporaryDirectory() as td:
            self._repo(td, self._BUG_SRC)        # migration + server/x.py (the bug @ line 2)
            with open(os.path.join(td, "server", "decoy.py"), "w", encoding="utf-8") as fh:
                fh.write("def close_group():\n    return 500\n")
            cres = mock.Mock(converged=True,
                             attributed_defect={"file": "server/x.py", "lines": "1-2"},
                             causal_check={"verdict": "consistent"}, missing_link=None)
            cres.as_dict.return_value = {
                "converged": True,
                "attributed_defect": {"file": "server/x.py", "lines": "1-2"},
                "winning_path": [{"file": "server/x.py", "lines": "1-2"}],
            }
            out = os.path.join(td, "verdicts.json")
            with mock.patch("hive.decompose.call_worker", return_value=_wr(self._DECOMPOSE2)), \
                 mock.patch("hive.judge.call_worker", return_value=_wr(self._DECOY_LOCATED)), \
                 mock.patch("hive.investigate.retrieve", side_effect=self._retrieve_decoy), \
                 mock.patch.object(INV, "run_converge", return_value=cres):
                result = INV.run_investigate(
                    seed_text="disposing a group returns 500", recipe_path=None,
                    code_root=td, docs_root=None, output_path=out, cfg=cfg, ledger=None)
        # Lever A could not fire (server/x.py never in a pre-converge bundle); lever B did.
        fk = self._fk_verdicts(result)
        self.assertEqual(len(fk), 1, [v["verdict"] for v in result["verdicts"]])
        self.assertEqual(fk[0]["verdict"]["lines"], "2")     # the insert_event line
        self.assertEqual(_norm(fk[0]["verdict"]["file"]), "server/x.py")
        # Promoted to THE converge attribution, carrying the FK mechanism.
        ad = result["converge"]["attributed_defect"]
        self.assertEqual(ad.get("via"), "fk-misrouting")
        self.assertEqual(_norm(ad["file"]), "server/x.py")
        self.assertEqual(ad["lines"], "2")

    def test_fk_sibling_facets_are_rendered_as_coverage_targets(self):
        # The 0082 fixed rule needs BOTH dispose and close callsites. If converge crowns
        # one FK facet, the sibling must ride as an additional defect so specify cannot
        # ship a one-site half-fix.
        converge = {
            "converged": True,
            "attributed_defect": {
                "file": "server/x.py", "lines": "2", "via": "fk-misrouting",
                "why": "dispose group_id into events.doc_id",
            },
            "causal_check": {"verdict": "consistent"},
            "additional_defects": [],
        }
        verdicts = [
            {"verdict": {"located": True, "file": "server/x.py", "lines": "2",
                         "via": "fk-misrouting", "reason": "dispose"}},
            {"verdict": {"located": True, "file": "server/x.py", "lines": "5",
                         "via": "fk-misrouting", "reason": "close"}},
        ]

        INV._promote_fk_sibling_facets(converge, verdicts)

        extras = converge["additional_defects"]
        self.assertEqual(len(extras), 1)
        self.assertEqual(extras[0]["lines"], "5")
        honey = INV.render_local_honey(
            {"axes_total": 2, "axes_judged": 2, "seed_kind": "fix",
             "verdicts": verdicts, "converge": converge},
            "dispose and close return 500")
        self.assertIn("## Converge-attributed edit targets", honey)
        self.assertIn("- server/x.py:2", honey)
        self.assertIn("- server/x.py:5", honey)


def _norm(p: str) -> str:
    return (p or "").replace("\\", "/")


if __name__ == "__main__":
    unittest.main()
