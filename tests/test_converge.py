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
    "causal_check": {
        "verdict": "consistent",
        "data_state_assumptions": ["R approved; M in-progress with result_doc_id set"],
        "trace": "workflow_sequences.py: with M.result_doc_id set the CASE puts M first, "
                 "displacing the "
                 "pending slot — reproduces the off-by-one",
        "counterfactual": "fixing the ORDER BY removes the displacement because M no "
                          "longer outranks the pending slot",
        "refuted_peers": [
            {"file": "client/src/workflow_view.ts", "lines": "160-170",
             "why_not": "the holistic evidence selected the query path"},
        ],
        "need_data_state": []},
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

# The N170 shape: the converger reaches a node (get_effective_head's ORDER BY CASE)
# and would attribute the defect there, but the cause→symptom check shows that under
# the only data state the scenario allows the clause cannot produce the symptom.
CONTRADICTED_OUT = json.dumps({
    "converged": True,
    "path": [
        {"node": "endpoint", "file": "api/workflow_head_routes.py", "lines": "93-102"},
        {"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57",
         "symbol": "get_effective_head"},
    ],
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57",
                          "why": "ORDER BY CASE WHEN result_doc_id IS NOT NULL prefers a "
                                 "linked slot, shifting the head one step"},
    "causal_check": {
        "verdict": "contradicted",
        "data_state_assumptions": [
            "R approved → excluded by WHERE; M, DS, D not started → result_doc_id NULL"],
        "trace": "all candidate rows have result_doc_id NULL, so the CASE expression "
                 "ties at 1 for every row; the tie breaks on sort_order which already "
                 "orders M first — the ORDER BY CASE cannot skip M. The reported skip "
                 "cannot be produced here.",
        "need_data_state": []},
    "missing_link": None,
})

# NR173 shape: same contradiction as above, but the converger does NOT dead-end — it
# reasons that the symptom must still have a home elsewhere and points the next hunt
# there (the front-end render path that overrides the already-correct query result).
CONTRADICTED_WITH_LEAD_OUT = json.dumps({
    "converged": True,
    "path": [
        {"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57",
         "symbol": "get_effective_head"},
    ],
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57",
                          "why": "ORDER BY CASE suspected of shifting the head"},
    "causal_check": {
        "verdict": "contradicted",
        "data_state_assumptions": ["all candidate rows result_doc_id NULL"],
        "trace": "the query already returns the pending memo as head; this node cannot "
                 "produce the observed skip — the symptom originates elsewhere.",
        "need_data_state": []},
    "missing_link": {"between": ["get_effective_head", "workflow bar render"],
                     "need": {"symbols": ["renderWorkflowBar", "currentStep"],
                              "greps": ["current.*step", "head"],
                              "file_globs": ["web/**"]}},
})

# The redirect re-pass lands the defect on a DIFFERENT node — the render path — proving
# the refutation became the next, better-aimed search rather than a rejection.
REDIRECT_CONVERGED_OUT = json.dumps({
    "converged": True,
    "path": [
        {"node": "fe", "file": "web/workflow_bar.tsx", "lines": "20-40",
         "symbol": "renderWorkflowBar"},
    ],
    "attributed_defect": {"node": "fe", "file": "web/workflow_bar.tsx", "lines": "20-40",
                          "why": "render keys 'current' off item_seq, skipping the "
                                 "sort_order-0 memo"},
    "causal_check": {
        "verdict": "consistent",
        "data_state_assumptions": ["memo item_seq highest though sort_order 0"],
        "trace": "workflow_bar.tsx ordering by item_seq puts DS first, painting the memo done — "
                 "reproduces the observed skip.",
        "counterfactual": "keying the render by sort_order removes the skip because the "
                          "memo remains the current item",
        "refuted_peers": [
            {"file": "api/workflow_head_routes.py", "lines": "93-102",
             "why_not": "redirect evidence moved the symptom to the render path"},
            {"file": "db/workflow_sequences.py", "lines": "45-57",
             "why_not": "the query was causally contradicted before redirecting"},
        ],
        "need_data_state": []},
    "missing_link": None,
})

# Outcome depends on stored row state the static evidence cannot determine.
UNDECIDABLE_OUT = json.dumps({
    "converged": True,
    "path": [
        {"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57",
         "symbol": "get_effective_head"},
    ],
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57", "why": "ORDER BY may mis-rank the head"},
    "causal_check": {
        "verdict": "undecidable",
        "data_state_assumptions": ["depends on whether M carries a non-null result_doc_id"],
        "trace": "if M.result_doc_id is set the CASE displaces it; if NULL it does not — "
                 "the static evidence does not show the row state.",
        "need_data_state": ["the workflow_sequence_items rows for the failing doc: "
                            "result_doc_id and doc_review_status per slot"]},
    "missing_link": None,
})

# converged + attributed but the converger SKIPPED the mandated causal_check.
NO_CAUSAL_OUT = json.dumps({
    "converged": True,
    "path": [{"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57"}],
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57", "why": "ORDER BY wrong"},
    "missing_link": None,
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

    def test_run_uses_one_causal_provenance_arbiter(self):
        with mock.patch.object(
                C, "_causal_provenance_arbiter",
                wraps=C._causal_provenance_arbiter) as arbiter, \
             mock.patch.object(C, "call_worker", return_value=_wr(CONVERGED_OUT)):
            C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                           bundles=BUNDLES, provider="deepinfra", model="m")
        arbiter.assert_called_once()


# ── N172: undecidable → live DB data read → re-rule on fact ────────────────────
import sqlite3
import tempfile
from hive.config import DbConnection

# First pass: undecidable AND names the exact rows to read (data_reads). This is the
# N172 shape — the verdict hinges on doc_review_status, which static evidence can't see.
UNDECIDABLE_WITH_READS_OUT = json.dumps({
    "converged": True,
    "path": [{"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57",
              "symbol": "get_effective_head"}],
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57", "why": "ORDER BY may mis-rank the head"},
    "causal_check": {
        "verdict": "undecidable",
        "data_state_assumptions": ["depends on the head doc's doc_review_status"],
        "trace": "if the head doc is unapproved it is included in the candidate set; "
                 "static evidence does not show its stored review status.",
        "need_data_state": ["doc_review_status for the failing head doc"],
        "data_reads": [{"table": "documents",
                        "where": {"doc_id": "D1"},
                        "columns": ["doc_review_status"]}]},
    "missing_link": {"between": ["handler", "db_fn"],
                     "need": {"symbols": ["get_effective_head"], "greps": [], "file_globs": []}},
})


# N174 #2 shape: the first pass already CONVERGED (consistent) but — belt-and-suspenders
# under db_available — ALSO listed data_reads. If those reads come back EMPTY the re-pass
# can only say "undecidable"; that must NOT cancel the static convergence.
CONVERGED_WITH_READS_OUT = json.dumps({
    "converged": True,
    "path": [{"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57",
              "symbol": "get_effective_head"}],
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57", "why": "ORDER BY prefers in-progress"},
    "causal_check": {
        "verdict": "consistent",
        "data_state_assumptions": ["R approved"],
        "trace": "workflow_sequences.py reproduces the off-by-one",
        "counterfactual": "fixing the ordering removes the off-by-one because the pending "
                          "slot remains first",
        "refuted_peers": [
            {"file": "api/workflow_head_routes.py", "lines": "93-102",
             "why_not": "the handler only forwards the selected row"},
        ],
        "need_data_state": [],
        "data_reads": [{"table": "documents", "where": {"doc_id": "NOPE"},
                        "columns": ["doc_review_status"]}]},
    "missing_link": None,
})
# A re-pass that cannot decide and names NO further reads (so the read loop ends).
UNDECIDABLE_NO_READS_OUT = json.dumps({
    "converged": True,
    "path": [{"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57"}],
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57", "why": "ORDER BY may mis-rank"},
    "causal_check": {"verdict": "undecidable",
                     "data_state_assumptions": [], "trace": "rows empty — cannot tell",
                     "need_data_state": ["the deciding row"], "data_reads": []},
    "missing_link": None,
})


# N180: a CONSISTENT verdict certified on a live DB read, attributing to the query node
# (db/workflow_sequences.py) while LEAVING the SEED_ANCHOR fragment (the render-layer key
# hypothesis at api/workflow_head_routes.py) OFF the path and UNMENTIONED in the trace.
# The data read returns a row (backed) — so the consistency is data-certified, exactly the
# shape the dropped-peer domain guard must demote.
N180_DATA_CERT_OUT = json.dumps({
    "converged": True,
    "path": [{"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57",
              "symbol": "build_module_query"}],
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57", "why": "query omits project_modules — add UNION"},
    "causal_check": {
        "verdict": "consistent",
        "data_state_assumptions": ["modules exist in project_modules"],
        "trace": "workflow_sequences.py returns the module rows after the UNION",
        "counterfactual": "adding the missing source returns the rows because the query "
                          "then includes project_modules",
        "refuted_peers": [],
        "need_data_state": [],
        "data_reads": [{"table": "documents", "where": {"doc_id": "D1"},
                        "columns": ["doc_review_status"]}]},
    "missing_link": None,
})


def _tmp_db_with_doc(review_status):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "t.db")
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE documents (doc_id TEXT, doc_review_status TEXT)")
    c.execute("INSERT INTO documents VALUES ('D1', ?)", (review_status,))
    c.commit(); c.close()
    return DbConnection(kind="sqlite", path=path)


class TestConvergeDataRead(unittest.TestCase):
    """The undecidable→DB-read→re-rule loop. The DB read is REAL (temp sqlite); only
    the model call is mocked."""

    def test_undecidable_reads_db_and_rerules_contradicted(self):
        """N172: real rows refute the band-aid → contradicted → NOT converged.

        The missing-link band-aid re-pass must NOT run once the data has ruled.
        """
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            # pass 1: undecidable+reads ; pass 2 (has confirmed block): contradicted
            return _wr(CONTRADICTED_OUT if "Confirmed data state" in prompt
                       else UNDECIDABLE_WITH_READS_OUT)

        db = _tmp_db_with_doc("wf_in_progress")
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="head off-by-one for D1",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        # exactly two model calls: undecidable, then the fact-grounded re-pass
        self.assertEqual(len(prompts), 2)
        # the re-pass prompt carried the ACTUAL value read from the DB
        self.assertIn("Confirmed data state", prompts[1])
        self.assertIn("wf_in_progress", prompts[1])
        # ruled on fact → contradicted, not converged (band-aid stopped)
        self.assertFalse(res.converged)
        self.assertEqual(res.causal_check["verdict"], "contradicted")

    def test_undecidable_reads_db_and_rerules_consistent(self):
        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            return _wr(CONVERGED_OUT if "Confirmed data state" in prompt
                       else UNDECIDABLE_WITH_READS_OUT)

        db = _tmp_db_with_doc("")
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="head off-by-one for D1",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        self.assertTrue(res.converged)
        self.assertEqual(res.causal_check["verdict"], "consistent")

    def test_empty_read_does_not_cancel_a_converged_verdict(self):
        """N174 #2: a data re-pass that comes back EMPTY is a FAILED CONFIRMATION,
        not a convergence cancellation — the static consistent verdict must STAND."""
        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            # pass 1 converges (consistent) + lists reads; the empty re-pass is undecidable
            return _wr(UNDECIDABLE_NO_READS_OUT if "Confirmed data state" in prompt
                       else CONVERGED_WITH_READS_OUT)

        db = _tmp_db_with_doc("approved")  # holds only D1; the read targets 'NOPE' → empty
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="head off-by-one", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        # the converged consistent verdict survives the empty data re-pass
        self.assertTrue(res.converged)
        self.assertEqual(res.causal_check["verdict"], "consistent")
        # but the report is honest: a read was attempted and returned nothing
        self.assertTrue(res.data_state_attempted)
        self.assertFalse(res.data_state_backed)

    def test_fact_backed_contradiction_still_overturns_convergence(self):
        """The N173 guard is preserved: when reads ACTUALLY return rows and the re-pass
        rules contradicted on them, that fact DOES overturn even a converged first pass."""
        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            return _wr(CONTRADICTED_OUT if "Confirmed data state" in prompt
                       else CONVERGED_WITH_READS_OUT.replace('"doc_id": "NOPE"',
                                                              '"doc_id": "D1"'))

        db = _tmp_db_with_doc("approved")  # D1 exists → the read returns a row (backed)
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="head off-by-one", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        self.assertFalse(res.converged)
        self.assertEqual(res.causal_check["verdict"], "contradicted")
        self.assertTrue(res.data_state_backed)

    def test_n180_data_certified_consistent_dropping_peer_is_demoted(self):
        """N180: a `consistent` certified on a live DB read must NOT stay converged while a
        DISTINCT-locus located peer (the render-layer key hypothesis) was dropped unrefuted.
        A DB read proves rows exist, never that a render symptom is resolved."""
        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            return _wr(N180_DATA_CERT_OUT)
        db = _tmp_db_with_doc("approved")  # documents row D1 exists → the read is BACKED
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="the module selector renders empty",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        self.assertTrue(res.data_state_backed)        # the verdict WAS data-certified
        self.assertFalse(res.converged)               # demoted — no ready half-fix ships
        self.assertIn("dropped_peer", res.causal_check)
        self.assertEqual(res.causal_check["dropped_peer"]["axis_id"], "SEED_ANCHOR")
        self.assertIn("api/workflow_head_routes.py",
                      res.causal_check["dropped_peer"]["file"])

    def test_n180_guard_silent_when_peer_addressed_in_trace(self):
        """The REFUTE-BEFORE-DROP escape: if the converger mentions/refutes the peer in its
        trace, the drop is deliberate, not silent — the guard stays quiet and converged holds."""
        out = json.loads(N180_DATA_CERT_OUT)
        out["causal_check"]["trace"] += (" — the api/workflow_head_routes.py mapping is "
                                         "reachable but already reads the correct key, not "
                                         "the cause")
        payload = json.dumps(out)
        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            return _wr(payload)
        db = _tmp_db_with_doc("approved")
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="the module selector renders empty",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        self.assertTrue(res.converged)
        self.assertNotIn("dropped_peer", res.causal_check or {})

    def test_p0_demotes_unrefuted_peer_without_data_backing(self):
        """P0 generalizes N180: unrefuted distinct peers are unsafe in every domain."""
        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            return _wr(N180_DATA_CERT_OUT)
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="the module selector renders empty",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=None)
        self.assertFalse(res.data_state_backed)
        self.assertFalse(res.converged)
        self.assertIn("unrefuted_peer", res.causal_check or {})

    def test_no_db_conn_leaves_undecidable_unresolved(self):
        """Without a DB connection the data read is skipped — the static path stands."""
        calls = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            calls.append(prompt)
            return _wr(UNDECIDABLE_WITH_READS_OUT)

        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=None)
        # no confirmed-state re-pass; missing-link re-pass may run but never a DB read
        self.assertFalse(res.converged)
        self.assertTrue(all("Confirmed data state" not in p for p in calls))

    def test_db_read_failure_degrades_to_static(self):
        """A read error (table not present) must not crash; verdict stays undecidable."""
        bad_reads = json.loads(UNDECIDABLE_WITH_READS_OUT)
        bad_reads["causal_check"]["data_reads"] = [
            {"table": "no_such_table", "where": {"doc_id": "D1"}, "columns": ["x"]}]
        bad_reads["missing_link"] = None
        out = json.dumps(bad_reads)
        db = _tmp_db_with_doc("wf_in_progress")
        with mock.patch.object(C, "call_worker", return_value=_wr(out)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        self.assertFalse(res.converged)
        self.assertEqual(res.causal_check["verdict"], "undecidable")

    def test_fetch_data_state_renders_rows(self):
        db = _tmp_db_with_doc("approved")
        block = C._fetch_data_state(
            [{"table": "documents", "where": {"doc_id": "D1"},
              "columns": ["doc_review_status"]}], db)
        self.assertIn("doc_review_status", block)
        self.assertIn("approved", block)

    def test_wide_cell_is_truncated_in_rendered_block(self):
        """Row COUNT is hard-capped by LIMIT; a single huge cell (SELECT * over a wide
        TEXT column) must also be clipped so it can't bloat the prompt/token cost."""
        d = tempfile.mkdtemp()
        path = os.path.join(d, "wide.db")
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE docs (doc_id TEXT, meta TEXT)")
        c.execute("INSERT INTO docs VALUES ('D1', ?)", ("x" * 5000,))
        c.commit(); c.close()
        db = DbConnection(kind="sqlite", path=path)
        block = C._fetch_data_state([{"table": "docs", "where": {"doc_id": "D1"}}], db)
        self.assertIn("truncated", block)
        self.assertLess(len(block), 1000)            # 5000-char cell did NOT land whole

    def test_truncation_warning_when_set_hits_cap(self):
        """Reading exactly the cap means the set may be clipped — flag it so an ordering
        verdict isn't drawn on a partial set (silent truncation would mislead)."""
        d = tempfile.mkdtemp()
        path = os.path.join(d, "many.db")
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE items (sequence_id INTEGER, sort_order INTEGER)")
        for i in range(25):                          # more rows than the read cap
            c.execute("INSERT INTO items VALUES (1, ?)", (i,))
        c.commit(); c.close()
        db = DbConnection(kind="sqlite", path=path)
        block = C._fetch_data_state([{"table": "items", "where": {"sequence_id": 1},
                                      "columns": ["sort_order"]}], db)
        self.assertIn("TRUNCATED", block)

    def test_fetch_data_state_chains_reads(self):
        """Multi-hop: read a key from one table, carry it into the next read's WHERE.

        Mirrors the real need (resolve a row in table A, then read table B by the id A
        carried) WITHOUT any join logic or schema knowledge in the glue.
        """
        d = tempfile.mkdtemp()
        path = os.path.join(d, "chain.db")
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE seqs (head TEXT, seq_id INTEGER)")
        c.execute("CREATE TABLE items (seq_id INTEGER, label TEXT, result_doc_id TEXT)")
        c.execute("INSERT INTO seqs VALUES ('HEAD1', 7)")
        c.execute("INSERT INTO items VALUES (7, 'DS', NULL)")
        c.execute("INSERT INTO items VALUES (7, 'D', NULL)")
        c.commit(); c.close()
        db = DbConnection(kind="sqlite", path=path)

        reads = [
            {"id": "s", "table": "seqs", "where": {"head": "HEAD1"}, "columns": ["seq_id"]},
            {"id": "i", "table": "items",
             "where": {"seq_id": {"from": "s", "column": "seq_id"}},
             "columns": ["label", "result_doc_id"]},
        ]
        block = C._fetch_data_state(reads, db)
        # the second read was filtered by the value the first returned (seq_id=7);
        # identifiers render delimited now (reserved-word safe)
        self.assertIn('"seq_id" = 7', block)
        self.assertIn("label='DS'", block)
        # both item rows came back, and their NULL result_doc_id is visible as fact
        self.assertIn("result_doc_id=None", block)

    def test_fetch_data_state_chain_empty_upstream_skips(self):
        """When the upstream read returns nothing, the dependent read is skipped, not run
        with a bogus literal — and it degrades gracefully (no crash)."""
        d = tempfile.mkdtemp()
        path = os.path.join(d, "empty.db")
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE seqs (head TEXT, seq_id INTEGER)")
        c.execute("CREATE TABLE items (seq_id INTEGER, label TEXT)")
        c.commit(); c.close()
        db = DbConnection(kind="sqlite", path=path)
        reads = [
            {"id": "s", "table": "seqs", "where": {"head": "MISSING"}, "columns": ["seq_id"]},
            {"id": "i", "table": "items",
             "where": {"seq_id": {"from": "s", "column": "seq_id"}}, "columns": ["label"]},
        ]
        block = C._fetch_data_state(reads, db)
        self.assertIn("no upstream values", block)

    def test_resolve_where_multi_value_becomes_list(self):
        """Several distinct upstream values resolve to a list (→ IN), one to a scalar."""
        prior = {"src": [{"k": "A"}, {"k": "B"}, {"k": "A"}, {"k": None}]}
        resolved, skip = C._resolve_where({"col": {"from": "src", "column": "k"}}, prior)
        self.assertEqual(skip, "")
        self.assertEqual(resolved["col"], ["A", "B"])  # distinct, NULL dropped, order kept
        one, _ = C._resolve_where({"col": {"from": "src2", "column": "k"}},
                                  {"src2": [{"k": "only"}]})
        self.assertEqual(one["col"], "only")  # single value stays scalar

    def test_data_reads_parsed_into_causal_check(self):
        res = C._result_from(json.loads(UNDECIDABLE_WITH_READS_OUT), set())
        self.assertEqual(res.causal_check["data_reads"][0]["table"], "documents")
        self.assertEqual(res.causal_check["data_reads"][0]["where"], {"doc_id": "D1"})

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
            "causal_check": {"verdict": "consistent", "data_state_assumptions": [],
                             "trace": "reproduces",
                             "counterfactual": "correcting totally/unseen.py removes the "
                                               "symptom because its output changes",
                             "refuted_peers": [
                                 {"file": "api/workflow_head_routes.py", "lines": "93-102",
                                  "why_not": "not the mechanism selected"},
                                 {"file": "db/workflow_sequences.py", "lines": "45-57",
                                  "why_not": "not the mechanism selected"}],
                             "need_data_state": []},
            "missing_link": None})
        with mock.patch.object(C, "call_worker", return_value=_wr(out)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertTrue(res.converged)
        self.assertTrue(res.attributed_defect.get("ungrounded"))

    def test_contradicted_causal_rejects_attribution(self):
        """Reachable node whose code can't produce the symptom ⇒ NOT converged (N170)."""
        with mock.patch.object(C, "call_worker", return_value=_wr(CONTRADICTED_OUT)):
            res = C.run_converge(seed_text="R-head bar skips M, shows DS",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m")
        self.assertFalse(res.converged)                       # rejected, not a target
        self.assertEqual(res.causal_check["verdict"], "contradicted")
        self.assertIsNotNone(res.attributed_defect)           # suspected node kept
        self.assertIn("contradiction", res.summary)

    def test_contradicted_does_not_trigger_followup(self):
        """A causal contradiction is not a missing CODE link — no second pass/retrieve."""
        with mock.patch.object(C, "call_worker", return_value=_wr(CONTRADICTED_OUT)) as cw, \
             mock.patch.object(C, "retrieve_followup") as rf:
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m",
                                 code_root="/repo")
        rf.assert_not_called()
        self.assertEqual(cw.call_count, 1)
        self.assertFalse(res.converged)

    def test_undecidable_surfaces_need_data_state(self):
        """Outcome depends on stored row state ⇒ NOT converged, need_data_state carried."""
        with mock.patch.object(C, "call_worker", return_value=_wr(UNDECIDABLE_OUT)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertFalse(res.converged)
        self.assertEqual(res.causal_check["verdict"], "undecidable")
        self.assertTrue(res.causal_check["need_data_state"])
        self.assertIn("data state", res.summary)

    def test_missing_causal_check_fails_closed(self):
        """converged+attributed but no causal_check ⇒ unverified ⇒ NOT converged."""
        with mock.patch.object(C, "call_worker", return_value=_wr(NO_CAUSAL_OUT)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertFalse(res.converged)
        self.assertEqual(res.causal_check["verdict"], "unverified")

    def test_as_dict_carries_causal_check(self):
        with mock.patch.object(C, "call_worker", return_value=_wr(CONVERGED_OUT)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        d = res.as_dict()
        self.assertIn("causal_check", d)
        self.assertEqual(d["causal_check"]["verdict"], "consistent")

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

    def test_contradiction_with_lead_redirects_and_reconverges(self):
        """NR173: a refutation that NAMES where to look next re-hunts and can land the
        real defect on a DIFFERENT node — not a dead-end rejection."""
        outs = [_wr(CONTRADICTED_WITH_LEAD_OUT), _wr(REDIRECT_CONVERGED_OUT)]
        fu = {"seeds": [{"file": "web/workflow_bar.tsx", "lines": "20-40",
                         "text": "renderWorkflowBar: order by item_seq", "via": "need-grep"}],
              "call_chain": [], "stats": {}}
        with mock.patch.object(C, "call_worker", side_effect=outs) as cw, \
             mock.patch.object(C, "retrieve_followup", return_value=fu) as rf:
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m",
                                 code_root="/repo")
        rf.assert_called_once()
        self.assertEqual(cw.call_count, 2)                  # 1st + redirect re-converge
        self.assertTrue(res.converged)                      # adopted the redirect
        self.assertEqual(res.attributed_defect["node"], "fe")
        self.assertIn("workflow_bar", res.attributed_defect["file"])

    def test_converged_claim_without_node_is_demoted(self):
        out = json.dumps({"converged": True, "path": [],
                          "attributed_defect": None, "missing_link": None})
        with mock.patch.object(C, "call_worker", return_value=_wr(out)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertFalse(res.converged)


# ── N173: a configured DB must be READ, not fabricated around ───────────────────

# The N173 failure shape: the converger claims "consistent" while ASSUMING a stored
# value it never read (result_doc_id='doc123'). It DID name the rows in data_reads,
# so the pipeline can fetch them and re-rule on fact instead of trusting the fiction.
CONSISTENT_ON_ASSUMPTION_OUT = json.dumps({
    "converged": True,
    "path": [{"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57",
              "symbol": "get_effective_head"}],
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57", "why": "ORDER BY mis-ranks the head"},
    "causal_check": {
        "verdict": "consistent",
        "data_state_assumptions": ["assumes result_doc_id = 'doc123' (non-NULL)"],
        "trace": "workflow_sequences.py CASE displaces the pending slot when result_doc_id is set",
        "counterfactual": "fixing the CASE removes the displacement because non-null rows "
                          "no longer outrank the pending slot",
        "refuted_peers": [
            {"file": "api/workflow_head_routes.py", "lines": "93-102",
             "why_not": "the handler only forwards the selected row"},
        ],
        "need_data_state": [],
        "data_reads": [{"table": "documents", "where": {"doc_id": "D1"},
                        "columns": ["doc_review_status"]}]},
    "missing_link": None,
})


# ── M017 lever 2: data-stamp gate ──────────────────────────────────────────────
# A two-node path covering BOTH located files, so no peer is "dropped" (the N180 guard
# stays inert and these tests isolate the data-stamp gate).
_TWO_NODE_PATH = [
    {"node": "endpoint", "file": "api/workflow_head_routes.py", "lines": "93-102",
     "symbol": "GET /workflow/{doc_id}/head"},
    {"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57",
     "symbol": "get_effective_head"},
]

# The GAP lever 2 closes: a CONSISTENT verdict the converger itself flagged data_dependent
# (it rests on a stored value) but ruled on an ASSUMED value — NO data_reads, so the read
# loop never fires and nothing backs it.
M017_DATA_DEP_NO_READS_OUT = json.dumps({
    "converged": True,
    "path": _TWO_NODE_PATH,
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57", "why": "ORDER BY mis-ranks the head"},
    "causal_check": {
        "verdict": "consistent", "data_dependent": True,
        "data_state_assumptions": ["assumes M.result_doc_id is set (non-NULL)"],
        "trace": "workflow_sequences.py CASE displaces the pending slot when result_doc_id is set",
        "counterfactual": "fixing the CASE removes the displacement because M no longer "
                          "outranks the pending slot",
        "refuted_peers": [],
        "need_data_state": [], "data_reads": []},
    "missing_link": None,
})

# Data-dependent consistent that DOES name the deciding read — the loop runs it, the row
# exists (backed), and the re-pass rules consistent on FACT → it earns its stamp.
M017_DATA_DEP_WITH_READS_OUT = json.dumps({
    "converged": True,
    "path": _TWO_NODE_PATH,
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57", "why": "ORDER BY mis-ranks the head"},
    "causal_check": {
        "verdict": "consistent", "data_dependent": True,
        "data_state_assumptions": ["M.result_doc_id set per the read below"],
        "trace": "workflow_sequences.py CASE displaces the pending slot when result_doc_id is set",
        "counterfactual": "fixing the CASE removes the displacement because M no longer "
                          "outranks the pending slot",
        "refuted_peers": [],
        "need_data_state": [],
        "data_reads": [{"table": "documents", "where": {"doc_id": "D1"},
                        "columns": ["doc_review_status"]}]},
    "missing_link": None,
})

# A pure code-logic consistent: NOT data_dependent, names no reads — the gate must leave
# it alone (no stored value is in question).
M017_PURE_CODE_OUT = json.dumps({
    "converged": True,
    "path": _TWO_NODE_PATH,
    "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                          "lines": "45-57", "why": "off-by-one in the slice bound"},
    "causal_check": {
        "verdict": "consistent", "data_dependent": False,
        "data_state_assumptions": [],
        "trace": "workflow_sequences.py slice drops the last element regardless of stored state",
        "counterfactual": "fixing the slice bound retains the last element because the "
                          "exclusive endpoint includes the full sequence",
        "refuted_peers": [],
        "need_data_state": [], "data_reads": []},
    "missing_link": None,
})


class TestConvergeDataStamp(unittest.TestCase):
    """M017 lever 2: a ``consistent`` verdict that DEPENDS on stored data must be backed by
    a real DB read (a data-stamp), never ruled on an assumed value. Only the model call is
    mocked; the DB read is a real temp sqlite."""

    def test_data_dependent_consistent_without_read_is_demoted(self):
        """The gap: consistent + data_dependent + NO backing read → demoted, stamped."""
        db = _tmp_db_with_doc("approved")
        with mock.patch.object(C, "call_worker",
                               return_value=_wr(M017_DATA_DEP_NO_READS_OUT)):
            res = C.run_converge(seed_text="head off-by-one for D1",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        self.assertFalse(res.converged)                 # not a ready edit
        self.assertTrue(res.causal_check.get("data_unstamped"))
        self.assertFalse(res.data_state_backed)         # nothing was read
        self.assertIn("M017 data-stamp", res.summary)

    def test_data_dependent_consistent_backed_by_read_keeps_stamp(self):
        """The earned stamp: a real read returns the row → consistent on fact survives."""
        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            return _wr(M017_DATA_DEP_WITH_READS_OUT)   # same shape on both passes
        db = _tmp_db_with_doc("approved")               # documents row D1 exists → backed
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="head off-by-one for D1",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        self.assertTrue(res.data_state_backed)          # a real row backed it
        self.assertTrue(res.converged)                  # stamped → stays converged
        self.assertNotIn("data_unstamped", res.causal_check or {})

    def test_pure_code_consistent_is_not_touched(self):
        """A consistent ruled purely on code logic (data_dependent=False) is left alone even
        with a DB configured and no read run."""
        db = _tmp_db_with_doc("approved")
        with mock.patch.object(C, "call_worker", return_value=_wr(M017_PURE_CODE_OUT)):
            res = C.run_converge(seed_text="off-by-one slice",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        self.assertTrue(res.converged)
        self.assertNotIn("data_unstamped", res.causal_check or {})

    def test_gate_inert_without_db_conn(self):
        """No DB configured → nothing to read, nothing to enforce: the static verdict stands
        (no regression vs the pre-lever-2 behaviour)."""
        with mock.patch.object(C, "call_worker",
                               return_value=_wr(M017_DATA_DEP_NO_READS_OUT)):
            res = C.run_converge(seed_text="head off-by-one",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=None)
        self.assertTrue(res.converged)
        self.assertNotIn("data_unstamped", res.causal_check or {})


class TestConvergeN173DbMandate(unittest.TestCase):
    """N173: when a DB is configured the converger is told so and must not fabricate
    stored values; any named read is executed and the verdict re-ruled on fact."""

    def test_db_available_signalled_in_first_prompt(self):
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONVERGED_OUT)

        db = _tmp_db_with_doc("approved")
        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                           provider="deepinfra", model="m", code_root="/repo", db_conn=db)
        self.assertIn("LIVE DATABASE AVAILABLE", prompts[0])
        self.assertIn("doc123", prompts[0])  # the forbidden-fabrication example

    def test_no_db_no_mandate_in_prompt(self):
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONVERGED_OUT)

        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                           provider="deepinfra", model="m", db_conn=None)
        self.assertNotIn("LIVE DATABASE AVAILABLE", prompts[0])

    def test_consistent_on_assumption_is_overruled_by_real_rows(self):
        """The core N173 fix: a 'consistent' verdict built on an ASSUMED stored value
        still triggers the read (data_reads were named) and is re-ruled on the real
        row — here the fact CONTRADICTS the fabrication, so it is NOT converged."""
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONTRADICTED_OUT if "Confirmed data state" in prompt
                       else CONSISTENT_ON_ASSUMPTION_OUT)

        db = _tmp_db_with_doc("wf_in_progress")
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="head off-by-one for D1",
                                 verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        self.assertEqual(len(prompts), 2)                 # 1st (assumed) + fact re-pass
        self.assertIn("wf_in_progress", prompts[1])        # the REAL value, not doc123
        self.assertFalse(res.converged)                    # fact overruled the fiction
        self.assertEqual(res.causal_check["verdict"], "contradicted")

    def test_real_rows_carried_onto_result(self):
        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            return _wr(CONVERGED_OUT if "Confirmed data state" in prompt
                       else CONSISTENT_ON_ASSUMPTION_OUT)

        db = _tmp_db_with_doc("approved")
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        self.assertTrue(res.data_state_attempted)
        self.assertTrue(res.data_state_backed)
        self.assertIn("approved", res.data_state_block)
        d = res.as_dict()
        self.assertIn("approved", d["data_state_block"])
        self.assertTrue(d["data_state_backed"])

    def test_read_attempted_but_empty_is_honest_not_fabricated(self):
        """A named read that matches no row leaves the verdict unchanged and records
        the honest 'attempted, no rows' fact — no re-rule on a fiction."""
        out = json.loads(CONSISTENT_ON_ASSUMPTION_OUT)
        out["causal_check"]["verdict"] = "undecidable"
        out["causal_check"]["data_reads"] = [
            {"table": "documents", "where": {"doc_id": "NOPE"}, "columns": ["doc_review_status"]}]
        out["converged"] = False
        calls = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            calls.append(prompt)
            return _wr(json.dumps(out))

        db = _tmp_db_with_doc("approved")
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        # The iterative loop re-asks with the HONEST empty result ("no rows matched"),
        # never a fabricated value; and since the model re-asks for the SAME read with no
        # new angle, the loop stops after one re-pass (no infinite spin).
        repass = [p for p in calls if "Confirmed data state" in p]
        self.assertEqual(len(repass), 1)
        self.assertIn("no rows matched", repass[0])
        self.assertNotIn("doc123", repass[0])           # no invented value injected
        self.assertTrue(res.data_state_attempted)
        self.assertFalse(res.data_state_backed)
        self.assertIn("no rows matched", res.data_state_block)
        self.assertFalse(res.converged)


class TestConvergeSchemaGuard(unittest.TestCase):
    """NR174: a read naming a table/column that does not exist in the live schema must be
    SKIPPED before it becomes broken SQL (the 'WHERE column = ...' / wrong-table cases),
    recorded honestly — never run, never silently matching nothing."""

    def setUp(self):
        self.db = _tmp_db_with_doc("approved")  # documents(doc_id, doc_review_status)
        self.schema = C._introspect_schema(self.db)

    def test_unknown_table_rejected(self):
        reads = [{"id": "r", "table": "items", "where": {"doc_id": "D1"},
                  "columns": ["doc_review_status"]}]
        block, any_rows, _ = C._run_data_reads(reads, self.db, self.schema)
        self.assertIn("unknown table 'items'", block)
        self.assertFalse(any_rows)

    def test_unknown_column_rejected(self):
        # the NR174 hallucination: WHERE column = 'result_doc_id'
        reads = [{"id": "r", "table": "documents", "where": {"column": "result_doc_id"},
                  "columns": ["doc_review_status"]}]
        block, any_rows, _ = C._run_data_reads(reads, self.db, self.schema)
        self.assertIn("unknown column 'column'", block)
        self.assertFalse(any_rows)

    def test_valid_read_passes_guard(self):
        reads = [{"id": "r", "table": "documents", "where": {"doc_id": "D1"},
                  "columns": ["doc_review_status"]}]
        block, any_rows, _ = C._run_data_reads(reads, self.db, self.schema)
        self.assertTrue(any_rows)
        self.assertIn("approved", block)

    def test_empty_schema_is_noop_guard(self):
        # no schema (introspection failed) → guard cannot reject, read still runs
        reads = [{"id": "r", "table": "documents", "where": {"doc_id": "D1"},
                  "columns": ["doc_review_status"]}]
        block, any_rows, _ = C._run_data_reads(reads, self.db, {})
        self.assertTrue(any_rows)


class TestConvergeSchemaInjection(unittest.TestCase):
    """NR174: the converger mis-named the table (asked for 'items', real table is
    'workflow_sequence_items') so the read came back empty. The live schema is now
    introspected and injected so it names real objects."""

    def test_schema_block_injected_when_db_available(self):
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONVERGED_OUT)

        d = tempfile.mkdtemp()
        path = os.path.join(d, "t.db")
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE workflow_sequence_items "
                  "(id INTEGER, sequence_id INTEGER, result_doc_id TEXT)")
        c.commit(); c.close()
        db = DbConnection(kind="sqlite", path=path)
        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                           provider="deepinfra", model="m", code_root="/repo", db_conn=db)
        self.assertIn("DB SCHEMA — the live database", prompts[0])  # the injected block
        # the FULL real name is shown — the abbreviation it guessed before is impossible
        self.assertIn("workflow_sequence_items(", prompts[0])
        self.assertIn("result_doc_id", prompts[0])
        # N176: the schema-shape cross-check directive is present
        self.assertIn("CROSS-CHECK schema-shape claims", prompts[0])

    def test_no_schema_block_without_db(self):
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONVERGED_OUT)

        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                           provider="deepinfra", model="m", db_conn=None)
        self.assertNotIn("DB SCHEMA — the live database", prompts[0])

    def test_schema_introspection_failure_degrades_silently(self):
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONVERGED_OUT)

        # a connection pointing at a nonexistent file: list_schema raises, block is ''
        db = DbConnection(kind="sqlite", path="/no/such/file.db")
        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                           provider="deepinfra", model="m", code_root="/repo", db_conn=db)
        # still signals the DB is available; just no authoritative schema list
        self.assertNotIn("DB SCHEMA — the live database", prompts[0])
        self.assertIn("LIVE DATABASE AVAILABLE", prompts[0])

    def test_iterative_read_narrows_then_rules(self):
        """The 'read more' lever: a first read that can't decide → the model NARROWS the
        query and reads AGAIN, and only the second (deciding) row lets it rule. Proves
        the loop iterates instead of giving up after one shot."""
        d = tempfile.mkdtemp()
        path = os.path.join(d, "iter.db")
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE items (seq INTEGER, kind TEXT, val TEXT)")
        c.execute("INSERT INTO items VALUES (1, 'A', 'x')")
        c.execute("INSERT INTO items VALUES (1, 'B', 'decider')")
        c.commit(); c.close()
        db = DbConnection(kind="sqlite", path=path)

        broad = json.dumps({
            "converged": False,
            "path": [{"node": "db_fn", "file": "db/x.py", "lines": "1-2"}],
            "attributed_defect": {"node": "db_fn", "file": "db/x.py", "lines": "1-2",
                                  "why": "depends on the B row"},
            "causal_check": {"verdict": "undecidable", "data_state_assumptions": [],
                             "trace": "rows tie; need the B row", "need_data_state": [],
                             "data_reads": [{"table": "items", "where": {"seq": 1},
                                             "columns": ["kind"]}]},
            "missing_link": None})
        narrowed = json.loads(broad)
        narrowed["causal_check"]["data_reads"] = [
            {"table": "items", "where": {"seq": 1, "kind": "B"}, "columns": ["val"]}]
        narrowed = json.dumps(narrowed)

        calls = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            calls.append(prompt)
            if "Confirmed data state" not in prompt:
                return _wr(broad)                       # first pass: broad read
            if "'decider'" not in prompt and "decider" not in prompt:
                return _wr(narrowed)                    # saw broad rows → narrow & re-read
            return _wr(CONVERGED_OUT)                   # saw the deciding row → rule

        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db)
        self.assertEqual(len(calls), 3)                 # first + 2 read rounds
        self.assertTrue(res.converged)                  # the narrowed read settled it
        self.assertIn("decider", res.data_state_block)  # the deciding row was fetched


class TestHoneyPastesRealRows(unittest.TestCase):
    """The honey must PASTE the rows converge read, or honestly say the read was empty."""

    def _result(self, converge):
        return {"axes_judged": 2, "axes_total": 2, "seed_kind": "fix",
                "converge": converge, "verdicts": LOCATED_VERDICTS}

    def test_confirmed_rows_pasted_under_consistent(self):
        converge = {
            "converged": True,
            "path": [{"node": "db_fn", "file": "db/workflow_sequences.py",
                      "lines": "45-57", "symbol": "get_effective_head"}],
            "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                                  "lines": "45-57", "why": "ORDER BY wrong"},
            "causal_check": {"verdict": "consistent",
                             "data_state_assumptions": ["head row result_doc_id NULL"],
                             "trace": "reproduces", "need_data_state": []},
            "missing_link": None,
            "data_state_attempted": True, "data_state_backed": True,
            "data_state_block": "- query: SELECT result_doc_id FROM "
                                "workflow_sequence_items WHERE group_id = 'g1'\n"
                                "  -> result_doc_id=None, label='DS'"}
        honey = render_local_honey(self._result(converge), "fix the head")
        self.assertIn("Live DB data confirmed", honey)
        self.assertIn("workflow_sequence_items", honey)
        self.assertIn("result_doc_id=None", honey)
        self.assertNotIn("doc123", honey)

    def test_empty_read_reported_honestly(self):
        converge = {
            "converged": False, "path": [],
            "attributed_defect": {"node": "db_fn", "file": "x.py", "lines": "1-2",
                                  "why": "maybe"},
            "causal_check": {"verdict": "undecidable", "data_state_assumptions": [],
                             "trace": "depends on stored row", "need_data_state": ["rows"]},
            "missing_link": None,
            "data_state_attempted": True, "data_state_backed": False,
            "data_state_block": "- query: SELECT x FROM t WHERE id = 'NOPE'\n"
                                "  -> (no rows matched)"}
        honey = render_local_honey(self._result(converge), "fix it")
        self.assertIn("NO usable rows", honey)
        self.assertIn("(no rows matched)", honey)


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
        # discourages a needless re-locate loop, while requiring the author to confirm
        # the claimed mechanism against live source (N177 phantom-attribution guard)
        self.assertIn("do not loop back merely to re-locate", honey.lower())
        self.assertIn("confirm the claimed mechanism against the live source", honey.lower())

    def test_honey_renders_n180_dropped_peer_section(self):
        """A demoted (dropped-peer) convergence renders as a RE-EXAMINE target routing to
        needs_reinvestigation — NOT as a ready primary edit at the data-certified node."""
        converge = {
            "converged": False,
            "path": [{"node": "db_fn", "file": "db/workflow_sequences.py",
                      "lines": "45-57", "symbol": "build_module_query"}],
            "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                                  "lines": "45-57", "why": "query omits project_modules"},
            "causal_check": {
                "verdict": "consistent",
                "trace": "after the UNION rows return [N180 guard] ...",
                "dropped_peer": {"axis_id": "SEED_ANCHOR",
                                 "file": "api/workflow_head_routes.py", "lines": "93-102",
                                 "reason": "response key is module_id; FE reads module"}},
            "missing_link": None}
        honey = render_local_honey(self._result(converge, "fix"),
                                   "the module selector renders empty")
        self.assertIn("dropped a competing hypothesis", honey)
        self.assertIn("api/workflow_head_routes.py:93-102", honey)
        self.assertIn("needs_reinvestigation", honey)
        # must NOT be presented as a ready primary edit at the data-certified node
        self.assertNotIn("Primary edit target", honey)

    def test_honey_renders_m017_data_unstamped_section(self):
        """A demoted (data-unstamped) convergence renders as a CONFIRM-WITH-DATA target
        routing to needs_reinvestigation — NOT a ready primary edit at the assumed node."""
        converge = {
            "converged": False,
            "path": [{"node": "db_fn", "file": "db/workflow_sequences.py",
                      "lines": "45-57", "symbol": "get_effective_head"}],
            "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                                  "lines": "45-57", "why": "ORDER BY mis-ranks the head"},
            "causal_check": {
                "verdict": "consistent", "data_dependent": True, "data_unstamped": True,
                "data_state_assumptions": ["assumes M.result_doc_id is set"],
                "trace": "with result_doc_id set the CASE displaces the pending slot "
                         "[M017 data-stamp] ..."},
            "missing_link": None}
        honey = render_local_honey(self._result(converge, "fix"), "head off-by-one")
        self.assertIn("UNREAD stored value", honey)
        self.assertIn("needs_reinvestigation", honey)
        self.assertIn("assumed (UNREAD) data state", honey)
        # must NOT be presented as a ready primary edit at the assumed node
        self.assertNotIn("Primary edit target", honey)

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

    def test_contradicted_causal_section_warns_not_primary(self):
        converge = {
            "converged": False,
            "path": [{"node": "db_fn", "file": "db/workflow_sequences.py",
                      "lines": "45-57", "symbol": "get_effective_head"}],
            "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                                  "lines": "45-57", "why": "ORDER BY CASE"},
            "causal_check": {"verdict": "contradicted",
                             "data_state_assumptions": ["M, DS, D all result_doc_id NULL"],
                             "trace": "rows tie on CASE; sort_order orders M first",
                             "need_data_state": []},
            "missing_link": None}
        honey = render_local_honey(self._result(converge, "fix"), "fix the head")
        self.assertIn("CAUSAL CHECK did not confirm", honey)
        self.assertNotIn("Primary edit target", honey)
        self.assertIn("Do NOT author an edit", honey)
        self.assertIn("needs_reinvestigation", honey)

    def test_undecidable_causal_section_lists_need_data_state(self):
        converge = {
            "converged": False,
            "path": [],
            "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                                  "lines": "45-57", "why": "maybe"},
            "causal_check": {"verdict": "undecidable", "data_state_assumptions": [],
                             "trace": "depends on M.result_doc_id",
                             "need_data_state": ["rows for the failing doc: result_doc_id per slot"]},
            "missing_link": None}
        honey = render_local_honey(self._result(converge, "fix"), "fix it")
        self.assertIn("CAUSAL CHECK did not confirm", honey)
        self.assertIn("Data state / fixture required", honey)
        self.assertIn("result_doc_id per slot", honey)
        self.assertNotIn("Primary edit target", honey)

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


class TestConvergeLiveCodeGrounding(unittest.TestCase):
    """N177: converge rules on a tool-OFF view, so it once fabricated a code mechanism
    (a phantom argument mismatch) that the live source did not exhibit. The fix lifts the
    CURRENT source at the located loci and injects it as authoritative ground truth."""

    def _write(self, td, rel, text):
        import tempfile  # noqa: F401 (td provided by caller)
        path = os.path.join(td, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_lift_live_code_reads_cited_loci(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            self._write(td, "client/src/view.ts",
                        "l1\nl2\nfunction buildStepStates(a, b, c, allDone = false) {}\nl4\n")
            located = [_verdict("D", True, "client/src/view.ts", "3-3", "fe mapping")]
            block = C._lift_live_code(located, td)
            self.assertIn("client/src/view.ts", block)
            self.assertIn("allDone = false", block)   # the live signature is in the block
            self.assertIn("(live)", block)

    def test_lift_live_code_skips_missing_file_and_no_root(self):
        self.assertEqual(C._lift_live_code(
            [_verdict("X", True, "nope.ts", "1-1", "r")], "/does/not/exist"), "")
        self.assertEqual(C._lift_live_code(
            [_verdict("X", True, "a.ts", "1-1", "r")], None), "")

    def test_run_converge_injects_live_code_block(self):
        import tempfile
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONVERGED_OUT)

        with tempfile.TemporaryDirectory() as td:
            self._write(td, "api/workflow_head_routes.py",
                        "\n" * 92 + "def get_workflow_head(): return effective\n")
            self._write(td, "db/workflow_sequences.py",
                        "\n" * 44 + "def get_effective_head(): ORDER BY sort_order\n")
            with mock.patch.object(C, "call_worker", side_effect=fake):
                C.run_converge(seed_text="trace", verdicts=LOCATED_VERDICTS,
                               bundles=BUNDLES, provider="deepinfra", model="m",
                               code_root=td)
        self.assertIn("Confirmed code", prompts[0])
        self.assertIn("get_effective_head", prompts[0])
        self.assertIn("THIS block wins", prompts[0])

    def test_no_live_code_block_without_code_root(self):
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONVERGED_OUT)

        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="trace", verdicts=LOCATED_VERDICTS,
                           bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertNotIn("Confirmed code", prompts[0])


class TestConvergeHttpBridge(unittest.TestCase):
    """N183: the FE response-mapping and the BE getter that serves it are localised in
    different axes, but the edge between them is an HTTP request (no call-chain hop), so
    the holistic stitch reports a missing_link. The bridge resolves the FE fetch-URL to
    its BE route deterministically and hands converge that edge as FACT."""

    def _write(self, td, rel, text):
        path = os.path.join(td, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _scaffold(self, td):
        self._write(td, "client/src/NewRequirementModal.vue",
                    "<script>\nconst load = async () => {\n"
                    "  const resp = await getRequest('/api/v1/projects')\n"
                    "  projects.value = resp.projects\n}\n</script>\n")
        self._write(td, "server/routes.py",
                    "router = APIRouter(prefix=\"/api/v1\")\n\n\n"
                    "@router.get('/projects')\n"
                    "def get_projects_with_modules():\n"
                    "    return {'projects': query_projects()}\n")

    def test_bridge_resolves_fe_url_to_be_route(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            self._scaffold(td)
            located = [
                _verdict("FE", True, "client/src/NewRequirementModal.vue", "3-4",
                         "maps resp.projects"),
                _verdict("BE", True, "server/routes.py", "5-6", "getter"),
            ]
            block = C._http_binding_bridges(located, [], td)
            self.assertIn("/api/v1/projects", block)
            self.assertIn("server/routes.py", block)
            self.assertIn("FE client", block)

    def test_bridge_empty_without_code_root(self):
        self.assertEqual(C._http_binding_bridges([], [], None), "")

    def test_bridge_empty_when_no_fetch_url(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            self._write(td, "a.py", "def f():\n    return 1\n")
            located = [_verdict("X", True, "a.py", "1-2", "r")]
            self.assertEqual(C._http_binding_bridges(located, [], td), "")

    def test_run_converge_injects_http_edge_block(self):
        import tempfile
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONVERGED_OUT)

        with tempfile.TemporaryDirectory() as td:
            self._scaffold(td)
            verdicts = [
                _verdict("FE", True, "client/src/NewRequirementModal.vue", "3-4",
                         "maps resp.projects"),
                _verdict("BE", True, "server/routes.py", "5-6", "getter"),
            ]
            with mock.patch.object(C, "call_worker", side_effect=fake):
                C.run_converge(seed_text="selector empty", verdicts=verdicts,
                               bundles=[], provider="deepinfra", model="m", code_root=td)
        self.assertIn("HTTP request edges", prompts[0])
        self.assertIn("/api/v1/projects", prompts[0])
        self.assertIn("Do NOT emit a missing_link", prompts[0])

    def test_no_http_edge_block_without_binding(self):
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONVERGED_OUT)

        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="trace", verdicts=LOCATED_VERDICTS,
                           bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertNotIn("HTTP request edges", prompts[0])


# ── N179: seed-negation grounding + multiple independent defects ───────────────
# A genuine multi-locus convergence: the primary defect PLUS two SEPARATE ones in
# different code (the highlight mapping, the step-state computation, the status badge).
MULTI_LOCUS_OUT = json.dumps({
    "converged": True,
    "path": [
        {"node": "fe", "file": "client/src/main/DocWorkflow.vue", "lines": "80-90",
         "symbol": "stepClass"},
    ],
    "attributed_defect": {"node": "fe", "file": "client/src/app.css", "lines": "12-14",
                          "why": "the active-step selector paints blue, not the seed's "
                                 "corrected colour"},
    "additional_defects": [
        {"node": "fe", "file": "client/src/main/DocInfoPanel.vue", "lines": "40-52",
         "why": "the status badge never flips wf_in_progress → done"},
        {"node": "handler", "file": "server/workflow_decision_service.py", "lines": "70-95",
         "why": "the computed step index is off by one"},
    ],
    "causal_check": {"verdict": "consistent", "data_state_assumptions": [],
                     "trace": "app.css reproduces the selector symptom",
                     "counterfactual": "correcting the primary selector removes its "
                                       "reported symptom because it emits the right class",
                     "refuted_peers": [
                         {"file": "api/workflow_head_routes.py", "lines": "93-102",
                          "why_not": "not one of these independent UI defects"},
                         {"file": "db/workflow_sequences.py", "lines": "45-57",
                          "why_not": "not one of these independent UI defects"}],
                     "need_data_state": []},
    "missing_link": None,
})


class TestSeedNegationAndMultiLocus(unittest.TestCase):
    """N179: converge must not re-assert a seed-NEGATED value as the intent, and a
    scenario with several INDEPENDENT defects must surface each, not collapse to one."""

    def test_prompt_carries_seed_negation_grounding(self):
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONVERGED_OUT)

        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="yellow is a hallucination; it should be blue",
                           verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                           provider="deepinfra", model="m")
        self.assertIn("Seed is ground truth", prompts[0])
        self.assertIn("INVERTS the requirement", prompts[0])

    def test_prompt_carries_multi_locus_contract(self):
        prompts = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            prompts.append(prompt)
            return _wr(CONVERGED_OUT)

        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="three separate things are wrong",
                           verdicts=LOCATED_VERDICTS, bundles=BUNDLES,
                           provider="deepinfra", model="m")
        self.assertIn("additional_defects", prompts[0])
        self.assertIn("MULTIPLE INDEPENDENT loci", prompts[0])

    def test_additional_defects_parsed(self):
        with mock.patch.object(C, "call_worker", return_value=_wr(MULTI_LOCUS_OUT)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertTrue(res.converged)
        self.assertEqual(len(res.additional_defects), 2)
        files = {d["file"] for d in res.additional_defects}
        self.assertIn("client/src/main/DocInfoPanel.vue", files)
        self.assertIn("server/workflow_decision_service.py", files)

    def test_additional_defects_in_as_dict(self):
        with mock.patch.object(C, "call_worker", return_value=_wr(MULTI_LOCUS_OUT)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertIn("additional_defects", res.as_dict())
        self.assertEqual(len(res.as_dict()["additional_defects"]), 2)

    def test_additional_defect_unmatched_file_flagged_not_dropped(self):
        out = json.dumps({
            "converged": True,
            "path": [{"node": "fe", "file": "api/workflow_head_routes.py", "lines": "93-102"}],
            "attributed_defect": {"node": "fe", "file": "api/workflow_head_routes.py",
                                  "lines": "93-102", "why": "x"},
            "additional_defects": [
                {"node": "other", "file": "totally/unknown/thing.py", "lines": "1-2",
                 "why": "separate bug"}],
            "causal_check": {"verdict": "consistent", "data_state_assumptions": [],
                             "trace": "t",
                             "counterfactual": "correcting the handler removes the symptom "
                                               "because its response changes",
                             "refuted_peers": [
                                 {"file": "db/workflow_sequences.py", "lines": "45-57",
                                  "why_not": "not the response mechanism"}],
                             "need_data_state": []},
            "missing_link": None,
        })
        with mock.patch.object(C, "call_worker", return_value=_wr(out)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertEqual(len(res.additional_defects), 1)
        self.assertTrue(res.additional_defects[0].get("ungrounded"))

    def test_additional_defect_duplicating_primary_is_dropped(self):
        out = json.dumps({
            "converged": True,
            "path": [{"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57"}],
            "attributed_defect": {"node": "db_fn", "file": "db/workflow_sequences.py",
                                  "lines": "45-57", "why": "x"},
            "additional_defects": [
                {"node": "db_fn", "file": "db/workflow_sequences.py", "lines": "45-57",
                 "why": "same locus restated"}],
            "causal_check": {"verdict": "consistent", "data_state_assumptions": [],
                             "trace": "t",
                             "counterfactual": "correcting the query removes the symptom "
                                               "because its selected row changes",
                             "refuted_peers": [
                                 {"file": "api/workflow_head_routes.py", "lines": "93-102",
                                  "why_not": "the handler only forwards the row"}],
                             "need_data_state": []},
            "missing_link": None,
        })
        with mock.patch.object(C, "call_worker", return_value=_wr(out)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertEqual(res.additional_defects, [])

    def test_single_defect_has_empty_additional(self):
        with mock.patch.object(C, "call_worker", return_value=_wr(CONVERGED_OUT)):
            res = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                 bundles=BUNDLES, provider="deepinfra", model="m")
        self.assertEqual(res.additional_defects, [])

    def test_honey_renders_multi_locus_targets(self):
        result = {
            "verdicts": LOCATED_VERDICTS, "axes_judged": 2, "axes_total": 2,
            "seed_kind": "fix",
            "converge": json.loads(MULTI_LOCUS_OUT),
        }
        # carry the parsed-and-grounded converge dict the way investigate does
        with mock.patch.object(C, "call_worker", return_value=_wr(MULTI_LOCUS_OUT)):
            cres = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                  bundles=BUNDLES, provider="deepinfra", model="m")
        result["converge"] = cres.as_dict()
        honey = render_local_honey(result, "the colour, the index AND the badge are wrong")
        self.assertIn("MULTIPLE INDEPENDENT defects", honey)
        self.assertIn("Additional independent defects", honey)
        self.assertIn("## Converge-attributed edit targets", honey)
        self.assertIn("client/src/main/DocInfoPanel.vue", honey)
        self.assertIn("server/workflow_decision_service.py", honey)

    def test_honey_single_defect_emits_no_target_section(self):
        with mock.patch.object(C, "call_worker", return_value=_wr(CONVERGED_OUT)):
            cres = C.run_converge(seed_text="s", verdicts=LOCATED_VERDICTS,
                                  bundles=BUNDLES, provider="deepinfra", model="m")
        result = {"verdicts": LOCATED_VERDICTS, "axes_judged": 2, "axes_total": 2,
                  "seed_kind": "fix", "converge": cres.as_dict()}
        honey = render_local_honey(result, "one thing is wrong")
        self.assertNotIn("## Converge-attributed edit targets", honey)
        self.assertNotIn("Additional independent defects", honey)


class TestPremiseRefutedGuard(unittest.TestCase):
    """M035: a data-dependent ``consistent`` whose own read chain collapsed (a chained
    read found NO upstream rows → the premise rows are absent) must NOT stay converged —
    ``data_backed`` (satisfied by an incidental id-lookup) cannot see this."""

    def test_run_data_reads_flags_chain_broke_when_no_upstream(self):
        db = _tmp_db_with_doc("approved")  # documents holds only D1
        reads = [
            # parent read matches nothing → produces no upstream values
            {"id": "p", "table": "documents", "where": {"doc_id": "NOPE"},
             "columns": ["doc_id"]},
            # child chains on the empty parent → skipped (no upstream values)
            {"id": "c", "table": "documents",
             "where": {"doc_id": {"from": "p", "column": "doc_id"}},
             "columns": ["doc_review_status"]},
        ]
        block, any_rows, chain_broke = C._run_data_reads(reads, db, C._introspect_schema(db))
        self.assertFalse(any_rows)
        self.assertTrue(chain_broke)
        self.assertIn("no upstream values", block)

    def test_run_data_reads_chain_intact_is_not_broke(self):
        db = _tmp_db_with_doc("approved")  # D1 exists → parent yields a value to chain
        reads = [
            {"id": "p", "table": "documents", "where": {"doc_id": "D1"},
             "columns": ["doc_id"]},
            {"id": "c", "table": "documents",
             "where": {"doc_id": {"from": "p", "column": "doc_id"}},
             "columns": ["doc_review_status"]},
        ]
        _, any_rows, chain_broke = C._run_data_reads(reads, db, C._introspect_schema(db))
        self.assertTrue(any_rows)
        self.assertFalse(chain_broke)

    def _consistent_res(self, data_dependent=True):
        return C.ConvergeResult(
            converged=True,
            attributed_defect={"node": "n", "file": "q.json", "lines": "1-2"},
            causal_check={"verdict": "consistent", "data_dependent": data_dependent,
                          "trace": "ruled on the ordering"})

    def test_guard_demotes_consistent_when_chain_broke(self):
        res = C._premise_refuted_guard(self._consistent_res(), True, chain_broke=True)
        self.assertFalse(res.converged)
        self.assertTrue(res.causal_check["data_premise_refuted"])
        self.assertIn("premise refuted", res.summary)

    def test_guard_noop_when_chain_intact(self):
        res = C._premise_refuted_guard(self._consistent_res(), True, chain_broke=False)
        self.assertTrue(res.converged)
        self.assertNotIn("data_premise_refuted", res.causal_check)

    def test_guard_noop_when_not_data_dependent(self):
        # a pure code-logic consistent ruling is untouched even if a chain broke
        res = C._premise_refuted_guard(
            self._consistent_res(data_dependent=False), True, chain_broke=True)
        self.assertTrue(res.converged)

    def test_guard_noop_when_no_db(self):
        res = C._premise_refuted_guard(self._consistent_res(), False, chain_broke=True)
        self.assertTrue(res.converged)

    def test_guard_noop_on_contradicted(self):
        res = C.ConvergeResult(
            converged=False,
            attributed_defect={"node": "n", "file": "q.json", "lines": "1"},
            causal_check={"verdict": "contradicted", "data_dependent": True})
        out = C._premise_refuted_guard(res, True, chain_broke=True)
        self.assertNotIn("data_premise_refuted", out.causal_check)


class TestCounterfactualCompleteness(unittest.TestCase):
    """P0: consistent verdicts must earn certification and account for every peer."""

    LOCATED = [
        _verdict("A", True, "server/query.py", "10-20", "query hypothesis"),
        _verdict("B", True, "client/render.ts", "30-40", "binding hypothesis"),
    ]

    def _res(self, *, counterfactual="fixing query.py changes the selected row",
             refuted_peers=None):
        return C.ConvergeResult(
            converged=True,
            path=[{"node": "db_fn", "file": "server/query.py", "lines": "10-20"}],
            attributed_defect={"node": "db_fn", "file": "server/query.py",
                               "lines": "10-20"},
            causal_check={
                "verdict": "consistent",
                "trace": "query.py selects the wrong row",
                "counterfactual": counterfactual,
                "refuted_peers": refuted_peers or [],
            })

    def test_unrefuted_distinct_peer_is_demoted(self):
        out = C._counterfactual_complete_guard(self._res(), self.LOCATED)
        self.assertFalse(out.converged)
        self.assertEqual(out.causal_check["unrefuted_peer"]["file"], "client/render.ts")

    def test_explicitly_refuted_peer_stays_converged(self):
        peers = [{"file": "client/render.ts", "lines": "30-40",
                  "why_not": "the consumer reads the key emitted by the handler"}]
        out = C._counterfactual_complete_guard(
            self._res(refuted_peers=peers), self.LOCATED)
        self.assertTrue(out.converged)

    def test_empty_counterfactual_is_demoted(self):
        out = C._counterfactual_complete_guard(
            self._res(counterfactual="   ", refuted_peers=[
                {"file": "client/render.ts", "why_not": "not causal"}]), self.LOCATED)
        self.assertFalse(out.converged)
        self.assertTrue(out.causal_check["counterfactual_incomplete"])

    def test_kill_switch_disables(self):
        with mock.patch.dict(os.environ, {"HIVE_NO_COUNTERFACTUAL": "1"}):
            out = C._counterfactual_complete_guard(self._res(counterfactual=""), self.LOCATED)
        self.assertTrue(out.converged)

    def test_offpath_winning_peer_does_not_block(self):
        # TR909: the attributed locus is the /api/v1/projects producer; a synthetic
        # winning-path peer harvested from an UNRELATED url (/api/v1/document auth) must
        # NOT count as a competing locus — different endpoint, not this symptom's cause.
        located = [
            _verdict("HTTP_WINNING_PATH:/api/v1/projects", True,
                     "server/query.py", "10-20", "projects producer"),
            _verdict("HTTP_WINNING_PATH:/api/v1/document", True,
                     "server/auth_outbound.py", "38-70", "jwt verify on another path"),
        ]
        out = C._counterfactual_complete_guard(self._res(), located)
        self.assertTrue(out.converged)

    def test_samepath_winning_peer_still_blocks(self):
        # A competing winning-path peer on the SAME url is a real second locus → blocks.
        located = [
            _verdict("HTTP_WINNING_PATH:/api/v1/projects", True,
                     "server/query.py", "10-20", "projects producer"),
            _verdict("HTTP_WINNING_PATH:/api/v1/projects", True,
                     "server/other.py", "1-9", "competing producer on same url"),
        ]
        out = C._counterfactual_complete_guard(self._res(), located)
        self.assertFalse(out.converged)
        self.assertEqual(out.causal_check["unrefuted_peer"]["file"], "server/other.py")

    def test_offpath_filter_fail_open_when_attributed_not_on_http_path(self):
        # Attributed locus is NOT a lifted winning-path producer → URL set unknown →
        # fail open: a distinct synthetic peer is treated by legacy file-only behaviour.
        located = [
            _verdict("HTTP_WINNING_PATH:/api/v1/document", True,
                     "client/render.ts", "30-40", "unrelated synthetic peer"),
        ]
        out = C._counterfactual_complete_guard(self._res(), located)
        self.assertFalse(out.converged)

    def test_malformed_causal_check_never_raises(self):
        res = C.ConvergeResult(
            converged=True,
            attributed_defect={"file": "server/query.py", "lines": "10-20"},
            causal_check="not-a-dict")
        out = C._counterfactual_complete_guard(res, [None, {"verdict": "bad"}])
        self.assertFalse(out.converged)
        parsed = C._coerce_causal({
            "verdict": "consistent",
            "data_state_assumptions": 7,
            "need_data_state": {"bad": "shape"},
            "refuted_peers": "bad",
        })
        self.assertEqual(parsed["data_state_assumptions"], [])
        self.assertEqual(parsed["refuted_peers"], [])

    def test_prompt_and_parser_carry_new_contract(self):
        prompt = C.build_converge_prompt("s", self.LOCATED, [], [])
        self.assertIn("counterfactual", prompt)
        self.assertIn("refuted_peers", prompt)
        parsed = C._coerce_causal({
            "verdict": "consistent",
            "counterfactual": "fixing query.py removes the bad row",
            "refuted_peers": [{"file": "client/render.ts", "lines": "30-40",
                               "why_not": "binding agrees"}],
        })
        self.assertEqual(parsed["counterfactual"], "fixing query.py removes the bad row")
        self.assertEqual(parsed["refuted_peers"][0]["file"], "client/render.ts")


class TestFragmentFactCards(unittest.TestCase):
    """P3: deterministic cards expose producer and reachability facts to the model."""

    def test_producer_and_off_path_decoy_cards(self):
        producer = _verdict("FE_STRIP", True, "server/documents.py", "375-397", "r")
        decoy = _verdict("HEAD_SQL", True, "server/queries.json", "127-129", "r")
        bundles = [{
            "axis_id": "FE_STRIP",
            "code_snippets": [
                {"file": "server/router.py", "lines": "10-20",
                 "symbol": "documents_handler", "text": "return build_document()"},
            ],
            "call_chain": [
                {"file": "server/documents.py", "lines": "375-397",
                 "via": "field-producer", "field": "workflow_head_type",
                 "symbol": "build_document",
                 "text": 'out["workflow_head_type"] = head_type'},
            ],
        }, {
            "axis_id": "HEAD_SQL",
            "code_snippets": [
                {"file": "server/queries.json", "lines": "127-129",
                 "text": '"get_effective_head": "SELECT ..."'},
            ],
            "call_chain": [],
        }]
        cards = C._fragment_fact_cards([producer, decoy], [], bundles, None)
        self.assertIn("workflow_head_type", cards)
        self.assertIn("server/router.py documents_handler", cards)
        decoy_card = cards.split("- axis HEAD_SQL", 1)[1]
        self.assertIn("produces FE-bound field(s): (none)", decoy_card)
        self.assertIn("(not linked to any other located fragment)", decoy_card)

    def test_empty_input_is_empty(self):
        self.assertEqual(C._fragment_fact_cards([], [], [], None), "")

    def test_malformed_window_never_raises(self):
        located = [_verdict("A", True, "x.py", "1-2", "r")]
        cards = C._fragment_fact_cards(
            located, [None, {"file": 7, "via": "field-producer"}],
            [{"code_snippets": "bad", "call_chain": [None]}], None)
        self.assertIsInstance(cards, str)

    def test_prompt_places_fact_block_after_located_fragments(self):
        block = "[Fragment facts — deterministic annotations computed by the pipeline; " \
                "treat as FACT]\n- axis A / x.py:1-2"
        prompt = C.build_converge_prompt(
            "s", [_verdict("A", True, "x.py", "1-2", "r")], [], [],
            fragment_fact_block=block)
        self.assertLess(prompt.index("[Located fragments"), prompt.index("[Fragment facts"))
        self.assertIn("outranks a lexically-similar fragment", prompt)


class TestWinningHttpPathGrounding(unittest.TestCase):
    """T909: mounted winner -> response producer is deterministic converge evidence."""

    def _bundles(self):
        return [{
            "axis_id": "HTTP",
            "code_snippets": [],
            "call_chain": [
                {"via": "http-binding", "winning": True, "ambiguous": False,
                 "url": "/api/v1/projects", "verb": "get",
                 "file": "server/settings/project_settings.py", "lines": "45-49",
                 "symbol": "list_projects_endpoint", "text": "def list_projects_endpoint(): ..."},
                {"via": "http-producer", "winning": True,
                 "url": "/api/v1/projects", "verb": "get", "path_depth": 1,
                 "producer": True, "file": "server/db/projects.py", "lines": "19-27",
                 "symbol": "list_projects",
                 "text": "def list_projects(): return fetch('SELECT * FROM projects')"},
            ],
        }]

    def test_deepest_response_producer_becomes_located_locus(self):
        nodes = C._winning_http_path_nodes(self._bundles())
        loci = C._winning_producer_loci(nodes)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(loci), 1)
        self.assertEqual(loci[0]["verdict"]["file"], "server/db/projects.py")
        self.assertEqual(loci[0]["verdict"]["via"], "http-winning-path")

    def test_honey_emits_machine_readable_path_and_attribution(self):
        result = {
            "verdicts": [], "axes_judged": 0, "axes_total": 0, "seed_kind": "fix",
            "converge": {
                "converged": True,
                "winning_path": C._winning_http_path_nodes(self._bundles()),
                "path": [],
                "attributed_defect": {
                    "node": "db_fn", "file": "server/db/projects.py",
                    "lines": "19-27", "why": "modules omitted",
                },
                "additional_defects": [],
                "causal_check": {"verdict": "consistent", "trace": "projects.py omits modules"},
            },
        }
        honey = render_local_honey(result, "modules are missing")
        self.assertIn("hive-winning-http-path:", honey)
        self.assertIn('"role": "producer"', honey)
        self.assertIn("hive-converge-attribution:", honey)


class TestTraceGrounding(unittest.TestCase):
    """P4: consistent prose must engage the attributed file or a live-code symbol."""

    def _res(self, trace, counterfactual=""):
        return C.ConvergeResult(
            converged=True,
            attributed_defect={"node": "db_fn", "file": "server/query.py", "lines": "1-4"},
            causal_check={"verdict": "consistent", "trace": trace,
                          "counterfactual": counterfactual})

    def test_file_basename_keeps_convergence(self):
        out = C._trace_grounding_guard(
            self._res("query.py selects the stale row"), None)
        self.assertTrue(out.converged)

    def test_live_symbol_keeps_convergence(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "server"))
        with open(os.path.join(d, "server", "query.py"), "w", encoding="utf-8") as f:
            f.write("def select_effective_head(rows):\n    return rows[0]\n")
        out = C._trace_grounding_guard(
            self._res("select_effective_head returns the stale row"), d)
        self.assertTrue(out.converged)

    def test_generic_trace_is_demoted(self):
        out = C._trace_grounding_guard(
            self._res("the code is consistent with the symptom"), None)
        self.assertFalse(out.converged)
        self.assertIn("trace_ungrounded", out.causal_check)

    def test_kill_switch_disables(self):
        with mock.patch.dict(os.environ, {"HIVE_NO_TRACE_GROUNDING": "1"}):
            out = C._trace_grounding_guard(self._res("generic"), None)
        self.assertTrue(out.converged)

    def test_empty_trace_and_attribution_never_raise(self):
        res = C.ConvergeResult(
            converged=True, attributed_defect=None,
            causal_check={"verdict": "consistent", "trace": None,
                          "counterfactual": None})
        out = C._trace_grounding_guard(res, None)
        self.assertFalse(out.converged)
        self.assertIn("trace_ungrounded", out.causal_check)


class TestEvidenceSufficiency(unittest.TestCase):
    """P5: abstain on exactly-floor, wholly ungrounded consistent guesses."""

    LOCATED = [
        _verdict("A", True, "a.py", "1-2", "r"),
        _verdict("B", True, "b.py", "3-4", "r"),
    ]

    def _res(self):
        return C.ConvergeResult(
            converged=True,
            path=[{"node": "other", "file": "a.py", "lines": "1-2"}],
            attributed_defect={"node": "other", "file": "a.py", "lines": "1-2"},
            causal_check={"verdict": "consistent", "trace": "a.py emits the value",
                          "counterfactual": "fixing a.py removes the symptom",
                          "refuted_peers": [{"file": "b.py", "why_not": "not causal"}]})

    def test_floor_without_grounding_is_demoted(self):
        out = C._evidence_sufficiency_guard(
            self._res(), self.LOCATED, [],
            min_located=2, data_backed=False)
        self.assertFalse(out.converged)
        self.assertTrue(out.causal_check["low_confidence"])

    def test_field_producer_grounding_keeps_convergence(self):
        windows = [{"file": "a.py", "via": "field-producer", "field": "result_value"}]
        out = C._evidence_sufficiency_guard(
            self._res(), self.LOCATED, windows,
            min_located=2, data_backed=False)
        self.assertTrue(out.converged)

    def test_data_backing_keeps_convergence(self):
        out = C._evidence_sufficiency_guard(
            self._res(), self.LOCATED, [],
            min_located=2, data_backed=True)
        self.assertTrue(out.converged)

    def test_more_than_floor_is_untouched(self):
        located = self.LOCATED + [_verdict("C", True, "c.py", "5-6", "r")]
        out = C._evidence_sufficiency_guard(
            self._res(), located, [],
            min_located=2, data_backed=False)
        self.assertTrue(out.converged)

    def test_kill_switch_disables(self):
        with mock.patch.dict(os.environ, {"HIVE_NO_SUFFICIENCY_GATE": "1"}):
            out = C._evidence_sufficiency_guard(
                self._res(), self.LOCATED, [],
                min_located=2, data_backed=False)
        self.assertTrue(out.converged)

    def _arbiter(self, **kw):
        return C._causal_provenance_arbiter(
            self._res(), self.LOCATED,
            fp_windows=[], http_ds_windows=[], data_backed=False,
            data_chain_broke=False, db_available=False, code_root=None,
            windows=[], min_located=2, **kw)

    def test_arbiter_demotes_thin_holistic_floor(self):
        """Through the arbiter, a holistic floor-level guess still abstains."""
        self.assertFalse(self._arbiter(split_origin=False).converged)

    def test_arbiter_exempts_split_origin_from_sufficiency(self):
        """A split's per-locus elimination IS grounding — P5 must not demote it."""
        self.assertTrue(self._arbiter(split_origin=True).converged)


class TestAttributionStability(unittest.TestCase):
    """P2: already-produced holistic/split disagreement is a humility signal."""

    @staticmethod
    def _res(file):
        return C.ConvergeResult(
            converged=True,
            attributed_defect={"node": "other", "file": file, "lines": "1-2"},
            causal_check={"verdict": "consistent", "trace": f"{file} is causal"})

    def test_disagreement_demotes(self):
        out = C._attribution_stability_guard(
            self._res("server/query.py"), self._res("client/render.ts"))
        self.assertFalse(out.converged)
        self.assertIn("attribution_unstable", out.causal_check)

    def test_agreement_keeps_convergence(self):
        out = C._attribution_stability_guard(
            self._res("server/query.py"), self._res("C:/repo/server/query.py"))
        self.assertTrue(out.converged)

    def test_missing_comparison_is_noop(self):
        out = C._attribution_stability_guard(self._res("server/query.py"), None)
        self.assertTrue(out.converged)

    def test_kill_switch_disables(self):
        with mock.patch.dict(os.environ, {"HIVE_NO_STABILITY_CHECK": "1"}):
            out = C._attribution_stability_guard(
                self._res("server/query.py"), self._res("client/render.ts"))
        self.assertTrue(out.converged)


class TestFieldProvenanceGuard(unittest.TestCase):
    """M035 §4 head case: re-point a mis-attribution to the code that PRODUCES the
    FE-bound response field the symptom is about (field-producer provenance)."""

    DOCS = "server/modules/flow_gate/documents/routers/documents.py"
    DECOY = "server/sql/queries/queries.json"

    def _windows(self, producer_file=None, text="out[\"workflow_head_type\"] = head_type"):
        # one field-producer evidence window for the head field
        return [{"file": producer_file or self.DOCS, "lines": "378-394",
                 "via": "field-producer", "field": "workflow_head_type", "text": text}]

    def _located(self, *files):
        return [{"axis_id": f"AX{i}", "verdict": {"located": True, "file": f,
                 "lines": "378-394" if f == self.DOCS else "129-129", "reason": "r"}}
                for i, f in enumerate(files)]

    def _res_attr(self, file, lines="129-129", converged=True):
        return C.ConvergeResult(
            converged=converged,
            attributed_defect={"node": "n", "file": file, "lines": lines},
            causal_check={"verdict": "consistent", "data_dependent": True,
                          "trace": "data-certified on a sql read"})

    def test_repoints_decoy_to_producer(self):
        # converge attributed the seed-anchored decoy; the producer is a located peer.
        res = self._res_attr(self.DECOY)
        out = C._field_provenance_guard(
            res, self._located(self.DECOY, self.DOCS), self._windows())
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.DOCS))
        self.assertTrue(out.converged)
        self.assertIn("field_provenance_repointed", out.causal_check)
        self.assertIn("workflow_head_type",
                      out.causal_check["field_provenance_repointed"]["fields"])

    def test_overrides_a_demotion(self):
        # the structural fact outranks an upstream guard's converged=False demotion.
        res = self._res_attr(self.DECOY, converged=False)
        out = C._field_provenance_guard(
            res, self._located(self.DECOY, self.DOCS), self._windows())
        self.assertTrue(out.converged)
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.DOCS))

    def test_noop_when_already_at_producer(self):
        res = self._res_attr(self.DOCS, lines="378-394")
        out = C._field_provenance_guard(
            res, self._located(self.DECOY, self.DOCS), self._windows())
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.DOCS))
        self.assertNotIn("field_provenance_repointed", out.causal_check)

    def test_confirms_producer_restores_demoted_convergence(self):
        # attribution is ALREADY the producer but an upstream guard demoted it over an
        # unrelated peer → assert convergence on the grounded producer (job 2).
        res = self._res_attr(self.DOCS, lines="378-394", converged=False)
        out = C._field_provenance_guard(
            res, self._located(self.DECOY, self.DOCS), self._windows())
        self.assertTrue(out.converged)
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.DOCS))
        self.assertIn("field_provenance_confirmed", out.causal_check)
        self.assertNotIn("field_provenance_repointed", out.causal_check)

    def test_picks_field_rich_producer_not_incidental_one(self):
        # regression for the m035 misfire: a second, UNRELATED single-field producer
        # (pipeline_service fills `in_progress`) must NOT win over the symptom's
        # field-rich producer (documents.py fills the whole workflow_head_* family).
        PIPE = "server/modules/flow_gate/workflow/pipeline_service.py"
        fp = [
            {"file": self.DOCS, "lines": "378-394", "via": "field-producer",
             "field": "workflow_head_type", "text": 'out["workflow_head_type"] = h'},
            {"file": self.DOCS, "lines": "354-370", "via": "field-producer",
             "field": "workflow_head_status", "text": 'out["workflow_head_status"] = s'},
            {"file": PIPE, "lines": "288-304", "via": "field-producer",
             "field": "in_progress", "text": 'out["in_progress"] = x'},
        ]
        # converge correctly attributed the field-rich producer; guard must leave it.
        res = self._res_attr(self.DOCS, lines="375-397", converged=False)
        out = C._field_provenance_guard(
            res, self._located(self.DECOY, self.DOCS, PIPE), fp)
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.DOCS))
        self.assertTrue(out.converged)

    def test_noop_when_producer_not_located(self):
        # never invent a target: the producer must be a judge-located candidate.
        res = self._res_attr(self.DECOY)
        out = C._field_provenance_guard(res, self._located(self.DECOY), self._windows())
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.DECOY))

    def test_noop_when_no_field_producer_evidence(self):
        res = self._res_attr(self.DECOY)
        out = C._field_provenance_guard(res, self._located(self.DECOY, self.DOCS), [])
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.DECOY))

    def test_silent_when_decoy_on_producer_path(self):
        # the "bug is downstream of the producer" case: the producer READS the attributed
        # file (its stem appears in the producer window) → leave the attribution alone.
        store = "server/db/store.py"
        res = self._res_attr(store, lines="40-44")
        windows = self._windows(text="rows = store.get_modules()  # see store.py\n"
                                     "out[\"workflow_head_type\"] = head_type")
        out = C._field_provenance_guard(
            res, self._located(store, self.DOCS), windows)
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(store))

    def test_kill_switch_disables(self):
        os.environ["HIVE_NO_FIELD_PROVENANCE"] = "1"
        try:
            res = self._res_attr(self.DECOY)
            out = C._field_provenance_guard(
                res, self._located(self.DECOY, self.DOCS), self._windows())
            self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.DECOY))
        finally:
            del os.environ["HIVE_NO_FIELD_PROVENANCE"]


class TestHttpDatasourceProvenanceGuard(unittest.TestCase):
    """M036: a gated FE collection is filled through a real HTTP endpoint whose
    datasource hardcodes the field empty. Re-point off-path list_modules decoys to the
    endpoint datasource, using only resolved HTTP/call-chain evidence."""

    FE = "client/src/main/components/NewRequirementModal.vue"
    ROUTE = "server/modules/flow_gate/api/v1/legacy_misc_routes.py"
    SVC = "server/modules/flow_gate/process_service.py"
    DB = "server/modules/flow_gate/db.py"
    STORE = "server/modules/flow_gate/store.py"
    DECOY = "server/modules/flow_gate/api/v1/list_routes.py"

    def _windows(self):
        return [
            {"file": self.FE, "lines": "35-43", "text":
             '<div v-if="currentModules.length > 0" class="form-group">\n'
             '<option v-for="m in currentModules" :key="m.id" :value="m.id">'},
            {"file": self.FE, "lines": "239-306", "text":
             "const res = await getRequest<unknown>('/api/v1/projects')\n"
             "currentModules.value = selectedProject?.modules ?? []\n"},
            {"file": self.ROUTE, "lines": "82-87", "via": "http-binding", "text":
             "# RESOLVED BINDING (hive): GET /api/v1/projects <- client /api/v1/projects\n"
             "async def api_projects():\n"
             "    projects = process_service.get_projects_with_modules()\n"
             "    return {'projects': projects}\n"},
            {"file": self.SVC, "lines": "2090-2106", "via": "call-chain",
             "symbol": "get_projects_with_modules", "text":
             "def get_projects_with_modules() -> list[dict]:\n"
             "    allowed = db.get_allowed_projects()\n"
             "    for row in allowed:\n"
             "        m = (row.get('module') or '').strip()\n"
             "        if m:\n"
             "            project_map[p].append(m)\n"},
            {"file": self.DB, "lines": "218-221", "via": "call-chain",
             "symbol": "get_allowed_projects", "text":
             "def get_allowed_projects() -> list[dict]:\n"
             "    return _store.get_allowed_projects()\n"},
            {"file": self.STORE, "lines": "1021-1035", "via": "call-chain",
             "symbol": "get_allowed_projects", "text":
             "def get_allowed_projects(self) -> List[Dict[str, Any]]:\n"
             "    return conn.execute(\"SELECT project_id AS project, project_name, '' AS module\"\n"
             "                        \" FROM projects WHERE is_active = 1\").fetchall()\n"},
            {"file": self.DECOY, "lines": "101-144", "via": "call-chain",
             "symbol": "list_modules", "text":
             "def list_modules(request, p):\n"
             "    rows = store._fetch_all('SELECT DISTINCT module FROM groups WHERE project_id = ?', [p])\n"},
        ]

    def _located(self, *files):
        return [{"axis_id": f"AX{i}", "verdict": {"located": True, "file": f,
                 "lines": "1021-1035" if f == self.STORE else "101-144",
                 "reason": "r"}}
                for i, f in enumerate(files)]

    def _res_attr(self, file, lines="101-144", converged=True):
        return C.ConvergeResult(
            converged=converged,
            attributed_defect={"node": "n", "file": file, "lines": lines},
            causal_check={"verdict": "consistent", "data_dependent": False,
                          "trace": "seed-anchored on list_modules"})

    def test_repoints_list_modules_decoy_to_endpoint_datasource(self):
        res = self._res_attr(self.DECOY)
        out = C._http_datasource_provenance_guard(
            res, self._located(self.DECOY, self.STORE), self._windows())
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.STORE))
        self.assertTrue(out.converged)
        self.assertIn("http_datasource_provenance_repointed", out.causal_check)
        self.assertEqual(out.causal_check["http_datasource_provenance_repointed"]["field"],
                         "module")
        self.assertEqual(out.causal_check["http_datasource_provenance_repointed"]["url"],
                         "/api/v1/projects")

    def test_noop_when_datasource_not_located(self):
        res = self._res_attr(self.DECOY)
        out = C._http_datasource_provenance_guard(
            res, self._located(self.DECOY), self._windows())
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.DECOY))
        self.assertNotIn("http_datasource_provenance_repointed", out.causal_check)

    def test_noop_when_already_at_datasource(self):
        res = self._res_attr(self.STORE, lines="1021-1035")
        out = C._http_datasource_provenance_guard(
            res, self._located(self.DECOY, self.STORE), self._windows())
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.STORE))
        self.assertNotIn("http_datasource_provenance_repointed", out.causal_check)

    def test_confirms_demoted_datasource_attribution(self):
        res = self._res_attr(self.STORE, lines="1021-1035", converged=False)
        out = C._http_datasource_provenance_guard(
            res, self._located(self.DECOY, self.STORE), self._windows())
        self.assertTrue(out.converged)
        self.assertIn("http_datasource_provenance_confirmed", out.causal_check)

    def test_kill_switch_disables(self):
        os.environ["HIVE_NO_HTTP_DATASOURCE_PROVENANCE"] = "1"
        try:
            res = self._res_attr(self.DECOY)
            out = C._http_datasource_provenance_guard(
                res, self._located(self.DECOY, self.STORE), self._windows())
            self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.DECOY))
        finally:
            del os.environ["HIVE_NO_HTTP_DATASOURCE_PROVENANCE"]

    def _windows_fe(self, fe_assign: str):
        """Backend chain from _windows(), but with a custom FE assignment line so the edge
        extractor is exercised on shapes other than FlowGate's ``?.field ?? []`` (generality:
        Hive is a general engine, not a FlowGate-specific one)."""
        ws = self._windows()
        ws[1] = {"file": self.FE, "lines": "239-306", "text":
                 "const res = await getRequest('/api/v1/projects')\n" + fe_assign + "\n"}
        return ws

    def test_generalizes_to_chained_property_access(self):
        # Old regex grabbed the FIRST dotted token (``data``) and required a trailing ``[]``;
        # this shape (chained access, no ``[]``) must still resolve the real field ``modules``.
        out = C._http_datasource_provenance_guard(
            self._res_attr(self.DECOY), self._located(self.DECOY, self.STORE),
            self._windows_fe("currentModules.value = res.data.modules"))
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.STORE))
        self.assertEqual(out.causal_check["http_datasource_provenance_repointed"]["field"],
                         "module")

    def test_generalizes_via_variable_name_fallback(self):
        # No property access on the RHS at all (indirect local); the gated variable's own name
        # tail (``currentModules`` → ``modules`` → ``module``) carries the field.
        out = C._http_datasource_provenance_guard(
            self._res_attr(self.DECOY), self._located(self.DECOY, self.STORE),
            self._windows_fe("currentModules.value = mapped"))
        self.assertEqual(C._norm(out.attributed_defect["file"]), C._norm(self.STORE))
        self.assertEqual(out.causal_check["http_datasource_provenance_repointed"]["field"],
                         "module")


# ── M020 follow-up: per-locus SPLIT elimination converge ────────────────────────
def _focal_file(prompt: str) -> str | None:
    """Extract the FOCAL locus's file from a per-locus split prompt (None if holistic)."""
    marker = "[FOCAL LOCUS — rule on THIS one]\n"
    i = prompt.find(marker)
    if i < 0:
        return None
    return prompt[i + len(marker):].split(":", 1)[0].strip()


def _locus_out(verdict, data_dependent=False, why="x", reads=None, trace="t") -> str:
    return json.dumps({"verdict": verdict, "data_dependent": data_dependent,
                       "why": why, "trace": trace, "data_reads": reads or []})


def _is_holistic(prompt: str) -> bool:
    return "STITCH those fragments" in prompt


# A render-vs-data pair (the M035 shape): a backend SQL/ordering locus that the live data
# refutes, plus a front-end render locus that actually carries the bug.
SPLIT_VERDICTS = [
    _verdict("SQL", True, "db/workflow_sequences.py", "45-57",
             "ORDER BY may put a stale row first"),
    _verdict("FE", True, "client/src/workflow_view.ts", "160-170",
             "active step painted highlight (yellow) not current (blue)"),
]
SPLIT_BUNDLES = [
    {"axis_id": "SQL",
     "code_snippets": [{"file": "db/workflow_sequences.py", "lines": "45-57",
                        "text": "ORDER BY CASE WHEN result_doc_id IS NOT NULL ..."}],
     "call_chain": []},
    {"axis_id": "FE",
     "code_snippets": [{"file": "client/src/workflow_view.ts", "lines": "160-170",
                        "text": "head -> 'highlight'"}],
     "call_chain": []},
]


class TestSplitConverge(unittest.TestCase):
    """M020 follow-up: split the holistic stitch into one narrow cause→symptom question
    per located locus, then COMBINE by deterministic elimination. Adopts a result only on
    a clean elimination (exactly one survivor); anything else falls back to holistic."""

    def test_clean_elimination_attributes_lone_survivor(self):
        """Exactly one locus rules consistent → it is attributed; no holistic call."""
        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            ff = _focal_file(prompt)
            if ff == "db/workflow_sequences.py":
                return _wr(_locus_out("contradicted", why="data shows expected row"))
            if ff == "client/src/workflow_view.ts":
                return _wr(_locus_out("consistent", why="paints highlight not current"))
            return _wr(CONVERGED_OUT)  # holistic — must NOT be reached

        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="active step is yellow; should be blue",
                                 verdicts=SPLIT_VERDICTS, bundles=SPLIT_BUNDLES,
                                 provider="deepinfra", model="m", split_enabled=True)
        self.assertTrue(res.converged)
        self.assertEqual(res.attributed_defect["file"], "client/src/workflow_view.ts")
        self.assertIn("split elimination", res.summary)
        # the trace records HOW the competing locus was eliminated (auditable)
        self.assertIn("eliminated competing loci", res.causal_check["trace"])
        self.assertIn("db/workflow_sequences.py", res.causal_check["trace"])

    def test_no_survivor_falls_back_to_holistic(self):
        """Zero loci consistent → split abstains → holistic converge runs instead."""
        saw_holistic = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            if _is_holistic(prompt):
                saw_holistic.append(prompt)
                return _wr(CONVERGED_OUT)
            return _wr(_locus_out("contradicted"))

        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="s", verdicts=SPLIT_VERDICTS,
                                 bundles=SPLIT_BUNDLES, provider="deepinfra", model="m",
                                 split_enabled=True, min_located=1)
        self.assertTrue(saw_holistic, "holistic converge should run on a split abstain")
        # holistic CONVERGED_OUT attributes to db/workflow_sequences.py
        self.assertTrue(res.converged)
        self.assertEqual(res.attributed_defect["file"], "db/workflow_sequences.py")

    def test_multiple_survivors_falls_back_to_holistic(self):
        """≥2 loci consistent → ambiguous → abstain → holistic runs."""
        saw_holistic = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            if _is_holistic(prompt):
                saw_holistic.append(prompt)
                return _wr(CONVERGED_OUT)
            return _wr(_locus_out("consistent"))  # both survive

        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="s", verdicts=SPLIT_VERDICTS,
                           bundles=SPLIT_BUNDLES, provider="deepinfra", model="m",
                           split_enabled=True)
        self.assertTrue(saw_holistic, "ambiguous split must fall back to holistic")

    def test_over_cap_abstains_without_locus_calls(self):
        """More located loci than max_loci → cannot soundly eliminate → abstain WITHOUT
        spending per-locus calls (cheaper) and run holistic."""
        calls = {"locus": 0, "holistic": 0}

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            if _is_holistic(prompt):
                calls["holistic"] += 1
                return _wr(CONVERGED_OUT)
            calls["locus"] += 1
            return _wr(_locus_out("consistent"))

        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="s", verdicts=SPLIT_VERDICTS,
                           bundles=SPLIT_BUNDLES, provider="deepinfra", model="m",
                           split_enabled=True, split_max_loci=1)
        self.assertEqual(calls["locus"], 0, "no per-locus calls when over the cap")
        self.assertEqual(calls["holistic"], 1)

    def test_disabled_never_calls_locus(self):
        """split_enabled defaults False → only the holistic converge prompt is used."""
        focal_seen = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            if _focal_file(prompt):
                focal_seen.append(prompt)
            return _wr(CONVERGED_OUT)

        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="s", verdicts=SPLIT_VERDICTS,
                           bundles=SPLIT_BUNDLES, provider="deepinfra", model="m")
        self.assertEqual(focal_seen, [])

    def test_custom_split_model_is_used(self):
        """split_provider/model override the per-locus calls (scout-style model knob)."""
        seen = []

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            if _focal_file(prompt):
                seen.append((provider, model))
            return _wr(_locus_out("contradicted"))  # abstain → holistic

        with mock.patch.object(C, "call_worker", side_effect=fake):
            C.run_converge(seed_text="s", verdicts=SPLIT_VERDICTS,
                           bundles=SPLIT_BUNDLES, provider="deepinfra", model="big",
                           split_enabled=True, split_provider="openai",
                           split_model="cheap-120b")
        self.assertTrue(seen)
        self.assertTrue(all(p == ("openai", "cheap-120b") for p in seen))

    def test_premise_refuted_inline_makes_data_locus_lose(self):
        """The M035 end-to-end shape: the SQL locus rules consistent on a stored value but
        its read chain BREAKS (the rows are absent in the live DB) → inline premise-refuted
        flips it to contradicted; the FE render locus survives → defect attributed to FE."""
        chain_reads = [
            {"id": "p", "table": "documents", "where": {"doc_id": "NOPE"},
             "columns": ["doc_id"]},
            {"id": "c", "table": "documents",
             "where": {"doc_id": {"from": "p", "column": "doc_id"}},
             "columns": ["doc_review_status"]},
        ]

        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            ff = _focal_file(prompt)
            if ff == "db/workflow_sequences.py":
                # claims consistent on a stored value, naming a chain-breaking read; even
                # on the re-ask it (wrongly) insists consistent — the inline guard demotes it
                return _wr(_locus_out("consistent", data_dependent=True,
                                      reads=chain_reads, why="stale row puts M first"))
            if ff == "client/src/workflow_view.ts":
                return _wr(_locus_out("consistent", why="paints highlight not current"))
            return _wr(CONVERGED_OUT)  # holistic must NOT be reached

        db = _tmp_db_with_doc("approved")  # has only D1 → 'NOPE' parent read matches nothing
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="active step yellow; should be blue",
                                 verdicts=SPLIT_VERDICTS, bundles=SPLIT_BUNDLES,
                                 provider="deepinfra", model="m",
                                 code_root="/repo", db_conn=db, split_enabled=True)
        self.assertTrue(res.converged)
        self.assertEqual(res.attributed_defect["file"], "client/src/workflow_view.ts")

    def test_split_result_carries_through_guards_unharmed(self):
        """A clean code-logic survivor (not data_dependent) must survive the trailing
        dropped-peer / data-stamp guards (they must not demote a sound split result)."""
        def fake(provider, model, prompt, cwd=None, timeout=300, **kw):
            ff = _focal_file(prompt)
            if ff == "db/workflow_sequences.py":
                return _wr(_locus_out("contradicted"))
            if ff == "client/src/workflow_view.ts":
                return _wr(_locus_out("consistent", data_dependent=False))
            return _wr(CONVERGED_OUT)

        db = _tmp_db_with_doc("approved")
        with mock.patch.object(C, "call_worker", side_effect=fake):
            res = C.run_converge(seed_text="s", verdicts=SPLIT_VERDICTS,
                                 bundles=SPLIT_BUNDLES, provider="deepinfra", model="m",
                                 db_conn=db, split_enabled=True)
        self.assertTrue(res.converged)
        self.assertNotIn("data_unstamped", res.causal_check)
        self.assertNotIn("dropped_peer", res.causal_check)


if __name__ == "__main__":
    unittest.main()
