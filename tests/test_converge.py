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
        "trace": "with M.result_doc_id set the CASE puts M first, displacing the "
                 "pending slot — reproduces the off-by-one",
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
        "trace": "ordering by item_seq puts DS first, painting the memo done — "
                 "reproduces the observed skip.",
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
        "trace": "reproduces the off-by-one",
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
                             "trace": "reproduces", "need_data_state": []},
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
        "trace": "with result_doc_id set the CASE displaces the pending slot",
        "need_data_state": [],
        "data_reads": [{"table": "documents", "where": {"doc_id": "D1"},
                        "columns": ["doc_review_status"]}]},
    "missing_link": None,
})


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
        block, any_rows = C._run_data_reads(reads, self.db, self.schema)
        self.assertIn("unknown table 'items'", block)
        self.assertFalse(any_rows)

    def test_unknown_column_rejected(self):
        # the NR174 hallucination: WHERE column = 'result_doc_id'
        reads = [{"id": "r", "table": "documents", "where": {"column": "result_doc_id"},
                  "columns": ["doc_review_status"]}]
        block, any_rows = C._run_data_reads(reads, self.db, self.schema)
        self.assertIn("unknown column 'column'", block)
        self.assertFalse(any_rows)

    def test_valid_read_passes_guard(self):
        reads = [{"id": "r", "table": "documents", "where": {"doc_id": "D1"},
                  "columns": ["doc_review_status"]}]
        block, any_rows = C._run_data_reads(reads, self.db, self.schema)
        self.assertTrue(any_rows)
        self.assertIn("approved", block)

    def test_empty_schema_is_noop_guard(self):
        # no schema (introspection failed) → guard cannot reject, read still runs
        reads = [{"id": "r", "table": "documents", "where": {"doc_id": "D1"},
                  "columns": ["doc_review_status"]}]
        block, any_rows = C._run_data_reads(reads, self.db, {})
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


if __name__ == "__main__":
    unittest.main()
