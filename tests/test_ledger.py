"""Unit tests for hive.ledger — SQLite telemetry ledger."""
import os, sqlite3, sys, tempfile, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive.ledger import Ledger, NullLedger, open_ledger, estimate_tokens


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_short_english(self):
        result = estimate_tokens("hello world foo bar")
        self.assertGreater(result, 0)

    def test_long_text_char_based(self):
        text = "a" * 400
        result = estimate_tokens(text)
        self.assertGreaterEqual(result, 100)


class TestLedgerTablesCreated(unittest.TestCase):
    """Tables are created idempotently."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_ledger.db")

    def test_tables_created(self):
        ldg = Ledger(self.db_path)
        ldg.close()
        conn = sqlite3.connect(self.db_path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        self.assertIn("runs", tables)
        self.assertIn("worker_calls", tables)

    def test_tables_idempotent(self):
        """Opening twice doesn't error."""
        Ledger(self.db_path).close()
        Ledger(self.db_path).close()


class TestLedgerInsertAndAggregate(unittest.TestCase):
    """Insert run + worker_calls; verify aggregates."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_ledger.db")
        self.ldg = Ledger(self.db_path)
        self.ldg.start_run(seed="test_seed.md", codebase="/repo",
                           model_queen="gpt-5-mini", model_swarm="gpt-5-mini")

    def tearDown(self):
        self.ldg.close()

    def test_run_row_exists(self):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT * FROM runs").fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)

    def test_run_status_running(self):
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT status FROM runs").fetchone()
        conn.close()
        self.assertEqual(row[0], "running")

    def test_record_call_inserts_row(self):
        prompt = "hello world " * 10
        output = "result text " * 5
        self.ldg.record_call("queen", "decompose", "copilot", "gpt-5-mini",
                              prompt=prompt, output=output, latency_s=1.5)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT * FROM worker_calls").fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)

    def test_in_chars_exact(self):
        prompt = "x" * 50
        output = "y" * 30
        self.ldg.record_call("swarm", "A", "copilot", "gpt-5-mini",
                              prompt=prompt, output=output, latency_s=0.5)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT in_chars, out_chars FROM worker_calls").fetchone()
        conn.close()
        self.assertEqual(row[0], 50)
        self.assertEqual(row[1], 30)

    def test_est_tokens_populated(self):
        prompt = "word " * 100
        output = "result " * 50
        self.ldg.record_call("swarm", "B", "copilot", "gpt-5-mini",
                              prompt=prompt, output=output, latency_s=1.0)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT est_tokens FROM worker_calls").fetchone()
        conn.close()
        self.assertIsNotNone(row[0])
        self.assertGreater(row[0], 0)

    def test_real_tokens_null(self):
        """real_tokens is always NULL for copilot (CLI exposes no token counts)."""
        self.ldg.record_call("swarm", "C", "copilot", "gpt-5-mini",
                              prompt="test", output="out", latency_s=0.1)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT real_tokens FROM worker_calls").fetchone()
        conn.close()
        self.assertIsNone(row[0])

    def test_real_tokens_stored_when_provided(self):
        """A provider that reports usage (e.g. deepinfra) populates real_tokens."""
        self.ldg.record_call("judge", "A", "deepinfra", "openai/gpt-oss-120b",
                             prompt="p", output="o", latency_s=1.0, real_tokens=206)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT real_tokens FROM worker_calls").fetchone()
        conn.close()
        self.assertEqual(row[0], 206)

    def test_finish_run_total_real_tokens_sums_reported(self):
        """total_real_tokens sums only calls that reported tokens; copilot adds nothing."""
        self.ldg.record_call("judge", "A", "deepinfra", "openai/gpt-oss-120b",
                             prompt="p", output="o", latency_s=1.0, real_tokens=200)
        self.ldg.record_call("judge", "B", "deepinfra", "openai/gpt-oss-120b",
                             prompt="p", output="o", latency_s=1.0, real_tokens=50)
        self.ldg.record_call("swarm", "C", "copilot", "gpt-5-mini",
                             prompt="p", output="o", latency_s=1.0)  # no real_tokens
        self.ldg.finish_run()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT total_real_tokens FROM runs").fetchone()
        conn.close()
        self.assertEqual(row[0], 250)

    def test_finish_run_total_real_tokens_null_when_none_reported(self):
        """A copilot-only run leaves total_real_tokens NULL (prior behavior preserved)."""
        self.ldg.record_call("swarm", "A", "copilot", "gpt-5-mini",
                             prompt="p", output="o", latency_s=1.0)
        self.ldg.finish_run()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT total_real_tokens FROM runs").fetchone()
        conn.close()
        self.assertIsNone(row[0])

    def test_finish_run_updates_status(self):
        self.ldg.finish_run(honey_path="/out/honey.md", axes_n=3, rounds=1,
                            conflicts_n=2, remaining_n=0, parse_errs=0, status="done")
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT status, axes_n, honey_path FROM runs").fetchone()
        conn.close()
        self.assertEqual(row[0], "done")
        self.assertEqual(row[1], 3)
        self.assertEqual(row[2], "/out/honey.md")

    def test_record_local_inserts_local_row(self):
        """A local (free) step lands as provider='local' with the mechanism + note."""
        self.ldg.record_local("retrieve", "A1", mechanism="ripgrep",
                              detail="hits=12 snippets=4", out_chars=900, latency_s=0.03)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT provider, model, stage, out_chars, comb_path, ok,"
                           " status FROM worker_calls").fetchone()
        conn.close()
        self.assertEqual(row[0], "local")
        self.assertEqual(row[1], "ripgrep")
        self.assertEqual(row[2], "retrieve")
        self.assertEqual(row[3], 900)
        self.assertEqual(row[4], "hits=12 snippets=4")
        self.assertEqual(row[5], 1)
        self.assertEqual(row[6], "done")

    def test_record_local_excluded_from_cost_aggregate(self):
        """Local rows are visible but never inflate the run's token/char totals."""
        self.ldg.record_call("judge", "A", "deepinfra", "m",
                             prompt="p" * 40, output="o" * 20, latency_s=1.0,
                             real_tokens=100)
        self.ldg.record_local("db_read", "converge", mechanism="sqlite",
                             out_chars=5000, latency_s=0.01)
        self.ldg.finish_run()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT total_in_chars, total_out_chars, total_real_tokens"
                           " FROM runs").fetchone()
        n_rows = conn.execute("SELECT COUNT(*) FROM worker_calls").fetchone()[0]
        conn.close()
        # Both rows are present, but only the model call feeds the cost totals.
        self.assertEqual(n_rows, 2)
        self.assertEqual(row[0], 40)    # in_chars: judge only, local excluded
        self.assertEqual(row[1], 20)    # out_chars: judge only, local's 5000 excluded
        self.assertEqual(row[2], 100)   # real_tokens: judge only

    def test_record_local_noop_without_run(self):
        """No run started → no _run_id → record_local silently no-ops (no crash)."""
        ldg = Ledger(self.db_path + ".x")
        ldg.record_local("retrieve", "A1")  # no start_run
        conn = sqlite3.connect(self.db_path + ".x")
        n = conn.execute("SELECT COUNT(*) FROM worker_calls").fetchone()[0]
        conn.close()
        ldg.close()
        self.assertEqual(n, 0)

    def test_finish_run_aggregates(self):
        self.ldg.record_call("swarm", "A", "copilot", "gpt-5-mini",
                              prompt="a"*100, output="b"*50, latency_s=1.0)
        self.ldg.record_call("swarm", "B", "copilot", "gpt-5-mini",
                              prompt="c"*200, output="d"*100, latency_s=2.0)
        self.ldg.finish_run()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT total_in_chars, total_out_chars FROM runs").fetchone()
        conn.close()
        self.assertEqual(row[0], 300)
        self.assertEqual(row[1], 150)


class TestLedgerBeginFinishCall(unittest.TestCase):
    """begin_call inserts a 'running' row at start; finish_call completes it."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_ledger.db")
        self.ldg = Ledger(self.db_path)
        self.ldg.start_run(seed="s.md", codebase="/r",
                           model_queen="gpt-5-mini", model_swarm="gpt-5-mini")

    def tearDown(self):
        self.ldg.close()

    def _row(self, call_id):
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT status, started_at, in_chars, out_chars, est_tokens, ok, err"
            " FROM worker_calls WHERE id=?", (call_id,)).fetchone()
        conn.close()
        return row

    def test_begin_call_returns_id(self):
        call_id = self.ldg.begin_call("queen", "decompose", "copilot", "gpt-5-mini",
                                      prompt="x" * 40)
        self.assertIsNotNone(call_id)

    def test_begin_row_is_wait_with_started_at(self):
        # The row starts at 'wait' (registered, not yet executing) — it only
        # becomes 'running' once the handler owns its slot (mark_running). This is
        # what stops a codex call parked on the serialization lock from showing a
        # false 'running'.
        call_id = self.ldg.begin_call("queen", "decompose", "copilot", "gpt-5-mini",
                                      prompt="x" * 40)
        status, started_at, in_chars, out_chars, est, ok, err = self._row(call_id)
        self.assertEqual(status, "wait")
        self.assertTrue(started_at)            # ISO timestamp present
        self.assertEqual(in_chars, 40)         # fixed at begin
        self.assertIsNone(out_chars)           # not yet known
        self.assertIsNone(ok)

    def test_mark_running_flips_wait_to_running(self):
        call_id = self.ldg.begin_call("specify", "specify", "codex", "gpt-5.4-mini",
                                      prompt="x" * 40)
        self.assertEqual(self._row(call_id)[0], "wait")
        self.ldg.mark_running(call_id)
        self.assertEqual(self._row(call_id)[0], "running")

    def test_mark_running_does_not_clobber_finished(self):
        # A late/lost wakeup must never resurrect a completed row back to running.
        call_id = self.ldg.begin_call("specify", "specify", "codex", "gpt-5.4-mini",
                                      prompt="x" * 40)
        self.ldg.finish_call(call_id, output="o" * 10, latency_s=1.0)
        self.assertEqual(self._row(call_id)[0], "done")
        self.ldg.mark_running(call_id)         # arrives after finish — must be a no-op
        self.assertEqual(self._row(call_id)[0], "done")

    def test_mark_running_none_id_noop(self):
        self.ldg.mark_running(None)            # ledger unavailable / begin failed

    def test_finish_call_marks_done(self):
        call_id = self.ldg.begin_call("converge", "converge", "copilot", "gpt-5-mini",
                                      prompt="word " * 20)
        self.ldg.finish_call(call_id, output="out " * 10, latency_s=2.5)
        status, _started, in_chars, out_chars, est, ok, err = self._row(call_id)
        self.assertEqual(status, "done")
        self.assertEqual(out_chars, len("out " * 10))
        self.assertEqual(ok, 1)
        self.assertGreater(est, 0)             # prompt-est + output-est

    def test_finish_call_failed_status(self):
        call_id = self.ldg.begin_call("converge", "converge", "codex", "gpt-5.4-mini",
                                      prompt="p" * 100)
        self.ldg.finish_call(call_id, output="", latency_s=0.0, ok=False,
                             err="timed out after 180 seconds")
        status, _s, in_chars, out_chars, _est, ok, err = self._row(call_id)
        self.assertEqual(status, "failed")     # the §6 codex-timeout case
        self.assertEqual(ok, 0)
        self.assertIn("timed out", err)
        self.assertEqual(in_chars, 100)        # input size survives even on timeout

    def test_finish_call_none_id_noop(self):
        # A None call_id (ledger unavailable / begin failed) must not raise.
        self.ldg.finish_call(None, output="o", latency_s=1.0)

    def test_begin_finish_feed_run_aggregate(self):
        cid = self.ldg.begin_call("swarm", "A", "copilot", "gpt-5-mini", prompt="a" * 100)
        self.ldg.finish_call(cid, output="b" * 50, latency_s=1.0)
        self.ldg.finish_run()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT total_in_chars, total_out_chars FROM runs").fetchone()
        conn.close()
        self.assertEqual(row[0], 100)
        self.assertEqual(row[1], 50)

    def test_record_call_still_one_shot_done(self):
        """The record_call wrapper begins+finishes in one shot → a 'done' row."""
        self.ldg.record_call("assemble", "assemble", "copilot", "gpt-5-mini",
                             prompt="p" * 20, output="o" * 10, latency_s=1.0)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT status, in_chars, out_chars FROM worker_calls").fetchone()
        conn.close()
        self.assertEqual(row[0], "done")
        self.assertEqual(row[1], 20)
        self.assertEqual(row[2], 10)


class TestLedgerMigration(unittest.TestCase):
    """An older worker_calls table (no status/started_at) is migrated on open."""

    def test_legacy_db_gets_new_columns(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "legacy.db")
        # Build a pre-migration worker_calls table lacking status/started_at.
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE worker_calls (id INTEGER PRIMARY KEY, run_id INTEGER,"
            " stage TEXT, axis_id TEXT, provider TEXT, model TEXT,"
            " in_chars INTEGER, out_chars INTEGER, est_tokens INTEGER,"
            " real_tokens INTEGER, latency_s REAL, comb_path TEXT, ok INTEGER, err TEXT)")
        conn.commit()
        conn.close()
        # Opening the Ledger should ALTER in the missing columns without error.
        ldg = Ledger(db_path)
        ldg.start_run(seed="s", codebase="/r", model_queen="m", model_swarm="m")
        cid = ldg.begin_call("queen", "decompose", "copilot", "m", prompt="x" * 10)
        ldg.finish_call(cid, output="y" * 5, latency_s=1.0)
        ldg.close()
        conn = sqlite3.connect(db_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(worker_calls)")}
        row = conn.execute("SELECT status, started_at FROM worker_calls").fetchone()
        conn.close()
        self.assertIn("status", cols)
        self.assertIn("started_at", cols)
        self.assertEqual(row[0], "done")
        self.assertTrue(row[1])


class TestLedgerDisabledIsNoop(unittest.TestCase):
    """When ledger.enabled=False, NullLedger is returned and no DB is created."""

    def test_null_ledger_returned(self):
        ldg = open_ledger(enabled=False, db_path="/nonexistent/path.db")
        self.assertIsInstance(ldg, NullLedger)

    def test_null_ledger_noop(self):
        ldg = NullLedger()
        ldg.start_run("seed", "codebase", "q-model", "s-model")
        cid = ldg.begin_call("stage", "ax", "copilot", "m", prompt="p")
        self.assertIsNone(cid)
        ldg.finish_call(cid, output="o", latency_s=1.0)
        ldg.record_call("stage", "ax", "copilot", "m", prompt="p", output="o", latency_s=1.0)
        ldg.finish_run(status="done")
        ldg.close()
        # No exception = pass


class TestLedgerSqliteFailureNoRaise(unittest.TestCase):
    """SQLite errors in record_call/finish_run must not raise."""

    def test_no_raise_on_bad_db(self):
        ldg = Ledger.__new__(Ledger)
        ldg._conn = None
        ldg._run_id = None
        ldg._calls = []
        ldg._start_ts = 0
        # All methods must silently no-op
        ldg.record_call("s", "a", "copilot", "m", prompt="p", output="o", latency_s=0.1)
        ldg.finish_run(status="done")
        ldg.close()


if __name__ == "__main__":
    unittest.main()
