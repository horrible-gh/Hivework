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


class TestLedgerDisabledIsNoop(unittest.TestCase):
    """When ledger.enabled=False, NullLedger is returned and no DB is created."""

    def test_null_ledger_returned(self):
        ldg = open_ledger(enabled=False, db_path="/nonexistent/path.db")
        self.assertIsInstance(ldg, NullLedger)

    def test_null_ledger_noop(self):
        ldg = NullLedger()
        ldg.start_run("seed", "codebase", "q-model", "s-model")
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
